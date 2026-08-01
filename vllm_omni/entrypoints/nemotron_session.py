# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Transport-neutral engine binding for one Nemotron realtime session.

The binding owns the direct ``AsyncOmni.generate`` integration while
the model-local session remains the sole authority for cadence
segmentation, carrier receipts, locale selection, and final-tail
construction. Callers submit arbitrary normalized pieces and never
predict how many model updates a piece completes.

All vLLM-coupled imports are lazy so the lifecycle contract remains
GPU-free unit-testable.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Awaitable, Callable
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable
from uuid import uuid4

if TYPE_CHECKING:
    import numpy as np
    import numpy.typing as npt

    from vllm_omni.model_executor.models.nemotron_asr.session import (
        NemotronRealtimeSession,
        ReceiptLedger,
    )

    FloatSamples = npt.NDArray[np.float32]

Render = Callable[[Any], Awaitable[Any]]
AcceptanceCallback = Callable[[int], None]


@runtime_checkable
class SessionLease(Protocol):
    """Transport-neutral lifecycle for one engine session."""

    async def feed(
        self,
        samples: FloatSamples,
        *,
        on_accepted: AcceptanceCallback | None = None,
    ) -> list[str]:
        """Submit one whole piece and return completed hypotheses."""
        ...

    async def flush(self) -> str:
        """Drain the final-tail and return the final transcript."""
        ...

    async def update_locale(self, locale: str) -> None:
        """Select the locale used by the next carrier mint."""
        ...

    async def abort(self) -> None:
        """Terminate an abnormal session."""
        ...

    async def finish(self) -> None:
        """Complete normal engine cleanup."""
        ...

    async def release(self) -> None:
        """Release the lease idempotently after terminal cleanup."""
        ...


@runtime_checkable
class SessionFactory(Protocol):
    """Construct transport-neutral sessions at caller-selected geometry."""

    async def open(self, *, cadence: str, locale: str) -> SessionLease:
        """Open one engine session after validating model controls."""
        ...


def _render_factory(engine: Any) -> Render:
    """Build the prompt-to-``StreamingInput`` render callable."""
    from vllm.engine.protocol import StreamingInput
    from vllm.renderers.inputs.preprocess import parse_model_prompt

    model_config = engine.model_config
    renderer = engine.renderer

    async def render(prompt: Any) -> Any:
        parsed = parse_model_prompt(model_config, prompt)
        (engine_input,) = await renderer.render_cmpl_async([parsed])
        return StreamingInput(prompt=engine_input)

    return render


def _streaming_sampling_params(engine: Any) -> list[Any]:
    """Coerce the engine defaults through the realtime sampling path."""
    from vllm_omni.entrypoints.utils import coerce_param_message_types

    coerced: list[Any] = coerce_param_message_types(list(engine.default_sampling_params_list), is_streaming=True)
    return coerced


# @spec PORT-RTC-002, PORT-RTC-004, PORT-RTC-005, PORT-RTC-006
class NemotronSessionLease:
    """One model session bound to one long-lived engine request.

    ``feed`` awaits the model-owned :class:`PieceReceipt`, not cadence
    arithmetic. Its optional callback fires immediately after that
    receipt acknowledges the submitted piece, before the receipt's
    carrier tickets are awaited. The background consumer completes one
    ticket at each observed park and fails the ledger on every engine,
    render, or premature-end failure.
    """

    def __init__(
        self,
        *,
        engine: Any,
        session: NemotronRealtimeSession,
        request_id: str,
        render: Render | None = None,
    ) -> None:
        """Bind one validated model-local session to the engine."""
        ledger = session.ledger
        if ledger is None:
            raise ValueError(
                "the session must be minted with_ledger=True: the "
                "receipt ledger is the feed/park synchronization "
                "authority (PORT-RTC-002)"
            )
        self._engine = engine
        self._session = session
        self._ledger: ReceiptLedger = ledger
        self._request_id = request_id
        self._render = render
        self._park_id = session.park_token_id
        self._audio: asyncio.Queue[Any] = asyncio.Queue()
        self._input_stream: asyncio.Queue[list[int]] = asyncio.Queue()
        self._task: asyncio.Task[None] | None = None
        self._done = asyncio.Event()
        self._text = ""
        self._error: BaseException | None = None
        self._ended = False
        self._audio_closed = False
        self._flush_parked = False
        self._aborted = False
        self._released = False

    @property
    def session(self) -> NemotronRealtimeSession:
        """The validated model-local session."""
        return self._session

    @property
    def request_id(self) -> str:
        """The engine request id driven by this lease."""
        return self._request_id

    async def feed(
        self,
        samples: FloatSamples,
        *,
        on_accepted: AcceptanceCallback | None = None,
    ) -> list[str]:
        """Submit one whole piece and await its completed carriers.

        Args:
            samples: One contiguous normalized audio piece.
            on_accepted: Optional nonblocking notification invoked once
                with the accepted sample count immediately after the
                model's piece receipt arrives.

        Returns:
            One cumulative hypothesis per completed carrier, in order.
        """
        self._ensure_started()
        self._require_live()
        self._audio.put_nowait(samples)
        receipt = await self._ledger.next_piece()
        if on_accepted is not None:
            on_accepted(len(samples))

        results: list[str] = []
        try:
            for ticket in receipt.tickets:
                results.append(await ticket.done)
        finally:
            for ticket in receipt.tickets:
                if ticket.done.done() and not ticket.done.cancelled():
                    ticket.done.exception()
        return results

    async def flush(self) -> str:
        """Drain the final-tail and return all terminal output."""
        self._ensure_started()
        if self._aborted:
            raise RuntimeError("the session was aborted")
        if self._done.is_set() and not self._audio_closed:
            raise self._error or RuntimeError("generation ended before the session's normal finalization")
        await self._drain()
        return self._text

    async def update_locale(self, locale: str) -> None:
        """Validate and select the locale for the next carrier mint."""
        self._session.select_prompt(locale)

    async def abort(self) -> None:
        """Abort the request and fail every blocked ledger waiter."""
        if self._aborted:
            return
        self._aborted = True
        self._ledger.fail(RuntimeError("the session was aborted"))
        if self._task is None:
            return
        ended = self._ended
        if not self._task.done():
            self._task.cancel()
        await asyncio.gather(self._task, return_exceptions=True)
        if not ended:
            await self._engine.abort(self._request_id)

    async def finish(self) -> None:
        """Complete normal cleanup, draining first when necessary."""
        if self._task is None or self._aborted:
            return
        await self._drain()

    async def release(self) -> None:
        """Release this lease exactly once.

        The current engine exposes no separate provider reservation
        handle. Keeping this idempotent operation in the contract lets a
        future provider-owned reservation attach here without changing
        transports or the model lifecycle.
        """
        if self._released:
            return
        self._released = True

    def _ensure_started(self) -> None:
        """Start the background engine consumer on first use."""
        if self._task is None:
            self._task = asyncio.get_running_loop().create_task(self._consume())

    def _require_live(self) -> None:
        """Reject work no consumer can acknowledge."""
        if self._aborted:
            raise RuntimeError("the session was aborted")
        if self._audio_closed:
            raise RuntimeError("audio is closed; the session is finalizing")
        if self._done.is_set():
            raise self._error or RuntimeError("generation ended before the session's normal finalization")

    async def _drain(self) -> None:
        """Close accepted audio and wait for generation end."""
        if not self._audio_closed:
            self._audio_closed = True
            self._audio.put_nowait(None)
        await self._done.wait()
        if self._task is not None:
            await asyncio.gather(self._task, return_exceptions=True)
        if self._error is not None:
            raise self._error

    async def _audio_frames(self) -> AsyncIterator[Any]:
        """Yield accepted pieces until the close marker."""
        while True:
            frame = await self._audio.get()
            if frame is None:
                return
            yield frame

    async def _rendered(self, render: Render) -> AsyncIterator[Any]:
        """Segment and render model-minted prompts."""
        from vllm_omni.model_executor.models.nemotron_asr.streaming import (
            buffer_stream,
        )

        prompts = buffer_stream(self._audio_frames(), self._input_stream, self._session)
        async for prompt in prompts:
            yield await render(prompt)

    async def _consume(self) -> None:
        """Drive generation and resolve ledger tickets at legal parks."""
        try:
            render = self._render or _render_factory(self._engine)
            outputs = self._engine.generate(
                prompt=self._rendered(render),
                request_id=self._request_id,
                sampling_params_list=_streaming_sampling_params(self._engine),
            )
            async for output in outputs:
                if getattr(output, "stage_id", 0) not in (None, 0):
                    continue
                stage_outputs = getattr(output, "outputs", None)
                if not stage_outputs:
                    continue
                first = stage_outputs[0]
                ids = list(first.token_ids)
                if ids:
                    self._input_stream.put_nowait(ids)
                self._text += first.text or ""
                if self._park_id in ids:
                    if self._ledger.pending:
                        self._ledger.complete_next(self._text)
                    elif self._audio_closed and not self._flush_parked:
                        # @spec PORT-RTC-002, PORT-RTC-005
                        # Explicit FLUSH has no carrier ticket. Its park
                        # is the terminal barrier after the final-tail
                        # ticket has already completed.
                        self._flush_parked = True
                    else:
                        raise RuntimeError("park arrived without a carrier ticket or terminal FLUSH (PORT-RTC-002)")
            self._ended = True
        except Exception as error:
            self._error = error
        finally:
            self._done.set()
            if self._error is not None:
                self._ledger.fail(self._error)
            elif not self._audio_closed or self._ledger.pending or not self._flush_parked:
                self._ledger.fail(RuntimeError("generation ended before the session's normal finalization"))


# @spec PORT-RTC-001, PORT-RTC-003, PORT-RTC-007
class NemotronSessionFactory:
    """Construct engine-bound sessions without a parallel admission gate."""

    def __init__(
        self,
        *,
        engine: Any,
        max_pending_carriers: int | None = None,
        request_id_prefix: str = "nemotron-session",
    ) -> None:
        self._engine = engine
        self._max_pending_carriers = max_pending_carriers
        self._request_id_prefix = request_id_prefix

    async def open(self, *, cadence: str, locale: str) -> NemotronSessionLease:
        """Validate model controls and construct one ordinary session."""
        from vllm_omni.model_executor.models.nemotron_asr.session import (
            NemotronRealtimeSession,
        )

        session = NemotronRealtimeSession.from_model_config(
            self._engine.model_config,
            cadence=cadence,
            locale=locale,
            with_ledger=True,
            max_pending_carriers=self._max_pending_carriers,
        )
        return NemotronSessionLease(
            engine=self._engine,
            session=session,
            request_id=f"{self._request_id_prefix}-{uuid4()}",
        )
