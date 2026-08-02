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
import time
from collections.abc import AsyncIterator, Awaitable, Callable
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable
from uuid import uuid4

from vllm_omni.entrypoints.session_lifecycle import (
    SessionLifecycleDeadline,
    SessionLifecycleKind,
)
from vllm_omni.model_executor.models.nemotron_asr.transcript import (
    SegmentCompletion,
    TerminalResult,
)

if TYPE_CHECKING:
    import numpy as np
    import numpy.typing as npt

    from vllm_omni.model_executor.models.nemotron_asr.endpointing import (
        EndpointPolicy,
    )
    from vllm_omni.model_executor.models.nemotron_asr.session import (
        NemotronRealtimeSession,
        ReceiptLedger,
        StreamingObserver,
    )

    FloatSamples = npt.NDArray[np.float32]

Render = Callable[[Any], Awaitable[Any]]
AcceptanceCallback = Callable[[int], None]
SegmentCallback = Callable[[SegmentCompletion], None]


@runtime_checkable
class SessionLease(Protocol):
    """Transport-neutral lifecycle for one engine session."""

    async def feed(
        self,
        samples: FloatSamples,
        *,
        on_accepted: AcceptanceCallback | None = None,
        on_segment: SegmentCallback | None = None,
    ) -> list[str]:
        """Submit one whole piece and return completed hypotheses."""
        ...

    async def force_segment(self) -> None:
        """Queue one semantic segment boundary."""
        ...

    async def flush(self) -> TerminalResult:
        """Drain the final-tail and return the terminal result."""
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

    async def open(
        self,
        *,
        cadence: str,
        locale: str,
        endpoint_policy: EndpointPolicy | None = None,
    ) -> SessionLease:
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
        persistent_state_service: Any | None = None,
        state_lease: Any | None = None,
        release_operation_id: str | None = None,
        idle_timeout_s: float | None = None,
        finalization_timeout_s: float | None = None,
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
        self._persistent_state_service = persistent_state_service
        self._state_lease = state_lease
        self._release_operation_id = release_operation_id
        self._park_id = session.park_token_id
        self._audio: asyncio.Queue[Any] = asyncio.Queue()
        self._input_stream: asyncio.Queue[list[int]] = asyncio.Queue()
        self._task: asyncio.Task[None] | None = None
        self._done = asyncio.Event()
        self._terminal_result: TerminalResult | None = None
        self._active_segment_callback: SegmentCallback | None = None
        self._error: BaseException | None = None
        self._ended = False
        self._audio_closed = False
        self._flush_parked = False
        self._segment_generation = 0
        self._aborted = False
        self._released = False
        self._session_finished_observed = False
        self._terminal_reason: str | None = None
        self._terminal_code: str | None = None
        self._state_cleanup_lock = asyncio.Lock()
        self._pending_claim_task: asyncio.Task[None] | None = None
        self._session_lifecycle = SessionLifecycleDeadline(
            idle_timeout_s=idle_timeout_s,
            finalization_timeout_s=finalization_timeout_s,
            on_expire=self._expire_session_lifecycle,
        )
        if (
            self._persistent_state_service is not None
            and self._state_lease is not None
        ):
            timeout_s = float(
                self._persistent_state_service.pending_claim_timeout_s
            )
            self._pending_claim_task = asyncio.create_task(
                self._expire_pending_claim(timeout_s),
                name=f"persistent-state-claim-{request_id}",
            )
        self._arm_session_lifecycle_timeout("idle")

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
        on_segment: SegmentCallback | None = None,
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
        self._require_live()
        self._ensure_started()
        if self._active_segment_callback is not None:
            raise RuntimeError("feed calls must be serialized")
        self._active_segment_callback = on_segment
        self._audio.put_nowait(samples)
        try:
            receipt = await self._ledger.next_piece()
            if on_accepted is not None:
                on_accepted(len(samples))

            results: list[str] = []
            for ticket in receipt.tickets:
                results.append(await ticket.done)
            return results
        finally:
            self._active_segment_callback = None
            receipt = locals().get("receipt")
            if receipt is not None:
                for ticket in receipt.tickets:
                    if ticket.done.done() and not ticket.done.cancelled():
                        ticket.done.exception()

    async def force_segment(self) -> None:
        """Queue one ordered semantic boundary without accepting samples."""
        self._require_live()
        self._ensure_started()
        self._session.force_segment()
        # Wake the stream consumer; empty arrays are internal control wakes
        # and never enter accepted-audio accounting.
        import numpy as np

        self._audio.put_nowait(np.empty(0, dtype=np.float32))

    async def flush(self) -> TerminalResult:
        """Drain the final-tail and return all terminal output."""
        self._require_live()
        self._ensure_started()
        if self._aborted:
            raise RuntimeError("the session was aborted")
        if self._done.is_set() and not self._audio_closed:
            raise self._error or RuntimeError("generation ended before the session's normal finalization")
        await self._drain()
        if self._terminal_result is None:
            raise RuntimeError("terminal result was not produced")
        return self._terminal_result

    async def update_locale(self, locale: str) -> None:
        """Validate and select the locale for the next carrier mint."""
        self._require_live()
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
        """Release the exact manager lease once through its cleanup lane."""
        async with self._state_cleanup_lock:
            if self._released:
                return
            timer = self._pending_claim_task
            if timer is not None and timer is not asyncio.current_task():
                timer.cancel()
                await asyncio.gather(timer, return_exceptions=True)
            await self._cancel_session_lifecycle_timeout()
            if self._task is not None and not self._task.done():
                await self.abort()
            self._observe_session_finished_once(
                self._terminal_reason
                or (
                    "aborted"
                    if self._aborted or not self._ended
                    else "completed"
                )
            )
            if (
                self._persistent_state_service is not None
                and self._state_lease is not None
                and self._release_operation_id is not None
            ):
                await self._persistent_state_service.release(
                    operation_id=self._release_operation_id,
                    lease=self._state_lease,
                    reason=self._terminal_code or "session_release",
                )
            self._released = True

    async def _cancel_session_lifecycle_timeout(self) -> None:
        await self._session_lifecycle.close()

    def _arm_session_lifecycle_timeout(
        self,
        kind: SessionLifecycleKind,
    ) -> None:
        self._session_lifecycle.arm(kind)

    async def _expire_session_lifecycle(
        self,
        kind: SessionLifecycleKind,
    ) -> None:
        code = "idle_timeout" if kind == "idle" else "finalization_timeout"
        self._terminal_reason = "aborted" if kind == "idle" else "error"
        self._terminal_code = code
        self._error = TimeoutError(code)
        if kind == "idle":
            self._aborted = True
        await self.release()

    def _observe_session_finished_once(self, reason: str) -> None:
        if self._session_finished_observed:
            return
        observer = self._session.observer
        if observer is None:
            self._session_finished_observed = True
            return
        from vllm_omni.metrics.streaming_transport import observe_safely

        outcome = "aborted" if reason == "aborted" else "error"
        observe_safely(
            observer.clear_all_outstanding,
            self._session.session_key,
            outcome=outcome,
        )
        observe_safely(
            observer.session_finished,
            session_key=self._session.session_key,
            reason=reason,
        )
        self._session_finished_observed = True

    async def _expire_pending_claim(self, timeout_s: float) -> None:
        await asyncio.sleep(timeout_s)
        service = self._persistent_state_service
        lease = self._state_lease
        if service is None or lease is None:
            return
        async with self._state_cleanup_lock:
            if self._released:
                return
            won = await service.claim_pending_cleanup(lease)
            if not won:
                return
            await self.abort()
            self._observe_session_finished_once("error")
            assert self._release_operation_id is not None
            await service.release(
                operation_id=self._release_operation_id,
                lease=lease,
                reason="pending_claim_timeout",
            )
            self._released = True

    def _ensure_started(self) -> None:
        """Start the background engine consumer on first use."""
        if self._task is None:
            self._task = asyncio.get_running_loop().create_task(self._consume())

    def _require_live(self) -> None:
        """Reject work no consumer can acknowledge."""
        if self._session_lifecycle.expired:
            raise self._error or RuntimeError("session lifecycle has expired")
        if self._aborted:
            raise RuntimeError("the session was aborted")
        if self._audio_closed:
            raise RuntimeError("audio is closed; the session is finalizing")
        if self._done.is_set():
            raise self._error or RuntimeError("generation ended before the session's normal finalization")

    async def _drain(self) -> None:
        """Close accepted audio and wait for generation end."""
        if not self._audio_closed:
            if self._session_lifecycle.expired:
                raise self._error or RuntimeError(
                    "session lifecycle has expired"
                )
            self._session.begin_finalize()
            self._audio_closed = True
            self._audio.put_nowait(None)
            self._arm_session_lifecycle_timeout("finalization")
        await self._done.wait()
        await self._cancel_session_lifecycle_timeout()
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

        prompts = buffer_stream(
            self._audio_frames(),
            self._input_stream,
            self._session,
            before_audio_accept=lambda: self._require_unexpired_lifecycle(),
            on_audio_accepted=lambda: self._arm_session_lifecycle_timeout(
                "idle"
            ),
        )
        async for prompt in prompts:
            streaming_input = await render(prompt)
            if self._state_lease is not None:
                rendered_prompt = dict(streaming_input.prompt)
                information = dict(
                    rendered_prompt.get("additional_information") or {}
                )
                information["persistent_state_binding"] = {
                    "engine_epoch": self._state_lease.engine_epoch,
                    "session_key": self._state_lease.session_key,
                    "generation": self._state_lease.generation,
                    "schema_id": self._state_lease.schema_id,
                    "profile_id": self._state_lease.profile_id,
                    "binding_token": self._state_lease.binding_token,
                }
                policy = self._session.endpoint_policy
                information["endpoint_policy"] = {
                    "mode": policy.mode,
                    "threshold_frames": policy.threshold_frames,
                    "residue_frames": policy.residue_frames,
                }
                rendered_prompt["additional_information"] = information
                streaming_input.prompt = rendered_prompt
            yield streaming_input

    def _require_unexpired_lifecycle(self) -> None:
        if self._session_lifecycle.expired:
            raise self._error or RuntimeError("session lifecycle has expired")

    async def _consume(self) -> None:
        """Drive generation and resolve ledger tickets at legal parks."""
        from vllm_omni.metrics.streaming_transport import observe_safely

        observer = self._session.observer
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
                in_flight = self._session.accepted_audio.in_flight_unit
                if ids:
                    self._input_stream.put_nowait(ids)
                self._session.transcript.commit_result(first.text or "")
                completion = getattr(output, "segment_completion", None)
                if completion is None and self._session.eou_token_id in ids:
                    self._segment_generation += 1
                    completion = SegmentCompletion(
                        generation=self._segment_generation,
                        text="",
                        reason=(
                            "forced"
                            if getattr(in_flight, "kind", None) == "forced_eou"
                            else "model"
                        ),
                    )
                if completion is not None:
                    self._segment_generation = max(
                        self._segment_generation,
                        int(completion.generation),
                    )
                    committed_completion = self._session.transcript.complete_segment(
                        generation=int(completion.generation),
                        reason=completion.reason,
                    )
                    if (
                        committed_completion is not None
                        and self._active_segment_callback is not None
                    ):
                        self._active_segment_callback(completion)
                if self._park_id in ids:
                    # PORT-OBS-003: the single-in-flight-handle
                    # correlation authority — resolves to the CHUNK this
                    # park completes, or ``None`` for the terminal FLUSH
                    # park (no in-flight unit left to resolve, correctly
                    # ignored).
                    if observer is not None:
                        handle = observe_safely(observer.complete_inflight, self._session.session_key)
                        if handle is not None:
                            observe_safely(observer.unit_parked, handle, park_stamp_s=time.monotonic())
                    if self._ledger.pending:
                        self._ledger.complete_next(
                            self._session.transcript.complete_text
                        )
                    elif self._audio_closed and not self._flush_parked:
                        # @spec PORT-RTC-002, PORT-RTC-005
                        # Explicit FLUSH has no carrier ticket. Its park
                        # is the terminal barrier after the final-tail
                        # ticket has already completed.
                        self._flush_parked = True
                    else:
                        raise RuntimeError("park arrived without a carrier ticket or terminal FLUSH (PORT-RTC-002)")
            self._terminal_result = self._session.transcript.finish_terminal()
            self._ended = True
        except Exception as error:
            self._error = error
        finally:
            self._done.set()
            # Explicit finalization state (review round 2026-07-28, F2):
            # completed means the session's NORMAL finalization actually
            # happened — audio closed, no pending tickets, FLUSH parked.
            # A generation that merely stopped early fails the ledger
            # AND reports error; leaving the reason derived only from
            # "no exception caught" previously published completed for
            # premature exhaustion.
            finalized = self._audio_closed and not self._ledger.pending and self._flush_parked
            if self._error is not None:
                self._ledger.fail(self._error)
            elif not finalized:
                self._ledger.fail(RuntimeError("generation ended before the session's normal finalization"))
            # PORT-OBS-006: the lease's single idempotent terminal-
            # disposition section — this coroutine body runs exactly
            # once per lease, so no separate guard flag is needed.
            # ``self._aborted`` is already correctly set by the time this
            # runs even under an abort/finish race, because ``abort()``
            # sets it BEFORE cancelling (and awaiting) this same task.
            if self._terminal_reason is not None:
                reason = self._terminal_reason
            elif self._aborted:
                reason = "aborted"
            elif self._error is not None or not finalized:
                reason = "error"
            else:
                reason = "completed"
            self._observe_session_finished_once(reason)


# @spec PORT-RTC-001, PORT-RTC-003, PORT-RTC-007
class NemotronSessionFactory:
    """Construct engine-bound sessions without a parallel admission gate."""

    def __init__(
        self,
        *,
        engine: Any,
        persistent_state_service: Any | None = None,
        max_pending_carriers: int | None = None,
        request_id_prefix: str = "nemotron-session",
        observer: StreamingObserver | None = None,
        runtime_config: Any | None = None,
    ) -> None:
        self._engine = engine
        if persistent_state_service is None:
            getter = getattr(engine, "get_persistent_state_service", None)
            if getter is not None:
                persistent_state_service = getter()
        self._persistent_state_service = persistent_state_service
        if runtime_config is None and persistent_state_service is not None:
            runtime_config = getattr(
                persistent_state_service, "runtime_config", None
            )
        if persistent_state_service is not None and runtime_config is None:
            raise ValueError(
                "persistent-state session factory requires the resolved "
                "streaming runtime envelope"
            )
        self._runtime_config = runtime_config
        self._max_pending_carriers = max_pending_carriers
        self._request_id_prefix = request_id_prefix
        # PORT-OBS-003: injected at every session this factory opens
        # ("the transport-neutral factory/lease binding, which injects the
        # same observer at session construction"). Not yet consumed by
        # NemotronRealtimeSession beyond storage.
        self._observer = observer

    @property
    def persistent_state_service(self) -> Any | None:
        """The engine-installed admission authority, without side effects."""
        return self._persistent_state_service

    async def open(
        self,
        *,
        cadence: str,
        locale: str,
        endpoint_policy: EndpointPolicy | None = None,
    ) -> NemotronSessionLease:
        """Validate model controls and construct one ordinary session."""
        from vllm_omni.model_executor.models.nemotron_asr.session import (
            NemotronRealtimeSession,
        )

        # Validate every caller-controlled value before consuming manager
        # capacity.  The service-bound construction below stamps the
        # acknowledged lease identity into the otherwise identical session.
        runtime = self._runtime_config
        session_limits: dict[str, Any] = {}
        if runtime is not None:
            session_limits = {
                "accepted_audio_budget_s": runtime.accepted_audio_budget_s,
                "accepted_audio_capacity_samples": (
                    runtime.accepted_audio_capacity_samples
                ),
                "max_retained_transcript_bytes": (
                    runtime.max_retained_transcript_bytes
                ),
                "max_session_samples": runtime.max_session_samples,
            }
        NemotronRealtimeSession.from_model_config(
            self._engine.model_config,
            cadence=cadence,
            locale=locale,
            endpoint_policy=endpoint_policy,
            with_ledger=True,
            max_pending_carriers=self._max_pending_carriers,
            observer=None,
            **session_limits,
        )
        service = self._persistent_state_service
        if service is None:
            raise RuntimeError("persistent-state service is not installed")
        check_health = getattr(
            service,
            "check_admission",
            getattr(service, "check_health", None),
        )
        if check_health is not None:
            await check_health()
        inventory = getattr(service, "inventory", None) or {}
        request_id = f"{self._request_id_prefix}-{uuid4()}"
        reserve_operation_id = uuid4().hex
        release_operation_id = uuid4().hex
        state_lease = await service.reserve(
            operation_id=reserve_operation_id,
            session_key=request_id,
            schema_id=str(inventory.get("schema_id", "state-manifest-v1")),
            profile_id=str(inventory.get("profile_id", "default")),
        )
        try:
            session = NemotronRealtimeSession.from_model_config(
                self._engine.model_config,
                cadence=cadence,
                locale=locale,
                endpoint_policy=endpoint_policy,
                with_ledger=True,
                max_pending_carriers=self._max_pending_carriers,
                observer=self._observer,
                request_id=request_id,
                engine_epoch=str(state_lease.engine_epoch),
                lease_generation=int(state_lease.generation),
                **session_limits,
            )
            return NemotronSessionLease(
                engine=self._engine,
                session=session,
                request_id=request_id,
                persistent_state_service=service,
                state_lease=state_lease,
                release_operation_id=release_operation_id,
                idle_timeout_s=(
                    runtime.session_idle_timeout_s
                    if runtime is not None
                    else None
                ),
                finalization_timeout_s=(
                    runtime.session_finalization_timeout_s
                    if runtime is not None
                    else None
                ),
            )
        except BaseException:
            await service.release(
                operation_id=release_operation_id,
                lease=state_lease,
                reason="configuration_error",
            )
            raise


# @spec PORT-RTC-003, PORT-RTC-007
def create_nemotron_session_factory(engine_client: Any) -> SessionFactory:
    """Validate a compatible engine and return its public session factory.

    This is the transport-neutral construction boundary for external
    frontends. Validation performs no engine request, lease allocation, or
    admission transition.

    PORT-RTC-003 pins this PUBLIC signature to exactly ``(engine_client)``
    — no observer parameter here. Observer injection (PORT-OBS-003) lives
    ONLY on the internal ``NemotronSessionFactory`` constructor; a
    model-aware downstream participant that wants observation wires it by
    constructing ``NemotronSessionFactory(engine=..., observer=...)``
    directly rather than through this public constructor.
    """
    model_config = getattr(engine_client, "model_config", None)
    if model_config is None:
        raise ValueError("Nemotron session factory requires an engine model_config")
    hf_config = getattr(model_config, "hf_config", model_config)
    architectures = getattr(hf_config, "architectures", None)
    if (
        not isinstance(architectures, (list, tuple))
        or "Nemotron3_5AsrForRNNT" not in architectures
    ):
        raise ValueError(
            "Nemotron session factory requires the "
            "Nemotron3_5AsrForRNNT architecture"
        )

    from vllm_omni.model_executor.models.nemotron_asr.session import (
        NemotronRealtimeSession,
    )

    NemotronRealtimeSession.from_model_config(model_config)
    return NemotronSessionFactory(engine=engine_client)
