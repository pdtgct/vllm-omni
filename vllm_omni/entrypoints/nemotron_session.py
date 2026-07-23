# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The transport-neutral Nemotron session module — the engine binding.

The ONE engine-coupled home the RFC-1 brief splits out (round-5,
decision 2): HTTP, gRPC, and realtime transports all obtain their
sessions here, while the engine-free orchestrator
(``ephemeral_session.py``) stays engine-free by contract. Three pieces:

- :class:`ServingConcurrencyLimiter` — the app-scope load shedder,
  constructed once at application init and injected into every
  transport (never per-transport);
- :class:`NemotronSessionFactory` — satisfies the orchestrator's
  ``SessionFactory`` protocol BY SHAPE (PORT-EPH-004: the protocol is
  deliberately not imported here); admission goes through the injected
  limiter, and the canonical single-shot geometry is the model
  package's ``EPHEMERAL_CADENCE`` — sourced from the model, never a
  literal in serving code (PORT-REGIME-002);
- :class:`NemotronSessionLease` — the concrete ``SessionLease`` (again
  by shape), the ledger-adopting successor to RFC-2's prototype
  ``EngineTranscriber``: ``feed`` awaits the session's
  :class:`ReceiptLedger` statements instead of predicting parks from
  cadence arithmetic (the F6/R2 payoff, ING-FE-006), and any
  generation/render failure fails the ledger so a blocked consumer is
  released with the ORIGINAL error, never hung (round-4 R2).

Importable WITHOUT vllm: every vllm / engine-coupled import is lazy
(inside functions), mirroring core's ``transcribe_realtime`` render
precedent (``vllm/entrypoints/speech_to_text/realtime/serving.py``),
so the module stays file-path loadable for GPU-free tests.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Awaitable, Callable
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from vllm_omni.entrypoints.ephemeral_session import AdmissionBusyError

if TYPE_CHECKING:
    import numpy as np
    import numpy.typing as npt

    from vllm_omni.model_executor.models.nemotron_asr.session import (
        NemotronRealtimeSession,
        ReceiptLedger,
    )

    #: One contiguous block of decoded float32 PCM (the seam's ``feed``
    #: argument); the lease only ever enqueues it.
    FloatSamples = npt.NDArray[np.float32]

#: Prompt -> engine-submittable ``StreamingInput`` (the serving path's
#: parse -> render -> wrap, built lazily by :func:`_render_factory`).
Render = Callable[[Any], Awaitable[Any]]


def _render_factory(engine: Any) -> Render:
    """Build the prompt-to-``StreamingInput`` render callable.

    Mirrors core's ``OpenAIServingRealtime.transcribe_realtime`` exactly
    (``realtime/serving.py``): each minted prompt goes through
    ``parse_model_prompt`` against the engine's REAL process model
    config, then ``renderer.render_cmpl_async``, then a
    ``StreamingInput`` wrap. The vllm imports live here, at call time,
    so the module imports without vllm.

    Args:
        engine: The ``AsyncOmni`` handle (typed ``Any`` to stay
            import-light); supplies ``model_config`` and ``renderer``.

    Returns:
        An async callable rendering one prompt to one ``StreamingInput``.
    """
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
    """The engine's default sampling params, coerced for streaming.

    The same coercion the realtime connection applies before
    ``engine.generate`` (``realtime_connection.py``): cumulative
    outputs become deltas so the binding's text accumulation is exact.
    Lazy import — the real ``utils`` module is vllm-coupled.

    Args:
        engine: The ``AsyncOmni`` handle.

    Returns:
        The coerced sampling params list for ``generate``.
    """
    from vllm_omni.entrypoints.utils import coerce_param_message_types

    coerced: list[Any] = coerce_param_message_types(
        list(engine.default_sampling_params_list), is_streaming=True
    )
    return coerced


class ServingConcurrencyLimiter:
    """App-scope counting load shedder over concurrent serving leases.

    Constructed ONCE at application init and injected into every
    transport's factory, so a single instance sheds over the whole
    serving pool; the cap value arrives as a constructor argument (its
    ENV wiring rides the serving adapter).

    DECIDED disclaimers (round-5 gate, RFC-1 brief §B) — all three are
    normative and none may be weakened here:

    - **Load shedding only.** A counter of live leases against a
      configured cap. It cannot acknowledge actual allocation from the
      state-block pool, cannot derive capacity from that pool, and
      cannot protect a request from scheduler preemption.
    - **It never emits ``ADMITTED`` and never claims residency.** A
      counted session is not a resident session.
    - **It is NOT PORT-STATE-004 compliance.** True compliance is the
      provider-owned reservation protocol (reserve the state
      allocation; authoritative acknowledgement; non-recompute-
      preemptible; allocation pressure FAILS the session rather than
      requeueing it) — the owed engine work the brief records. Until
      that lands, the endpoints this limiter fronts stay experimental
      and disabled by default.
    """

    def __init__(self, *, max_concurrent: int) -> None:
        if max_concurrent < 1:
            raise ValueError(
                f"max_concurrent must be positive, got {max_concurrent}"
            )
        self._max_concurrent = max_concurrent
        self._live = 0

    @property
    def max_concurrent(self) -> int:
        """The configured cap on concurrently live leases."""
        return self._max_concurrent

    @property
    def live(self) -> int:
        """Currently held slots (acquired, not yet released)."""
        return self._live

    def acquire(self) -> None:
        """Count one live lease, or shed.

        Raises:
            AdmissionBusyError: When the cap is reached — no slot is
                held, so the caller has nothing to release.
        """
        if self._live >= self._max_concurrent:
            raise AdmissionBusyError(
                f"serving concurrency cap reached "
                f"({self._max_concurrent} live leases); load shed"
            )
        self._live += 1

    def release(self) -> None:
        """Free exactly one held slot.

        Callers release once per successful :meth:`acquire` (the lease
        guards its own idempotence); the limiter guards the protocol:

        Raises:
            RuntimeError: On release with no slot held — a
                double-release would silently widen the cap.
        """
        if self._live <= 0:
            raise RuntimeError(
                "release without a held slot: each acquire is released "
                "exactly once, and the live count never goes negative"
            )
        self._live -= 1


class NemotronSessionLease:
    """The concrete engine-bound session lease.

    Satisfies the orchestrator's ``SessionLease`` protocol BY SHAPE
    (``feed``/``flush``/``update_locale``/``abort``/``finish``/
    ``release``; PORT-EPH-004 — the protocol is deliberately not
    imported). One instance = one session = one long-lived engine
    request driven through ``engine.generate`` over the model's own
    segmenter (``buffer_stream``), exactly as core's realtime serving
    does.

    Ledger adoption (the F6/R2 payoff, replacing the prototype
    ``EngineTranscriber``'s waiter arithmetic): the binding holds NO
    cadence arithmetic. ``feed`` enqueues the piece and awaits the
    session ledger's :class:`PieceReceipt`, then each named
    :class:`CarrierTicket` — PORT's own statement of what was minted —
    while the background consumer completes the oldest pending ticket
    at each observed park (``ReceiptLedger.complete_next``) and fails
    the ledger on ANY generation/render failure or premature stream end
    so a blocked ``feed``/``flush`` raises the original error, never
    hangs (round-4 R2 / round-5 V3).

    The prototype's liveness guards are preserved (round-5 review):
    aborted / audio-closed / generation-ended states raise immediately
    instead of parking a waiter the consumer can no longer answer
    (ING-LIFE-010). ``release`` returns the limiter slot exactly once
    and is idempotent; it is a separate call so the orchestrator can
    free the slot in a single ``finally`` after terminal cleanup.
    """

    def __init__(
        self,
        *,
        engine: Any,
        session: NemotronRealtimeSession,
        limiter: ServingConcurrencyLimiter,
        request_id: str,
        render: Render | None = None,
    ) -> None:
        """Bind one admitted session to the engine.

        Args:
            engine: The ``AsyncOmni`` handle (``generate``/``abort``/
                ``model_config``/``renderer``; typed ``Any`` to stay
                import-light).
            session: The admitted model-local session; MUST carry an
                armed receipt ledger (``with_ledger=True``) — the
                ledger is this binding's entire synchronization story.
            limiter: The shared app-scope limiter whose slot this lease
                holds; :meth:`release` returns it exactly once.
            render: Prompt renderer override for engine-free tests;
                ``None`` builds the real one lazily via
                :func:`_render_factory` at consumer start.

        Raises:
            ValueError: If the session carries no receipt ledger.
        """
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
        self._limiter = limiter
        self._request_id = request_id
        self._render = render
        self._park_id = session.park_token_id
        # Audio frames to the segmenter; ``None`` closes the stream.
        self._audio: asyncio.Queue[Any] = asyncio.Queue()
        # Output token-id echoes feeding the segmenter's
        # hold-until-park backpressure (PORT-SESS-001).
        self._input_stream: asyncio.Queue[list[int]] = asyncio.Queue()
        self._task: asyncio.Task[None] | None = None
        self._done = asyncio.Event()
        self._text = ""
        self._error: BaseException | None = None
        self._ended = False
        self._audio_closed = False
        self._aborted = False
        self._released = False

    @property
    def session(self) -> NemotronRealtimeSession:
        """The admitted model-local session (geometry, prompt, ledger)."""
        return self._session

    @property
    def request_id(self) -> str:
        """The engine request id this lease drives."""
        return self._request_id

    async def feed(self, samples: FloatSamples) -> list[str]:
        """Advance one accepted piece; await its ledger receipt.

        The piece reaches the segmenter as ONE frame; the ledger's
        acknowledgement names exactly the carrier tickets that frame
        minted (zero for a sub-cadence piece), and each ticket resolves
        with the cumulative hypothesis stamped at its park — no
        prediction, no arithmetic (ING-FE-006, PORT-RTC-002).

        Args:
            samples: One contiguous float32 block (may be empty).

        Returns:
            One cumulative hypothesis per completed cadence, in order.

        Raises:
            RuntimeError: If the session is aborted, finalizing, or the
                generation already ended (the liveness guards).
            BaseException: The ORIGINAL engine/render failure, when the
                consumer failed the ledger.
        """
        self._ensure_started()
        self._require_live()
        self._audio.put_nowait(samples)
        receipt = await self._ledger.next_piece()
        results: list[str] = []
        try:
            for ticket in receipt.tickets:
                results.append(await ticket.done)
        finally:
            # A mid-burst failure fails the later tickets too; retrieve
            # their exceptions so none surfaces as unretrieved noise.
            for ticket in receipt.tickets:
                if ticket.done.done() and not ticket.done.cancelled():
                    ticket.done.exception()
        return results

    async def flush(self) -> str:
        """Drain the final tail; return the final transcript.

        Residual-free (ING-FE-006): the segmenter holds the accepted
        remainder and mints the explicit final-tail transaction — the
        zero-sample one included — when the audio stream closes
        (PORT-SESS-003). Waits for the generation to END, not merely
        the final park, so late output still lands in the transcript.

        Raises:
            RuntimeError: If the session was aborted, or generation
                ended before normal finalization.
            BaseException: The stored original failure, if any.
        """
        self._ensure_started()
        if self._aborted:
            raise RuntimeError("the session was aborted")
        if self._done.is_set() and not self._audio_closed:
            raise self._error or RuntimeError(
                "generation ended before the session's normal finalization"
            )
        await self._drain()
        return self._text

    async def update_locale(self, locale: str) -> None:
        """Select the session-control prompt for the NEXT mint.

        Delegates to ``session.select_prompt`` — the ONE validator
        (PORT-LID-001): an unknown locale rejects the update and the
        prior selection stands.

        Args:
            locale: A locale of the served checkpoint's prompt
                dictionary.

        Raises:
            ValueError: If the locale is unknown to the checkpoint.
        """
        self._session.select_prompt(locale)

    async def abort(self) -> None:
        """Free engine state immediately (the non-normal end).

        Fails the ledger (idempotent — an earlier engine failure keeps
        its original error), cancels the consumer, and aborts the
        engine request if generation had not already ended. Idempotent;
        never releases the limiter slot — :meth:`release` is the
        separate, orchestrator-sequenced call (ING-LIFE-010).
        """
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
        """Terminal call on the normal end (ING-LIFE-010).

        After ``flush`` this is a no-op; if ``flush`` never ran it ends
        the open generation gracefully through the segmenter's explicit
        (zero-sample) final-tail transaction — never an abort.
        """
        if self._task is None or self._aborted:
            return
        await self._drain()

    async def release(self) -> None:
        """Return the limiter slot exactly once; idempotent.

        Sequenced by the orchestrator AFTER the terminal ``finish`` or
        ``abort``; the lease guards its own exactly-once so the shared
        limiter count stays truthful.
        """
        if self._released:
            return
        self._released = True
        self._limiter.release()

    def _ensure_started(self) -> None:
        """Start the background consumer on first use."""
        if self._task is None:
            self._task = asyncio.get_running_loop().create_task(
                self._consume()
            )

    def _require_live(self) -> None:
        """Refuse feeds the consumer can no longer answer (never hang).

        A piece enqueued after the audio closed or the generation ended
        would wait forever on a receipt nobody will publish; raise the
        stored original error, or name the state, instead
        (ING-LIFE-010).
        """
        if self._aborted:
            raise RuntimeError("the session was aborted")
        if self._audio_closed:
            raise RuntimeError("audio is closed; the session is finalizing")
        if self._done.is_set():
            raise self._error or RuntimeError(
                "generation ended before the session's normal finalization"
            )

    async def _drain(self) -> None:
        """Close the audio stream and wait for generation end."""
        if not self._audio_closed:
            self._audio_closed = True
            self._audio.put_nowait(None)
        await self._done.wait()
        if self._task is not None:
            await asyncio.gather(self._task, return_exceptions=True)
        if self._error is not None:
            raise self._error

    async def _audio_frames(self) -> AsyncIterator[Any]:
        """The segmenter's audio stream: frames until the close marker."""
        while True:
            frame = await self._audio.get()
            if frame is None:
                return
            yield frame

    async def _rendered(self, render: Render) -> AsyncIterator[Any]:
        """The engine's prompt stream: segment, then render each mint.

        The session object rides the segmenter's ``model_config``
        position (PORT-RTC-001), so mint-time controls — prompt index,
        geometry, park — are the session's own; the render path keeps
        the engine's REAL model config inside ``render``. Lazy model-
        package import keeps the module vllm-free at import time.
        """
        from vllm_omni.model_executor.models.nemotron_asr.streaming import (
            buffer_stream,
        )

        prompts = buffer_stream(
            self._audio_frames(), self._input_stream, self._session
        )
        async for prompt in prompts:
            yield await render(prompt)

    async def _consume(self) -> None:
        """Drive the engine request; complete ledger tickets at parks.

        Echoes every output token batch onto the segmenter's
        ``input_stream`` (its hold-until-park backpressure stays THE
        pacing mechanism, PORT-REGIME-001), accumulates the cumulative
        hypothesis, and completes the oldest pending ticket per
        observed park. Any failure — render, generate, or a broken
        park/ticket chain — is stored and fails the ledger, as does a
        premature stream end, so no consumer-side awaiter ever hangs.
        """
        try:
            render = self._render or _render_factory(self._engine)
            outputs = self._engine.generate(
                prompt=self._rendered(render),
                request_id=self._request_id,
                sampling_params_list=_streaming_sampling_params(
                    self._engine
                ),
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
                    self._ledger.complete_next(self._text)
            self._ended = True
        except Exception as error:
            self._error = error
        finally:
            self._done.set()
            if self._error is not None:
                self._ledger.fail(self._error)
            elif not self._audio_closed or self._ledger.pending:
                # Premature end (stream closed under an open session,
                # or a park never arrived for a minted ticket): fail
                # the ledger so blocked awaiters are released, never
                # hung. No-op if abort already failed it.
                self._ledger.fail(
                    RuntimeError(
                        "generation ended before the session's normal "
                        "finalization"
                    )
                )


class NemotronSessionFactory:
    """The concrete, model-aware session factory.

    Satisfies the orchestrator's ``SessionFactory`` protocol BY SHAPE
    (PORT-EPH-004 — deliberately not imported here). Admission goes
    through the injected :class:`ServingConcurrencyLimiter` (load
    shedding only — see its disclaimers); on any failure after the
    slot is counted, the slot is returned before the error propagates,
    so a failed open never leaks capacity.

    The canonical single-shot geometry lives HERE, in the model-aware
    factory, sourced from the model package's ``EPHEMERAL_CADENCE`` —
    callers of :meth:`open_ephemeral` pass no cadence, so no transport
    can pick the wrong one (PORT-REGIME-002).
    """

    def __init__(
        self,
        *,
        engine: Any,
        limiter: ServingConcurrencyLimiter,
        max_pending_carriers: int | None = None,
        request_id_prefix: str = "nemotron-session",
    ) -> None:
        """Bind the factory to one engine and the shared limiter.

        Args:
            engine: The ``AsyncOmni`` handle (typed ``Any`` to stay
                import-light); supplies the served ``model_config``
                every session is validated against.
            limiter: The SHARED app-scope limiter — one instance across
                every transport, injected at app init.
            max_pending_carriers: Optional override for the ledger's
                backlog bound; ``None`` keeps the model package's
                default. Locale/cadence defaults ride the model
                package, not this factory.
            request_id_prefix: Prefix for minted engine request ids.
        """
        self._engine = engine
        self._limiter = limiter
        self._max_pending_carriers = max_pending_carriers
        self._request_id_prefix = request_id_prefix

    async def open_ephemeral(self, *, locale: str) -> NemotronSessionLease:
        """Lease one canonical single-shot session (PORT-REGIME-002).

        The cadence is the model package's ``EPHEMERAL_CADENCE``; the
        caller supplies only ``locale``.

        Args:
            locale: A locale of the served checkpoint's prompt
                dictionary.

        Returns:
            The admitted lease.

        Raises:
            AdmissionBusyError: When the limiter is at cap (no lease
                minted, nothing to release).
            ValueError: On an unknown locale (the slot is returned).
        """
        from vllm_omni.model_executor.models.nemotron_asr.session import (
            EPHEMERAL_CADENCE,
        )

        return await self.open(cadence=EPHEMERAL_CADENCE, locale=locale)

    async def open(
        self, *, cadence: str, locale: str
    ) -> NemotronSessionLease:
        """Lease one realtime session at ``cadence``.

        Args:
            cadence: A published cadence label; validated by the model
                package's own geometry authority (PORT-SESS-002).
            locale: A locale of the served checkpoint's prompt
                dictionary.

        Returns:
            The admitted lease.

        Raises:
            AdmissionBusyError: When the limiter is at cap.
            ValueError: On an unknown cadence or locale (the counted
                slot is returned before the error propagates).
        """
        self._limiter.acquire()
        try:
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
                limiter=self._limiter,
                request_id=f"{self._request_id_prefix}-{uuid4()}",
            )
        except BaseException:
            # No lease was handed out, so nobody else can return the
            # slot: give it back here or it leaks forever.
            self._limiter.release()
            raise
