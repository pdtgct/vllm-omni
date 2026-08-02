# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The realtime input segmenter — pure, engine-free (PORT-SESS-001/002/003).

The chunking behaviour that backs ``buffer_realtime_audio``: fixed
segmenter, hold-until-park backpressure, and the NeMo tail rules. It
carries no vLLM coupling (numpy + async generators only), so it stays on
the macOS loader path and CPU-tested while the model class that exposes
it — via the thin ``SupportsRealtime.buffer_realtime_audio`` classmethod
— is engine/pod-gated (see ``static-analysis-for-vllm-coupled-code``).
"""

from __future__ import annotations

from collections.abc import AsyncGenerator, Callable
from typing import Any

import numpy as np

from vllm_omni.metrics.streaming_transport import observe_safely
from vllm_omni.model_executor.models.nemotron_asr.manifests import (
    ENVELOPE_HEADER_FIELDS,
    FRONTEND_CONSTANTS,
)
from vllm_omni.model_executor.models.nemotron_asr.session import (
    ACCEPTED_AUDIO_BUDGET_DEFAULT_S,
    NemotronRealtimeSession,
    StreamingObserver,
)

_ENVELOPE_VERSION = 2.0
_HEADER_SLOTS = len(ENVELOPE_HEADER_FIELDS)


def mint_envelope(
    samples: np.ndarray,
    *,
    geometry_id: int,
    final_tail: bool,
    prompt_index: int,
    chunk_sequence: int,
    admission_ms_mod: int,
) -> np.ndarray:
    """Mint one CHUNK envelope: the versioned header + raw samples.

    This is the serving tier's twin admission record (design §Phase-6c
    transaction seams): the worker-side plan provider reads the same
    header host-side to stamp the session registry, and the device path
    re-validates it against that authority. Header integers must stay
    exactly FP32-representable (PORT-INT-004). ``admission_ms_mod`` is
    the caller's job to capture at true cadence-completion time, not at
    mint time — see :func:`buffer_stream`'s ready-queue.
    """
    header = np.zeros(_HEADER_SLOTS, dtype=np.float32)
    header[0] = _ENVELOPE_VERSION
    header[1] = float(samples.shape[0])
    header[2] = float(geometry_id)
    header[3] = 1.0 if final_tail else 0.0
    header[4] = float(prompt_index)
    header[5] = float(chunk_sequence)
    header[6] = float(admission_ms_mod)
    return np.concatenate([header, samples.astype(np.float32, copy=False)])


async def buffer_stream(
    audio_stream: Any,
    input_stream: Any,
    model_config: Any,
    *,
    observer: StreamingObserver | None = None,
    accepted_audio_budget_s: float | None = None,
    session_key: str | None = None,
    final_tail_ready_stamp_s: float | None = None,
    before_audio_accept: Callable[[], None] | None = None,
    on_audio_accepted: Callable[[], None] | None = None,
) -> AsyncGenerator[dict[str, Any], None]:
    """Chunk client audio: one yield = one StreamingUpdate.

    Fixed segmenter built once at generator start from the session's
    admitted chunk config (PORT-SESS-002); holds chunk N+1 until the
    park token id for chunk N appears on ``input_stream``
    (buffer-until-drained, PORT-SESS-001 — defense in depth over core's
    park-time queue consumption); applies the NeMo tail rules on
    finalize (PORT-SESS-003: exactly one actual-residual final-tail,
    including an explicit zero-sample transaction; never zero-padded).

    Args:
        observer: PORT-OBS-003: overrides ``session.observer`` when given
            (e.g. the ledgerless native path, where ``model_config`` is a
            bare config with no session object to carry one).
        final_tail_ready_stamp_s: PORT-OBS-004: the caller-captured
            monotonic finalize-acceptance stamp. When given, the
            final-tail ``unit_ready`` call uses exactly this value rather
            than reconstructing one here at generator resumption.
        accepted_audio_budget_s: PORT-SESS-001 Decision 1: overrides the
            per-session accepted-audio queue budget when ``model_config``
            is a bare config (a new session is constructed here) — the
            native path's only construction point for one. Ignored when
            ``model_config`` is already a :class:`NemotronRealtimeSession`
            (that session's own configured budget governs). ``None``
            (the default) leaves :data:`ACCEPTED_AUDIO_BUDGET_DEFAULT_S`
            in force, keeping today's behavior exactly.
        before_audio_accept: Optional synchronous lifecycle fence invoked
            immediately before the session mutates accepted-audio state.
        on_audio_accepted: Optional synchronous lifecycle callback invoked
            immediately after whole-piece acceptance succeeds. Together
            these callbacks linearize idle-timeout expiry against acceptance
            without holding a lock across an engine or transport await.

    Raises:
        ValueError: If a piece would push the session's accepted-audio
            queue occupancy past ``session.accepted_audio_budget_s``
            (design §Park, Finalization, and Backpressure, Decision 1,
            amended PORT-SESS-001) — the whole piece is rejected before
            any of its complete cadences are accepted.
    """
    # Every session control is a typed attribute of the session object
    # riding the model_config position (PORT-RTC-001); anything else is
    # the channel-less standard path, whose defaults live only in the
    # factory.
    if isinstance(model_config, NemotronRealtimeSession):
        session = model_config
    else:
        accepted_audio_capacity_samples = None
        if accepted_audio_budget_s is not None:
            if accepted_audio_budget_s <= 0:
                raise ValueError("accepted_audio_budget_s must be positive")
            accepted_audio_capacity_samples = int(
                accepted_audio_budget_s * FRONTEND_CONSTANTS["sample_rate"]
            )
        session = NemotronRealtimeSession.from_model_config(
            model_config,
            accepted_audio_capacity_samples=accepted_audio_capacity_samples,
            request_id=session_key or "unbound",
            observer=observer,
            accepted_audio_budget_s=(
                accepted_audio_budget_s
                if accepted_audio_budget_s is not None
                else ACCEPTED_AUDIO_BUDGET_DEFAULT_S
            ),
        )
    geometry = session.geometry
    geometry_id = geometry.geometry_id
    park_id = session.park_token_id
    placeholder_id = session.audio_chunk_token_id
    ledger = session.ledger
    # PORT-OBS-003: the explicit override takes precedence over the
    # session's own observer (the ledgerless native path threads one in
    # without a session-owning observer to fall back to).
    active_observer: StreamingObserver | None = observer if observer is not None else session.observer
    session_key = session.session_key

    async def hold_until_park() -> None:
        while True:
            ids = await input_stream.get()
            if park_id in ids:
                return

    authority = session.accepted_audio

    def prompt(unit: Any) -> dict[str, Any]:
        # TokensPrompt shape: one placeholder token per chunk
        # (PORT-INT-003 / D-BU-1) — a bare multi_modal_data dict is
        # invalid on the real render path. The mm payload is the minted
        # ENVELOPE (header + raw samples), the serving tier's twin
        # admission record (design §Phase-6c transaction seams).
        if unit.kind == "forced_eou":
            if session.eou_token_id is None:
                raise ValueError("forced endpoint lacks an EOU control token")
            return {"prompt_token_ids": [session.eou_token_id]}
        # The session-control prompt is re-read at each mint so the
        # last valid update ordered before mint is the one stamped
        # (PORT-LID-001); a live per-session config view (ING-VEH-007)
        # makes mid-session locale updates visible exactly here.
        envelope = mint_envelope(
            unit.samples,
            geometry_id=geometry_id,
            final_tail=unit.kind == "final_tail",
            prompt_index=session.prompt_index,
            chunk_sequence=unit.carrier_sequence,
            admission_ms_mod=unit.admission_ms_mod,
        )
        return {
            "prompt_token_ids": [placeholder_id],
            "multi_modal_data": {"audio": envelope},
        }

    async def dispatch_ready() -> AsyncGenerator[dict[str, Any], None]:
        while authority.ready_units:
            unit = authority.dispatch_next()
            if unit is None:
                raise RuntimeError("ready audio could not become in-flight")
            handle = session.take_ready_handle(unit.logical_sequence)
            if active_observer is not None and handle is not None:
                observe_safely(active_observer.unit_minted, handle)
            yield prompt(unit)
            await hold_until_park()
            authority.park(
                request_id=authority.request_id,
                engine_epoch=authority.engine_epoch,
                lease_generation=authority.lease_generation,
                logical_sequence=unit.logical_sequence,
                carrier_sequence=unit.carrier_sequence,
            )

    # Acceptance and ready-stamping are synchronous under the session's
    # single authority. Every cadence completed by one caller piece is
    # therefore visible before the first corresponding prompt is yielded,
    # while delivery remains park-gated and FIFO.
    async for frame in audio_stream:
        if frame.shape[0] == 0:
            async for rendered in dispatch_ready():
                yield rendered
            continue
        prior_sequences = {
            unit.logical_sequence for unit in authority.ready_units
        }
        if before_audio_accept is not None:
            before_audio_accept()
        session.accept_audio(frame)
        if on_audio_accepted is not None:
            on_audio_accepted()
        new_units = tuple(
            unit
            for unit in authority.ready_units
            if unit.logical_sequence not in prior_sequences
        )
        if ledger is not None:
            for unit in new_units:
                if unit.kind != "forced_eou":
                    ledger.mint(
                        final_tail=unit.kind == "final_tail",
                        admission_ms_mod=unit.admission_ms_mod,
                        handle=session.ready_handle(unit.logical_sequence),
                    )
        if ledger is not None:
            ledger.acknowledge_piece(int(frame.shape[0]))
        async for rendered in dispatch_ready():
            yield rendered
    # Finalization is an explicit protocol transaction even when the
    # residual is shorter than the frontend's minimum commit or is
    # exactly zero. The frontend owns the zero-frame decision; the
    # session transition still needs the final marker (PORT-SESS-003).
    if not authority.snapshot().finalizing:
        finalize_at_ns = (
            None
            if final_tail_ready_stamp_s is None
            else int(final_tail_ready_stamp_s * 1_000_000_000)
        )
        session.begin_finalize(finalize_at_ns=finalize_at_ns)
    final_tail = next(
        (
            unit
            for unit in reversed(authority.ready_units)
            if unit.kind == "final_tail"
        ),
        None,
    )
    if ledger is not None and final_tail is not None:
        ledger.mint(
            final_tail=True,
            admission_ms_mod=final_tail.admission_ms_mod,
            handle=session.ready_handle(final_tail.logical_sequence),
        )
    async for rendered in dispatch_ready():
        yield rendered
    # @spec PORT-DEC-009, PORT-REGIME-004, PORT-SESS-003
    # The engine's later non-resumable end marker closes the request
    # without guaranteeing a model step. PORT therefore submits its
    # model-level FLUSH explicitly, after the final-tail transaction
    # has committed at legal park.
    yield {"prompt_token_ids": [session.flush_token_id]}
    # The generic AsyncOmni end marker is queued only after this
    # model-level barrier has itself committed. Otherwise a lifecycle
    # close could overtake an accepted-but-unprocessed FLUSH.
    await hold_until_park()
