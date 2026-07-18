# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The advance seam split (ledger P5-1; PORT-ADV-001/003).

``run_forward_step`` (``forward_step.py``) is being replaced by two
narrower operations that separate the storage-agnostic checkpoint
transition from the outer page-pool transaction:

- :func:`advance_session` — the ONE storage- and adapter-agnostic
  CHUNK transition (PORT-ADV-001). It owns the incremental frontend:
  the caller hands it raw PCM plus validated controls
  (:class:`ChunkBatch`) and gathered checkpoint state
  (:class:`SessionStateBatch`, including the frontend raw/mel tails and
  counters), and it advances the bounded frontend, encoder,
  conditioning, and RNN-T label loop. It never sees a pool, a
  ``state_indices`` tensor, a queue, or a book.
- :func:`advance_model_rows` — the ONE shared outer transaction
  (PORT-ADV-003) every native/fallback/probe caller passes through. It
  receives the scheduler's row mapping (:class:`RowPlan`) — not a
  pre-joined index tensor — so it itself composes decode-then-prefill
  indices, derives freshness from scheduler metadata (never recycled
  page contents), runs whole-call structural preflight (PORT-STATE-007)
  before any resident read, classifies CHUNK/REPLAY/FLUSH,
  gathers-only-for-CHUNK (PORT-STATE-008), calls ``advance_session``,
  materializes/validates captures, and projects the committed
  adapter-neutral :class:`AdvanceResult` through the selected emission
  adapter in one transaction.

This module is a tests-first stub (Phase 5): the functions raise
``NotImplementedError`` with complete typed signatures so pod tests
pin the future contract now and turn green in Phase 6, when
``forward_step.py`` / ``run_forward_step`` are deleted in the same
change (ledger P5-1 — a semantic split, never a second legacy forward
path).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from vllm_omni.model_executor.models.nemotron_asr.nemotron_asr import (
        NemotronASRCore,
    )
    from vllm_omni.model_executor.models.nemotron_asr.rnnt import (
        DecodeState,
    )

#: An RNN-T bucket decode: ``(conditioned_frames, enc_lengths,
#: predictor, joint, state) -> (token_ids, token_lengths, next_state)``.
#: Both production candidates (``decode_compact_active``,
#: ``decode_dense_masked``) satisfy it; selection is a startup
#: dispatch-table decision from the measured dense-vs-compact profile
#: (PORT-DEC-008), never a hardcoded default.
RnntDecodeFn = Callable[
    ...,
    "tuple[torch.Tensor, torch.Tensor, DecodeState]",
]

#: Chunk-envelope header layout (design §Chunk envelope): the versioned
#: FP32 carrier row is ``[version, valid_samples, geometry_id,
#: final_tail, prompt_index, chunk_sequence, samples...]``. Every header
#: integer must be exactly representable in FP32 (< 2**24). Slot order
#: is the LLD's listing order and is part of the pinned contract.
ENVELOPE_VERSION = 1
(
    ENV_VERSION,
    ENV_VALID_SAMPLES,
    ENV_GEOMETRY_ID,
    ENV_FINAL_TAIL,
    ENV_PROMPT_INDEX,
    ENV_CHUNK_SEQUENCE,
) = range(6)
ENVELOPE_HEADER_SLOTS = 6


@dataclass(frozen=True)
class ChunkBatch:
    """Raw PCM plus validated controls for a batch of CHUNK rows.

    The multimodal processor is stateless: it packs raw samples and
    controls, never mel (PORT-REGIME-001 / design §Chunk envelope). One
    row per CHUNK — every cadence unit ``buffer_realtime_audio`` mints,
    including the single explicit final-tail. Transport packet
    boundaries never mint a CHUNK.

    ``samples``: ``(B, S)`` FP32 mono, padded only inside the transport
    row to the largest admitted raw cadence; padding is never
    semantically appended to the audio.
    ``valid_samples``: ``(B,)`` long, the true sample count per row.
    ``geometry_id``: ``(B,)`` long, the admitted geometry; every
    envelope must match the session's admitted geometry before any
    state mutation.
    ``final_tail``: ``(B,)`` bool, the one explicit final-tail marker.
    ``prompt_index``: ``(B,)`` long, the LID prompt slot.
    ``chunk_sequence``: ``(B,)`` long, the monotonically increasing
    expected chunk sequence, validated against the session book.

    All control fields are validated (integer-valued, in range,
    sequence order) before the row is used.
    """

    samples: torch.Tensor
    valid_samples: torch.Tensor
    geometry_id: torch.Tensor
    final_tail: torch.Tensor
    prompt_index: torch.Tensor
    chunk_sequence: torch.Tensor


@dataclass
class SessionStateBatch:
    """Gathered checkpoint state :func:`advance_session` reads and
    mutates in place into the next state (design §advance_session).

    Storage-agnostic: the caller gathers page-backed (or any other)
    tensor views into these fields; NO pools, ``state_indices``, queue,
    or book live here — those are :func:`advance_model_rows`' concern.

    Frontend continuity (the incremental STFT frontend advances inside
    ``advance_session``):
      ``raw_tail``: ``(B, R)`` FP32 bounded raw-sample tail;
      ``mel_tail``: ``(B, 128, 9)`` FP32 committed-boundary mel tail;
      ``frontend_counters``: ``(B, 8)`` int64 control counters, in
      ``manifests.FRONTEND_COUNTER_FIELDS`` order (the manifest is the
      single source of the slot names and order): total valid samples,
      committed mel frames, encoded mel frames, raw-tail origin,
      raw-tail length, mel-tail length, expected chunk sequence, and
      finalization state.

    Encoder / conv (per layer, list index = encoder layer):
      ``channel``: each ``(B, window, d_model)`` left-context window;
      ``window_valid``: each ``(B, 1)`` int32 valid length;
      ``time``: each ``(B, d_model, conv_tail)`` depthwise-conv tail.

    Predictor:
      ``h`` / ``c``: each ``(B, pred_layers, pred_hidden)`` LSTM state;
      ``last_label``: ``(B,)`` long, the predictor's most recent label
      (the RNN-T loop's continuation across chunks).
    """

    raw_tail: torch.Tensor
    mel_tail: torch.Tensor
    frontend_counters: torch.Tensor
    channel: list[torch.Tensor]
    window_valid: list[torch.Tensor]
    time: list[torch.Tensor]
    h: torch.Tensor
    c: torch.Tensor
    last_label: torch.Tensor


@dataclass(frozen=True)
class PreparedCaptures:
    """GPU-resident named captures: padded storage + exact valid
    lengths, never Python lists (PORT-PERF-001). A finalized
    zero-frame CHUNK stages all three tensors with zero valid length
    — the checkpoint record survives even when no model work ran.

    ``frontend_mel``: ``(B, n_mels, T_mel_pad)``; ``mel_lengths``:
    ``(B,)`` long. ``encoder_raw`` / ``encoder_conditioned``:
    ``(B, T_enc_pad, d_model)``; ``encoder_lengths``: ``(B,)`` long.
    """

    frontend_mel: torch.Tensor
    mel_lengths: torch.Tensor
    encoder_raw: torch.Tensor
    encoder_conditioned: torch.Tensor
    encoder_lengths: torch.Tensor


@dataclass(frozen=True)
class AdvanceResult:
    """The adapter-neutral, GPU-RESIDENT result of one
    :func:`advance_session` call (no host synchronization on the
    result path, PORT-PERF-001).

    ``token_ids``: ``(B, K)`` int32, padded; ``token_lengths``:
    ``(B,)`` int32 — row ``b``'s burst is ``token_ids[b,
    :token_lengths[b]]``, ordered, NONBLANK, bounded by
    ``valid_encoder_frames × max_symbols_per_step`` for the row's
    geometry (PORT-INT-002), NOT by the attention window.
    ``row_status``: ``(B,)`` int32 — the device-resolved per-row
    status bitmask (PORT-ADV-004; bit names are
    ``frontend.ROW_STATUS_*``, 0 == committed clean). A row with any
    bit set mutated no state and its burst is zero-length; the
    transaction consumes this tensor through its existing
    asynchronous output readback at the commit boundary, mapping
    protocol bits to PORT-STATE-008's row tier and invariant bits to
    port-defect escalation — never a raise or device assertion from
    inside the transition.
    ``captures``: :class:`PreparedCaptures` or ``None`` when capture
    is disabled.

    The result carries no MRV1/runner projection: the selected
    emission adapter turns it into runner rows via an explicit
    :class:`EmissionProjection`; the outer transaction is the sole
    scatter owner.
    """

    token_ids: torch.Tensor
    token_lengths: torch.Tensor
    row_status: torch.Tensor | None = None
    captures: PreparedCaptures | None = None

    @property
    def row_valid(self) -> torch.Tensor | None:
        """``(B,)`` bool view of ``row_status``: True == clean."""
        if self.row_status is None:
            return None
        return self.row_status == 0


@dataclass(frozen=True)
class RowPlan:
    """The scheduler's authoritative row mapping (design §Initialization
    and row-to-page mapping).

    :func:`advance_model_rows` receives this — NOT a pre-joined index
    tensor — so it composes decode-then-prefill indices itself and can
    prove the composition (PORT-STATE-007), derives freshness from
    scheduler metadata rather than recycled page contents, and knows
    liveness and per-row geometry without a content sentinel.

    ``state_indices_d``: ``(num_decodes_padded, K)`` decode block
    indices; only ``[:num_decodes, 0]`` are real (extra columns are a
    speculative-decode configuration error, not values to flatten;
    CUDA-graph padding rows are excluded by ``num_decodes``).
    ``num_decodes`` / ``num_prefills``: real row counts.
    ``state_indices_p``: ``(num_prefills,)`` prefill block indices.
    ``has_initial_states_p``: ``(num_prefills,)`` bool; ``~mask`` is the
    fresh-row mask (first-chunk prefills initialize from metadata).
    ``null_block_id``: the reserved block that is never a live session
    page.
    ``num_pool_blocks``: the resident block-pool size (in-range bound).
    ``live_block_ids``: ``(L,)`` long, the scheduler's liveness
    authority — the block ids currently allocated to resident sessions
    (including blocks allocated to this call's fresh prefills). Every
    real row's index must be a member; a valid-but-non-live index is a
    whole-call structural defect. Liveness is never inferred from page
    contents.
    ``geometry_id``: ``(num_decodes + num_prefills,)`` long, the
    admitted geometry per real row, in decode-then-prefill order.
    ``prompt_index``: ``(num_prefills,)`` long, the admitted prompt for
    each fresh prefill row (metadata-sourced session-book init).
    """

    state_indices_d: torch.Tensor
    num_decodes: int
    state_indices_p: torch.Tensor
    num_prefills: int
    has_initial_states_p: torch.Tensor
    null_block_id: int
    num_pool_blocks: int
    live_block_ids: torch.Tensor
    geometry_id: torch.Tensor
    prompt_index: torch.Tensor


@dataclass
class EmissionContext:
    """Per-call row context + GATHERED emission scratch the adapter
    consumes. The transaction gathers ``queue``/``book`` copies for
    the batch and later scatters the projection's updated scratch —
    the adapter NEVER mutates resident pools (the transaction is the
    sole scatter owner).

    ``roles``: ``(N,)`` int (CHUNK/REPLAY/FLUSH); ``input_ids``:
    ``(N,)`` long; ``chunk_rows``: ``(B,)`` long positions of the
    CHUNK rows within the N-row batch; ``queue`` ``(N, cap)`` /
    ``book`` ``(N, 7)``: gathered emission scratch.
    """

    roles: torch.Tensor
    input_ids: torch.Tensor
    chunk_rows: torch.Tensor
    queue: torch.Tensor
    book: torch.Tensor


@dataclass(frozen=True)
class EmissionProjection:
    """What the adapter returns: the runner rows plus the UPDATED
    emission scratch for the transaction to scatter at commit.

    ``rows``: ``(N, H)`` runner output (e.g. MRV1 decision carriers);
    ``queue`` / ``book``: the updated scratch, same shapes as the
    context's.
    """

    rows: torch.Tensor
    queue: torch.Tensor
    book: torch.Tensor


#: A model-local emission adapter: projects a committed
#: :class:`AdvanceResult` under an :class:`EmissionContext` into an
#: :class:`EmissionProjection`. Selection happens once at model init
#: (runner-mode configuration); the transaction owns every resident
#: scatter.
EmissionAdapter = Callable[
    ["AdvanceResult", "EmissionContext"], "EmissionProjection"
]


def make_mrv1_adapter(
    *,
    hidden_size: int,
    park_id: int,
    blank_id: int,
) -> EmissionAdapter:
    """Build the MRV1 emission adapter (the correctness fallback,
    PORT-DEC-010's burst adapter arrives behind the same seam).

    The returned adapter is POOL-FREE: it consumes the context's
    gathered queue/book scratch and the committed
    :class:`AdvanceResult`, and returns an
    :class:`EmissionProjection` — first burst label as each CHUNK
    row's decision carrier, the remainder queued with pending-echo and
    expected-label stamped, ``park_id`` for drained rows. The
    transaction scatters.

    Raises:
        NotImplementedError: Always, at this tests-first stub — lands
            in Phase 6.
    """
    raise NotImplementedError(
        "make_mrv1_adapter lands in Phase 6 with the forward_step.py "
        "deletion (ledger P5-1)"
    )


# @spec PORT-ADV-001, PORT-ADV-004
def advance_session(
    core: NemotronASRCore,
    batch: ChunkBatch,
    state: SessionStateBatch,
    *,
    geometry: int,
    decode_fn: RnntDecodeFn,
    capture: bool = False,
) -> AdvanceResult:
    """The ONE storage- and adapter-agnostic CHUNK transition.

    CHUNK-only: REPLAY and FLUSH rows never enter this function. It owns
    every stateful checkpoint step — the incremental bounded frontend
    (advancing ``state``'s raw/mel tails and counters from ``batch``'s
    raw samples and controls), encoder-cache advancement, language
    conditioning, predictor advancement, and RNN-T emission ordering —
    so no caller reimplements any of them (PORT-ADV-001). It reads and
    mutates ``state`` in place into the next state and returns the
    adapter-neutral :class:`AdvanceResult`.

    Length-aware (PORT-ADV-004): one call serves one profile+geometry
    bucket carrying mixed session-first / continuing / final /
    zero-frame rows. Every shape is host-derived from the bucket
    geometry (padded frontend width ``C + 6`` — a legal final residual
    is strictly under one cadence per PORT-SESS-001/003 — and encoder
    width from the subsampling formula); validity is per-row length
    tensors. Every per-row failure — wrong chunk sequence, audio
    after finalization, an envelope geometry different from the
    bucket's, an oversized final residual, or a frontend design
    invariant — is a masked no-op reported as a bit in
    ``AdvanceResult.row_status``; the transaction consumes that
    tensor at its commit readback. The transition never raises on a
    device predicate and never uses a device assertion. The call
    issues no host/device synchronization outside ``decode_fn``.

    Args:
        core: the assembled pipeline (encoder / lid / predictor /
            joint / featurizer).
        batch: raw PCM plus validated controls, one row per CHUNK.
        state: gathered checkpoint state, read and mutated into the
            next state.
        geometry: the bucket's admitted geometry id — a host value,
            per the profile+geometry bucket contract (PORT-PERF-001).
        decode_fn: the RNN-T decode to run — REQUIRED and never
            defaulted here (PORT-DEC-008): ``decode_compact_active``
            (eager, synchronizes on compaction) and
            ``decode_dense_masked`` (fixed-trip, sync-free) stay
            unselected candidates until the measured dense-vs-compact
            profile; startup then builds a dispatch table keyed by
            geometry, padded batch tier, and execution/precision
            profile, and passes the selected callable per bucket.
        capture: the explicit capture policy (PORT-HOOK-001). OFF by
            default: performance runs return ``captures=None`` with no
            capture-only allocations and no extended lifetime for the
            raw/conditioned encoder tensors. ON stages the three named
            tensors at the bucket's fixed padded widths with exact
            logical lengths — including zero length for a finalized
            zero-frame CHUNK.

    Returns:
        The GPU-resident :class:`AdvanceResult` (padded token
        tensors, per-row status, prepared captures).

    Raises:
        ValueError: if ``geometry`` is not an admitted geometry id
            (a host configuration error, not a row condition).
    """
    from vllm_omni.model_executor.models.nemotron_asr.encoder import (
        stream_step,
    )
    from vllm_omni.model_executor.models.nemotron_asr.frontend import (
        CTR_COMMITTED_MEL_FRAMES,
        CTR_ENCODED_MEL_FRAMES,
        CTR_EXPECTED_CHUNK_SEQUENCE,
        MEL_TAIL_FRAMES,
        ROW_STATUS_FINAL_OVERSIZE,
        ROW_STATUS_GEOMETRY,
        ROW_STATUS_SEQUENCE,
        advance_frontend,
    )
    from vllm_omni.model_executor.models.nemotron_asr.manifests import (
        CADENCES,
    )
    from vllm_omni.model_executor.models.nemotron_asr.rnnt import (
        DecodeState,
    )

    n_rows = batch.samples.shape[0]
    device = batch.samples.device
    lookaheads = [right for (_, right) in CADENCES.values()]
    if not 0 <= geometry < len(lookaheads):
        raise ValueError(f"unknown bucket geometry id {geometry}")
    lookahead = lookaheads[geometry]
    cadence = 8 * (lookahead + 1)
    # The bucket's padded frontend width: a first row commits C-7, a
    # continuing row C, and a legal final tail at most C+6 — the
    # segmentation contract (PORT-SESS-001/003) drains complete
    # cadences as regular units before minting the actual residual,
    # so a final carrier holds strictly less than one cadence unit
    # (design §Exact Bounded Frontend).
    pad_frames = cadence + 6
    mel_width = MEL_TAIL_FRAMES + pad_frames

    counters = state.frontend_counters
    # Envelope-protocol status bits (PORT-ADV-004), all on device: the
    # envelope geometry must match the bucket, the sequence must match
    # the session's expected counter, and a final residual must be
    # strictly smaller than one cadence unit (an oversized final is an
    # ingress defect, never priced into the padded width). The
    # frontend composes in its own FINALIZED and invariant bits.
    hop = core.featurizer.hop_length
    incoming = (
        (batch.geometry_id.to(device) != geometry).to(torch.int32)
        * ROW_STATUS_GEOMETRY
    )
    incoming |= (
        batch.chunk_sequence.to(device)
        != counters[:, CTR_EXPECTED_CHUNK_SEQUENCE]
    ).to(torch.int32) * ROW_STATUS_SEQUENCE
    incoming |= (
        batch.final_tail.to(device)
        & (batch.valid_samples.to(device) >= cadence * hop)
    ).to(torch.int32) * ROW_STATUS_FINAL_OVERSIZE
    # Regular-cadence targets, vectorized: B_{seq+1} = (8L+1) + seq·C
    # (final rows ignore targets inside the frontend).
    targets = (8 * lookahead + 1) + batch.chunk_sequence * cadence
    session_first = batch.chunk_sequence == 0

    # Pre-consume snapshot: the fixed-width mel-tail prefix is the
    # encoder's pre-encode cache (NeMo semantics: the FULL nine-slot
    # tail, zeros where not yet valid, with the CONFIGURED two-output
    # post-first-chunk drop — never a dynamic overlap); session-first
    # rows take no prefix and drop nothing.
    prefix = state.mel_tail.clone()

    new_frames, counts, row_status = advance_frontend(
        core.featurizer,
        batch.samples,
        batch.valid_samples,
        batch.final_tail,
        targets,
        raw_tail=state.raw_tail,
        mel_tail=state.mel_tail,
        counters=state.frontend_counters,
        pad_frames=pad_frames,
        row_status=incoming,
    )
    row_ok = row_status == 0
    counters[:, CTR_EXPECTED_CHUNK_SEQUENCE] += row_ok.to(torch.int64)

    # Per-row encoder input on the bucket's fixed grid: column j of
    # row b reads [prefix | new][j + 9 - p_b] — p_b is 9 for
    # continuing rows (the full nine-slot pre-encode cache) and 0 for
    # session-first rows. One batched gather, no phase branch.
    p = torch.where(
        session_first,
        torch.zeros_like(counts),
        torch.full_like(counts, MEL_TAIL_FRAMES),
    )
    combined = torch.cat([prefix, new_frames], dim=2)
    col = torch.arange(mel_width, device=device).view(1, 1, -1)
    gidx = (col + (MEL_TAIL_FRAMES - p).view(-1, 1, 1)).clamp(
        max=combined.shape[2] - 1
    )
    mel = combined.gather(
        2, gidx.expand(n_rows, combined.shape[1], mel_width)
    )
    mel_len = p + counts
    mel = torch.where(
        col < mel_len.view(-1, 1, 1), mel, mel.new_zeros(())
    )

    # Per-row pre-encode drop and logical encoder lengths; the padded
    # encoder width is the host formula on the fixed mel width.
    drop = torch.where(
        session_first,
        torch.zeros_like(counts),
        torch.full_like(counts, PRE_ENCODE_DROP),
    )
    enc_lengths = torch.where(
        counts > 0,
        torch.clamp(
            core.encoder.pre_encode.output_lengths(mel_len) - drop,
            min=0,
        ),
        torch.zeros_like(counts),
    )
    out_width = int(
        core.encoder.pre_encode.output_lengths(
            torch.tensor([mel_width])
        )[0]
    )

    caches = _GatheredCaches(state)
    with torch.no_grad():
        enc = stream_step(
            # _GatheredCaches is StreamingCaches' structural twin over
            # the gathered batch; stream_step reads only the shared
            # .channel/.time/.valid surface (the forward_step.py
            # precedent, migration-proven bit-for-bit).
            core.encoder, mel, caches,  # type: ignore[arg-type]
            out_offsets=drop,
            out_lengths=enc_lengths,
            out_width=out_width,
        )
        # Row-wise language conditioning in ONE call: the
        # conditioner takes the (B,) prompt tensor directly (no
        # per-prompt fragmentation or host set construction).
        conditioned = core.lid(enc, prompt_index=batch.prompt_index)
        # Padded-position zeroing for the conditioned stream (the
        # conditioner may bias padded rows away from zero; decode
        # masks by length, but captures and determinism want zeros).
        fcol = torch.arange(out_width, device=device).view(1, -1, 1)
        conditioned = torch.where(
            fcol < enc_lengths.view(-1, 1, 1),
            conditioned,
            conditioned.new_zeros(()),
        )
        decode = DecodeState(
            h=state.h.transpose(0, 1).contiguous(),
            c=state.c.transpose(0, 1).contiguous(),
            last_label=state.last_label,
        )
        token_ids, token_lengths, decode = decode_fn(
            conditioned, enc_lengths, core.predictor, core.joint, decode
        )
    state.h.copy_(decode.h.transpose(0, 1))
    state.c.copy_(decode.c.transpose(0, 1))
    state.last_label.copy_(decode.last_label)
    counters[:, CTR_ENCODED_MEL_FRAMES] = torch.where(
        row_ok,
        counters[:, CTR_COMMITTED_MEL_FRAMES],
        counters[:, CTR_ENCODED_MEL_FRAMES],
    )

    if not capture:
        return AdvanceResult(
            token_ids=token_ids,
            token_lengths=token_lengths,
            row_status=row_status,
        )
    # Capture lengths come from logical lengths (PORT-HOOK-001): a
    # zero-commit row stages all three tensors at zero length — its
    # mel capture is fully zero even where the encoder input carried
    # the pre-encode prefix.
    cap_mel_len = torch.where(
        counts > 0, mel_len, torch.zeros_like(mel_len)
    )
    return AdvanceResult(
        token_ids=token_ids,
        token_lengths=token_lengths,
        row_status=row_status,
        captures=PreparedCaptures(
            frontend_mel=torch.where(
                col < cap_mel_len.view(-1, 1, 1),
                mel,
                mel.new_zeros(()),
            ),
            mel_lengths=cap_mel_len,
            encoder_raw=enc,
            encoder_conditioned=conditioned,
            encoder_lengths=enc_lengths,
        ),
    )


#: The configured pre-encode overlap dropped from every non-first
#: chunk's encoder output (the checkpoint's nine-mel cache -> two
#: subsampled outputs; a value, never derived per row).
PRE_ENCODE_DROP = 2


class _StackedRows:
    """Per-layer gathered views behind stacked-tensor indexing:
    ``stream_step`` reads ``caches.channel.shape[2]`` and does
    ``caches.channel[idx]`` reads / slice-assign writes; reads return
    the (B, ...) view, writes copy through to the gathered tensors."""

    def __init__(self, views: list[torch.Tensor]) -> None:
        self._views = views

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self._views[idx]

    def __setitem__(self, idx: int, value: torch.Tensor) -> None:
        self._views[idx].copy_(value)

    @property
    def shape(self) -> tuple[int, ...]:
        return (len(self._views), *self._views[0].shape)


class _GatheredCaches:
    """``StreamingCaches``' surface over a gathered
    :class:`SessionStateBatch` (channel/time/valid/left_context) —
    ``stream_step`` advances the gathered views directly, so the
    golden-proven advance IS the scratch write."""

    def __init__(self, state: SessionStateBatch) -> None:
        self.channel = _StackedRows(state.channel)
        self.time = _StackedRows(state.time)
        self._window_valid = state.window_valid
        self.left_context = state.channel[0].shape[1]

    @property
    def valid(self) -> torch.Tensor:
        return self._window_valid[0].reshape(-1).to(torch.long)

    @valid.setter
    def valid(self, value: torch.Tensor) -> None:
        for slot in self._window_valid:
            slot.copy_(
                value.reshape(slot.shape).to(slot.dtype)
            )


# @spec PORT-ADV-003, PORT-STATE-007, PORT-STATE-008
def advance_model_rows(
    core: NemotronASRCore,
    input_ids: torch.Tensor,
    inputs_embeds: torch.Tensor,
    plan: RowPlan,
    *,
    channel_pools: list[torch.Tensor],
    time_pools: list[torch.Tensor],
    len_pools: list[torch.Tensor],
    h_pool: torch.Tensor,
    c_pool: torch.Tensor,
    queue_pool: torch.Tensor,
    book_pool: torch.Tensor,
    frontend_raw_pool: torch.Tensor,
    frontend_mel_pool: torch.Tensor,
    frontend_counter_pool: torch.Tensor,
    adapter: EmissionAdapter,
    placeholder_id: int,
    park_id: int,
    feat: int,
    capture_capacity: int = 0,
) -> torch.Tensor:
    """The ONE shared outer transaction (PORT-ADV-003).

    Fixed order (design §advance_session, steps 1–6):

    1. compose decode-then-prefill indices from ``plan`` and validate
       every structural row/page property before any page read
       (PORT-STATE-007): row count vs ``input_ids``/``inputs_embeds``;
       non-null (``!= null_block_id``), live, in-range
       (``< num_pool_blocks``) indices; uniqueness; speculative-column
       exclusion (extra decode columns); real/padding-row agreement;
    2. initialize fresh rows (``~plan.has_initial_states_p``) from
       scheduler metadata — never recycled page contents — then gather
       only the small session/emission book to validate replay echoes
       and classify each real row CHUNK / REPLAY / FLUSH;
    3. group CHUNK rows by execution profile + immutable geometry and
       gather their full frontend/encoder/predictor scratch; valid
       length stays a per-row tensor, not a batch-key component;
    4. run :func:`advance_session` over each CHUNK bucket into scratch;
    5. materialize and validate named captures beside each
       :class:`AdvanceResult`, reserve bounded result-sink capacity
       (``capture_capacity`` rows; 0 disables capture — with capture
       enabled, materialization/validation/reservation failure is
       pre-commit fatal for the transaction), then perform
       trusted-identity row validation and project the committed
       result through ``adapter``;
    6. commit recurrent/session state, emission bookkeeping (the
       persistent replay book), and returnable output together;
       publish the prepared capture record only for committed CHUNK
       rows, in request/chunk and checkpoint order.

    Suppression (PORT-STATE-008): any structural defect fails the whole
    call before a read; a trusted-identity row failure suppresses only
    that row; a bucket failure suppresses its bucket; a device failure
    suppresses all; no row scatters unless its emission result can be
    returned. Kernels never mutate resident pages directly.

    Returns:
        The runner's row output as projected by ``adapter`` (e.g. the
        ``(N, H)`` decision-carrier for MRV1).

    Raises:
        NotImplementedError: Always, at this tests-first stub.
    """
    raise NotImplementedError(
        "advance_model_rows lands in Phase 6 with the forward_step.py "
        "deletion (ledger P5-1)"
    )
