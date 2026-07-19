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

Both operations are implemented as of Phase 6c (the four contract
seams — plan provenance, decode resolver, capture reservation, and
the row-status handoff — are the design's §Phase-6c transaction
seams). ``forward_step.py`` / ``run_forward_step`` remain the legacy
path and regression oracle until the model ``forward`` is rewired
through :func:`advance_model_rows` and the pod parity matrix passes;
they are deleted in that same change (ledger P5-1 — a semantic split,
never a second legacy forward path).
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol

import torch

logger = logging.getLogger(__name__)

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
    ``is_chunk``: ``(num_decodes + num_prefills,)`` bool, the HOST
    role authority — True for a row whose scheduled token is the
    minted placeholder (design §Phase-6c transaction seams: sync-free
    bucketing requires host-side roles; the device token is the
    cross-check via ``ROW_STATUS_ROLE_MISMATCH``, never the bucketing
    authority).
    ``admission_generation``: ``(num_decodes + num_prefills,)`` long,
    the registry's admission generation per row — the capture
    record's block-reuse/ABA guard.

    All plan tensors are HOST-side (CPU) scheduler/registry authority:
    structural preflight validates them without any device
    synchronization, and the transaction uploads validated indices
    asynchronously for its device gathers.
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
    is_chunk: torch.Tensor
    admission_generation: torch.Tensor


@dataclass
class EmissionContext:
    """Per-call row context + GATHERED emission scratch the adapter
    consumes. The transaction gathers ``queue``/``book`` copies for
    the batch and later scatters the projection's updated scratch —
    the adapter NEVER mutates resident pools (the transaction is the
    sole scatter owner).

    ``roles``: ``(N,)`` long (``ROLE_CHUNK``/``ROLE_REPLAY``/
    ``ROLE_FLUSH``); ``input_ids``: ``(N,)`` long; ``chunk_rows``:
    ``(B,)`` long positions of the CHUNK rows within the N-row batch
    (merged-result row ``j`` is call row ``chunk_rows[j]``); ``queue``
    ``(N, cap)`` / ``book`` ``(N, 7)``: gathered emission scratch;
    ``row_status``: ``(N,)`` int32 — the transaction-composed FINAL
    per-row status (transition bits already folded in for CHUNK
    rows). The adapter trusts it and never recomputes protocol logic:
    any nonzero row emits park and keeps its scratch untouched.
    """

    roles: torch.Tensor
    input_ids: torch.Tensor
    chunk_rows: torch.Tensor
    queue: torch.Tensor
    book: torch.Tensor
    row_status: torch.Tensor


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

#: Transaction-owned per-row status bits — the PORT-ADV-004 vocabulary
#: extension settled at the Phase-6c preflight (design §Phase-6c
#: transaction seams; PORT-DEC-007 as amended). The frontend owns bits
#: 1..256 (``frontend.ROW_STATUS_*``); these continue the same int32
#: mask for defects the OUTER transaction detects around the CHUNK
#: transition. Any set bit makes the row a safe masked no-op — park
#: emission only, no state scatter — and the session is resolved at
#: the single batched asynchronous status readback, never by a raise
#: or device assertion on the trusted-identity path.
ROW_STATUS_ECHO_MISMATCH = 512
ROW_STATUS_QUEUE_NOT_DRAINED = 1024
ROW_STATUS_ROLE_MISMATCH = 2048
ROW_STATUS_ENVELOPE = 4096
ROW_STATUS_BOOK_IDENTITY = 8192
ROW_STATUS_SESSION_PROTOCOL = 16384

#: Model-row roles at a scheduler step (the transaction's own
#: vocabulary; values match the retiring ``forward_ops`` constants for
#: continuity). CHUNK comes from HOST authority (``RowPlan.is_chunk``);
#: REPLAY is an echo-armed non-chunk row; FLUSH is everything else and
#: is validated (drained + finalized) before it may emit park.
ROLE_CHUNK = 0
ROLE_REPLAY = 1
ROLE_FLUSH = 2


@dataclass(frozen=True)
class DecodeRequest:
    """One profile+geometry bucket's dispatch query (design §Phase-6c
    transaction seams).

    ``geometry``: the bucket's admitted geometry id.
    ``execution_batch_size``: the PADDED execution tier the decode
    actually runs at — never merely the live-row count.
    ``graph_covers_decode``: invocation-specific graph coverage; a
    covering graph structurally forces the capture-eligible arm.
    ``ready_decode_buckets``: decode buckets ready in this engine
    iteration — the multi-bucket compact-override input (the dispatch
    record's step-4 obligation: compact's occupancy-1.000 host lock is
    priced for single buckets only).
    """

    geometry: int
    execution_batch_size: int
    graph_covers_decode: bool
    ready_decode_buckets: int


@dataclass(frozen=True)
class ResolvedDecode:
    """The resolver's typed answer: the selected arm name (telemetry),
    its bound callable, and the reason when a safety rule — graph
    coverage or the multi-bucket override — displaced the table's raw
    selection."""

    arm: str
    decode_fn: RnntDecodeFn
    override_reason: str | None = None


#: The immutable startup-constructed dispatch seam: the transaction
#: invokes it exactly once per CHUNK bucket. Construction validates
#: table identity (``DispatchTable.validate_runtime``) and compiles an
#: O(1) lookup; the hot path performs no parsing, sorting, logging, or
#: synchronization. A transaction without a resolver fails before any
#: resident-state read (PORT-DEC-008: no hardcoded default).
DecodeResolver = Callable[[DecodeRequest], ResolvedDecode]


def make_table_resolver(
    table: Any,
    *,
    lane: str,
    arms: Mapping[str, RnntDecodeFn],
    max_batch: int,
) -> DecodeResolver:
    """Compile a validated :class:`DispatchTable` into a resolver.

    Startup-only: for every geometry id (``manifests.CADENCES`` order)
    and every execution batch size ``1..max_batch``, the measured-tier
    entries are resolved ONCE — nearest measured tier, sync-free arm
    when the bracketing tiers disagree (the crossover-gap rule) —
    into a dense lookup, with one aggregate log line per geometry
    instead of per-lookup warnings. The returned resolver then:

    - forces ``dense-graphed`` whenever the invocation's engine graph
      covers decode (compact is capture-ineligible by construction);
    - forces the sync-free eager arm whenever more than one decode
      bucket is ready in the iteration (the multi-bucket override),
      recording ``override_reason`` for telemetry;
    - otherwise returns the compiled table arm.

    The caller has already run ``table.validate_runtime`` for its
    deployment profile; this factory trusts the table's content.

    Args:
        table: the validated ``decode_dispatch.DispatchTable``.
        lane: the execution/precision lane key (e.g. ``"fp32"``).
        arms: arm name -> decode callable bindings (``rnnt``'s two
            candidates; ``dense-graphed`` binds the dense callable).
        max_batch: the largest execution batch size to compile
            (resolved ``max_num_seqs``); larger requests clamp.

    Returns:
        The frozen resolver.

    Raises:
        KeyError: a geometry with no measured tiers, or an arm the
            ``arms`` mapping does not bind — startup failures, never
            hot-path ones.
    """
    from vllm_omni.model_executor.models.nemotron_asr.decode_dispatch import (
        SYNC_FREE_ARMS,
    )
    from vllm_omni.model_executor.models.nemotron_asr.manifests import (
        CADENCES,
    )

    labels = list(CADENCES)
    for required in ("dense-graphed", "dense-eager"):
        if required not in arms:
            raise KeyError(f"no callable bound for arm {required!r}")
    compiled: list[tuple[str, ...]] = []
    for label in labels:
        tiers = sorted(
            t
            for (g, t, la) in table.entries
            if g == label and la == lane
        )
        if not tiers:
            raise KeyError(f"no measured tiers for {label!r}/{lane!r}")
        by_batch: list[str] = [""]
        fallbacks = 0
        for batch in range(1, max_batch + 1):
            if batch in tiers:
                by_batch.append(table.entries[(label, batch, lane)])
                continue
            fallbacks += 1
            lower = max((t for t in tiers if t < batch), default=None)
            upper = min((t for t in tiers if t > batch), default=None)
            if lower is None:
                assert upper is not None
                by_batch.append(table.entries[(label, upper, lane)])
            elif upper is None:
                by_batch.append(table.entries[(label, lower, lane)])
            else:
                low_arm = table.entries[(label, lower, lane)]
                high_arm = table.entries[(label, upper, lane)]
                if low_arm != high_arm:
                    by_batch.append(
                        low_arm
                        if low_arm in SYNC_FREE_ARMS
                        else high_arm
                    )
                else:
                    by_batch.append(
                        low_arm
                        if (batch - lower) <= (upper - batch)
                        else high_arm
                    )
        for arm in set(by_batch[1:]):
            if arm not in arms:
                raise KeyError(f"no callable bound for arm {arm!r}")
        logger.info(
            "decode dispatch %s/%s: compiled batches 1..%d "
            "(%d unmeasured resolved by nearest/crossover-gap)",
            label, lane, max_batch, fallbacks,
        )
        compiled.append(tuple(by_batch))
    compiled_t = tuple(compiled)
    sync_free_eager = "dense-eager"

    def resolve(request: DecodeRequest) -> ResolvedDecode:
        if request.graph_covers_decode:
            return ResolvedDecode(
                arm="dense-graphed",
                decode_fn=arms["dense-graphed"],
                override_reason="graph-covers-decode",
            )
        batch = min(max(request.execution_batch_size, 1), max_batch)
        arm = compiled_t[request.geometry][batch]
        if (
            request.ready_decode_buckets > 1
            and arm not in SYNC_FREE_ARMS
        ):
            return ResolvedDecode(
                arm=sync_free_eager,
                decode_fn=arms[sync_free_eager],
                override_reason="multi-bucket-serialization-guard",
            )
        return ResolvedDecode(arm=arm, decode_fn=arms[arm])

    return resolve


@dataclass(frozen=True)
class CapturePlan:
    """The exact bounded footprint of one call's prepared capture
    records: record count AND payload bytes (PORT-HOOK-001 — a sink
    reserves both before any resident-state commit)."""

    rows: int
    payload_bytes: int


@dataclass(frozen=True)
class CaptureRecord:
    """One committed CHUNK row's named-capture record.

    Host identity: ``row`` (call row position), ``block_id``, the
    registry's ``admission_generation`` (the block-reuse/ABA guard),
    and the admitted ``geometry``. Device identity/tensors stay
    GPU-resident scalar views/rows of the prepared padded captures —
    publication is an in-memory handoff; any D2H or durable I/O is
    the consumer's concern outside the transaction.
    """

    row: int
    block_id: int
    admission_generation: int
    geometry: int
    chunk_sequence: torch.Tensor
    prompt_index: torch.Tensor
    row_status: torch.Tensor
    frontend_mel: torch.Tensor
    mel_length: torch.Tensor
    encoder_raw: torch.Tensor
    encoder_conditioned: torch.Tensor
    encoder_length: torch.Tensor


class CaptureReservation(Protocol):
    """Reserved sink slots for one call's prepared records.

    ``publish`` transfers already-prepared records into the reserved
    slots — by construction no-fail: no allocation, serialization,
    copying, or durable I/O after commit. ``cancel`` releases the
    reservation idempotently on any earlier failure.
    """

    def publish(self, records: Sequence[CaptureRecord]) -> None: ...

    def cancel(self) -> None: ...


class CaptureSink(Protocol):
    """A model/worker-owned bounded capture sink (PORT-HOOK-001).

    ``reserve`` is called pre-commit with the exact
    :class:`CapturePlan`; failure to reserve is pre-commit fatal for
    the transaction. Implementations are bounded and concurrency-safe;
    ``capture_sink=None`` disables capture creation entirely,
    including capture-only allocations.
    """

    def reserve(self, plan: CapturePlan) -> CaptureReservation: ...


class StatusSink(Protocol):
    """The transaction's single batched asynchronous status handoff.

    Called exactly once per call at the commit boundary with the
    device-resident ``(N,)`` int32 row status in plan row order. The
    implementation stages a non-blocking D2H into model-owned pinned
    storage; the host consumes it at the normal scheduling boundary —
    protocol bits terminate the session, invariant bits escalate as
    port defects (PORT-STATE-008 tiers). Never synchronizes in the
    hot path.
    """

    def stage(self, row_status: torch.Tensor) -> None: ...


def make_mrv1_adapter(
    *,
    hidden_size: int,
    park_id: int,
    blank_id: int,
) -> EmissionAdapter:
    """Build the MRV1 emission adapter (the correctness fallback,
    PORT-DEC-010's burst adapter arrives behind the same seam).

    The returned adapter is POOL-FREE and a pure tensor transform: it
    consumes the context's gathered queue/book scratch and the merged
    committed :class:`AdvanceResult`, and returns an
    :class:`EmissionProjection` — the first burst label as each CHUNK
    row's decision carrier (head=1 past it, pending-echo and
    expected-label armed), the next queued label for a validated
    REPLAY row, and ``park_id`` for drained, FLUSH, zero-burst
    (PORT-DEC-004), and status-masked rows. Park never arms an echo
    (PORT-DEC-003) and blank never appears (bursts are nonblank by
    the PORT-DEC-001 decode contract — ``blank_id`` documents the
    pinned factory identity). The adapter issues no host/device
    synchronization and mutates no resident pool; the transaction
    scatters.
    """
    del blank_id  # bursts are nonblank by contract; id pins identity

    def adapter(
        result: AdvanceResult, context: EmissionContext
    ) -> EmissionProjection:
        book = context.book
        queue = context.queue
        n = book.shape[0]
        device = book.device
        from vllm_omni.model_executor.models.nemotron_asr.rnnt import (
            BOOK_EXPECTED_LABEL,
            BOOK_PENDING_ECHO,
            QUEUE_HEAD,
            QUEUE_LEN,
        )

        ok = context.row_status == 0
        is_replay = context.roles == ROLE_REPLAY
        head = book[:, QUEUE_HEAD].long()
        length = book[:, QUEUE_LEN].long()

        queue_out = queue.clone()
        book_out = book.clone()
        decisions = torch.full(
            (n,), park_id, dtype=torch.long, device=device
        )

        # ---- CHUNK rows: burst into the queue, first label out ----
        chunk_rows = context.chunk_rows
        b = int(chunk_rows.shape[0])  # host shape, not a sync
        if b:
            burst = result.token_ids.to(queue.dtype)
            blen = result.token_lengths.long()
            k = int(burst.shape[1])
            cok = ok.index_select(0, chunk_rows)
            old_q = queue_out.index_select(0, chunk_rows)
            new_q = torch.zeros_like(old_q)
            if k:
                new_q[:, : min(k, new_q.shape[1])] = burst[
                    :, : new_q.shape[1]
                ]
            queue_out.index_copy_(
                0, chunk_rows,
                torch.where(cok.unsqueeze(1), new_q, old_q),
            )
            has = blen > 0
            first = (
                burst[:, 0].long() if k else torch.zeros_like(blen)
            )
            decision_c = torch.where(
                has, first, torch.full_like(blen, park_id)
            )
            old_b = book_out.index_select(0, chunk_rows)
            new_b = old_b.clone()
            dt = new_b.dtype
            new_b[:, QUEUE_HEAD] = has.to(dt)  # 1 past the emitted
            new_b[:, QUEUE_LEN] = blen.to(dt)
            new_b[:, BOOK_PENDING_ECHO] = has.to(dt)
            new_b[:, BOOK_EXPECTED_LABEL] = torch.where(
                has, first, torch.zeros_like(first)
            ).to(dt)
            book_out.index_copy_(
                0, chunk_rows,
                torch.where(cok.unsqueeze(1), new_b, old_b),
            )
            decisions.index_copy_(
                0, chunk_rows,
                torch.where(
                    cok, decision_c, torch.full_like(decision_c, park_id)
                ),
            )

        # ---- REPLAY rows: validated echo → next label or park ----
        rep = ok & is_replay
        drained = head >= length
        cap = queue.shape[1]
        next_label = (
            queue.gather(
                1, head.clamp(max=max(cap - 1, 0)).unsqueeze(1)
            )
            .squeeze(1)
            .long()
        )
        emit_rep = rep & ~drained
        decisions = torch.where(emit_rep, next_label, decisions)
        dt = book_out.dtype
        book_out[:, QUEUE_HEAD] = torch.where(
            emit_rep, (head + 1).to(dt), book_out[:, QUEUE_HEAD]
        )
        book_out[:, BOOK_EXPECTED_LABEL] = torch.where(
            emit_rep,
            next_label.to(dt),
            book_out[:, BOOK_EXPECTED_LABEL],
        )
        # An emitting replay keeps its echo armed for the emitted
        # label; a drained replay parks and clears it (park never
        # arms an echo). FLUSH and masked rows keep their book.
        book_out[:, BOOK_PENDING_ECHO] = torch.where(
            emit_rep,
            torch.ones_like(book_out[:, BOOK_PENDING_ECHO]),
            torch.where(
                rep & drained,
                torch.zeros_like(book_out[:, BOOK_PENDING_ECHO]),
                book_out[:, BOOK_PENDING_ECHO],
            ),
        )

        # Decision carrier: slot 0 of each runner row. Written
        # directly, not through the legacy checked helper — its
        # min/max range guard is a host synchronization
        # (PORT-ADV-004); the id range is a STARTUP invariant
        # (num_logits < 2**24 keeps every id fp32-integer-exact).
        rows = torch.zeros(
            n, hidden_size, dtype=torch.float32, device=device
        )
        rows[:, 0] = decisions.to(rows.dtype)
        return EmissionProjection(
            rows=rows, queue=queue_out, book=book_out
        )

    return adapter


# @spec PORT-ADV-001, PORT-ADV-004
def advance_session(
    core: NemotronASRCore,
    batch: ChunkBatch,
    state: SessionStateBatch,
    *,
    geometry: int,
    decode_fn: RnntDecodeFn,
    capture: bool = False,
    row_status: torch.Tensor | None = None,
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
        row_status: optional ``(B,)`` int32 incoming status carrying
            transaction-owned protocol bits (``ROW_STATUS_QUEUE_NOT_
            DRAINED``, ``ROW_STATUS_ROLE_MISMATCH``, ``ROW_STATUS_
            ENVELOPE``, ``ROW_STATUS_BOOK_IDENTITY``) composed BEFORE
            the transition's own envelope-protocol predicates; a row
            arriving with any bit set is a masked no-op end to end.
            ``None`` means all rows arrive clean. Never mutated.

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
    if row_status is None:
        incoming = torch.zeros(
            n_rows, dtype=torch.int32, device=device
        )
    else:
        incoming = row_status.to(
            device=device, dtype=torch.int32
        ).clone()
    incoming |= (
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


def _h2d(t: torch.Tensor, device: torch.device) -> torch.Tensor:
    """Host-authority tensor to the compute device without a hot-path
    synchronization: pinned staging + non-blocking copy on CUDA (a
    reusable model-owned staging buffer is a later allocation-audit
    refinement), plain move elsewhere."""
    if device.type == "cuda":
        return t.pin_memory().to(device, non_blocking=True)
    return t.to(device)


def _structural_preflight(
    plan: RowPlan, n_ids: int, n_embeds: int
) -> torch.Tensor:
    """PORT-STATE-007: whole-call structural validation, HOST-side
    over the plan's CPU authority tensors, before any page read.

    Returns:
        The composed decode-then-prefill CPU index tensor.

    Raises:
        ValueError: naming the first structural defect.
    """
    n_real = plan.num_decodes + plan.num_prefills
    if n_ids != n_real or n_embeds != n_real:
        raise ValueError(
            f"row count mismatch: plan has {n_real} real rows, "
            f"input_ids {n_ids}, inputs_embeds {n_embeds}"
        )
    d = plan.state_indices_d
    if d.dim() != 2:
        raise ValueError(
            f"state_indices_d must be (rows, K); got dim {d.dim()}"
        )
    if d.shape[1] != 1:
        raise ValueError(
            f"{d.shape[1]} decode index columns: extra speculative "
            "columns are a configuration error, never values to "
            "flatten (PORT-STATE-007)"
        )
    if d.shape[0] < plan.num_decodes:
        raise ValueError(
            f"state_indices_d has {d.shape[0]} rows for "
            f"{plan.num_decodes} real decodes"
        )
    real_d = d[: plan.num_decodes, 0]
    pad_d = d[plan.num_decodes :, 0]
    if pad_d.numel() and bool((pad_d != plan.null_block_id).any()):
        raise ValueError(
            "graph-padding rows must carry the null block id "
            f"{plan.null_block_id} (real/padding mismatch)"
        )
    p = plan.state_indices_p
    if p.dim() != 1 or p.shape[0] != plan.num_prefills:
        raise ValueError(
            f"state_indices_p shape {tuple(p.shape)} for "
            f"{plan.num_prefills} prefills"
        )
    if plan.has_initial_states_p.shape[0] != plan.num_prefills:
        raise ValueError("has_initial_states_p shape mismatch")
    per_row = (
        ("geometry_id", plan.geometry_id, n_real),
        ("is_chunk", plan.is_chunk, n_real),
        ("admission_generation", plan.admission_generation, n_real),
        ("prompt_index", plan.prompt_index, plan.num_prefills),
    )
    for name, t, want in per_row:
        if t.shape[0] != want:
            raise ValueError(
                f"plan.{name} has {t.shape[0]} rows, expected {want}"
            )
    idx = torch.cat([real_d, p])
    if idx.numel():
        if bool((idx == plan.null_block_id).any()):
            raise ValueError("null block id claimed by a real row")
        if bool((idx < 0).any()) or bool(
            (idx >= plan.num_pool_blocks).any()
        ):
            raise ValueError(
                f"state index out of range [0, {plan.num_pool_blocks})"
            )
        if int(torch.unique(idx).numel()) != int(idx.numel()):
            raise ValueError(
                "duplicate state index across the decode/prefill "
                "composition"
            )
        if bool((~torch.isin(idx, plan.live_block_ids)).any()):
            raise ValueError(
                "state index outside the live (registry-allocated) "
                "block set"
            )
    return idx


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
    decode_resolver: DecodeResolver,
    placeholder_id: int,
    park_id: int,
    capture_sink: CaptureSink | None = None,
    status_sink: StatusSink | None = None,
    graph_covers_decode: bool = False,
) -> torch.Tensor:
    """The ONE shared outer transaction (PORT-ADV-003).

    Fixed order (design §advance_session, steps 1–6, with the
    Phase-6c seams — design §Phase-6c transaction seams):

    1. compose decode-then-prefill indices from ``plan`` and validate
       every structural row/page property HOST-SIDE before any page
       read (PORT-STATE-007): row count vs ``input_ids``/
       ``inputs_embeds``; non-null (``!= null_block_id``), live,
       in-range (``< num_pool_blocks``) indices; uniqueness;
       speculative-column exclusion (extra decode columns);
       real/padding-row agreement — then upload the validated indices
       asynchronously for the device gathers;
    2. initialize fresh rows (``~plan.has_initial_states_p``) from
       plan authority — never recycled page contents — then gather
       only the small session/emission book; evaluate replay echoes
       and the host-role cross-check ON DEVICE into the transaction's
       ``ROW_STATUS_*`` bits (echo mismatch is row-local after
       trusted preflight, PORT-DEC-007 as amended); classify each
       real row CHUNK / REPLAY / FLUSH;
    3. group CHUNK rows by execution profile + immutable geometry
       (host authority: ``plan.is_chunk`` + ``plan.geometry_id``) and
       gather their full frontend/encoder/predictor scratch; valid
       length stays a per-row tensor, not a batch-key component;
    4. resolve the decode callable once per bucket through
       ``decode_resolver`` (typed :class:`DecodeRequest` — padded
       execution tier, invocation graph coverage, ready decode
       buckets) and run :func:`advance_session` over each bucket into
       scratch;
    5. prepare immutable capture records beside each
       :class:`AdvanceResult` and reserve their exact bounded
       footprint through ``capture_sink`` BEFORE any resident commit
       (reservation failure is pre-commit fatal; ``None`` disables
       capture with zero capture-only allocations), then project the
       committed result through ``adapter``;
    6. commit recurrent/session state, emission bookkeeping (the
       persistent replay book), and returnable output together as
       masked no-allocation writes; stage the single batched
       asynchronous row-status handoff through ``status_sink``; and
       publish the prepared records into the reserved slots — a
       no-fail in-memory handoff — only for committed CHUNK rows, in
       request/chunk and checkpoint order.

    Suppression (PORT-STATE-008): any structural defect fails the
    whole call before a read; an expected per-row defect resolves to
    a device status bit and a masked park-only no-op; a bucket
    failure suppresses its bucket; an unexpected Python/CUDA failure
    aborts the whole call before any scatter and propagates (a CUDA
    context failure is worker-fatal, never row suppression). No row
    scatters unless its emission result can be returned. Kernels
    never mutate resident pages directly.

    Returns:
        The runner's row output as projected by ``adapter`` (e.g. the
        ``(N, H)`` decision-carrier for MRV1).

    Raises:
        ValueError: a structural mapping defect, a missing/invalid
            resolver, or a trusted-identity projection-shape defect —
            always before any resident mutation.
    """
    from vllm_omni.model_executor.models.nemotron_asr.frontend import (
        CTR_FINALIZED,
    )
    from vllm_omni.model_executor.models.nemotron_asr.manifests import (
        CADENCES,
    )
    from vllm_omni.model_executor.models.nemotron_asr.rnnt import (
        BOOK_EXPECTED_LABEL,
        BOOK_GEOMETRY,
        BOOK_PENDING_ECHO,
        QUEUE_HEAD,
        QUEUE_LAST_LABEL,
        QUEUE_LEN,
        QUEUE_PROMPT,
    )

    if not callable(decode_resolver):
        raise ValueError(
            "advance_model_rows requires a decode resolver: dispatch "
            "is never a hardcoded default (PORT-DEC-008)"
        )
    n_real = plan.num_decodes + plan.num_prefills
    hidden = int(inputs_embeds.shape[1])
    device = inputs_embeds.device
    idx_cpu = _structural_preflight(
        plan, int(input_ids.shape[0]), int(inputs_embeds.shape[0])
    )
    lookaheads = [right for (_, right) in CADENCES.values()]
    if n_real and (
        bool((plan.geometry_id < 0).any())
        or bool((plan.geometry_id >= len(lookaheads)).any())
    ):
        raise ValueError("plan.geometry_id outside the admitted set")

    # ---- step 2: small book/counter gather + metadata fresh init ----
    didx = _h2d(idx_cpu, device)
    book = book_pool.index_select(0, didx)
    queue = queue_pool.index_select(0, didx)
    counters_small = frontend_counter_pool.index_select(0, didx)
    fresh_local = (~plan.has_initial_states_p).nonzero(as_tuple=True)[0]
    fresh_set: set[int] = set()
    if fresh_local.numel():
        fresh_pos_cpu = plan.num_decodes + fresh_local
        fresh_set = set(fresh_pos_cpu.tolist())
        finit = torch.zeros(
            int(fresh_local.numel()), book.shape[1], dtype=torch.int64
        )
        finit[:, QUEUE_LAST_LABEL] = int(core.blank_id)
        finit[:, QUEUE_PROMPT] = plan.prompt_index.index_select(
            0, fresh_local
        )
        finit[:, BOOK_GEOMETRY] = plan.geometry_id.index_select(
            0, fresh_pos_cpu
        )
        fresh_dev = _h2d(fresh_pos_cpu, device)
        book.index_copy_(0, fresh_dev, _h2d(finit, device).to(book.dtype))
        queue.index_fill_(0, fresh_dev, 0)
        counters_small.index_fill_(0, fresh_dev, 0)

    # ---- device role/protocol composition (no host readback) ----
    is_chunk_dev = _h2d(plan.is_chunk, device)
    plan_geom_dev = _h2d(plan.geometry_id, device)
    ids_dev = input_ids.long()
    pending = book[:, BOOK_PENDING_ECHO] != 0
    remaining = (
        book[:, QUEUE_LEN].long() - book[:, QUEUE_HEAD].long()
    )
    finalized = counters_small[:, CTR_FINALIZED] != 0
    status = torch.zeros(n_real, dtype=torch.int32, device=device)
    status |= (
        is_chunk_dev != (ids_dev == placeholder_id)
    ).to(torch.int32) * ROW_STATUS_ROLE_MISMATCH
    status |= (
        (~is_chunk_dev)
        & pending
        & (ids_dev != book[:, BOOK_EXPECTED_LABEL].long())
    ).to(torch.int32) * ROW_STATUS_ECHO_MISMATCH
    status |= (
        is_chunk_dev & (pending | (remaining > 0))
    ).to(torch.int32) * ROW_STATUS_QUEUE_NOT_DRAINED
    status |= (
        (~is_chunk_dev)
        & (~pending)
        & ((remaining > 0) | (~finalized))
    ).to(torch.int32) * ROW_STATUS_SESSION_PROTOCOL
    status |= (
        book[:, BOOK_GEOMETRY].long() != plan_geom_dev
    ).to(torch.int32) * ROW_STATUS_BOOK_IDENTITY
    roles = torch.full(
        (n_real,), ROLE_FLUSH, dtype=torch.long, device=device
    )
    roles = torch.where(
        (~is_chunk_dev) & pending,
        torch.full_like(roles, ROLE_REPLAY),
        roles,
    )
    roles = torch.where(
        is_chunk_dev, torch.full_like(roles, ROLE_CHUNK), roles
    )

    # ---- step 3/4: geometry buckets → gather → advance_session ----
    bucket_map: dict[int, list[int]] = {}
    for pos in plan.is_chunk.nonzero(as_tuple=True)[0].tolist():
        bucket_map.setdefault(int(plan.geometry_id[pos]), []).append(pos)
    ordered = sorted(bucket_map.items())
    ready = len(ordered)
    hop = int(core.featurizer.hop_length)
    capture_on = capture_sink is not None
    executed: list[dict[str, Any]] = []
    for g, pos_list in ordered:
        resolved = decode_resolver(
            DecodeRequest(
                geometry=g,
                execution_batch_size=len(pos_list),
                graph_covers_decode=graph_covers_decode,
                ready_decode_buckets=ready,
            )
        )
        rows_dev = _h2d(
            torch.tensor(pos_list, dtype=torch.long), device
        )
        blocks_dev = didx.index_select(0, rows_dev)
        cadence = 8 * (lookaheads[g] + 1)
        s_g = cadence * hop
        if ENVELOPE_HEADER_SLOTS + s_g > hidden:
            raise ValueError(
                f"carrier width {hidden} cannot hold geometry {g}'s "
                f"{s_g}-sample cadence"
            )
        env = inputs_embeds.index_select(0, rows_dev)
        hdr = env[:, :ENVELOPE_HEADER_SLOTS]
        final_col = hdr[:, ENV_FINAL_TAIL]
        valid_col = hdr[:, ENV_VALID_SAMPLES]
        prompt_col = hdr[:, ENV_PROMPT_INDEX]
        env_bad = hdr[:, ENV_VERSION] != ENVELOPE_VERSION
        env_bad |= (hdr != hdr.trunc()).any(dim=1)
        env_bad |= (final_col != 0) & (final_col != 1)
        env_bad |= valid_col > s_g
        env_bad |= (final_col == 0) & (valid_col != s_g)
        env_bad |= (prompt_col < 0) | (
            prompt_col >= core.lid.num_prompts
        )
        tail = env[:, ENVELOPE_HEADER_SLOTS + s_g :]
        if tail.shape[1]:
            env_bad |= (tail != 0).any(dim=1)
        incoming = status.index_select(0, rows_dev) | (
            env_bad.to(torch.int32) * ROW_STATUS_ENVELOPE
        )
        batch = ChunkBatch(
            samples=env[
                :, ENVELOPE_HEADER_SLOTS : ENVELOPE_HEADER_SLOTS + s_g
            ],
            valid_samples=valid_col.long(),
            geometry_id=hdr[:, ENV_GEOMETRY_ID].long(),
            final_tail=final_col != 0,
            prompt_index=prompt_col.long(),
            chunk_sequence=hdr[:, ENV_CHUNK_SEQUENCE].long(),
        )
        state = SessionStateBatch(
            raw_tail=frontend_raw_pool.index_select(0, blocks_dev),
            mel_tail=frontend_mel_pool.index_select(0, blocks_dev),
            frontend_counters=counters_small.index_select(0, rows_dev),
            channel=[
                pool.index_select(0, blocks_dev)
                for pool in channel_pools
            ],
            window_valid=[
                pool.index_select(0, blocks_dev) for pool in len_pools
            ],
            time=[
                pool.index_select(0, blocks_dev) for pool in time_pools
            ],
            h=h_pool.index_select(0, blocks_dev),
            c=c_pool.index_select(0, blocks_dev),
            last_label=book.index_select(0, rows_dev)[
                :, QUEUE_LAST_LABEL
            ].long(),
        )
        fresh_in_bucket = [
            i for i, pos in enumerate(pos_list) if pos in fresh_set
        ]
        if fresh_in_bucket:
            fdev = _h2d(
                torch.tensor(fresh_in_bucket, dtype=torch.long), device
            )
            scratch = [
                state.raw_tail, state.mel_tail, state.h, state.c,
                *state.channel, *state.window_valid, *state.time,
            ]
            for t in scratch:
                t.index_fill_(0, fdev, 0)
        result = advance_session(
            core, batch, state,
            geometry=g,
            decode_fn=resolved.decode_fn,
            capture=capture_on,
            row_status=incoming,
        )
        rs = (
            result.row_status
            if result.row_status is not None
            else incoming
        )
        status.index_copy_(0, rows_dev, rs)
        # The transition's advanced last_label goes back into the book
        # copy (masked rows returned it unchanged — a bit-level no-op).
        brows = book.index_select(0, rows_dev)
        brows[:, QUEUE_LAST_LABEL] = state.last_label.to(brows.dtype)
        book.index_copy_(0, rows_dev, brows)
        executed.append({
            "geometry": g,
            "pos": pos_list,
            "rows_dev": rows_dev,
            "blocks_dev": blocks_dev,
            "state": state,
            "result": result,
            "batch": batch,
            "rs": rs,
        })

    # ---- step 5: prepared records + reservation BEFORE commit ----
    chunk_all = [pos for _, pos_list in ordered for pos in pos_list]
    chunk_all_dev = _h2d(
        torch.tensor(chunk_all, dtype=torch.long), device
    )
    b_total = len(chunk_all)
    reservation: CaptureReservation | None = None
    records: list[CaptureRecord] = []
    if capture_on and b_total:
        for ex in executed:
            caps = ex["result"].captures
            if caps is None:
                raise ValueError(
                    "capture enabled but the transition staged no "
                    "captures (PORT-HOOK-001 pre-commit fatal)"
                )
            for i, pos in enumerate(ex["pos"]):
                records.append(
                    CaptureRecord(
                        row=pos,
                        block_id=int(idx_cpu[pos]),
                        admission_generation=int(
                            plan.admission_generation[pos]
                        ),
                        geometry=int(ex["geometry"]),
                        chunk_sequence=ex["batch"].chunk_sequence[i],
                        prompt_index=ex["batch"].prompt_index[i],
                        row_status=ex["rs"][i],
                        frontend_mel=caps.frontend_mel[i],
                        mel_length=caps.mel_lengths[i],
                        encoder_raw=caps.encoder_raw[i],
                        encoder_conditioned=caps.encoder_conditioned[i],
                        encoder_length=caps.encoder_lengths[i],
                    )
                )
        records.sort(key=lambda r: r.row)
        payload = sum(
            r.frontend_mel.numel() * r.frontend_mel.element_size()
            + r.encoder_raw.numel() * r.encoder_raw.element_size()
            + r.encoder_conditioned.numel()
            * r.encoder_conditioned.element_size()
            for r in records
        )
        assert capture_sink is not None
        reservation = capture_sink.reserve(
            CapturePlan(rows=len(records), payload_bytes=payload)
        )

    # ---- step 5b/6: projection, then the no-fail masked commit ----
    k_max = max(
        (int(ex["result"].token_ids.shape[1]) for ex in executed),
        default=0,
    )
    merged_ids = torch.zeros(
        b_total, k_max, dtype=torch.int32, device=device
    )
    merged_len = torch.zeros(
        b_total, dtype=torch.int32, device=device
    )
    row0 = 0
    for ex in executed:
        nb = len(ex["pos"])
        ids_b = ex["result"].token_ids
        merged_ids[row0 : row0 + nb, : ids_b.shape[1]] = ids_b
        merged_len[row0 : row0 + nb] = ex["result"].token_lengths
        row0 += nb
    merged = AdvanceResult(
        token_ids=merged_ids,
        token_lengths=merged_len,
        row_status=status.index_select(0, chunk_all_dev),
    )
    context = EmissionContext(
        roles=roles,
        input_ids=ids_dev,
        chunk_rows=chunk_all_dev,
        queue=queue,
        book=book,
        row_status=status,
    )
    try:
        projection = adapter(merged, context)
        if tuple(projection.rows.shape) != (n_real, hidden):
            raise ValueError(
                "adapter returned rows shaped "
                f"{tuple(projection.rows.shape)}, expected "
                f"{(n_real, hidden)}"
            )
        if tuple(projection.queue.shape) != tuple(queue.shape) or (
            tuple(projection.book.shape) != tuple(book.shape)
        ):
            raise ValueError(
                "adapter returned queue/book scratch with a different "
                "shape than it was given"
            )
        queue_commit = projection.queue.to(queue_pool.dtype)
        book_commit = projection.book.to(book_pool.dtype)
        # Commit: index_copy_ only — no allocation, masked rows wrote
        # bit-identical scratch, so this is the transactional scatter.
        for ex in executed:
            blocks = ex["blocks_dev"]
            st = ex["state"]
            for layer, pool in enumerate(channel_pools):
                pool.index_copy_(0, blocks, st.channel[layer])
            for layer, pool in enumerate(time_pools):
                pool.index_copy_(0, blocks, st.time[layer])
            for layer, pool in enumerate(len_pools):
                pool.index_copy_(
                    0, blocks,
                    st.window_valid[layer].to(pool.dtype),
                )
            h_pool.index_copy_(0, blocks, st.h)
            c_pool.index_copy_(0, blocks, st.c)
            frontend_raw_pool.index_copy_(0, blocks, st.raw_tail)
            frontend_mel_pool.index_copy_(0, blocks, st.mel_tail)
            frontend_counter_pool.index_copy_(
                0, blocks, st.frontend_counters
            )
        queue_pool.index_copy_(0, didx, queue_commit)
        book_pool.index_copy_(0, didx, book_commit)
    except BaseException:
        if reservation is not None:
            reservation.cancel()
        raise
    # ---- the single batched asynchronous status handoff ----
    if status_sink is not None:
        status_sink.stage(status)
    if reservation is not None:
        reservation.publish(records)
    return projection.rows
