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
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from vllm_omni.model_executor.models.nemotron_asr.nemotron_asr import (
        NemotronASRCore,
    )

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
class AdvanceResult:
    """The adapter-neutral result of one :func:`advance_session` call.

    ``bursts``: one ordered, bounded, NONBLANK label list per CHUNK row
    (``bursts[i]`` never contains ``blank_id``); each burst is bounded
    by ``valid_encoder_frames × max_symbols_per_step`` for that row's
    geometry (PORT-INT-002), NOT by the attention window.
    ``captures``: a dict with EXACTLY the keys ``frontend_mel``,
    ``encoder_raw``, ``encoder_conditioned`` — each a list of one
    tensor per row, in row order. Empty when capture is disabled.

    The result carries no MRV1/runner projection: a model-local
    emission adapter (selected in :func:`advance_model_rows`) turns
    these bursts into either MRV1 one-token rows plus the persistent
    replay book or a padded variable-length runner result.
    """

    bursts: list[list[int]]
    captures: dict[str, list[torch.Tensor]] = field(default_factory=dict)


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


#: A model-local emission adapter: projects a committed
#: :class:`AdvanceResult` into the runner's row shape (MRV1 one-token
#: rows + persistent replay book, or a padded variable-length burst).
#: Selection happens once at model init (runner-mode configuration),
#: and the selected adapter is passed into every
#: :func:`advance_model_rows` call.
EmissionAdapter = Callable[["AdvanceResult"], torch.Tensor]


def make_mrv1_adapter(
    *,
    queue_pool: torch.Tensor,
    book_pool: torch.Tensor,
    hidden_size: int,
    park_id: int,
    blank_id: int,
) -> EmissionAdapter:
    """Build the MRV1 emission adapter (the correctness fallback,
    PORT-DEC-010's burst adapter arrives behind the same seam).

    The returned adapter projects a committed :class:`AdvanceResult`
    into MRV1 one-token decision-carrier rows: first burst label out
    now, the remainder into the bound persistent replay queue/book
    (pending-echo and expected-label stamped for echo verification),
    ``park_id`` once a session's queue is drained.

    Raises:
        NotImplementedError: Always, at this tests-first stub — lands
            in Phase 6.
    """
    raise NotImplementedError(
        "make_mrv1_adapter lands in Phase 6 with the forward_step.py "
        "deletion (ledger P5-1)"
    )


# @spec PORT-ADV-001
def advance_session(
    core: NemotronASRCore,
    batch: ChunkBatch,
    state: SessionStateBatch,
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

    Args:
        core: the assembled pipeline (encoder / lid / predictor /
            joint / featurizer).
        batch: raw PCM plus validated controls, one row per CHUNK.
        state: gathered checkpoint state, read and mutated into the
            next state.

    Returns:
        Per-row nonblank bursts and named captures.

    Raises:
        ValueError: on a control-field violation (non-integer value,
            out-of-range geometry/prompt, wrong chunk sequence), a
            frontend protocol error, or a design-margin violation
            (via ``advance_frontend``).
    """
    from vllm_omni.model_executor.models.nemotron_asr.encoder import (
        stream_step,
    )
    from vllm_omni.model_executor.models.nemotron_asr.frontend import (
        CTR_COMMITTED_MEL_FRAMES,
        CTR_ENCODED_MEL_FRAMES,
        CTR_EXPECTED_CHUNK_SEQUENCE,
        advance_frontend,
        cadence_boundary,
    )
    from vllm_omni.model_executor.models.nemotron_asr.manifests import (
        CADENCES,
    )
    from vllm_omni.model_executor.models.nemotron_asr.rnnt import (
        DecodeState,
        greedy_decode_batch,
    )

    n_rows = batch.samples.shape[0]
    lookaheads = [right for (_, right) in CADENCES.values()]

    # Validate controls (design §Chunk envelope): integer-valued,
    # in-range, in sequence — before any state mutation.
    targets = torch.zeros(n_rows, dtype=torch.long)
    for b in range(n_rows):
        geometry = int(batch.geometry_id[b])
        if not 0 <= geometry < len(lookaheads):
            raise ValueError(f"row {b}: geometry id {geometry} unknown")
        seq = int(batch.chunk_sequence[b])
        expected = int(
            state.frontend_counters[b, CTR_EXPECTED_CHUNK_SEQUENCE]
        )
        if seq != expected:
            raise ValueError(
                f"row {b}: chunk sequence {seq}, expected {expected}"
            )
        if int(batch.valid_samples[b]) < 0:
            raise ValueError(f"row {b}: negative valid_samples")
        targets[b] = cadence_boundary(
            seq + 1, lookahead=lookaheads[geometry]
        )

    # Pre-consume snapshot: encoder input = mel-tail prefix (the
    # pre-encode cache; empty on session-first) + this update's newly
    # committed frames.
    tail_valid = [
        min(
            int(state.frontend_counters[b, CTR_ENCODED_MEL_FRAMES]),
            state.mel_tail.shape[2],
        )
        for b in range(n_rows)
    ]
    prefixes = [
        state.mel_tail[b, :, state.mel_tail.shape[2] - tail_valid[b]:]
        .clone()
        for b in range(n_rows)
    ]

    new_frames, _counts = advance_frontend(
        core.featurizer,
        batch.samples,
        batch.valid_samples,
        batch.final_tail,
        targets,
        raw_tail=state.raw_tail,
        mel_tail=state.mel_tail,
        counters=state.frontend_counters,
    )

    # One geometry bucket per call is the caller's contract
    # (advance_model_rows groups by execution profile + geometry), so
    # every row shares the same encoder-input width here.
    mel_inputs = [
        torch.cat([prefixes[b], new_frames[b]], dim=1)
        for b in range(n_rows)
    ]
    widths = {m.shape[1] for m in mel_inputs}
    if len(widths) != 1:
        raise ValueError(
            f"heterogeneous encoder-input widths in one bucket: "
            f"{sorted(widths)} (caller must bucket by geometry/phase)"
        )
    mel = torch.stack(mel_inputs)
    drops = {-(-tail_valid[b] // 8) for b in range(n_rows)}
    if len(drops) != 1:
        raise ValueError(
            "heterogeneous pre-encode overlap in one bucket "
            f"(tail_valid={tail_valid})"
        )
    drop_extra = drops.pop()

    caches = _GatheredCaches(state)
    with torch.no_grad():
        enc = stream_step(
            # _GatheredCaches is StreamingCaches' structural twin over
            # the gathered batch; stream_step reads only the shared
            # .channel/.time/.valid surface (the forward_step.py
            # precedent, migration-proven bit-for-bit).
            core.encoder, mel, caches,  # type: ignore[arg-type]
            drop_extra=drop_extra,
        )
        conditioned = core.lid(enc, prompt_index=batch.prompt_index)
        decode = DecodeState(
            h=state.h.transpose(0, 1).contiguous(),
            c=state.c.transpose(0, 1).contiguous(),
            last_label=state.last_label,
        )
        bursts, decode = greedy_decode_batch(
            conditioned, core.predictor, core.joint, decode
        )
    state.h.copy_(decode.h.transpose(0, 1))
    state.c.copy_(decode.c.transpose(0, 1))
    state.last_label.copy_(decode.last_label)
    for b in range(n_rows):
        state.frontend_counters[b, CTR_ENCODED_MEL_FRAMES] = int(
            state.frontend_counters[b, CTR_COMMITTED_MEL_FRAMES]
        )
        state.frontend_counters[b, CTR_EXPECTED_CHUNK_SEQUENCE] += 1

    captures = {
        "frontend_mel": [mel[b] for b in range(n_rows)],
        "encoder_raw": [enc[b] for b in range(n_rows)],
        "encoder_conditioned": [conditioned[b] for b in range(n_rows)],
    }
    return AdvanceResult(bursts=bursts, captures=captures)


class _GatheredCaches:
    """``StreamingCaches``' surface over a gathered
    :class:`SessionStateBatch` (channel/time/valid/left_context) —
    ``stream_step`` advances the gathered views directly, so the
    golden-proven advance IS the scratch write."""

    def __init__(self, state: SessionStateBatch) -> None:
        self.channel = state.channel
        self.time = state.time
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
