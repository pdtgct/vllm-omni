# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The advance seam split (ledger P5-1; PORT-ADV-001/003).

``run_forward_step`` (``forward_step.py``) is being replaced by two
narrower operations that separate the storage-agnostic checkpoint
transition from the outer page-pool transaction:

- :func:`advance_session` — the ONE storage- and adapter-agnostic
  CHUNK transition (PORT-ADV-001). It never sees a pool, a
  ``state_indices`` tensor, a queue, or a book: callers gather tensor
  views (page-backed or otherwise) into a :class:`GatheredState` and
  read the mutated state back out.
- :func:`advance_model_rows` — the ONE shared outer transaction
  (PORT-ADV-003) every native/fallback/probe caller passes through:
  whole-call structural preflight (PORT-STATE-007) before any resident
  read, CHUNK/REPLAY/FLUSH classification, gather-only-for-CHUNK
  (PORT-STATE-008), the ``advance_session`` call, and a single commit.

This module is a tests-first stub (Phase 5): both functions raise
``NotImplementedError`` with complete typed signatures so pod tests
pin the future contract now and turn green in Phase 6, when
``forward_step.py`` / ``run_forward_step`` are deleted in the same
change (ledger P5-1 — a semantic split, never a second legacy forward
path).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from vllm_omni.model_executor.models.nemotron_asr.nemotron_asr import (
        NemotronASRCore,
    )


@dataclass(frozen=True)
class ChunkBatch:
    """Per-CHUNK-row gathered inputs for :func:`advance_session`.

    ``mel``: ``(B, feat, T)`` the packed mel for every CHUNK row.
    ``prompt_index``: ``(B,)`` long, the LID prompt slot per row.
    ``session_first``: ``(B,)`` bool — a fresh (never-advanced)
    session, so the encoder/predictor scratch in the paired
    :class:`GatheredState` reads as zeroed rather than resumed.
    ``drop_extra``: the pre-encode-overlap column count to drop from
    non-first chunks (0 for an all-session-first batch is the caller's
    choice, not this type's).
    """

    mel: torch.Tensor
    prompt_index: torch.Tensor
    session_first: torch.Tensor
    drop_extra: int


@dataclass
class GatheredState:
    """Storage-agnostic scratch :func:`advance_session` reads AND
    mutates in place as the next state.

    Per-encoder-layer ``channel`` / ``time`` / ``valid`` mirror
    ``StreamingCaches``' attribute surface (the window/conv/len-slot
    scratch); ``h`` / ``c`` are the predictor LSTM state; ``last_label``
    is ``(B,)`` long, the predictor's most recent input label. NO
    pools, NO ``state_indices``, NO queue/book of any kind lives here —
    those are ``advance_model_rows``' concern, never this type's.
    """

    channel: list[torch.Tensor]
    time: list[torch.Tensor]
    valid: list[torch.Tensor]
    h: torch.Tensor
    c: torch.Tensor
    last_label: torch.Tensor


@dataclass(frozen=True)
class AdvanceResult:
    """The result of one :func:`advance_session` call.

    ``bursts``: one ordered, bounded, NONBLANK label list per CHUNK
    row (``bursts[i]`` never contains ``blank_id``). ``captures``: a
    dict with EXACTLY the keys ``frontend_mel``, ``encoder_raw``, and
    ``encoder_conditioned`` — each a list of one tensor per row, in
    row order.
    """

    bursts: list[list[int]]
    captures: dict[str, list[torch.Tensor]] = field(default_factory=dict)


# @spec PORT-ADV-001
def advance_session(
    core: NemotronASRCore, batch: ChunkBatch, state: GatheredState
) -> AdvanceResult:
    """The ONE storage- and adapter-agnostic checkpoint transition.

    CHUNK-only: REPLAY and FLUSH rows never enter this function — the
    caller (``advance_model_rows``) filters them out before gathering
    ``batch``/``state``. No caller may independently implement
    frontend boundaries, encoder-cache advancement, language
    conditioning, predictor advancement, or emission ordering; every
    native, fallback, and probe path invokes this one transition
    (PORT-ADV-001).

    Args:
        core: the assembled pipeline (encoder / lid / predictor /
            joint).
        batch: per-CHUNK-row gathered inputs.
        state: the per-row scratch this call reads and mutates in
            place into the next state.

    Returns:
        The per-row nonblank bursts and named captures.

    Raises:
        NotImplementedError: Always, at this tests-first stub — lands
            in Phase 6 with ``forward_step.py``'s deletion.
    """
    raise NotImplementedError(
        "advance_session lands in Phase 6 with the forward_step.py "
        "deletion (ledger P5-1)"
    )


# @spec PORT-ADV-003, PORT-STATE-007, PORT-STATE-008
def advance_model_rows(
    core: NemotronASRCore,
    input_ids: torch.Tensor,
    inputs_embeds: torch.Tensor,
    *,
    channel_pools: list[torch.Tensor],
    time_pools: list[torch.Tensor],
    len_pools: list[torch.Tensor],
    h_pool: torch.Tensor,
    c_pool: torch.Tensor,
    queue_pool: torch.Tensor,
    book_pool: torch.Tensor,
    state_indices: torch.Tensor,
    num_real_rows: int,
    placeholder_id: int,
    park_id: int,
    feat: int,
    drop_extra: int,
) -> torch.Tensor:
    """The ONE shared outer transaction over page-backed session state.

    Every native and fallback model-row call, and every probe, passes
    through this operation (PORT-ADV-003):

    1. Whole-call structural preflight BEFORE any resident-state read
       (PORT-STATE-007): wrong row count against ``input_ids`` /
       ``inputs_embeds``; a null, non-live, or out-of-range index; a
       duplicate index; an extra speculative-decode column
       (``input_ids`` must be 1-D); a padding row inside the first
       ``num_real_rows``; or a real/padding mismatch. Any defect fails
       the WHOLE call — no state is read or written first.
    2. Decode-then-prefill index composition, MRV1 echo validation, and
       CHUNK/REPLAY/FLUSH classification from the small session/
       emission book alone.
    3. Fresh books for session-first rows, initialized without reading
       a recycled page's stale contents.
    4. Full frontend/encoder/predictor scratch gathered ONLY for CHUNK
       subbatches (PORT-STATE-008) — REPLAY/FLUSH touch only their
       required session/emission fields, and resident state is never
       mutated during compute.
    5. One :func:`advance_session` invocation over the gathered CHUNK
       scratch.
    6. Bucket-scratch compute with row/bucket/device failure
       suppression (PORT-STATE-008): a row failure suppresses only
       that row, a bucket failure suppresses every row in the bucket,
       a device failure suppresses every bucket, and no row scatters
       unless its emission result can be returned.
    7. Adapter projection and a single commit of the returnable
       output, the recurrent/session state, and the prepared capture
       record.

    Args:
        core: the assembled pipeline.
        input_ids: ``(N,)`` the per-row token id (a chunk placeholder,
            a replay label, or the flush sentinel).
        inputs_embeds: ``(N, H)`` the per-row carrier (packed mel for
            chunk rows, zeros otherwise).
        channel_pools, time_pools, len_pools: one entry per encoder
            layer, the engine-bound window/conv/len page pools.
        h_pool, c_pool: the predictor LSTM page pools.
        queue_pool, book_pool: the replay-queue and session-book page
            pools.
        state_indices: ``(N,)`` the page block index per row.
        num_real_rows: the count of leading rows that are real
            (non-padding); rows beyond this index are graph padding.
        placeholder_id: the token id marking a CHUNK row.
        park_id: the token id the engine treats as the finish
            sentinel once a session's queue is drained.
        feat: the mel-bin count (carrier unpacking).
        drop_extra: the pre-encode overlap dropped from non-first
            chunks.

    Returns:
        ``(N, H)`` — the same decision-carrier contract as the
        current ``run_forward_step``: each row's hidden output holds
        the id that row emits this step.

    Raises:
        NotImplementedError: Always, at this tests-first stub — lands
            in Phase 6 with the ``forward_step.py`` deletion.
        ValueError: (Phase 6 behavior, documented here for the pinned
            contract) on any PORT-STATE-007 structural defect, before
            any resident-state read.
    """
    raise NotImplementedError(
        "advance_model_rows lands in Phase 6 with the forward_step.py "
        "deletion (ledger P5-1)"
    )
