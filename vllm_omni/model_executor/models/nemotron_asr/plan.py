# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The Task-5 plan provider: host row authority for the transaction.

`advance_model_rows` proves that its :class:`RowPlan` columns reproduce
each row's :class:`PreparedRowBinding`; this module is where those
bindings are MINTED truthfully. The model's ``prepare_row_plan_context``
runner hook observes the final host row order (request ids, scheduled
CPU token ids, per-request block ids, and each scheduled CHUNK's CPU
envelope header) and asks the :class:`SessionRegistry` to join them
into one immutable, consume-once :class:`PlanContext`; ``forward`` then
combines that context with the attention metadata's host row counts
into the transaction's ``RowPlan`` (design §Phase-6c transaction
seams, PORT-ADV-003 as amended).

Authority boundaries, stated exactly:

- The ENGINE is the authority for row order, decode/prefill counts,
  and block allocation (``CachedRequestState.block_ids`` at the pin).
- The REGISTRY is the authority for session identity across steps:
  admission generation, admitted geometry, the current session-control
  prompt, and block-remap rejection. It is stamped host-side from the
  FIRST minted CHUNK's envelope header — the serving tier's twin
  admission record read on CPU before any device work — and updated
  only by later minted CHUNKs.
- The DEVICE remains the cross-check: every envelope is re-validated
  against the plan authority by the transaction's row-status bits, so
  host/device drift masks rows instead of corrupting sessions.

Registry prompt/expectation updates are deliberately OPTIMISTIC: a row
that later fails on device terminates its session at the status
readback, so host state that ran ahead of a failed commit only
accelerates masking — it can never resurrect a dead session.

Torch + stdlib only (CPU tensors): the whole module runs under the
macOS loader chain and is differentially tested locally.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import torch

from vllm_omni.model_executor.models.nemotron_asr.advance import (
    ENV_CHUNK_SEQUENCE,
    ENV_FINAL_TAIL,
    ENV_GEOMETRY_ID,
    ENV_PROMPT_INDEX,
    ENV_VALID_SAMPLES,
    ENV_VERSION,
    ENVELOPE_HEADER_SLOTS,
    ENVELOPE_VERSION,
    PreparedRowBinding,
    RowPlan,
)


@dataclass(frozen=True)
class ObservedRow:
    """One scheduled model row as the runner hook sees it, host-side.

    ``envelope_header``: the first :data:`ENVELOPE_HEADER_SLOTS` values
    of the row's scheduled CPU envelope when a new CHUNK carrier is
    scheduled this step, else ``None``. ``has_prior_state`` mirrors the
    scheduler's freshness authority (``num_computed_tokens > 0``).
    """

    request_id: str
    block_id: int
    scheduled_token_id: int
    has_prior_state: bool
    envelope_header: tuple[float, ...] | None


@dataclass(frozen=True)
class PlanContext:
    """One step's immutable host row authority (consume-once).

    Parallel CPU columns plus the atomically minted bindings; ``forward``
    splits rows by the metadata's decode/prefill counts and hands the
    result to the transaction, whose preflight re-proves that every
    column reproduces its binding.
    """

    step: int
    request_ids: tuple[str, ...]
    bindings: tuple[PreparedRowBinding, ...]
    block_ids: torch.Tensor
    geometry_id: torch.Tensor
    prompt_index: torch.Tensor
    is_chunk: torch.Tensor
    has_prior_state: torch.Tensor
    admission_generation: torch.Tensor
    ready_deadline_ns: torch.Tensor
    live_block_ids: torch.Tensor


@dataclass
class _Session:
    block_id: int
    generation: int
    geometry_id: int
    prompt_index: int


def _header_int(header: tuple[float, ...], slot: int, name: str) -> int:
    value = header[slot]
    if not math.isfinite(value) or value != math.trunc(value):
        raise ValueError(
            f"envelope header {name} is not an exact integer: {value!r} "
            "— the serving tier mints headers, so this is a port defect"
        )
    return int(value)


class SessionRegistry:
    """Worker-side session identity authority (design §Phase-6c seams).

    Registers a session atomically at its first minted CHUNK, retains
    parked sessions, prunes on request completion, and rejects block
    remapping, duplicate block ownership, and re-admission of a live
    request before any state access. ``admission_generation`` is a
    process-monotonic counter — the capture/status ABA guard across
    block reuse.
    """

    def __init__(self) -> None:
        self._sessions: dict[str, _Session] = {}
        self._blocks: dict[int, str] = {}
        self._generation = 0

    def __len__(self) -> int:
        return len(self._sessions)

    def prune(self, resident_request_ids: Sequence[str]) -> None:
        """Drop sessions whose request is no longer worker-resident."""
        keep = set(resident_request_ids)
        for req_id in [r for r in self._sessions if r not in keep]:
            session = self._sessions.pop(req_id)
            self._blocks.pop(session.block_id, None)

    def live_block_ids(self) -> list[int]:
        """The registry's allocated-block liveness authority, sorted."""
        return sorted(self._blocks)

    def validate_lease(
        self, bindings: Sequence[PreparedRowBinding]
    ) -> None:
        """Reserve-time lease check: every binding must be CURRENT.

        Raises:
            ValueError: a binding whose request is unknown or whose
                block/generation no longer matches the registry — the
                composite reservation must fail pre-commit rather than
                stage status for a swapped or reused row.
        """
        for binding in bindings:
            session = self._sessions.get(binding.request_id)
            if session is None:
                raise ValueError(
                    f"lease check: request {binding.request_id!r} is "
                    "not registered"
                )
            if (
                session.block_id != binding.block_id
                or session.generation != binding.admission_generation
            ):
                raise ValueError(
                    f"lease check: request {binding.request_id!r} "
                    "binding does not match the current registry lease"
                )

    def lease_is_current(
        self, bindings: Sequence[PreparedRowBinding]
    ) -> bool:
        """No-fail stage-time revalidation of the same lease facts."""
        try:
            self.validate_lease(bindings)
        except ValueError:
            return False
        return True

    def _parse_chunk_header(
        self,
        row: ObservedRow,
        *,
        num_prompts: int,
        num_geometries: int,
    ) -> tuple[int, int]:
        header = row.envelope_header
        if header is None:
            raise ValueError(
                f"CHUNK row {row.request_id!r} has no scheduled envelope "
                "header — the serving tier mints one per CHUNK"
            )
        if len(header) != ENVELOPE_HEADER_SLOTS:
            raise ValueError(
                f"envelope header for {row.request_id!r} has "
                f"{len(header)} slots, expected {ENVELOPE_HEADER_SLOTS}"
            )
        if _header_int(header, ENV_VERSION, "version") != ENVELOPE_VERSION:
            raise ValueError(
                f"envelope header for {row.request_id!r} carries version "
                f"{header[ENV_VERSION]!r}, expected {ENVELOPE_VERSION}"
            )
        if _header_int(header, ENV_VALID_SAMPLES, "valid_samples") < 0:
            raise ValueError(
                f"envelope header for {row.request_id!r} carries a "
                "negative valid-sample count"
            )
        final = _header_int(header, ENV_FINAL_TAIL, "final_tail")
        if final not in (0, 1):
            raise ValueError(
                f"envelope header for {row.request_id!r} carries a "
                f"non-boolean final flag {final}"
            )
        _header_int(header, ENV_CHUNK_SEQUENCE, "chunk_sequence")
        geometry = _header_int(header, ENV_GEOMETRY_ID, "geometry_id")
        if not 0 <= geometry < num_geometries:
            raise ValueError(
                f"envelope header for {row.request_id!r} carries "
                f"geometry {geometry} outside the admitted set"
            )
        prompt = _header_int(header, ENV_PROMPT_INDEX, "prompt_index")
        if not 0 <= prompt < num_prompts:
            raise ValueError(
                f"envelope header for {row.request_id!r} carries prompt "
                f"{prompt} outside the prompt dictionary"
            )
        return geometry, prompt

    def bind_rows(
        self,
        rows: Sequence[ObservedRow],
        *,
        placeholder_id: int,
        num_prompts: int,
        num_geometries: int,
        now_ns: int,
    ) -> tuple[PreparedRowBinding, ...]:
        """Mint one binding per observed row, atomically per step.

        Registration, remap rejection, prompt-transition detection, and
        deadline minting happen here — the single writer of registry
        state. A defect at this tier (malformed header from OUR serving
        tier, block remap, duplicate ownership, unregistered
        continuation) raises: this is the host structural tier, before
        any resident read (PORT-STATE-007 posture).
        """
        if now_ns <= 0:
            raise ValueError("now_ns must be a positive monotonic stamp")
        seen: set[str] = set()
        bindings: list[PreparedRowBinding] = []
        for row in rows:
            if row.request_id in seen:
                raise ValueError(
                    f"request {row.request_id!r} appears twice in one "
                    "step's row plan"
                )
            seen.add(row.request_id)
            is_chunk = row.scheduled_token_id == placeholder_id
            if not is_chunk and row.envelope_header is not None:
                raise ValueError(
                    f"non-CHUNK row {row.request_id!r} carries a "
                    "scheduled envelope"
                )
            session = self._sessions.get(row.request_id)
            if is_chunk and not row.has_prior_state:
                if session is not None:
                    raise ValueError(
                        f"request {row.request_id!r} re-admitted while "
                        "still registered — resident sessions are never "
                        "recomputed (PORT-STATE-005)"
                    )
                owner = self._blocks.get(row.block_id)
                if owner is not None:
                    raise ValueError(
                        f"block {row.block_id} is already owned by "
                        f"{owner!r} — duplicate ownership"
                    )
                geometry, prompt = self._parse_chunk_header(
                    row,
                    num_prompts=num_prompts,
                    num_geometries=num_geometries,
                )
                self._generation += 1
                session = _Session(
                    block_id=row.block_id,
                    generation=self._generation,
                    geometry_id=geometry,
                    prompt_index=prompt,
                )
                self._sessions[row.request_id] = session
                self._blocks[row.block_id] = row.request_id
                bindings.append(
                    PreparedRowBinding(
                        request_id=row.request_id,
                        block_id=row.block_id,
                        admission_generation=session.generation,
                        geometry_id=geometry,
                        prompt_index=prompt,
                        prior_prompt_index=prompt,
                        allow_prompt_transition=False,
                        is_chunk=True,
                        ready_deadline_ns=now_ns,
                    )
                )
                continue
            if session is None:
                raise ValueError(
                    f"continuing row {row.request_id!r} is not "
                    "registered — the worker never observed its first "
                    "CHUNK"
                )
            if session.block_id != row.block_id:
                raise ValueError(
                    f"request {row.request_id!r} block remapped from "
                    f"{session.block_id} to {row.block_id} — resident "
                    "state never migrates outside park serialization"
                )
            if is_chunk:
                _, prompt = self._parse_chunk_header(
                    row,
                    num_prompts=num_prompts,
                    num_geometries=num_geometries,
                )
                prior = session.prompt_index
                # Optimistic host update: the transaction commits the
                # new prompt only for a clean row, and a failed row
                # terminates the session at the status readback.
                session.prompt_index = prompt
                bindings.append(
                    PreparedRowBinding(
                        request_id=row.request_id,
                        block_id=row.block_id,
                        admission_generation=session.generation,
                        geometry_id=session.geometry_id,
                        prompt_index=prompt,
                        prior_prompt_index=prior,
                        allow_prompt_transition=prompt != prior,
                        is_chunk=True,
                        ready_deadline_ns=now_ns,
                    )
                )
                continue
            bindings.append(
                PreparedRowBinding(
                    request_id=row.request_id,
                    block_id=row.block_id,
                    admission_generation=session.generation,
                    geometry_id=session.geometry_id,
                    prompt_index=session.prompt_index,
                    prior_prompt_index=session.prompt_index,
                    allow_prompt_transition=False,
                    is_chunk=False,
                    ready_deadline_ns=0,
                )
            )
        return tuple(bindings)


def prepare_plan_context(
    registry: SessionRegistry,
    rows: Sequence[ObservedRow],
    *,
    resident_request_ids: Sequence[str],
    placeholder_id: int,
    num_prompts: int,
    num_geometries: int,
    now_ns: int,
    step: int,
) -> PlanContext:
    """One hook invocation: prune, mint bindings, build the columns."""
    registry.prune(resident_request_ids)
    bindings = registry.bind_rows(
        rows,
        placeholder_id=placeholder_id,
        num_prompts=num_prompts,
        num_geometries=num_geometries,
        now_ns=now_ns,
    )
    return PlanContext(
        step=step,
        request_ids=tuple(b.request_id for b in bindings),
        bindings=bindings,
        block_ids=torch.tensor(
            [b.block_id for b in bindings], dtype=torch.int64
        ),
        geometry_id=torch.tensor(
            [b.geometry_id for b in bindings], dtype=torch.int64
        ),
        prompt_index=torch.tensor(
            [b.prompt_index for b in bindings], dtype=torch.int64
        ),
        is_chunk=torch.tensor(
            [b.is_chunk for b in bindings], dtype=torch.bool
        ),
        has_prior_state=torch.tensor(
            [row.has_prior_state for row in rows], dtype=torch.bool
        ),
        admission_generation=torch.tensor(
            [b.admission_generation for b in bindings], dtype=torch.int64
        ),
        ready_deadline_ns=torch.tensor(
            [b.ready_deadline_ns for b in bindings], dtype=torch.int64
        ),
        live_block_ids=torch.tensor(
            registry.live_block_ids(), dtype=torch.int64
        ),
    )


class PlanContextSlot:
    """Consume-once staging between the runner hook and ``forward``.

    ``stage`` replaces any unconsumed context (a context stranded by an
    exception or warm-up call can never leak into a later step);
    ``consume`` pops exactly once and fails loudly when nothing was
    staged this step.
    """

    def __init__(self) -> None:
        self._context: PlanContext | None = None

    def stage(self, context: PlanContext) -> None:
        self._context = context

    def consume(self) -> PlanContext:
        context = self._context
        self._context = None
        if context is None:
            raise ValueError(
                "no staged PlanContext: the prepare_row_plan_context "
                "runner hook did not run for this step (or the context "
                "was already consumed)"
            )
        return context


def build_row_plan(
    context: PlanContext,
    *,
    num_decodes: int,
    num_prefills: int,
    null_block_id: int,
    num_pool_blocks: int,
    execution_tier: int = 0,
) -> RowPlan:
    """Combine the consumed context with the metadata's host counts.

    The engine's decode/prefill split must agree with the context's
    prior-state column (a decode row without prior state, or a fresh
    row classified decode, is host/engine drift and fails the call
    here — before the transaction's own preflight re-proves the
    bindings).
    """
    n_real = len(context.bindings)
    if num_decodes < 0 or num_prefills < 0:
        raise ValueError("negative metadata row counts")
    if num_decodes + num_prefills != n_real:
        raise ValueError(
            f"metadata rows {num_decodes}+{num_prefills} disagree with "
            f"the staged context's {n_real} rows"
        )
    if bool((~context.has_prior_state[:num_decodes]).any()):
        raise ValueError(
            "a decode row has no prior state — engine/context row drift"
        )
    return RowPlan(
        state_indices_d=context.block_ids[:num_decodes].reshape(-1, 1),
        num_decodes=num_decodes,
        state_indices_p=context.block_ids[num_decodes:],
        num_prefills=num_prefills,
        has_initial_states_p=context.has_prior_state[num_decodes:],
        null_block_id=null_block_id,
        num_pool_blocks=num_pool_blocks,
        live_block_ids=context.live_block_ids,
        geometry_id=context.geometry_id,
        prompt_index=context.prompt_index,
        is_chunk=context.is_chunk,
        admission_generation=context.admission_generation,
        ready_deadline_ns=context.ready_deadline_ns,
        request_ids=context.request_ids,
        execution_tier=execution_tier,
        bindings=context.bindings,
    )
