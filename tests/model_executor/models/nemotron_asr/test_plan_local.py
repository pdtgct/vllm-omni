# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Exercised Task-5 plan-provider + commit-sink coverage, locally.

The plan provider (`plan.py`) and the production composite sink
(`commit_sink.py`) are torch/stdlib host logic — loaded by file path
under the stubbed package chain, the registry lifecycle, binding
minting, consume-once slot, RowPlan assembly, and the reserve/stage/
collect ticket protocol all execute on macOS. The decisive integration
pin: a registry-minted plan must pass `advance_model_rows`' own
structural preflight, which re-proves every binding.
"""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path
from typing import Any

import pytest
import torch

_PKG = (
    Path(__file__).resolve().parents[4]
    / "vllm_omni/model_executor/models/nemotron_asr"
)
_BASE = "vllm_omni.model_executor.models.nemotron_asr"


def _load_chain() -> dict[str, Any]:
    for name in (
        "vllm_omni",
        "vllm_omni.model_executor",
        "vllm_omni.model_executor.models",
        _BASE,
    ):
        if name not in sys.modules:
            sys.modules[name] = types.ModuleType(name)
    loaded: dict[str, Any] = {}
    for mod in (
        "precision", "masks", "featurizer", "encoder", "lid",
        "manifests", "frontend", "rnnt_cell", "rnnt",
        "decode_dispatch", "state_scatter", "advance", "plan",
        "commit_sink",
    ):
        spec = importlib.util.spec_from_file_location(
            f"{_BASE}.{mod}", _PKG / f"{mod}.py"
        )
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[f"{_BASE}.{mod}"] = module
        spec.loader.exec_module(module)
        loaded[mod] = module
    return loaded


mods = _load_chain()
advance = mods["advance"]
plan_mod = mods["plan"]
commit_sink_mod = mods["commit_sink"]

PLACEHOLDER_ID = 9001
NUM_PROMPTS = 4
NUM_GEOMETRIES = 5
NOW_NS = 1_000_000


def _header(
    *,
    valid: int = 1_280,
    geometry: int = 0,
    final: int = 0,
    prompt: int = 0,
    seq: int = 0,
    version: float = 1.0,
) -> tuple[float, ...]:
    return (
        float(version), float(valid), float(geometry), float(final),
        float(prompt), float(seq),
    )


def _row(
    req: str,
    block: int,
    *,
    chunk: bool = False,
    prior: bool = True,
    header: tuple[float, ...] | None = None,
) -> Any:
    if chunk and header is None:
        header = _header()
    return plan_mod.ObservedRow(
        request_id=req,
        block_id=block,
        scheduled_token_id=PLACEHOLDER_ID if chunk else 7,
        has_prior_state=prior,
        envelope_header=header,
    )


def _bind(registry: Any, rows: list[Any], *, now_ns: int = NOW_NS) -> Any:
    return registry.bind_rows(
        rows,
        placeholder_id=PLACEHOLDER_ID,
        num_prompts=NUM_PROMPTS,
        num_geometries=NUM_GEOMETRIES,
        now_ns=now_ns,
    )


# ---- SessionRegistry lifecycle -------------------------------------------


def test_fresh_chunk_registers_with_monotonic_generation() -> None:
    registry = plan_mod.SessionRegistry()
    [b1] = _bind(
        registry, [_row("a", 3, chunk=True, prior=False)]
    )
    [b2] = _bind(
        registry,
        [_row("b", 4, chunk=True, prior=False, header=_header(prompt=2))],
    )
    assert b1.admission_generation == 1
    assert b2.admission_generation == 2
    assert b1.is_chunk and b1.ready_deadline_ns == NOW_NS
    assert not b1.allow_prompt_transition
    assert b2.prompt_index == 2 and b2.prior_prompt_index == 2
    assert registry.live_block_ids() == [3, 4]


def test_duplicate_block_ownership_rejected() -> None:
    registry = plan_mod.SessionRegistry()
    _bind(registry, [_row("a", 3, chunk=True, prior=False)])
    with pytest.raises(ValueError, match="already owned"):
        _bind(registry, [_row("b", 3, chunk=True, prior=False)])


def test_readmission_of_registered_request_rejected() -> None:
    registry = plan_mod.SessionRegistry()
    _bind(registry, [_row("a", 3, chunk=True, prior=False)])
    with pytest.raises(ValueError, match="re-admitted"):
        _bind(registry, [_row("a", 3, chunk=True, prior=False)])


def test_block_remap_rejected_before_state_access() -> None:
    registry = plan_mod.SessionRegistry()
    _bind(registry, [_row("a", 3, chunk=True, prior=False)])
    with pytest.raises(ValueError, match="remapped"):
        _bind(registry, [_row("a", 5)])


def test_unregistered_continuation_rejected() -> None:
    registry = plan_mod.SessionRegistry()
    with pytest.raises(ValueError, match="not.*registered"):
        _bind(registry, [_row("ghost", 3)])


def test_non_chunk_row_with_envelope_rejected() -> None:
    registry = plan_mod.SessionRegistry()
    _bind(registry, [_row("a", 3, chunk=True, prior=False)])
    with pytest.raises(ValueError, match="carries a scheduled envelope"):
        _bind(registry, [_row("a", 3, header=_header(seq=1))])


def test_duplicate_request_in_one_step_rejected() -> None:
    registry = plan_mod.SessionRegistry()
    _bind(registry, [_row("a", 3, chunk=True, prior=False)])
    with pytest.raises(ValueError, match="twice"):
        _bind(registry, [_row("a", 3), _row("a", 3)])


def test_bind_rows_is_atomic_when_a_later_row_is_invalid() -> None:
    registry = plan_mod.SessionRegistry()
    with pytest.raises(ValueError, match="already owned"):
        _bind(
            registry,
            [
                _row("a", 3, chunk=True, prior=False),
                _row("b", 3, chunk=True, prior=False),
            ],
        )
    assert len(registry) == 0
    assert registry.live_block_ids() == []
    [binding] = _bind(
        registry, [_row("c", 4, chunk=True, prior=False)]
    )
    assert binding.admission_generation == 1


def test_prepare_context_rolls_back_prune_when_binding_fails() -> None:
    registry = plan_mod.SessionRegistry()
    _bind(registry, [_row("a", 3, chunk=True, prior=False)])
    _bind(registry, [_row("b", 4, chunk=True, prior=False)])
    with pytest.raises(ValueError, match="already owned"):
        plan_mod.prepare_plan_context(
            registry,
            [
                _row("c", 5, chunk=True, prior=False),
                _row("d", 5, chunk=True, prior=False),
            ],
            resident_request_ids=["a", "c", "d"],
            placeholder_id=PLACEHOLDER_ID,
            num_prompts=NUM_PROMPTS,
            num_geometries=NUM_GEOMETRIES,
            now_ns=NOW_NS,
            step=2,
        )
    assert registry.live_block_ids() == [3, 4]


def test_prepare_context_rejects_rows_missing_from_resident_authority() -> None:
    registry = plan_mod.SessionRegistry()
    with pytest.raises(ValueError, match="absent from worker-resident"):
        plan_mod.prepare_plan_context(
            registry,
            [_row("a", 3, chunk=True, prior=False)],
            resident_request_ids=[],
            placeholder_id=PLACEHOLDER_ID,
            num_prompts=NUM_PROMPTS,
            num_geometries=NUM_GEOMETRIES,
            now_ns=NOW_NS,
            step=1,
        )
    assert len(registry) == 0


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ({"version": 9.0}, "version"),
        ({"valid": -1}, "negative valid-sample"),
        ({"geometry": 7}, "outside the admitted set"),
        ({"prompt": 9}, "outside the prompt dictionary"),
        ({"final": 2}, "non-boolean final"),
        ({"seq": 1.5}, "not an exact integer"),
    ],
)
def test_malformed_minted_header_is_a_loud_port_defect(
    mutation: dict[str, Any], message: str
) -> None:
    registry = plan_mod.SessionRegistry()
    with pytest.raises(ValueError, match=message):
        _bind(
            registry,
            [
                _row(
                    "a", 3, chunk=True, prior=False,
                    header=_header(**mutation),
                )
            ],
        )


def test_prompt_transition_flags_exactly_on_change() -> None:
    registry = plan_mod.SessionRegistry()
    _bind(registry, [_row("a", 3, chunk=True, prior=False)])
    # Same prompt on the next minted CHUNK: no transition authorized.
    [same] = _bind(
        registry, [_row("a", 3, chunk=True, header=_header(seq=1))]
    )
    assert not same.allow_prompt_transition
    # A locale update surfaces on the next minted CHUNK's header.
    [changed] = _bind(
        registry,
        [_row("a", 3, chunk=True, header=_header(prompt=2, seq=2))],
    )
    assert changed.allow_prompt_transition
    assert changed.prior_prompt_index == 0
    assert changed.prompt_index == 2
    with pytest.raises(ValueError, match="status.*pending"):
        _bind(registry, [_row("a", 3)])
    registry.resolve_status(
        request_id="a",
        admission_generation=changed.admission_generation,
        clean=True,
    )
    # REPLAY continuation carries the persisted (new) prompt, never a
    # retroactive selection, and can never authorize a transition.
    [replay] = _bind(registry, [_row("a", 3)])
    assert not replay.is_chunk
    assert not replay.allow_prompt_transition
    assert replay.prompt_index == 2 and replay.prior_prompt_index == 2
    assert replay.ready_deadline_ns == 0


def test_failed_prompt_transition_preserves_prior_prompt() -> None:
    registry = plan_mod.SessionRegistry()
    _bind(registry, [_row("a", 3, chunk=True, prior=False)])
    [changed] = _bind(
        registry,
        [_row("a", 3, chunk=True, header=_header(prompt=2, seq=1))],
    )
    registry.resolve_status(
        request_id="a",
        admission_generation=changed.admission_generation,
        clean=False,
    )
    [replay] = _bind(registry, [_row("a", 3)])
    assert replay.prompt_index == 0
    assert replay.prior_prompt_index == 0


def test_outer_full_decode_graph_mode_is_rejected() -> None:
    class _RuntimeMode:
        name = "FULL"

    class _ConfiguredMode:
        def decode_mode(self) -> _RuntimeMode:
            return _RuntimeMode()

    with pytest.raises(ValueError, match="full CUDA graph"):
        plan_mod.reject_unsupported_outer_graph_mode(
            type("Compilation", (), {"cudagraph_mode": _ConfiguredMode()})()
        )


def test_outer_none_and_piecewise_graph_modes_are_accepted() -> None:
    for name in ("NONE", "PIECEWISE"):
        runtime = type("Runtime", (), {"name": name})()
        configured = type(
            "Configured",
            (),
            {"decode_mode": lambda self, value=runtime: value},
        )()
        plan_mod.reject_unsupported_outer_graph_mode(
            type("Compilation", (), {"cudagraph_mode": configured})()
        )


def test_prune_releases_sessions_and_blocks() -> None:
    registry = plan_mod.SessionRegistry()
    _bind(registry, [_row("a", 3, chunk=True, prior=False)])
    _bind(registry, [_row("b", 4, chunk=True, prior=False)])
    registry.prune(["b"])
    assert registry.live_block_ids() == [4]
    # Block 3 is reusable by a NEW session after release.
    [b3] = _bind(registry, [_row("c", 3, chunk=True, prior=False)])
    assert b3.admission_generation == 3  # ABA guard advanced


def test_lease_validation_tracks_registry_currency() -> None:
    registry = plan_mod.SessionRegistry()
    [binding] = _bind(registry, [_row("a", 3, chunk=True, prior=False)])
    registry.validate_lease([binding])
    assert registry.lease_is_current([binding])
    registry.prune([])
    with pytest.raises(ValueError, match="lease"):
        registry.validate_lease([binding])
    assert not registry.lease_is_current([binding])


# ---- consume-once slot and context assembly -------------------------------


def _context(
    registry: Any, rows: list[Any], *, step: int = 1
) -> Any:
    return plan_mod.prepare_plan_context(
        registry,
        rows,
        resident_request_ids=[row.request_id for row in rows],
        placeholder_id=PLACEHOLDER_ID,
        num_prompts=NUM_PROMPTS,
        num_geometries=NUM_GEOMETRIES,
        now_ns=NOW_NS,
        step=step,
    )


def test_slot_is_consume_once_and_replaces_stale_context() -> None:
    registry = plan_mod.SessionRegistry()
    slot = plan_mod.PlanContextSlot()
    first = _context(
        registry, [_row("a", 3, chunk=True, prior=False)], step=1
    )
    slot.stage(first)
    second = _context(registry, [_row("a", 3)], step=2)
    slot.stage(second)  # a stranded context is replaced, never leaked
    consumed = slot.consume()
    assert consumed.step == 2
    with pytest.raises(ValueError, match="no staged PlanContext"):
        slot.consume()


def test_context_columns_mirror_bindings_exactly() -> None:
    registry = plan_mod.SessionRegistry()
    context = _context(
        registry,
        [
            _row("a", 3, chunk=True, prior=False,
                 header=_header(prompt=1, geometry=2)),
            _row("b", 4, chunk=True, prior=False),
        ],
    )
    assert context.request_ids == ("a", "b")
    assert context.block_ids.tolist() == [3, 4]
    assert context.geometry_id.tolist() == [2, 0]
    assert context.prompt_index.tolist() == [1, 0]
    assert context.is_chunk.tolist() == [True, True]
    assert context.admission_generation.tolist() == [1, 2]
    assert context.live_block_ids.tolist() == [3, 4]
    for row, binding in enumerate(context.bindings):
        assert binding.request_id == context.request_ids[row]
        assert binding.block_id == int(context.block_ids[row])


# ---- RowPlan assembly + the transaction's own preflight -------------------


def test_registry_minted_plan_passes_transaction_preflight() -> None:
    # The decisive integration pin: the plan the provider builds must
    # satisfy the transaction's structural preflight, which re-proves
    # every PreparedRowBinding from the plan columns (PORT-ADV-003).
    registry = plan_mod.SessionRegistry()
    # Establish an ongoing session (decode row) + a fresh prefill.
    _bind(registry, [_row("a", 1, chunk=True, prior=False)])
    context = _context(
        registry,
        [
            _row("a", 1, chunk=True, header=_header(seq=1)),
            _row("b", 2, chunk=True, prior=False),
        ],
    )
    plan = plan_mod.build_row_plan(
        context,
        num_decodes=1,
        num_prefills=1,
        null_block_id=0,
        num_pool_blocks=4,
    )
    idx = advance._structural_preflight(plan, 2, 2)
    assert idx.tolist() == [1, 2]
    assert plan.has_initial_states_p.tolist() == [False]
    assert plan.request_ids == ("a", "b")


def test_build_row_plan_rejects_count_and_freshness_drift() -> None:
    registry = plan_mod.SessionRegistry()
    context = _context(
        registry, [_row("a", 1, chunk=True, prior=False)]
    )
    with pytest.raises(ValueError, match="disagree"):
        plan_mod.build_row_plan(
            context, num_decodes=1, num_prefills=1,
            null_block_id=0, num_pool_blocks=4,
        )
    with pytest.raises(ValueError, match="decode row has no prior"):
        plan_mod.build_row_plan(
            context, num_decodes=1, num_prefills=0,
            null_block_id=0, num_pool_blocks=4,
        )


# ---- the production composite sink ----------------------------------------


def _sink_and_plan() -> tuple[Any, Any, Any]:
    registry = plan_mod.SessionRegistry()
    [binding] = _bind(registry, [_row("a", 3, chunk=True, prior=False)])
    sink = commit_sink_mod.BoundedCommitSink(registry, max_rows=4)
    commit_plan = advance.CommitPlan(bindings=(binding,), capture=None)
    return registry, sink, commit_plan


def test_sink_reserve_stage_collect_roundtrip() -> None:
    registry, sink, commit_plan = _sink_and_plan()
    ticket = sink.reserve(commit_plan)
    status = torch.tensor([0], dtype=torch.int32)
    ticket.stage(status)
    reports, committed, lease_ok = sink.collect()
    assert [r.request_id for r in reports] == ["a"]
    assert reports[0].row_status == 0
    assert reports[0].admission_generation == 1
    assert committed == [] and lease_ok


def test_sink_exposes_only_status_clean_records() -> None:
    registry = plan_mod.SessionRegistry()
    bindings = _bind(
        registry,
        [
            _row("a", 3, chunk=True, prior=False),
            _row("b", 4, chunk=True, prior=False),
        ],
    )
    sink = commit_sink_mod.BoundedCommitSink(
        registry, max_rows=4, max_capture_rows=4,
        max_capture_bytes=1 << 20,
    )
    records = [
        _record("a", 3, 1, row=0),
        _record("b", 4, 2, row=1),
    ]
    commit_plan = advance.CommitPlan(
        bindings=tuple(bindings),
        capture=advance.CapturePlan(
            rows=2, payload_bytes=_payload_bytes(records)
        ),
        records=tuple(records),
    )
    ticket = sink.reserve(commit_plan)
    ticket.stage(torch.tensor([512, 0], dtype=torch.int32))
    reports, committed, lease_ok = sink.collect()
    assert [r.row_status for r in reports] == [512, 0]
    assert [r.request_id for r in committed] == ["b"]
    assert lease_ok


def _record(
    req: str, block: int, generation: int, *, row: int = 0
) -> Any:
    scalar = torch.zeros((), dtype=torch.int64)
    return advance.CaptureRecord(
        row=row,
        request_id=req,
        block_id=block,
        admission_generation=generation,
        geometry=0,
        chunk_sequence=scalar,
        prompt_index=scalar,
        row_status=torch.zeros((), dtype=torch.int32),
        frontend_mel=torch.zeros(2, 3),
        mel_length=scalar,
        encoder_raw=torch.zeros(1, 2),
        encoder_conditioned=torch.zeros(1, 2),
        encoder_length=scalar,
    )


def _payload_bytes(records: list[Any]) -> int:
    return sum(
        tensor.numel() * tensor.element_size()
        for record in records
        for tensor in (
            record.chunk_sequence,
            record.prompt_index,
            record.row_status,
            record.frontend_mel,
            record.mel_length,
            record.encoder_raw,
            record.encoder_conditioned,
            record.encoder_length,
        )
    )


def test_sink_rejects_second_reservation_until_collected() -> None:
    registry, sink, commit_plan = _sink_and_plan()
    ticket = sink.reserve(commit_plan)
    with pytest.raises(ValueError, match="unconsumed reservation"):
        sink.reserve(commit_plan)
    ticket.cancel()
    ticket.cancel()  # idempotent
    sink.reserve(commit_plan)  # released by cancel


def test_sink_reserve_validates_lease_and_bounds() -> None:
    registry, sink, commit_plan = _sink_and_plan()
    registry.prune([])
    with pytest.raises(ValueError, match="lease"):
        sink.reserve(commit_plan)
    registry2 = plan_mod.SessionRegistry()
    bindings = _bind(
        registry2,
        [
            _row(f"r{i}", i + 1, chunk=True, prior=False)
            for i in range(3)
        ],
    )
    small = commit_sink_mod.BoundedCommitSink(registry2, max_rows=2)
    with pytest.raises(ValueError, match="exceed the sink bound"):
        small.reserve(
            advance.CommitPlan(bindings=tuple(bindings), capture=None)
        )
    capped = commit_sink_mod.BoundedCommitSink(
        registry2, max_rows=4, max_capture_rows=0, max_capture_bytes=0
    )
    with pytest.raises(ValueError, match="capture rows exceed"):
        capped.reserve(
            advance.CommitPlan(
                bindings=tuple(bindings),
                capture=advance.CapturePlan(rows=1, payload_bytes=8),
            )
        )


def test_sink_reserve_rejects_capture_identity_drift() -> None:
    registry = plan_mod.SessionRegistry()
    [binding] = _bind(
        registry, [_row("a", 3, chunk=True, prior=False)]
    )
    sink = commit_sink_mod.BoundedCommitSink(
        registry,
        max_rows=2,
        max_capture_rows=2,
        max_capture_bytes=1 << 20,
    )
    bad = _record("other", 3, binding.admission_generation)
    with pytest.raises(ValueError, match="capture record identity"):
        sink.reserve(
            advance.CommitPlan(
                bindings=(binding,),
                capture=advance.CapturePlan(
                    rows=1, payload_bytes=_payload_bytes([bad])
                ),
                records=(bad,),
            )
        )


def test_sink_stage_records_lease_break_without_raising() -> None:
    # Nothing may fail after resident state committed: a lease broken
    # between reserve and stage is RECORDED, never raised.
    registry, sink, commit_plan = _sink_and_plan()
    ticket = sink.reserve(commit_plan)
    registry.prune([])  # simulate a lease break inside the window
    ticket.stage(torch.tensor([0], dtype=torch.int32))
    _, _, lease_ok = sink.collect()
    assert not lease_ok


def test_sink_collect_without_stage_raises() -> None:
    registry, sink, commit_plan = _sink_and_plan()
    sink.reserve(commit_plan)
    with pytest.raises(ValueError, match="no staged commit"):
        sink.collect()


def test_geometry_only_status_preserves_session_and_prompt() -> None:
    registry = plan_mod.SessionRegistry()
    _bind(registry, [_row("a", 3, chunk=True, prior=False)])
    [changed] = _bind(
        registry,
        [_row("a", 3, chunk=True, header=_header(prompt=2, seq=1))],
    )
    sink = commit_sink_mod.BoundedCommitSink(registry, max_rows=2)
    ticket = sink.reserve(
        advance.CommitPlan(bindings=(changed,), capture=None)
    )
    ticket.stage(torch.tensor([1], dtype=torch.int32))
    reports, _, lease_ok = sink.collect()

    statuses, failed = commit_sink_mod.resolve_status_reports(
        registry, reports, lease_ok=lease_ok
    )

    assert statuses == {"a": 1}
    assert failed == set()
    [replay] = _bind(registry, [_row("a", 3)])
    assert replay.prompt_index == 0


def test_nonrecoverable_status_is_terminal() -> None:
    registry, sink, commit_plan = _sink_and_plan()
    ticket = sink.reserve(commit_plan)
    ticket.stage(torch.tensor([512], dtype=torch.int32))
    reports, _, lease_ok = sink.collect()

    statuses, failed = commit_sink_mod.resolve_status_reports(
        registry, reports, lease_ok=lease_ok
    )

    assert statuses == {"a": 512}
    assert failed == {"a"}
