# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Dispatch-table generator contracts (PORT-DEC-008, dispatch-table
form; decisions/decode-dispatch-regime.md r3.2).

The generator consumes a p6b_decode_profile artifact and COMPUTES
selections — never eyeballed brackets: lanes come from
``generator_eligible_lanes``; compact may be selected only from
SELECTIVE cells and only when it beats the sync-free arm by the
recorded hysteresis at EVERY dcp activity level; nonselective cells
force the sync-free arm; unmeasured tiers resolve to the nearest
measured tier, sync-free on bracketing disagreement; a profile whose
engine graph covers decode never dispatches the capture-ineligible
candidate; fingerprint mismatch fails closed for performance-gated
deployments. Pure host logic — no torch, runs anywhere.
"""

from __future__ import annotations

import copy
import importlib.util
import sys
import types
from pathlib import Path
from typing import Any

import pytest

_PKG = (
    Path(__file__).resolve().parents[4]
    / "vllm_omni/model_executor/models/nemotron_asr"
)
_BASE = "vllm_omni.model_executor.models.nemotron_asr"
for _name in (
    "vllm_omni",
    "vllm_omni.model_executor",
    "vllm_omni.model_executor.models",
    _BASE,
):
    if _name not in sys.modules:
        sys.modules[_name] = types.ModuleType(_name)
_spec = importlib.util.spec_from_file_location(
    f"{_BASE}.decode_dispatch", _PKG / "decode_dispatch.py"
)
assert _spec is not None and _spec.loader is not None
_mod = importlib.util.module_from_spec(_spec)
sys.modules[f"{_BASE}.decode_dispatch"] = _mod
_spec.loader.exec_module(_mod)

DispatchTable = _mod.DispatchTable
FingerprintMismatch = _mod.FingerprintMismatch
generate_dispatch_table = _mod.generate_dispatch_table


def _cell(
    geometry: str,
    batch: int,
    lane: str,
    arm: str,
    activity: str,
    latency_us: float,
    *,
    selective: bool | None = None,
) -> dict[str, Any]:
    if selective is None:
        selective = activity in ("silence", "speech")
    return {
        "geometry": geometry,
        "batch": batch,
        "precision": lane,
        "arm": arm,
        "activity": activity,
        "latency_us": latency_us,
        "policy_realized": selective,
        "selective": selective,
        "tokens_match": True,
    }


def _report(cells: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "probe": "p6b_decode_profile",
        "fingerprint": {
            "device_name": "NVIDIA L4",
            "driver": "580.126.20",
            "torch": "2.11.0+cu130",
            "model_shape_digest": "sha256:abc",
            "fork_commit": "deadbeef",
            "dirty_tree": False,
        },
        "dcp": {
            "version": "dcp-v1",
            "targets": {"silence": 0.3, "speech": 3.0},
            "tolerance": {
                "silence": (0.0, 1.0), "speech": (1.5, 4.5),
            },
            "selective_activities": ("silence", "speech"),
            "fallback_rule": "nonselective -> sync-free",
        },
        "lane_results": {
            "fp32": {
                "mismatch_cells": 0,
                "performance_qualified": True,
                "dispatch_eligible": True,
            },
            "bf16-joint": {
                "mismatch_cells": 3,
                "performance_qualified": False,
                "dispatch_eligible": False,
            },
        },
        "generator_eligible_lanes": ["fp32"],
        "cells": cells,
    }


def _tier_cells(
    geometry: str,
    batch: int,
    *,
    dense_eager: float,
    dense_graphed: float,
    compact_silence: float,
    compact_speech: float,
    lane: str = "fp32",
    selective: bool = True,
) -> list[dict[str, Any]]:
    return [
        _cell(geometry, batch, lane, "dense-eager", "speech",
              dense_eager),
        _cell(geometry, batch, lane, "dense-graphed", "speech",
              dense_graphed),
        _cell(geometry, batch, lane, "compact-eager", "silence",
              compact_silence, selective=selective),
        _cell(geometry, batch, lane, "compact-eager", "speech",
              compact_speech, selective=selective),
    ]


# A two-tier fixture: small tier where dense dominates, large tier
# where compact beats dense-eager by far more than the hysteresis at
# both dcp levels.
def _two_tier_report() -> dict[str, Any]:
    cells = _tier_cells(
        "1120ms", 8,
        dense_eager=140_000, dense_graphed=65_000,
        compact_silence=100_000, compact_speech=115_000,
    ) + _tier_cells(
        "1120ms", 1024,
        dense_eager=890_000, dense_graphed=880_000,
        compact_silence=120_000, compact_speech=230_000,
    )
    return _report(cells)


def test_lanes_come_from_generator_eligible_lanes() -> None:
    table = generate_dispatch_table(_two_tier_report())
    assert table.lanes == ("fp32",)
    with pytest.raises(KeyError):
        table.select(
            "1120ms", 8, lane="bf16-joint",
            graph_covers_decode=False,
        )


def test_graph_covered_profile_never_dispatches_compact() -> None:
    # Sanity invariant, not a measured outcome: capture-ineligible
    # compact never enters a profile whose graph covers decode —
    # even at the tier where compact wins eager.
    table = generate_dispatch_table(_two_tier_report())
    assert table.select(
        "1120ms", 1024, lane="fp32", graph_covers_decode=True,
    ) == "dense-graphed"


def test_compact_needs_every_dcp_level_beyond_hysteresis() -> None:
    table = generate_dispatch_table(_two_tier_report())
    # Large tier: compact beats dense-eager by >25% at BOTH levels.
    assert table.select(
        "1120ms", 1024, lane="fp32", graph_covers_decode=False,
    ) == "compact-eager"
    # Small tier: compact loses to the best sync-free arm.
    assert table.select(
        "1120ms", 8, lane="fp32", graph_covers_decode=False,
    ) == "dense-eager"


def test_hysteresis_margin_blocks_thin_wins() -> None:
    report = _two_tier_report()
    # Make compact's speech win thinner than 25% at the large tier.
    for cell in report["cells"]:
        if (
            cell["batch"] == 1024
            and cell["arm"] == "compact-eager"
            and cell["activity"] == "speech"
        ):
            cell["latency_us"] = 700_000  # wins, but only ~21%
    table = generate_dispatch_table(report)
    assert table.select(
        "1120ms", 1024, lane="fp32", graph_covers_decode=False,
    ) == "dense-eager"


def test_nonselective_cell_forces_sync_free() -> None:
    report = _two_tier_report()
    for cell in report["cells"]:
        if (
            cell["batch"] == 1024
            and cell["arm"] == "compact-eager"
            and cell["activity"] == "silence"
        ):
            cell["policy_realized"] = False
            cell["selective"] = False
    table = generate_dispatch_table(report)
    assert table.select(
        "1120ms", 1024, lane="fp32", graph_covers_decode=False,
    ) == "dense-eager"


def test_unmeasured_tier_resolves_nearest_measured() -> None:
    table = generate_dispatch_table(_two_tier_report())
    # 12 is nearest to 8 (both dense) — dense-eager.
    assert table.select(
        "1120ms", 12, lane="fp32", graph_covers_decode=False,
    ) == "dense-eager"
    # 900 is nearest to 1024 (compact tier).
    assert table.select(
        "1120ms", 900, lane="fp32", graph_covers_decode=False,
    ) == "compact-eager"


def test_bracketing_disagreement_selects_sync_free() -> None:
    table = generate_dispatch_table(_two_tier_report())
    # 516 sits between tier 8 (dense) and tier 1024 (compact):
    # nearest is 1024, but the bracketing tiers disagree — the
    # crossover-gap rule selects the sync-free arm.
    assert table.select(
        "1120ms", 516, lane="fp32", graph_covers_decode=False,
    ) == "dense-eager"


def test_unknown_geometry_is_an_error() -> None:
    table = generate_dispatch_table(_two_tier_report())
    with pytest.raises(KeyError):
        table.select(
            "999ms", 8, lane="fp32", graph_covers_decode=False,
        )


def test_fingerprint_mismatch_fails_closed_when_gated() -> None:
    table = generate_dispatch_table(_two_tier_report())
    runtime = dict(table.fingerprint)
    runtime["device_name"] = "NVIDIA A100 80GB PCIe"
    with pytest.raises(FingerprintMismatch):
        table.validate_runtime(runtime, performance_gated=True)
    # Dev profile: warns (returns the mismatched keys), no raise.
    assert "device_name" in table.validate_runtime(
        runtime, performance_gated=False,
    )


def test_table_records_selection_constants() -> None:
    table = generate_dispatch_table(_two_tier_report())
    assert table.hysteresis_pct == 25.0
    assert table.policy_version == "dcp-v1"
    assert table.fingerprint["model_shape_digest"] == "sha256:abc"


def test_dirty_tree_artifact_is_rejected() -> None:
    report = _two_tier_report()
    report["fingerprint"]["dirty_tree"] = True
    with pytest.raises(ValueError, match="dirty"):
        generate_dispatch_table(report)


def test_old_schema_without_lane_results_is_rejected() -> None:
    report = _two_tier_report()
    del report["lane_results"]
    del report["generator_eligible_lanes"]
    with pytest.raises(ValueError, match="lane"):
        generate_dispatch_table(report)


def test_ineligible_lane_cells_are_ignored_entirely() -> None:
    report = _two_tier_report()
    # Add bf16 cells where compact "wins" hugely — they must not
    # produce entries (the lane is not generator-eligible).
    report["cells"] += _tier_cells(
        "1120ms", 1024, lane="bf16-joint",
        dense_eager=900_000, dense_graphed=880_000,
        compact_silence=50_000, compact_speech=60_000,
    )
    table = generate_dispatch_table(report)
    assert table.lanes == ("fp32",)


def test_missing_activity_level_forces_sync_free() -> None:
    report = _two_tier_report()
    report["cells"] = [
        c for c in report["cells"]
        if not (
            c["batch"] == 1024
            and c["arm"] == "compact-eager"
            and c["activity"] == "silence"
        )
    ]
    table = generate_dispatch_table(report)
    # Compact cannot prove the worst case without BOTH dcp levels.
    assert table.select(
        "1120ms", 1024, lane="fp32", graph_covers_decode=False,
    ) == "dense-eager"


def test_serialization_round_trip() -> None:
    table = generate_dispatch_table(_two_tier_report())
    blob = table.to_json()
    loaded = DispatchTable.from_json(blob)
    assert loaded.entries == table.entries
    assert loaded.fingerprint == table.fingerprint
    assert loaded.select(
        "1120ms", 1024, lane="fp32", graph_covers_decode=False,
    ) == "compact-eager"


def test_generation_is_deterministic() -> None:
    a = generate_dispatch_table(_two_tier_report())
    b = generate_dispatch_table(
        copy.deepcopy(_two_tier_report())
    )
    assert a.entries == b.entries
