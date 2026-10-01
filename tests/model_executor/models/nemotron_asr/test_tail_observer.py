# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""CPU checks for opt-in conditional-tail diagnostics (not timing evidence)."""

import importlib.util
import sys
from pathlib import Path

import pytest
import torch

_PKG = Path(__file__).resolve().parents[4] / "vllm_omni/model_executor/models/nemotron_asr"


def _load(name):
    spec = importlib.util.spec_from_file_location(f"nemotron_{name}_under_test", _PKG / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


ConditionalTailObserver = _load("tail_observer").ConditionalTailObserver
CADENCES = _load("manifests").CADENCES

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _observer(limit: int = 8):
    observer = ConditionalTailObserver(
        max_invocations=limit, device=torch.device("cpu"), cadences=CADENCES, max_symbols=10
    )
    observer.begin()
    return observer


def _observe(observer, counts, lengths, final=None, **kwargs):
    counts = torch.as_tensor(counts, dtype=torch.int32)
    live = counts.shape[0]
    observer.observe(
        rows=tuple(range(live)),
        geometry=next(i for i, (_, right) in enumerate(CADENCES.values()) if right + 1 == counts.shape[1]),
        tier=kwargs.pop("tier", live),
        arm="dense-graphed",
        chunk_graph=kwargs.pop("chunk_graph", False),
        counts=counts,
        lengths=torch.tensor(lengths, dtype=torch.int64),
        final_tail=torch.tensor(final if final is not None else [False] * live),
        **kwargs,
    )


def _collect(observer):
    observer.stage()
    observer.collect()
    return observer.receipt()


def test_exact_prefix_predicate_padding_final_flush_and_borrowed_ownership():
    observer = _observer()
    borrowed = torch.tensor([[1, 2, 0, 0], [2, 1, 10, 0]], dtype=torch.int32)
    _observe(observer, borrowed, [2, 3], [False, True], tier=4)
    # A later decoder invocation may overwrite its borrowed output in place.
    borrowed.zero_()
    _observe(observer, borrowed, [0, 0], [True, True], tier=4)
    receipt = _collect(observer)
    assert receipt["estimate_eligible"]
    first, second = receipt["records"]
    assert first["rows"] == [0, 1] and first["decoder_tier"] == 4
    assert first["final_tail_rows"] == 1
    assert [f["valid_lanes"] for f in first["frames"]] == [2, 2, 1, 0]
    assert [f["max_emissions"] for f in first["frames"]] == [2, 2, 10, 0]
    assert [f["tail_runs"] for f in first["frames"]] == [True, True, True, False]
    assert [f["final_valid_lanes"] for f in first["frames"]] == [1, 1, 1, 0]
    assert second["final_tail_rows"] == 2
    assert all(f["valid_lanes"] == 0 and not f["tail_runs"] for f in second["frames"])
    assert second["invocation"] != first["invocation"]


@pytest.mark.parametrize("bad", [-1, 11])
def test_out_of_range_counts_fail_closed(bad):
    observer = _observer()
    _observe(observer, [[bad]], [1])
    receipt = _collect(observer)
    assert not receipt["estimate_eligible"]
    assert receipt["invalid_invocations"] == 1


def test_invalid_frame_must_be_zero_and_lengths_are_clamped():
    observer = _observer()
    _observe(observer, [[0, 0], [1, 2]], [-7, 99])
    assert _collect(observer)["estimate_eligible"]
    observer.begin()
    _observe(observer, [[0, 1]], [1])
    assert not _collect(observer)["estimate_eligible"]


def test_missing_fields_and_overflow_are_visible_and_fail_closed():
    observer = _observer(limit=1)
    observer.observe(
        rows=(0,),
        geometry=0,
        tier=1,
        arm="legacy",
        chunk_graph=False,
        counts=None,
        lengths=None,
        final_tail=torch.tensor([False]),
    )
    _observe(observer, [[0]], [1])
    receipt = _collect(observer)
    assert receipt["attempted_invocations"] == 2
    assert receipt["collected_invocations"] == 1
    assert receipt["dropped_invocations"] == 1
    assert receipt["invalid_invocations"] == 1
    assert not receipt["estimate_eligible"]


def test_empty_bypassed_decode_has_no_opportunity_and_aborted_step_is_incomplete():
    observer = _observer()
    assert _collect(observer)["records"] == []
    assert not observer.receipt()["estimate_eligible"]
    _observe(observer, [[0]], [1])
    assert not observer.receipt()["coverage_complete"]
    observer.begin()  # prior transaction failed before collection
    _observe(observer, [[0]], [1])
    receipt = _collect(observer)
    assert receipt["dropped_invocations"] == 1
    assert not receipt["estimate_eligible"]


def test_only_valid_lane_reaching_two_emissions_runs_the_tail():
    observer = _observer()
    _observe(observer, [[0, 1, 2, 10], [1, 0, 1, 2]], [4, 4])
    receipt = _collect(observer)
    assert receipt["estimate_eligible"]
    assert [f["tail_runs"] for f in receipt["records"][0]["frames"]] == [False, False, True, True]
