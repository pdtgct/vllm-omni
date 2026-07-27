# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Phase-5 tests-first: the neutral streaming_transport batch-stat hops.

Specs: PORT-OBS-008/009 — the runner->scheduler batch-stat transport
hops (``drain_batch_stats_into_runner_output``,
``forward_batch_stats_to_engine_core_outputs``) live in the neutral,
Prometheus-free ``vllm_omni.metrics.streaming_transport`` module
(correction 3) so the GPU worker/scheduler never need Prometheus.

Both functions unconditionally raise ``NotImplementedError`` (Phase-5
stub), so the unit-level tests below are EXPECTED RED with that exact
failure mode. The runner/scheduler PRODUCTION call sites
(``vllm_omni/worker/gpu_ar_model_runner.py``'s ``OmniModelRunnerOutput``
construction; ``vllm_omni/core/sched/omni_ar_scheduler.py``'s
``update_from_output``) are deep GPU/VllmConfig-coupled code this Mac
cannot instantiate — per the "seam-existence + unit red pair" fallback,
each hop is pinned by (1) a seam-existence source-scan asserting the
production file references the transport function, currently FALSE
(red), plus (2) the unit-level test above driving the function directly
with a fake model/output pair.
"""

from __future__ import annotations

import inspect
from pathlib import Path

import pytest

from vllm_omni.metrics.streaming_transport import (
    drain_batch_stats_into_runner_output,
    forward_batch_stats_to_engine_core_outputs,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

_REPO_ROOT = Path(__file__).resolve().parents[2]


# ---- runner hop: model.consume_batch_stats() -> OmniModelRunnerOutput -------
# (PORT-OBS-008/009) — RED (unit level)


class _FakeModelWithBatchStats:
    def __init__(self, stats: list[tuple[int, int]] | None) -> None:
        self._stats = stats
        self.consume_calls = 0

    def consume_batch_stats(self) -> list[tuple[int, int]] | None:
        self.consume_calls += 1
        return self._stats


# @spec PORT-OBS-008, PORT-OBS-009
def test_runner_drain_attaches_the_consumed_list_to_the_runner_output() -> None:
    from vllm_omni.outputs import OmniModelRunnerOutput

    model = _FakeModelWithBatchStats([(0, 4), (2, 9)])
    runner_output = OmniModelRunnerOutput(req_ids=[], req_id_to_index={}, sampled_token_ids=[])

    drain_batch_stats_into_runner_output(model, runner_output)

    assert model.consume_calls == 1
    assert runner_output.streaming_chunk_batch_stats == [(0, 4), (2, 9)]


# @spec PORT-OBS-008, PORT-OBS-009
def test_runner_drain_forwards_none_when_not_collecting() -> None:
    from vllm_omni.outputs import OmniModelRunnerOutput

    model = _FakeModelWithBatchStats(None)
    runner_output = OmniModelRunnerOutput(req_ids=[], req_id_to_index={}, sampled_token_ids=[])

    drain_batch_stats_into_runner_output(model, runner_output)

    assert runner_output.streaming_chunk_batch_stats is None


# ---- runner hop: production seam-existence (PORT-OBS-008/009) — RED ----------
# GPUARModelRunner needs a full VllmConfig/device setup this Mac cannot
# stand up — pinned via seam-existence source-scan instead of driving the
# class directly.


# @spec PORT-OBS-008, PORT-OBS-009
def test_runner_module_references_the_drain_hop() -> None:
    """The production seam: ``GPUARModelRunner``'s ``OmniModelRunnerOutput``
    construction in ``vllm_omni/worker/gpu_ar_model_runner.py`` must call
    ``streaming_transport.drain_batch_stats_into_runner_output`` — not
    reference it today, so this is RED until Phase 6 wires the call."""
    path = _REPO_ROOT / "vllm_omni/worker/gpu_ar_model_runner.py"
    source = path.read_text()
    assert "drain_batch_stats_into_runner_output" in source, (
        "gpu_ar_model_runner.py must call streaming_transport.drain_batch_stats_into_runner_output "
        "when constructing OmniModelRunnerOutput"
    )


# ---- scheduler hop: OmniModelRunnerOutput -> OmniEngineCoreOutputs -----------
# gated by host statistics collection (PORT-OBS-008/009) — RED (unit level)


# @spec PORT-OBS-008, PORT-OBS-009
def test_scheduler_hop_attaches_when_stats_enabled() -> None:
    from vllm_omni.engine import OmniEngineCoreOutputs
    from vllm_omni.outputs import OmniModelRunnerOutput

    runner_output = OmniModelRunnerOutput(
        req_ids=[],
        req_id_to_index={},
        sampled_token_ids=[],
        streaming_chunk_batch_stats=[(0, 4)],
    )
    engine_core_outputs = OmniEngineCoreOutputs()

    forward_batch_stats_to_engine_core_outputs(runner_output, engine_core_outputs, stats_enabled=True)

    assert engine_core_outputs.streaming_chunk_batch_stats == [(0, 4)]


# @spec PORT-OBS-008, PORT-OBS-009
def test_scheduler_hop_leaves_none_when_stats_disabled() -> None:
    from vllm_omni.engine import OmniEngineCoreOutputs
    from vllm_omni.outputs import OmniModelRunnerOutput

    runner_output = OmniModelRunnerOutput(
        req_ids=[],
        req_id_to_index={},
        sampled_token_ids=[],
        streaming_chunk_batch_stats=[(0, 4)],
    )
    engine_core_outputs = OmniEngineCoreOutputs()

    forward_batch_stats_to_engine_core_outputs(runner_output, engine_core_outputs, stats_enabled=False)

    assert engine_core_outputs.streaming_chunk_batch_stats is None


# ---- scheduler hop: production seam-existence, log_stats True/False ----------
# (PORT-OBS-008/009) — RED. OmniARScheduler needs a full VllmConfig this
# Mac cannot stand up — pinned via seam-existence source-scan.


# @spec PORT-OBS-008, PORT-OBS-009
def test_scheduler_module_references_the_forward_hop_gated_by_log_stats() -> None:
    """The production seam: ``OmniARScheduler.update_from_output`` in
    ``vllm_omni/core/sched/omni_ar_scheduler.py`` must call
    ``streaming_transport.forward_batch_stats_to_engine_core_outputs``,
    gated by ``self.log_stats`` (the omni scheduler's own host-statistics
    switch) — not reference it today, so this is RED until Phase 6 wires
    the call for both the enabled and disabled cases."""
    path = _REPO_ROOT / "vllm_omni/core/sched/omni_ar_scheduler.py"
    source = path.read_text()
    assert "forward_batch_stats_to_engine_core_outputs" in source, (
        "omni_ar_scheduler.py must call streaming_transport.forward_batch_stats_to_engine_core_outputs"
    )
    assert "log_stats" in source, "the scheduler's existing log_stats switch must gate the forward call"


# ---- module-level docstring/signature sanity (real, GREEN) -------------------


# @spec PORT-OBS-008, PORT-OBS-009
def test_transport_functions_are_typed_and_documented() -> None:
    for fn in (drain_batch_stats_into_runner_output, forward_batch_stats_to_engine_core_outputs):
        assert fn.__doc__, f"{fn.__name__} must be documented"
        sig = inspect.signature(fn)
        assert list(sig.parameters), f"{fn.__name__} must declare its parameters"
