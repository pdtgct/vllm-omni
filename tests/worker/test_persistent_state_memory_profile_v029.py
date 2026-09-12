# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Allocator-safe v0.29 accounting at the persistent-state worker seam."""

from contextlib import contextmanager
from types import SimpleNamespace

import pytest
import torch
from vllm.utils.mem_utils import MemoryProfilingResult, MemorySnapshot

from vllm_omni.engine.persistent_state_config import PersistentStateRuntimeConfig
from vllm_omni.worker import base as worker_base
from vllm_omni.worker import persistent_state as worker_state

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _worker(monkeypatch, *, result, explicit=None, limit=20, override=None):
    spec = SimpleNamespace(page_size_bytes=100)
    config = SimpleNamespace(
        cache_config=SimpleNamespace(kv_cache_memory_bytes=explicit, num_gpu_blocks_override=override)
    )
    monkeypatch.setattr(worker_state, "discover_persistent_state_specs", lambda _config: {"state": spec})
    monkeypatch.setattr(
        PersistentStateRuntimeConfig,
        "from_vllm_config",
        lambda _config: SimpleNamespace(max_resident_sessions=limit, safety_reserve_slots=1),
    )
    monkeypatch.setattr(worker_base.current_omni_platform, "is_rocm", lambda: False)
    calls = []

    @contextmanager
    def profile(snapshot, *, weights_memory):
        assert snapshot is result.before_create
        assert weights_memory == 200
        calls.append("measure")
        yield result

    def profile_run():
        assert torch.is_inference_mode_enabled()
        calls.append("run")

    monkeypatch.setattr(worker_base, "memory_profiling", profile)
    worker = object.__new__(worker_base.OmniGPUWorkerBase)
    worker.cache_config = config.cache_config
    worker.vllm_config = config
    worker.init_snapshot = result.before_create
    worker.requested_memory = 2_000
    worker.local_rank = 0
    worker.model_runner = SimpleNamespace(
        model_memory_usage=200,
        profile_run=profile_run,
        get_kv_cache_spec=lambda: {"state": spec},
    )
    return worker, calls


def _profile(*, consumed=800, transient=200, old_peak=500, old_non_torch=-400):
    return MemoryProfilingResult(
        before_create=MemorySnapshot(device=torch.device("cpu"), auto_measure=False),
        total_consumed=consumed,
        transient_peak_headroom=transient,
        non_kv_cache_memory=consumed + transient,
        torch_peak_increase=old_peak,
        non_torch_increase=old_non_torch,
        weights_memory=200,
    )


@pytest.mark.parametrize("old_non_torch", [-400, 700])
def test_device_consumption_not_allocator_tracking_drives_pool(monkeypatch, old_non_torch):
    """@spec PORT-STATE-004: signed allocator bookkeeping cannot create space."""
    result = _profile(old_non_torch=old_non_torch)
    worker, calls = _worker(monkeypatch, result=result)

    assert worker.determine_available_memory() == 1_000
    assert worker.available_kv_cache_memory_bytes == 1_000
    assert worker.total_consumed == 800
    assert worker.peak_activation_memory == 200
    assert calls == ["measure", "run"]


def test_retained_profile_allocations_are_not_counted_twice(monkeypatch):
    """@spec PORT-STATE-004: warmup reporting adds only transient peak headroom."""
    result = _profile(consumed=900, transient=100, old_peak=600, old_non_torch=700)
    worker, _ = _worker(monkeypatch, result=result)

    assert worker.determine_available_memory() == 1_000
    # Core's later warmup report adds these fields; retained allocations are
    # already in total_consumed, so torch_peak_increase is not the addend.
    assert worker.total_consumed + worker.peak_activation_memory == 1_000


@pytest.mark.parametrize("explicit", [None, 1_000])
def test_both_budget_branches_keep_downward_resident_limit(monkeypatch, explicit):
    """@spec PORT-STATE-004: a smaller resident count returns unused budget."""
    worker, calls = _worker(monkeypatch, result=_profile(), explicit=explicit, limit=4)

    assert worker.determine_available_memory() == 600  # four resident + safety + null
    assert calls == (["run"] if explicit is not None else ["measure", "run"])


@pytest.mark.parametrize("explicit", [False, True])
@pytest.mark.parametrize("available,override", [(1_000, 11), (200, None)])
def test_budget_and_override_failures_remain_physical(monkeypatch, explicit, available, override):
    """@spec PORT-STATE-004: neither branch manufactures physical capacity."""
    result = _profile(consumed=2_000 - available, transient=0)
    worker, calls = _worker(
        monkeypatch,
        result=result,
        explicit=available if explicit else None,
        override=override,
    )

    with pytest.raises(ValueError, match="profiled|bound|null block"):
        worker.determine_available_memory()
    assert calls == (["run"] if explicit else ["measure", "run"])
