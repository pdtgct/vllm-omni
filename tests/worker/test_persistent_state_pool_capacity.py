# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Post-profile physical allocation contracts for persistent state."""

from types import SimpleNamespace

import pytest

from vllm_omni.worker import persistent_state as worker_state

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _config(*, override: int | None = None) -> SimpleNamespace:
    return SimpleNamespace(
        cache_config=SimpleNamespace(num_gpu_blocks_override=override),
    )


def _persistent_specs() -> dict[str, SimpleNamespace]:
    return {"state": SimpleNamespace(page_size_bytes=100)}


def _install_fakes(
    monkeypatch: pytest.MonkeyPatch,
    *,
    max_resident_sessions: int,
    safety_reserve_slots: int = 1,
) -> None:
    monkeypatch.setattr(
        worker_state,
        "discover_persistent_state_specs",
        lambda _config: {"state": SimpleNamespace(page_size_bytes=100)},
    )
    from vllm_omni.engine.persistent_state_config import (
        PersistentStateRuntimeConfig,
    )

    monkeypatch.setattr(
        PersistentStateRuntimeConfig,
        "from_vllm_config",
        lambda _config: SimpleNamespace(
            max_resident_sessions=max_resident_sessions,
            safety_reserve_slots=safety_reserve_slots,
        ),
    )


def test_post_profile_allocation_returns_memory_above_a_smaller_count(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """@spec PORT-STATE-004: a count may size down, never size up."""

    _install_fakes(monkeypatch, max_resident_sessions=4)
    resolved = worker_state.resolve_persistent_state_available_memory(
        _config(),
        available_memory_bytes=1_000,
        cache_specs=_persistent_specs(),
    )

    assert resolved == 600  # four real + safety + null


def test_post_profile_allocation_clamps_count_to_physical_memory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """@spec PORT-STATE-004: a logical request cannot create capacity."""

    _install_fakes(monkeypatch, max_resident_sessions=20)
    resolved = worker_state.resolve_persistent_state_available_memory(
        _config(),
        available_memory_bytes=1_000,
        cache_specs=_persistent_specs(),
    )

    assert resolved == 1_000  # eight real + safety + null


@pytest.mark.parametrize(
    ("available", "override"),
    ((1_000, 11), (200, None)),
)
def test_post_profile_allocation_rejects_impossible_physical_requests(
    monkeypatch: pytest.MonkeyPatch,
    available: int,
    override: int | None,
) -> None:
    """@spec PORT-STATE-004: override and minimum fit fail closed."""

    _install_fakes(monkeypatch, max_resident_sessions=8)
    with pytest.raises(ValueError, match="profiled|bound|null block"):
        worker_state.resolve_persistent_state_available_memory(
            _config(override=override),
            available_memory_bytes=available,
            cache_specs=_persistent_specs(),
        )


def test_non_persistent_worker_memory_is_unchanged(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The common worker remains a no-op for every ordinary model."""

    monkeypatch.setattr(
        worker_state,
        "discover_persistent_state_specs",
        lambda _config: {},
    )
    assert (
        worker_state.resolve_persistent_state_available_memory(
            _config(),
            available_memory_bytes=123_456,
            cache_specs={},
        )
        == 123_456
    )


def test_worker_returns_resolved_bytes_to_vllm_cache_construction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """@spec PORT-STATE-004: the real worker allocation seam consumes the resolver."""

    from vllm_omni.worker import base as worker_base

    seen: list[tuple[object, int]] = []

    def resolve(
        config: object,
        *,
        available_memory_bytes: int,
        cache_specs: object,
    ) -> int:
        assert cache_specs == {"state": "persistent"}
        seen.append((config, available_memory_bytes))
        return 600

    monkeypatch.setattr(
        worker_state,
        "resolve_persistent_state_available_memory",
        resolve,
    )
    monkeypatch.setattr(worker_base.current_omni_platform, "is_rocm", lambda: False)
    config = SimpleNamespace()
    worker = object.__new__(worker_base.OmniGPUWorkerBase)
    worker.cache_config = SimpleNamespace(kv_cache_memory_bytes=1_000)
    worker.model_runner = SimpleNamespace(
        profile_run=lambda: None,
        get_kv_cache_spec=lambda: {"state": "persistent"},
    )
    worker.vllm_config = config

    assert worker.determine_available_memory() == 600
    assert seen == [(config, 1_000)]


def test_mixed_cache_layout_fails_before_vllm_consumes_a_global_byte_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """@spec PORT-STATE-004: the page-size conversion is persistent-only."""

    _install_fakes(monkeypatch, max_resident_sessions=4)
    with pytest.raises(ValueError, match="persistent-only"):
        worker_state.resolve_persistent_state_available_memory(
            _config(),
            available_memory_bytes=1_000,
            cache_specs={**_persistent_specs(), "attention": object()},
        )
