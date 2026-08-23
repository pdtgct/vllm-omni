# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Feature-gated static encoder execution contracts."""

from types import SimpleNamespace
from typing import Any

import pytest
import torch

from vllm_omni.model_executor.models.nemotron_asr import (
    encoder_execution as encoder_execution_module,
)
from vllm_omni.model_executor.models.nemotron_asr.encoder_execution import (
    build_encoder_execution,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_absent_encoder_gate_is_the_unchanged_eager_baseline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # @spec PORT-PERF-009
    def forbidden_compile(*_args: Any, **_kwargs: Any) -> Any:
        pytest.fail("the absent baseline must not initialize torch.compile")

    monkeypatch.setattr(torch, "compile", forbidden_compile)
    resolved = build_encoder_execution(
        SimpleNamespace(),
        SimpleNamespace(),
        maximum_population=4,
        warmup_geometries=(0,),
    )
    assert resolved.arm == "eager"
    assert resolved.ready_receipt() == {
        "arm": "eager",
        "ready": True,
        "warmup_cells": [],
        "warmup_geometries": [],
        "warmup_populations": [],
    }


def test_explicit_eager_encoder_gate_does_not_compile(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # @spec PORT-PERF-009
    def forbidden_compile(*_args: Any, **_kwargs: Any) -> Any:
        pytest.fail("the explicit eager arm must not initialize torch.compile")

    monkeypatch.setattr(torch, "compile", forbidden_compile)
    resolved = build_encoder_execution(
        SimpleNamespace(),
        SimpleNamespace(encoder_execution_arm="eager"),
        maximum_population=4,
        warmup_geometries=(0,),
    )
    assert resolved.arm == "eager"


def test_compiled_static_uses_one_fail_closed_fullgraph_authority(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # @spec PORT-PERF-009
    calls: list[tuple[Any, dict[str, Any]]] = []

    def fake_compile(fn: Any, **kwargs: Any) -> Any:
        calls.append((fn, kwargs))
        return "compiled-transition"

    monkeypatch.setattr(torch, "compile", fake_compile)
    resolved = build_encoder_execution(
        SimpleNamespace(),
        SimpleNamespace(encoder_execution_arm="compiled-static"),
        maximum_population=4,
        warmup_geometries=(0, 1, 2, 3, 4),
    )
    assert resolved.arm == "compiled-static"
    assert callable(resolved.transition)
    assert len(calls) == 1
    assert resolved.warmup_populations == (1, 2, 3, 4)
    assert calls[0][1] == {
        "fullgraph": True,
        "dynamic": False,
        "options": {"triton.cudagraphs": False},
    }


def test_compiler_initialization_failure_is_not_hidden(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # @spec PORT-PERF-009
    def fail_compile(*_args: Any, **_kwargs: Any) -> Any:
        raise RuntimeError("synthetic compiler failure")

    monkeypatch.setattr(torch, "compile", fail_compile)
    with pytest.raises(RuntimeError, match="synthetic compiler failure"):
        build_encoder_execution(
            SimpleNamespace(),
            SimpleNamespace(encoder_execution_arm="compiled-static"),
            maximum_population=1,
            warmup_geometries=(0,),
        )


def test_unknown_encoder_execution_arm_fails_closed() -> None:
    # @spec PORT-PERF-009
    with pytest.raises(ValueError, match="unknown encoder_execution_arm"):
        build_encoder_execution(
            SimpleNamespace(),
            SimpleNamespace(encoder_execution_arm="auto-magic"),
            maximum_population=4,
            warmup_geometries=(0,),
        )


def test_compiled_static_derives_complete_population_authority_from_runner_ceiling(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # @spec PORT-PERF-009
    monkeypatch.setattr(torch, "compile", lambda fn, **_kwargs: fn)
    resolved = build_encoder_execution(
        SimpleNamespace(),
        SimpleNamespace(encoder_execution_arm="compiled-static"),
        maximum_population=4,
        warmup_geometries=(0, 2, 4),
    )
    assert resolved.warmup_populations == (1, 2, 3, 4)
    assert resolved.warmup_geometries == (0, 2, 4)


def test_compiled_static_seals_observed_signatures_and_rejects_lazy_compile(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # @spec PORT-PERF-009
    compiled_calls: list[tuple[int, int]] = []

    def fake_compile(fn: Any, **_kwargs: Any) -> Any:
        def compiled(*args: Any) -> Any:
            compiled_calls.append((int(args[0].shape[0]), int(args[4])))
            return fn(*args)

        return compiled

    monkeypatch.setattr(torch, "compile", fake_compile)
    execution = build_encoder_execution(
        SimpleNamespace(),
        SimpleNamespace(encoder_execution_arm="compiled-static"),
        maximum_population=2,
        warmup_geometries=(0,),
    )
    def run(population: int, *, mel_width: int = 4) -> None:
        caches = SimpleNamespace(
            channel=(torch.zeros(population, 2, 3),),
            time=(torch.zeros(population, 2, 3),),
            valid=torch.zeros(population, dtype=torch.long),
            left_context=2,
        )
        execution.transition(
            torch.zeros(population, mel_width, 5),
            caches,
            torch.zeros(population, dtype=torch.long),
            torch.ones(population, dtype=torch.long),
            3,
            torch.zeros(population, dtype=torch.long),
        )

    monkeypatch.setattr(
        encoder_execution_module,
        "execute_encoder_transition",
        lambda *_args: (torch.zeros(1), torch.zeros(1)),
    )
    execution.warmup_domain(
        expected_cells=((0, 1), (0, 2)),
        invoke=lambda _geometry, population: run(population),
    )
    run(1)
    assert compiled_calls == [(1, 3), (2, 3), (1, 3)]
    assert execution.ready_receipt() == {
        "arm": "compiled-static",
        "ready": True,
        "warmup_cells": [[0, 1], [0, 2]],
        "warmup_geometries": [0],
        "warmup_populations": [1, 2],
    }

    with pytest.raises(ValueError, match="was not warmed"):
        run(1, mel_width=5)
    assert compiled_calls == [(1, 3), (2, 3), (1, 3)]

    with pytest.raises(ValueError, match="population 3 was not declared"):
        execution.transition(
            torch.zeros(3, 4, 5),
            SimpleNamespace(
                channel=(torch.zeros(3, 2, 3),),
                time=(torch.zeros(3, 2, 3),),
                valid=torch.zeros(3, dtype=torch.long),
                left_context=2,
            ),
            torch.zeros(3, dtype=torch.long),
            torch.ones(3, dtype=torch.long),
            3,
            torch.zeros(3, dtype=torch.long),
        )
    assert compiled_calls == [(1, 3), (2, 3), (1, 3)]


def test_compiled_static_warmup_scopes_cache_budget_to_declared_domain(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # @spec PORT-PERF-009
    observed_limits: list[tuple[int, int]] = []

    def fake_compile(fn: Any, **_kwargs: Any) -> Any:
        def compiled(*args: Any) -> Any:
            observed_limits.append(
                (
                    int(torch._dynamo.config.cache_size_limit),
                    int(torch._dynamo.config.accumulated_cache_size_limit),
                )
            )
            return fn(*args)

        return compiled

    monkeypatch.setattr(torch, "compile", fake_compile)
    monkeypatch.setattr(torch._dynamo.config, "cache_size_limit", 8)
    monkeypatch.setattr(
        torch._dynamo.config,
        "accumulated_cache_size_limit",
        12,
    )
    monkeypatch.setattr(
        encoder_execution_module,
        "execute_encoder_transition",
        lambda *_args: (torch.zeros(1), torch.zeros(1)),
    )
    execution = build_encoder_execution(
        SimpleNamespace(),
        SimpleNamespace(encoder_execution_arm="compiled-static"),
        maximum_population=4,
        warmup_geometries=(0, 1, 2, 3, 4),
    )
    cells = tuple((geometry, population) for geometry in range(5) for population in range(1, 5))

    def invoke(geometry: int, population: int) -> None:
        caches = SimpleNamespace(
            channel=(torch.zeros(population, 2, 3),),
            time=(torch.zeros(population, 2, 3),),
            valid=torch.zeros(population, dtype=torch.long),
            left_context=2,
        )
        execution.transition(
            torch.zeros(population, 4 + geometry, 5),
            caches,
            torch.zeros(population, dtype=torch.long),
            torch.ones(population, dtype=torch.long),
            3 + geometry,
            torch.zeros(population, dtype=torch.long),
        )

    execution.profile_cell(
        geometry=4,
        population=4,
        invoke=lambda: invoke(4, 4),
    )
    assert observed_limits == [(20, 20)]
    assert torch._dynamo.config.cache_size_limit == 8
    assert torch._dynamo.config.accumulated_cache_size_limit == 12

    execution.warmup_domain(expected_cells=cells, invoke=invoke)

    assert observed_limits == [(20, 20)] * 21
    assert torch._dynamo.config.cache_size_limit == 8
    assert torch._dynamo.config.accumulated_cache_size_limit == 12
    assert execution.ready


def test_compiled_static_preseal_invocation_requires_declared_cell_authority(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # @spec PORT-PERF-009
    compiled_calls = 0

    def fake_compile(fn: Any, **_kwargs: Any) -> Any:
        def compiled(*args: Any) -> Any:
            nonlocal compiled_calls
            compiled_calls += 1
            return fn(*args)

        return compiled

    monkeypatch.setattr(torch, "compile", fake_compile)
    execution = build_encoder_execution(
        SimpleNamespace(),
        SimpleNamespace(encoder_execution_arm="compiled-static"),
        maximum_population=1,
        warmup_geometries=(0,),
    )
    caches = SimpleNamespace(
        channel=(torch.zeros(1, 2, 3),),
        time=(torch.zeros(1, 2, 3),),
        valid=torch.zeros(1, dtype=torch.long),
        left_context=2,
    )

    with pytest.raises(ValueError, match="lacks declared cell authority"):
        execution.transition(
            torch.zeros(1, 4, 5),
            caches,
            torch.zeros(1, dtype=torch.long),
            torch.ones(1, dtype=torch.long),
            3,
            torch.zeros(1, dtype=torch.long),
        )
    with pytest.raises(ValueError, match="geometry 1 was not declared"):
        execution.warmup_cell(
            geometry=1,
            population=1,
            invoke=lambda: None,
        )
    with pytest.raises(ValueError, match=r"missing=\[\(0, 1\)\]"):
        execution._seal()
    assert compiled_calls == 0


def test_compiled_static_warmup_restores_cache_budget_after_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # @spec PORT-PERF-009
    calls = 0
    fail_second = True

    def fake_compile(fn: Any, **_kwargs: Any) -> Any:
        def compiled(*args: Any) -> Any:
            nonlocal calls, fail_second
            calls += 1
            if fail_second and calls == 2:
                raise RuntimeError("synthetic specialization failure")
            return fn(*args)

        return compiled

    monkeypatch.setattr(torch, "compile", fake_compile)
    monkeypatch.setattr(torch._dynamo.config, "cache_size_limit", 1)
    monkeypatch.setattr(
        torch._dynamo.config,
        "accumulated_cache_size_limit",
        1,
    )
    execution = build_encoder_execution(
        SimpleNamespace(),
        SimpleNamespace(encoder_execution_arm="compiled-static"),
        maximum_population=1,
        warmup_geometries=(0, 1),
    )
    monkeypatch.setattr(
        encoder_execution_module,
        "execute_encoder_transition",
        lambda *_args: (torch.zeros(1), torch.zeros(1)),
    )

    def invoke(geometry: int, _population: int) -> Any:
        return execution.transition(
            torch.zeros(1, 4 + geometry, 5),
            SimpleNamespace(
                channel=(torch.zeros(1, 2, 3),),
                time=(torch.zeros(1, 2, 3),),
                valid=torch.zeros(1, dtype=torch.long),
                left_context=2,
            ),
            torch.zeros(1, dtype=torch.long),
            torch.ones(1, dtype=torch.long),
            3 + geometry,
            torch.zeros(1, dtype=torch.long),
        )

    with pytest.raises(RuntimeError, match="synthetic specialization failure"):
        execution.warmup_domain(
            expected_cells=((0, 1), (1, 1)),
            invoke=invoke,
        )

    assert torch._dynamo.config.cache_size_limit == 1
    assert torch._dynamo.config.accumulated_cache_size_limit == 1
    assert not execution.ready
    assert execution.ready_receipt()["warmup_cells"] == []

    fail_second = False
    execution.warmup_domain(
        expected_cells=((0, 1), (1, 1)),
        invoke=invoke,
    )
    assert execution.ready
