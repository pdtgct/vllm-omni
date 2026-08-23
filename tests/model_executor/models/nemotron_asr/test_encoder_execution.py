# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Feature-gated static encoder execution contracts."""

from types import SimpleNamespace
from typing import Any

import pytest
import torch

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
    )
    assert resolved.arm == "eager"
    assert resolved.ready_receipt() == {
        "arm": "eager",
        "ready": True,
        "warmup_cells": [],
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
        )


def test_unknown_encoder_execution_arm_fails_closed() -> None:
    # @spec PORT-PERF-009
    with pytest.raises(ValueError, match="unknown encoder_execution_arm"):
        build_encoder_execution(
            SimpleNamespace(),
            SimpleNamespace(encoder_execution_arm="auto-magic"),
            maximum_population=4,
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
    )
    assert resolved.warmup_populations == (1, 2, 3, 4)


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
    )
    caches = SimpleNamespace(
        channel=(torch.zeros(1, 2, 3),),
        time=(torch.zeros(1, 2, 3),),
        valid=torch.zeros(1, dtype=torch.long),
        left_context=2,
    )

    def run_one() -> None:
        execution.transition(
            torch.zeros(1, 4, 5),
            caches,
            torch.zeros(1, dtype=torch.long),
            torch.ones(1, dtype=torch.long),
            3,
            torch.zeros(1, dtype=torch.long),
        )

    monkeypatch.setattr(
        "vllm_omni.model_executor.models.nemotron_asr.encoder_execution.execute_encoder_transition",
        lambda *_args: (torch.zeros(1), torch.zeros(1)),
    )
    execution.warmup_cell(geometry=0, population=1, invoke=run_one)
    execution.seal(expected_cells=((0, 1),))
    run_one()
    assert compiled_calls == [(1, 3), (1, 3)]
    assert execution.ready_receipt() == {
        "arm": "compiled-static",
        "ready": True,
        "warmup_cells": [[0, 1]],
        "warmup_populations": [1, 2],
    }

    two = SimpleNamespace(
        channel=(torch.zeros(2, 2, 3),),
        time=(torch.zeros(2, 2, 3),),
        valid=torch.zeros(2, dtype=torch.long),
        left_context=2,
    )
    with pytest.raises(ValueError, match="was not warmed"):
        execution.transition(
            torch.zeros(2, 4, 5),
            two,
            torch.zeros(2, dtype=torch.long),
            torch.ones(2, dtype=torch.long),
            3,
            torch.zeros(2, dtype=torch.long),
        )
    assert compiled_calls == [(1, 3), (1, 3)]

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
    assert compiled_calls == [(1, 3), (1, 3)]
