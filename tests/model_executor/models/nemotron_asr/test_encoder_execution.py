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
    def forbidden_compile(*_args: Any, **_kwargs: Any) -> Any:
        pytest.fail("the absent baseline must not initialize torch.compile")

    monkeypatch.setattr(torch, "compile", forbidden_compile)
    resolved = build_encoder_execution(SimpleNamespace(), SimpleNamespace())
    assert resolved.arm == "eager"


def test_compiled_static_uses_one_fail_closed_fullgraph_authority(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[Any, dict[str, Any]]] = []

    def fake_compile(fn: Any, **kwargs: Any) -> Any:
        calls.append((fn, kwargs))
        return "compiled-transition"

    monkeypatch.setattr(torch, "compile", fake_compile)
    resolved = build_encoder_execution(
        SimpleNamespace(),
        SimpleNamespace(encoder_execution_arm="compiled-static"),
    )
    assert resolved.arm == "compiled-static"
    assert resolved.transition == "compiled-transition"
    assert len(calls) == 1
    assert calls[0][1] == {
        "fullgraph": True,
        "dynamic": False,
        "options": {"triton.cudagraphs": False},
    }


def test_compiler_initialization_failure_is_not_hidden(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_compile(*_args: Any, **_kwargs: Any) -> Any:
        raise RuntimeError("synthetic compiler failure")

    monkeypatch.setattr(torch, "compile", fail_compile)
    with pytest.raises(RuntimeError, match="synthetic compiler failure"):
        build_encoder_execution(
            SimpleNamespace(),
            SimpleNamespace(encoder_execution_arm="compiled-static"),
        )


def test_unknown_encoder_execution_arm_fails_closed() -> None:
    with pytest.raises(ValueError, match="unknown encoder_execution_arm"):
        build_encoder_execution(
            SimpleNamespace(),
            SimpleNamespace(encoder_execution_arm="auto-magic"),
        )
