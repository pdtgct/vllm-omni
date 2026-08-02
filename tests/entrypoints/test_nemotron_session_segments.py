# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests-first transport-neutral segmentation contract."""

from __future__ import annotations

import inspect
from typing import Any, NoReturn

import pytest

from vllm_omni.entrypoints import nemotron_session

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _fail(message: str) -> NoReturn:
    pytest.fail(message, pytrace=False)
    raise AssertionError(message)


def _require_attribute(owner: Any, name: str, spec_id: str) -> Any:
    try:
        return getattr(owner, name)
    except AttributeError:
        owner_name = getattr(owner, "__name__", type(owner).__name__)
        _fail(f"{spec_id} missing {owner_name}.{name}")


def test_segment_completion_is_small_model_neutral_value() -> None:
    # @spec PORT-SEG-006 / PORT-RTC-007
    completion_type = _require_attribute(
        nemotron_session,
        "SegmentCompletion",
        "PORT-SEG-006",
    )
    completion = completion_type(
        generation=3,
        text="hello",
        reason="model",
    )

    assert completion.generation == 3
    assert completion.text == "hello"
    assert completion.reason == "model"


def test_session_lease_preserves_hypotheses_and_adds_segment_callback() -> None:
    # @spec PORT-SEG-006 / PORT-RTC-007
    signature = inspect.signature(nemotron_session.SessionLease.feed)
    assert "samples" in signature.parameters
    assert "on_accepted" in signature.parameters
    assert "on_segment" in signature.parameters
    assert "list[str]" in str(signature.return_annotation)


def test_session_lease_exposes_force_segment_without_wire_vocabulary() -> None:
    # @spec PORT-SEG-004 / PORT-SEG-006 / PORT-RTC-007
    protocol_method = _require_attribute(
        nemotron_session.SessionLease,
        "force_segment",
        "PORT-SEG-006",
    )
    concrete_method = _require_attribute(
        nemotron_session.NemotronSessionLease,
        "force_segment",
        "PORT-SEG-006",
    )

    for method in (protocol_method, concrete_method):
        assert inspect.iscoroutinefunction(method)
        assert tuple(inspect.signature(method).parameters) == ("self",)


def test_flush_returns_complete_text_and_terminal_acknowledgement() -> None:
    # @spec PORT-SEG-006
    terminal_type = _require_attribute(
        nemotron_session,
        "TerminalResult",
        "PORT-SEG-006",
    )
    completion_type = _require_attribute(
        nemotron_session,
        "SegmentCompletion",
        "PORT-SEG-006",
    )
    result = terminal_type(
        complete_text="hello world",
        completion=completion_type(
            generation=2,
            text=" world",
            reason="terminal",
        ),
    )
    assert result.complete_text == "hello world"
    assert result.completion.text == " world"
    assert result.completion.reason == "terminal"
    assert "TerminalResult" in str(
        inspect.signature(nemotron_session.SessionLease.flush).return_annotation
    )


def test_factory_open_accepts_typed_endpoint_policy() -> None:
    # @spec PORT-SEG-001 / PORT-RTC-001 / PORT-RTC-007
    signature = inspect.signature(nemotron_session.SessionFactory.open)
    assert "cadence" in signature.parameters
    assert "locale" in signature.parameters
    assert "endpoint_policy" in signature.parameters
