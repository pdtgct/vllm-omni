# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests-first session wiring for endpointing and accepted-audio ownership."""

from __future__ import annotations

import asyncio
import inspect
from types import SimpleNamespace
from typing import Any, NoReturn

import numpy as np
import pytest

from vllm_omni.model_executor.models.nemotron_asr import session, streaming

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _fail(message: str) -> NoReturn:
    pytest.fail(message, pytrace=False)
    raise AssertionError(message)


def _endpointing_module() -> Any:
    try:
        return __import__(
            "vllm_omni.model_executor.models.nemotron_asr.endpointing",
            fromlist=["EndpointPolicy"],
        )
    except ModuleNotFoundError:
        _fail("PORT-SEG-001 missing EndpointPolicy")


def _require_attribute(owner: Any, name: str, spec_id: str) -> Any:
    try:
        return getattr(owner, name)
    except AttributeError:
        _fail(f"{spec_id} missing {owner.__name__}.{name}")


def _hf(**overrides: Any) -> SimpleNamespace:
    values: dict[str, Any] = {
        "num_asr_labels": 100,
        "vocab_size": 105,
        "eos_token_id": 101,
        "audio_chunk_token_id": 102,
        "eou_token_id": 103,
        "flush_token_id": 104,
        "prompt_dictionary": {"auto": 0, "en-US": 1},
        "num_prompts": 2,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _default_policy() -> Any:
    return _endpointing_module().EndpointPolicy.resolve(
        mode="greedy_blank",
        stop_history_ms=800,
        residue_frames=2,
        frame_stride_ms=80,
        history_capacity_frames=12,
    )


def _build(**overrides: Any) -> Any:
    values: dict[str, Any] = {
        "cadence": "560ms",
        "locale": "en-US",
        "endpoint_policy": _default_policy(),
        "endpoint_history_capacity_frames": 12,
        "accepted_audio_capacity_samples": 480_000,
    }
    values.update(overrides)
    return session.NemotronRealtimeSession.from_model_config(_hf(), **values)


def test_factory_binds_immutable_controls_policy_and_audio_authority() -> None:
    # @spec PORT-RTC-001 / PORT-SEG-001 / PORT-SESS-001
    built = _build()

    assert built.park_token_id == 101
    assert built.audio_chunk_token_id == 102
    assert built.eou_token_id == 103
    assert built.flush_token_id == 104
    assert built.endpoint_policy == _default_policy()
    assert built.accepted_audio.capacity_samples == 480_000
    assert not hasattr(built, "endpoint_book")
    assert not hasattr(built, "endpoint_history")
    for attribute in (
        "park_token_id",
        "audio_chunk_token_id",
        "eou_token_id",
        "flush_token_id",
        "endpoint_policy",
        "accepted_audio",
    ):
        with pytest.raises(AttributeError):
            setattr(built, attribute, None)


@pytest.mark.parametrize(
    "overrides",
    [
        {"eou_token_id": None},
        {"flush_token_id": None},
        {"eou_token_id": 101},
        {"flush_token_id": 102},
        {"eou_token_id": 100},
    ],
)
def test_factory_rejects_missing_colliding_or_label_space_control(
    overrides: dict[str, Any]
) -> None:
    # @spec PORT-RTC-001 / PORT-WGT-004
    with pytest.raises(ValueError, match="token|control|label|distinct"):
        session.NemotronRealtimeSession.from_model_config(
            _hf(**overrides),
            endpoint_policy=_default_policy(),
            endpoint_history_capacity_frames=12,
            accepted_audio_capacity_samples=480_000,
        )


def test_policy_exceeding_installed_capacity_fails_before_session_exists() -> None:
    # @spec PORT-RTC-001 / PORT-SEG-001
    oversized = _endpointing_module().EndpointPolicy.resolve(
        mode="greedy_blank",
        stop_history_ms=1_600,
        residue_frames=2,
        frame_stride_ms=80,
        history_capacity_frames=32,
    )
    with pytest.raises(ValueError, match="capacity"):
        _build(
            endpoint_policy=oversized,
            endpoint_history_capacity_frames=12,
        )


def test_segmenter_delegates_to_one_session_owned_audio_authority() -> None:
    # @spec PORT-SESS-001 / PORT-SESS-014
    async def scenario() -> None:
        chunk_samples = session.AdmittedGeometry.from_cadence(
            "560ms"
        ).chunk_samples
        built = _build(accepted_audio_capacity_samples=chunk_samples)

        async def over_capacity_piece() -> Any:
            yield np.zeros(2 * chunk_samples, dtype=np.float32)

        outputs = streaming.buffer_stream(
            over_capacity_piece(),
            asyncio.Queue(),
            built,
        )
        with pytest.raises(ValueError, match="buffer_overflow"):
            await anext(outputs)
        snapshot = built.accepted_audio.snapshot()
        assert snapshot.accepted_samples == 0
        assert snapshot.outstanding_samples == 0
        await outputs.aclose()

    asyncio.run(scenario())


def test_session_control_methods_delegate_to_one_authority_without_second_lock() -> None:
    # @spec PORT-SESS-014 / PORT-SEG-004
    built = _build()
    cls = session.NemotronRealtimeSession
    for name in (
        "accept_audio",
        "force_segment",
        "select_prompt",
        "begin_finalize",
    ):
        method = _require_attribute(cls, name, "PORT-SESS-014")
        assert not inspect.iscoroutinefunction(method), name

    samples = np.arange(
        built.geometry.chunk_samples + 1,
        dtype=np.float32,
    )
    built.accept_audio(samples)
    built.force_segment()
    assert built.select_prompt("auto") == 0
    built.begin_finalize()

    assert [unit.kind for unit in built.accepted_audio.ready_units] == [
        "regular",
        "forced_eou",
        "final_tail",
    ]
    assert built.accepted_audio.snapshot().accepted_samples == len(samples)
