# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Source-level pins for RFC-1's realtime-only contribution boundary."""

from pathlib import Path

import pytest

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

_ROOT = Path(__file__).resolve().parents[2]
_OMNI = _ROOT / "vllm_omni"
_MODEL = _OMNI / "model_executor/models/nemotron_asr"
_ENTRYPOINTS = _OMNI / "entrypoints"
_DEPLOY_CONFIG = _OMNI / "deploy/nemotron_asr.yaml"
_LEGACY_STAGE_CONFIG = _OMNI / "model_executor/stage_configs/nemotron_asr.yaml"


def test_nemotron_model_exposes_realtime_without_file_transcription() -> None:
    """PORT-REGIME-003: capability and route eligibility stay realtime-only."""
    source = (_MODEL / "nemotron_asr.py").read_text()

    assert "supports_realtime = True" in source
    for forbidden in (
        "supports_transcription =",
        "def get_speech_to_text_config(",
        "def get_generation_prompt(",
        "def validate_language(",
        "/v1/audio/transcriptions",
    ):
        assert forbidden not in source


def test_rfc1_contains_no_file_transcription_or_router_plugin_modules() -> None:
    """RFC-1 owns neither the file endpoint nor RFC-2's router vehicle."""
    removed = (
        _ENTRYPOINTS / "ephemeral_session.py",
        _ENTRYPOINTS / "openai/nemotron_transcription_rules.py",
        _ENTRYPOINTS / "openai/serving_nemotron_transcription.py",
        _ENTRYPOINTS / "openai/router_plugins.py",
    )
    assert [path.relative_to(_ROOT) for path in removed if path.exists()] == []

    api_server = (_ENTRYPOINTS / "openai/api_server.py").read_text()
    for forbidden in (
        "NEMOTRON_TRANSCRIPTION_ENV",
        "NEMOTRON_MAX_SESSIONS_ENV",
        "_nemotron_transcription_opt_in",
        "NemotronServingTranscription",
        "ServingConcurrencyLimiter",
        "load_router_plugins",
    ):
        assert forbidden not in api_server


def test_rfc1_contains_no_declared_task_advertisement_extension() -> None:
    """The generic task producer was introduced solely for the removed route."""
    assert not (_OMNI / "engine/task_advertisement.py").exists()

    production_files = (
        _OMNI / "config/stage_config.py",
        _OMNI / "engine/async_omni_engine.py",
        _OMNI / "engine/stage_client.py",
        _OMNI / "engine/stage_engine_core_client.py",
        _OMNI / "engine/stage_init_utils.py",
        _OMNI / "engine/stage_runtime.py",
        _OMNI / "diffusion/stage_diffusion_client.py",
        _OMNI / "diffusion/inline_stage_diffusion_client.py",
    )
    offenders = [path.relative_to(_ROOT) for path in production_files if "declared_tasks" in path.read_text()]
    assert offenders == []


def test_session_binding_has_no_ephemeral_cadence_or_admission_counter() -> None:
    """PORT-RTC-003/007: the retained binding is transport-neutral."""
    binding = (_ENTRYPOINTS / "nemotron_session.py").read_text()
    model_session = (_MODEL / "session.py").read_text()

    for forbidden in (
        "ServingConcurrencyLimiter",
        "AdmissionBusyError",
        "open_ephemeral",
        "EPHEMERAL_CADENCE",
        "1120",
        "ephemeral_session",
    ):
        assert forbidden not in binding
    assert "EPHEMERAL_CADENCE" not in model_session
    assert "class NemotronSessionFactory" in binding
    assert "class NemotronSessionLease" in binding


def test_nemotron_stage_config_pins_supported_bringup_lane() -> None:
    """PORT-INT-007: the supported Omni stage owns safe parity defaults."""
    from vllm_omni.config.stage_config import (
        load_deploy_config,
        merge_pipeline_deploy,
    )
    from vllm_omni.model_executor.models.nemotron_asr.pipeline import (
        NEMOTRON_ASR_PIPELINE,
    )

    assert _DEPLOY_CONFIG.exists()
    assert not _LEGACY_STAGE_CONFIG.exists()
    stage = merge_pipeline_deploy(
        NEMOTRON_ASR_PIPELINE,
        load_deploy_config(_DEPLOY_CONFIG),
    )[0].to_omegaconf()

    assert stage.engine_args.scheduler_cls == (
        "vllm_omni.model_executor.models.nemotron_asr.scheduler."
        "NemotronASRScheduler"
    )
    assert stage.engine_args.async_scheduling is False
    assert stage.engine_args.dtype == "float16"
    assert stage.engine_args.enforce_eager is True
    assert stage.engine_args.enable_prefix_caching is False
    assert "speculative_config" not in stage.engine_args
