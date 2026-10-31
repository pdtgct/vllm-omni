# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""The Nemotron VoiceChat duplex MRv2 deploy profile resolves to the expected keys.

``nemotron_labs_voicechat_duplex_mrv2.yaml`` moves only stage 0 (thinker) to
Model Runner V2 on CUDA; stages 1 (talker) and 2 (code2wav) stay on V1 and
non-CUDA platforms fall back to V1 everywhere. Admission limits are inherited
from ``nemotron_labs_voicechat_duplex.yaml``. CPU-only; no GPU validation.
"""

import pytest

from tests.helpers.stage_config import get_deploy_config_path
from vllm_omni.config.stage_config import (
    _apply_platform_overrides,
    load_deploy_config,
    merge_pipeline_deploy,
)
from vllm_omni.model_executor.models.nemotron_voicechat.pipeline import NEMOTRON_VOICECHAT_PIPELINE
from vllm_omni.platforms import current_omni_platform

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

_DEPLOY = "nemotron_labs_voicechat_duplex_mrv2.yaml"
_BASE = "nemotron_labs_voicechat_duplex.yaml"


def _resolve(monkeypatch, name: str, platform: str):
    monkeypatch.setattr(current_omni_platform, "device_name", platform)
    config = _apply_platform_overrides(load_deploy_config(get_deploy_config_path(name)), platform=platform)
    return config, merge_pipeline_deploy(NEMOTRON_VOICECHAT_PIPELINE, config)


def test_voicechat_duplex_mrv2_profile_requests_duplex_and_stage0_v2(monkeypatch) -> None:
    config, stages = _resolve(monkeypatch, _DEPLOY, "cuda")
    assert config.session_mode == "duplex"
    assert [s.yaml_engine_args["use_v2_model_runner"] for s in stages] == [True, False, False]


@pytest.mark.parametrize("platform", ["npu", "xpu", "rocm", "musa"])
def test_voicechat_duplex_mrv2_profile_keeps_non_cuda_on_v1(monkeypatch, platform) -> None:
    _, stages = _resolve(monkeypatch, _DEPLOY, platform)
    assert [s.yaml_engine_args["use_v2_model_runner"] for s in stages] == [False, False, False]


def test_voicechat_duplex_mrv2_profile_inherits_admission_limits(monkeypatch) -> None:
    config, stages = _resolve(monkeypatch, _DEPLOY, "cuda")
    base_config, base_stages = _resolve(monkeypatch, _BASE, "cuda")
    assert config.active_stream_window == base_config.active_stream_window == 1
    assert config.duplex_session == base_config.duplex_session
    assert config.duplex_session.max_sessions == 1
    assert len(stages) == len(base_stages) == 3
    for stage, base_stage in zip(stages, base_stages):
        assert stage.yaml_engine_args["max_num_seqs"] == base_stage.yaml_engine_args["max_num_seqs"] == 1
        assert stage.yaml_engine_args["async_scheduling"] == base_stage.yaml_engine_args["async_scheduling"]
