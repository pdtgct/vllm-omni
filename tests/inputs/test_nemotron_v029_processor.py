# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Native v0.29 apply/dummy paths must never invoke an HF audio processor."""

from types import SimpleNamespace
from typing import cast

import numpy as np
import pytest
import torch
from transformers import PretrainedConfig
from vllm.config import ModelConfig
from vllm.config.multimodal import MultiModalConfig
from vllm.multimodal.processing.context import InputProcessingContext, TimingContext
from vllm.multimodal.processing.inputs import ProcessorInputs

from vllm_omni.model_executor.models.nemotron_asr.processor import (
    NemotronASRDummyInputsBuilder,
    NemotronASRMultiModalProcessor,
    NemotronASRProcessingInfo,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]
PLACEHOLDER = 13089


@pytest.fixture
def processor(monkeypatch):
    mm_config = MultiModalConfig()
    model_config = SimpleNamespace(
        model="test-nemotron",
        hf_config=PretrainedConfig(audio_chunk_token_id=PLACEHOLDER),
        multimodal_config=mm_config,
        get_multimodal_config=lambda: mm_config,
    )
    context = InputProcessingContext(cast(ModelConfig, model_config), tokenizer=None)
    info = NemotronASRProcessingInfo(context)

    def forbid_hf_processor(**kwargs):
        pytest.fail("HF processor lookup is forbidden for raw-audio passthrough")

    monkeypatch.setattr(info, "get_hf_processor", forbid_hf_processor)
    return NemotronASRMultiModalProcessor(info, NemotronASRDummyInputsBuilder(info))


def assert_carrier(result, expected_audio):
    assert result["prompt_token_ids"] == [PLACEHOLDER]
    placeholders = result["mm_placeholders"]["audio"]
    assert len(placeholders) == 1
    assert placeholders[0].offset == 0
    assert placeholders[0].length == 1
    fields = result["mm_kwargs"]["audio"]
    assert len(fields) == 1
    audio = fields[0]["audio"].data
    assert isinstance(audio, torch.Tensor)
    assert audio.dtype == torch.float32
    torch.testing.assert_close(audio, torch.as_tensor(expected_audio), rtol=0, atol=0)


def test_native_dummy_apply_builds_one_carrier_without_hf_processor(processor):
    inputs = processor.dummy_inputs.get_dummy_processor_inputs(1, {"audio": 1}, {})
    assert inputs.prompt == []
    result = processor.apply(inputs, TimingContext(enabled=False))
    assert_carrier(result, np.zeros(16_000, dtype=np.float32))
    assert processor.cache is None


@pytest.mark.parametrize("tensor_input", [False, True])
def test_native_token_apply_preserves_audio_and_prompt_without_hf_processor(processor, tensor_input):
    audio = np.array([0.125, -0.25, 0.5, -0.875], dtype=np.float32)
    raw = torch.from_numpy(audio) if tensor_input else audio
    items = processor.info.parse_mm_data({"audio": raw}, validate=False)
    prompt = [PLACEHOLDER]
    result = processor.apply(ProcessorInputs(prompt, items), TimingContext(enabled=False))
    assert result["prompt_token_ids"] == prompt
    assert_carrier(result, audio)
