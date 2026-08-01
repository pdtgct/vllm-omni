# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Register NemotronASRConfig with transformers AutoConfig.

Eagerly imported by ``vllm_omni.transformers_utils.configs`` so the
``AutoConfig.register`` side-effect runs before any ``ModelConfig`` is
built — the streaming ASR ``config.json`` then loads without
``trust_remote_code`` (PORT-WGT-004).
"""

from transformers import AutoConfig

from vllm_omni.model_executor.models.nemotron_asr.configuration_nemotron_asr import (
    MODEL_TYPE,
    NemotronASRConfig,
)

AutoConfig.register(MODEL_TYPE, NemotronASRConfig)

__all__ = ["NemotronASRConfig"]
