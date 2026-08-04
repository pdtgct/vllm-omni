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
    CARD_MODEL_TYPE,
    MODEL_TYPE,
    NemotronASRConfig,
    NemotronCardServingConfig,
)

AutoConfig.register(MODEL_TYPE, NemotronASRConfig)

# The public HF card's model type is owned by transformers' own
# ``nemotron3_5_asr`` implementation, so it is NOT re-registered with
# AutoConfig here. vLLM consults its own config registry before
# AutoConfig; inserting the translating class there makes every
# process's ``ModelConfig`` load the public checkpoint through this
# port's schema (vLLM re-registers registry entries with
# ``exist_ok=True``, so the override is process-local and sanctioned).
from vllm.transformers_utils.config import (  # noqa: E402
    _CONFIG_REGISTRY,
)

_CONFIG_REGISTRY[CARD_MODEL_TYPE] = NemotronCardServingConfig

__all__ = ["NemotronASRConfig", "NemotronCardServingConfig"]
