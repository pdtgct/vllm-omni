# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Single-stage LLM_AR pipeline for the streaming ASR model.

One stage, no transfers (PORT-INT-001). ``sampling_constraints`` is
the model-owned, per-update sampling pin (PORT-DEC-005/007): on the
realtime route every ``StreamingUpdate`` carries no params, so each
update falls back to the merged stage-0 defaults — and the merge gives
``sampling_constraints`` the last word over deploy-YAML values
(``stage_config.merge_pipeline_deploy``). ``max_tokens`` must be
explicit here: the omni realtime connection does not consult
``realtime_max_tokens`` (unlike core's), and an unset value re-derives
per update as ``max_model_len - seq_len`` — unbounded.
"""

from vllm_omni.config.stage_config import (
    PipelineConfig,
    StageExecutionType,
    StagePipelineConfig,
)
from vllm_omni.model_executor.models.nemotron_asr.identity import ARCHITECTURE
from vllm_omni.model_executor.models.nemotron_asr.rnnt import (
    MAX_SYMBOLS_PER_STEP,
)

#: Worst-case per-burst budget: the largest chunk config (1120 ms,
#: 14 frames) times the per-frame emission cap, plus the park token
#: (PORT-INT-002; realtime_token_budget(frames_per_chunk=14)).
_MAX_BURST_TOKENS = 14 * MAX_SYMBOLS_PER_STEP + 1

def _derive_engine_args(hf_config: object) -> dict:
    """Derive the engine stop id from the served configuration.

    The park token — the served config's ``eos_token_id`` — must stop
    every burst. vLLM only inherits eos into request stop ids from the
    generation config, and whether that inheritance happens depends on
    which files the checkpoint ships (the public card ships a
    ``generation_config.json`` without eos, which blocks the
    from-model-config fallback the authored artifact relied on).
    Declaring the override here derives the value from the config and
    removes the file-shape dependency on every path.
    """
    eos = getattr(hf_config, "eos_token_id", None)
    if isinstance(eos, bool) or not isinstance(eos, int):
        return {}
    return {"override_generation_config": {"eos_token_id": eos}}


NEMOTRON_ASR_PIPELINE = PipelineConfig(
    model_type="nemotron_asr",
    model_arch=ARCHITECTURE,
    # The public HF card resolves here too: by its architectures entry
    # (this fallback) and by its model_type (the registry alias).
    hf_architectures=(ARCHITECTURE,),
    derive_engine_args=_derive_engine_args,
    stages=(
        StagePipelineConfig(
            stage_id=0,
            model_stage="asr",
            execution_type=StageExecutionType.LLM_AR,
            input_sources=(),
            final_output=True,
            final_output_type="text",
            owns_tokenizer=True,
            requires_multimodal_data=True,
            scheduler_cls=("vllm_omni.model_executor.models.nemotron_asr.scheduler.NemotronASRScheduler"),
            sampling_constraints={
                # The greedy pin (PORT-DEC-005): RNN-T greedy decode;
                # replay steps argmax forced-logits rows.
                "temperature": 0.0,
                # Constant per-burst budget (PORT-INT-002); see module
                # docstring for why this must be explicit.
                "max_tokens": _MAX_BURST_TOKENS,
                "detokenize": True,
            },
        ),
    ),
)
