# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""PORT-DEC-005/INT-002: optional EOU and PARK both consume emission budget.

These regressions use parent-native primitives and require no five-profile
constructor, so the budget correction can be reviewed as a prerequisite.
"""

from types import SimpleNamespace

import pytest
import torch
from vllm.sampling_params import SamplingParams

from vllm_omni.config.omni_config import VllmOmniConfig
from vllm_omni.config.stage_config import DeployConfig, StageDeployConfig, merge_pipeline_deploy
from vllm_omni.engine.stage_init_utils import (
    extract_legacy_stage_metadata,
    extract_stage_metadata_from_omni_stage_config,
)
from vllm_omni.entrypoints.omni_base import OmniBase
from vllm_omni.model_executor.models.nemotron_asr import advance, rnnt
from vllm_omni.model_executor.models.nemotron_asr.manifests import SESSION_LIMITS
from vllm_omni.model_executor.models.nemotron_asr.model_state_v2 import NemotronASRModelState
from vllm_omni.model_executor.models.nemotron_asr.native_burst import NativeBurstHandoff
from vllm_omni.model_executor.models.nemotron_asr.pipeline import NEMOTRON_ASR_PIPELINE

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _effective_stage_sampling(typed, *, deploy_budget=999, caller_budget=None):
    """Exercise actual stage merging, metadata extraction and request constraints."""
    deploy = DeployConfig(
        stages=[
            StageDeployConfig(
                stage_id=0,
                default_sampling_params={"max_tokens": deploy_budget, "temperature": 1.0},
            )
        ]
    )
    original_constraints = dict(NEMOTRON_ASR_PIPELINE.stages[0].sampling_constraints)
    if typed:
        config = VllmOmniConfig.from_pipeline_config(NEMOTRON_ASR_PIPELINE, user_deploy_config=deploy)
        stage = config.stage_configs[0]
        metadata = extract_stage_metadata_from_omni_stage_config(stage)
    else:
        stage = merge_pipeline_deploy(NEMOTRON_ASR_PIPELINE, deploy)[0].to_omegaconf()
        metadata = extract_legacy_stage_metadata(stage)
    base = OmniBase.__new__(OmniBase)
    base.engine = SimpleNamespace(num_stages=1, stage_configs=[stage])
    base.default_sampling_params_list = [metadata.default_sampling_params]
    base.sampling_constraints_list = base._get_sampling_constraints_list([stage])
    caller = None if caller_budget is None else SamplingParams(max_tokens=caller_budget, temperature=1.0)
    effective = base.resolve_sampling_params_list(caller, allow_delta_coercion=True)[0]
    assert deploy.stages[0].default_sampling_params["max_tokens"] == deploy_budget
    assert NEMOTRON_ASR_PIPELINE.stages[0].sampling_constraints == original_constraints
    if caller is not None:
        assert caller.max_tokens == caller_budget
    return effective, metadata.default_sampling_params, base.sampling_constraints_list[0]


def _admission_state(*, bound=142, max_model_len=143):
    state = object.__new__(NemotronASRModelState)
    state.model = SimpleNamespace(
        _native_burst_handoff=NativeBurstHandoff(max_tokens=bound),
        config=SimpleNamespace(eos_token_id=9000),
    )
    state.model_config = SimpleNamespace(max_model_len=max_model_len)
    return state


def _request(params):
    return SimpleNamespace(
        sampling_params=params,
        prefill_token_ids=[9001],
        prompt_token_ids=[9001],
        num_computed_tokens=0,
    )


@pytest.mark.parametrize("frames,bound", [(1, 12), (2, 22), (4, 42), (7, 72), (14, 142)])
def test_realtime_helper_reserves_optional_eou_and_park(frames, bound):
    # @spec PORT-DEC-005, PORT-INT-002
    assert rnnt.realtime_token_budget(frames_per_chunk=frames) == bound


@pytest.mark.parametrize("typed", [False, True], ids=["legacy", "typed"])
@pytest.mark.parametrize("deploy_budget", [141, 999])
@pytest.mark.parametrize("caller_budget", [None, 17, 999], ids=["defaults", "small-caller", "large-caller"])
def test_effective_stage_budget_keeps_both_control_tokens_after_request_merge(typed, deploy_budget, caller_budget):
    # @spec PORT-DEC-005, PORT-INT-002
    effective, defaults, constraints = _effective_stage_sampling(
        typed,
        deploy_budget=deploy_budget,
        caller_budget=caller_budget,
    )
    assert defaults.max_tokens == constraints["max_tokens"] == effective.max_tokens == 142
    assert effective.temperature == 0.0
    assert effective.detokenize is True
    assert effective.skip_special_tokens is False


@pytest.mark.parametrize("typed", [False, True], ids=["legacy", "typed"])
def test_real_merged_sampling_reaches_native_worker_admission(typed):
    # @spec PORT-DEC-012, PORT-INT-002
    effective, _, _ = _effective_stage_sampling(typed)
    state = _admission_state()
    state.validate_omni_request(_request(effective))
    assert effective.max_tokens == 142
    assert state.num_new_sampled_tokens_per_step == 1


@pytest.mark.parametrize("budget,accepted", [(141, False), (142, True)])
def test_effective_worker_budget_still_rejects_undersized_141(budget, accepted):
    # @spec PORT-DEC-012, PORT-INT-002
    state = _admission_state()
    request = _request(SamplingParams(max_tokens=budget, temperature=0))
    if accepted:
        state.validate_omni_request(request)
    else:
        with pytest.raises(ValueError, match="complete burst plus PARK"):
            state.validate_omni_request(request)
    assert request.num_computed_tokens == 0
    assert request.prefill_token_ids == [9001]


@pytest.mark.parametrize("max_model_len,accepted", [(142, False), (143, True)])
def test_whole_native_history_headroom_is_independent_of_sampling_fix(max_model_len, accepted):
    # @spec PORT-DEC-012, PORT-INT-002
    state = _admission_state(max_model_len=max_model_len)
    request = _request(SamplingParams(max_tokens=142, temperature=0))
    if accepted:
        state.validate_omni_request(request)
    else:
        with pytest.raises(ValueError, match="history"):
            state.validate_omni_request(request)
    assert request.prefill_token_ids == [9001]


@pytest.mark.parametrize("budget,accepted", [(21, False), (22, True), (142, True)])
def test_legacy_native_geometry_bound_remains_22(budget, accepted):
    # @spec PORT-DEC-014, PORT-INT-002
    from vllm_omni.model_executor.models.nemotron_asr.configuration_nemotron_asr import NemotronASRConfig
    from vllm_omni.model_executor.models.nemotron_asr.native_burst import native_burst_token_budget

    bound = native_burst_token_budget(NemotronASRConfig(supported_num_lookahead_tokens=[1]))
    assert bound == 22
    state = _admission_state(bound=bound, max_model_len=23)
    request = _request(SamplingParams(max_tokens=budget, temperature=0))
    if accepted:
        state.validate_omni_request(request)
    else:
        with pytest.raises(ValueError, match="complete burst plus PARK"):
            state.validate_omni_request(request)


@pytest.mark.parametrize("typed", [False, True], ids=["legacy", "typed"])
def test_compatibility_140_labels_eou_and_legal_park_fit_effective_budget(typed):
    # @spec PORT-DEC-003, PORT-DEC-005, PORT-INT-002
    effective, _, _ = _effective_stage_sampling(typed)
    blank, park, placeholder, eou = 12, 9000, 9001, 15
    cap = SESSION_LIMITS["queue_capacity"]
    assert cap == 141
    labels = [2, 3] * 70 + [eou]
    queue = torch.zeros(1, cap, dtype=torch.int32)
    book = torch.zeros(1, 7, dtype=torch.int64)
    book[0, rnnt.QUEUE_LAST_LABEL] = 3
    status = torch.zeros(1, dtype=torch.int32)
    result = advance.AdvanceResult(torch.tensor([labels], dtype=torch.int32), torch.tensor([cap]), status)
    context = advance.EmissionContext(
        torch.tensor([advance.ROLE_CHUNK]),
        torch.tensor([placeholder]),
        torch.tensor([0]),
        queue,
        book,
        torch.tensor([0]),
        status,
    )
    adapter = advance.make_mrv1_adapter(hidden_size=1, park_id=park, blank_id=blank)
    current = adapter(result, context)
    observed = [int(current.rows[0, 0])]
    while observed[-1] != park:
        assert len(observed) <= cap
        context = advance.EmissionContext(
            torch.tensor([advance.ROLE_REPLAY]),
            torch.tensor([observed[-1]]),
            torch.empty(0, dtype=torch.long),
            current.queue,
            current.book,
            torch.tensor([0]),
            status,
        )
        current = adapter(result, context)
        observed.append(int(current.rows[0, 0]))
    assert observed == labels + [park]
    assert len(observed) == 142
    assert current.book[0, rnnt.QUEUE_HEAD] == current.book[0, rnnt.QUEUE_LEN] == cap
    assert current.book[0, rnnt.BOOK_PENDING_ECHO] == 0
    assert len(observed) <= effective.max_tokens
