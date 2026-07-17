# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""α4 pod-tier: registry, config pass, sampling pin, allocator rung.

Specs: PORT-INT-001/002 (registration + resolution + budget),
PORT-DEC-005/007 (the declarative per-update pin and core's wholesale
replacement — the negative that makes it a per-update duty),
PORT-STATE-002/004 (attention-free allocator: uniform-type grouping
over raw heterogeneous pages). Consult: D-α4a/c/d.

Runs on the pod tiers (installed vllm + vllm_omni). CPU-only: the
two-stage config pass runs the same hooks on the CPU platform
(α2-verified), and grouping/allocation take injected memory numbers.
"""

from types import SimpleNamespace

import pytest

from vllm_omni.model_executor.models.nemotron_asr.precision import (
    FP32_BRINGUP,
)
from vllm_omni.model_executor.models.nemotron_asr.state_layers import (
    ConvCachePage,
    LSTMStatePage,
    ReplayQueuePage,
    WindowCachePage,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

ARCH = "Nemotron3_5AsrForRNNT"
WINDOW, D_MODEL, KERNEL = 56, 1024, 9


# ---- D-α4a: one registration, every consumer (PORT-INT-001) -------------------


def test_registry_resolves_the_model_class():
    # Wheel-side signature (vllm 0.24.0, ee0da84ab9 @
    # vllm/model_executor/models/registry.py _ModelRegistry
    # .resolve_model_cls): takes a required `model_config: ModelConfig`
    # second argument (duck-typed access only: `.model_impl` and
    # `.convert_type` are read internally) — the bare
    # `resolve_model_cls([ARCH])` this test used to call predates that
    # signature.
    from types import SimpleNamespace

    from vllm_omni.model_executor.models.registry import OmniModelRegistry

    duck_model_config = SimpleNamespace(model_impl="auto", convert_type="none")
    cls, resolved_arch = OmniModelRegistry.resolve_model_cls(
        [ARCH], duck_model_config
    )
    assert resolved_arch == ARCH
    assert cls.__name__ == "NemotronASRForRNNT"
    assert cls.supports_realtime is True
    assert cls.realtime_max_tokens == 141  # 14 × 10 + park


def test_pipeline_registered_and_single_stage():
    from vllm_omni.config.pipeline_registry import OMNI_PIPELINES
    from vllm_omni.config.stage_config import StageExecutionType

    pipeline = OMNI_PIPELINES["nemotron_asr"]
    assert pipeline.model_arch == ARCH
    assert len(pipeline.stages) == 1
    stage = pipeline.stages[0]
    assert stage.execution_type is StageExecutionType.LLM_AR
    assert stage.final_output_type == "text"
    assert stage.input_sources == ()


def test_sampling_constraints_carry_the_pin():
    # The declarative per-update pin (PORT-DEC-005/007): paramless
    # realtime updates fall back to merged stage defaults, and the
    # merge gives sampling_constraints the last word — so these three
    # keys ARE the pin. max_tokens explicit because the omni realtime
    # route never consults realtime_max_tokens (consult flag 2).
    from vllm_omni.config.pipeline_registry import OMNI_PIPELINES

    constraints = OMNI_PIPELINES["nemotron_asr"].stages[0].sampling_constraints
    assert constraints["temperature"] == 0.0
    assert constraints["max_tokens"] == 141
    assert constraints["detokenize"] is True
    for hostile in ("allowed_token_ids", "min_tokens", "logits_processors"):
        assert hostile not in constraints


def test_constraints_win_the_deploy_merge():
    # stage_config.merge: sampling.update(ps.sampling_constraints) —
    # the model's pipeline pin overrides deploy-YAML values on
    # collision. Drive the real merge with a hostile deploy default.
    #
    # Wheel-side shape (vllm_omni.config.stage_config @ the α4 pod
    # checkout): merge_pipeline_deploy takes a real `DeployConfig`
    # dataclass, not a raw nested dict (`_apply_platform_overrides`
    # reads `deploy.platforms`); per-stage overrides are
    # `StageDeployConfig` dataclass instances keyed by `stage_id`. The
    # merged result carries `default_sampling_params` inside
    # `StageConfig.yaml_extras`, not as a direct attribute.
    from vllm_omni.config.pipeline_registry import OMNI_PIPELINES
    from vllm_omni.config.stage_config import (
        DeployConfig,
        StageDeployConfig,
        merge_pipeline_deploy,
    )

    pipeline = OMNI_PIPELINES["nemotron_asr"]
    deploy = DeployConfig(
        stages=[
            StageDeployConfig(
                stage_id=0,
                default_sampling_params={"temperature": 0.9},
            )
        ]
    )
    merged = merge_pipeline_deploy(pipeline, deploy)
    stage0 = merged[0]
    assert stage0.yaml_extras["default_sampling_params"]["temperature"] == 0.0


# ---- D-α4c: the registry-driven two-stage config pass (PORT-STATE-002) --------


def _configured_vllm_config():
    """A real VllmConfig resolving our arch through the registry.

    The rung's point (ledger, α2 close-out): the architecture string
    alone drives the whole engine-config derivation.
    """
    raise NotImplementedError("α4 code phase: config-builder helper")


def test_two_stage_pass_sets_block_size_and_never_pads():
    # is_hybrid routes the pre-pass: mamba_block_size = max_model_len
    # (mode "none"); the platform hook early-returns on the all-SSM
    # model BEFORE the align phase, so mamba_page_size_padded stays
    # None — raw heterogeneous pages, padding-free (the measured
    # RAW_PURE_STATE_GROUPS_FORMED=True posture).
    cfg = _configured_vllm_config()
    assert cfg.cache_config.mamba_block_size == cfg.model_config.max_model_len
    assert cfg.cache_config.mamba_page_size_padded is None
    assert cfg.cache_config.enable_prefix_caching in (False, None)


# ---- D-α4c: attention-free allocator rung (PORT-STATE-004) --------------------


def four_kind_spec_dict(cfg) -> dict:
    pages = {
        "encoder.layers.0.window": WindowCachePage(
            prefix="encoder.layers.0.window",
            window=WINDOW,
            d_model=D_MODEL,
            policy=FP32_BRINGUP,
        ),
        "encoder.layers.0.conv": ConvCachePage(
            prefix="encoder.layers.0.conv",
            d_model=D_MODEL,
            kernel=KERNEL,
            policy=FP32_BRINGUP,
        ),
        "predictor.state": LSTMStatePage(
            prefix="predictor.state",
            pred_rnn_layers=2,
            pred_hidden=640,
            policy=FP32_BRINGUP,
        ),
        "decode.replay": ReplayQueuePage(
            prefix="decode.replay",
            max_symbols_per_step=10,
            max_frames_per_chunk=7,
            policy=FP32_BRINGUP,
        ),
    }
    return {name: page.get_kv_cache_spec(cfg) for name, page in pages.items()}


def raw_duck_config(max_model_len: int = 4096) -> SimpleNamespace:
    # The REAL attention-free outcome: block size from the pre-pass,
    # padding never set (no align phase ran).
    return SimpleNamespace(
        cache_config=SimpleNamespace(
            mamba_block_size=max_model_len,
            mamba_page_size_padded=None,
            mamba_cache_mode="none",
            # Wheel-side read (vllm 0.24.0 @ ee0da84ab9,
            # kv_cache_utils.py:945 may_override_num_blocks): real
            # CacheConfig always carries this field; the duck config
            # needs it too once get_kv_cache_config_from_groups is
            # driven for real.
            num_gpu_blocks_override=None,
        ),
        speculative_config=None,
        scheduler_config=SimpleNamespace(
            disable_hybrid_kv_cache_manager=False
        ),
        model_config=SimpleNamespace(max_model_len=max_model_len),
    )


def test_pure_state_dict_forms_one_uniform_type_group():
    from vllm.v1.core.kv_cache_utils import get_kv_cache_groups
    from vllm.v1.kv_cache_interface import UniformTypeKVCacheSpecs

    cfg = raw_duck_config()
    spec_dict = four_kind_spec_dict(cfg)
    # Raw pages are heterogeneous — four kinds, four sizes.
    assert len({s.page_size_bytes for s in spec_dict.values()}) == 4
    groups = get_kv_cache_groups(cfg, spec_dict)
    assert len(groups) == 1
    assert isinstance(groups[0].kv_cache_spec, UniformTypeKVCacheSpecs)
    assert sorted(groups[0].layer_names) == sorted(spec_dict)
    # The group page is the sum of the raw per-layer pages.
    assert groups[0].kv_cache_spec.page_size_bytes == sum(
        s.page_size_bytes for s in spec_dict.values()
    )


def test_allocator_sizes_per_layer_tensors_from_available_memory():
    from vllm.v1.core.kv_cache_utils import (
        get_kv_cache_config_from_groups,
        get_kv_cache_groups,
    )

    cfg = raw_duck_config()
    spec_dict = four_kind_spec_dict(cfg)
    groups = get_kv_cache_groups(cfg, spec_dict)
    page_sum = sum(s.page_size_bytes for s in spec_dict.values())
    budget = 64 * page_sum  # room for exactly 64 sessions
    # Wheel-side signature (vllm 0.24.0 @ ee0da84ab9,
    # vllm/v1/core/kv_cache_utils.py:1318): 3 positional args — the
    # per-layer kv_cache_spec dict is no longer accepted; the group's
    # embedded UniformTypeKVCacheSpecs already carries it.
    kv_config = get_kv_cache_config_from_groups(cfg, groups, budget)
    assert kv_config.num_blocks == 64
    tensors = kv_config.kv_cache_tensors
    assert len(tensors) == len(spec_dict)
    for tensor in tensors:
        assert len(tensor.shared_by) == 1
        (layer,) = tensor.shared_by
        assert tensor.size == spec_dict[layer].page_size_bytes * 64


def test_session_footprint_is_mib_class():
    # Criterion 2 of the vehicle decision, pinned as a regression: one
    # session (one block of every kind at the real 24-layer geometry)
    # stays in the ~MiB class. Here: 24 windows + 24 convs + LSTM +
    # queue at fp32.
    cfg = raw_duck_config()
    window = WindowCachePage(
        prefix="w", window=WINDOW, d_model=D_MODEL, policy=FP32_BRINGUP
    ).get_kv_cache_spec(cfg)
    conv = ConvCachePage(
        prefix="c", d_model=D_MODEL, kernel=KERNEL, policy=FP32_BRINGUP
    ).get_kv_cache_spec(cfg)
    lstm = LSTMStatePage(
        prefix="l", pred_rnn_layers=2, pred_hidden=640, policy=FP32_BRINGUP
    ).get_kv_cache_spec(cfg)
    queue = ReplayQueuePage(
        prefix="q",
        max_symbols_per_step=10,
        max_frames_per_chunk=14,
        policy=FP32_BRINGUP,
    ).get_kv_cache_spec(cfg)
    per_session = (
        24 * (window.page_size_bytes + conv.page_size_bytes)
        + lstm.page_size_bytes
        + queue.page_size_bytes
    )
    assert per_session < 8 * 1024 * 1024  # ~6.1 MiB design number


def test_zero_available_memory_raises_the_documented_error():
    from vllm.v1.core.kv_cache_utils import (
        check_enough_kv_cache_memory,
        get_kv_cache_groups,
    )

    cfg = raw_duck_config()
    spec_dict = four_kind_spec_dict(cfg)
    get_kv_cache_groups(cfg, spec_dict)  # groups must form first
    with pytest.raises(ValueError):
        check_enough_kv_cache_memory(cfg, spec_dict, 0)


# ---- D-α4b negative: core replaces params wholesale (PORT-DEC-007) ------------


def test_session_update_replaces_sampling_params_wholesale():
    # The reason the pin must ride EVERY update: core's
    # _update_request_as_session assigns update.sampling_params over
    # the session's. This documents core behavior at the pin — if it
    # changes, PORT-DEC-007's composition gets revisited.
    import inspect

    from vllm.v1.core.sched.scheduler import Scheduler

    source = inspect.getsource(Scheduler._update_request_as_session)
    assert "sampling_params = update.sampling_params" in source.replace(
        "session.", ""
    ).replace("request.", "")
