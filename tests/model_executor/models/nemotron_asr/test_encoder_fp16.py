# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""CPU dtype and state boundaries for the experimental encoder-only FP16 arm."""

from types import SimpleNamespace

import pytest
import torch

from vllm_omni.model_executor.models.nemotron_asr.encoder import FastConformerEncoder, StreamingCaches
from vllm_omni.model_executor.models.nemotron_asr.encoder_execution import (
    _compiled_transition_model_state,
    _compiled_transition_runner_device,
    execute_encoder_transition,
)
from vllm_omni.model_executor.models.nemotron_asr.lid import PromptConditioner
from vllm_omni.model_executor.models.nemotron_asr.precision import PrecisionPolicy, resolve_precision_policy

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _core():
    torch.manual_seed(31)
    policy = resolve_precision_policy(SimpleNamespace(experimental_encoder_compute_dtype="float16"))
    return SimpleNamespace(
        policy=policy,
        encoder=FastConformerEncoder(
            feat_in=16,
            d_model=32,
            d_ff=64,
            n_layers=2,
            n_heads=4,
            conv_kernel=5,
            subsampling_channels=16,
            att_context=(8, 1),
        )
        .to(policy.dtype_for("encoder_compute"))
        .eval(),
        lid=PromptConditioner(enc_hidden=32, num_prompts=4).eval(),
    )


@pytest.mark.parametrize("equivalent_policy", [False, True])
@torch.inference_mode()
def test_fp16_encoder_retains_fp32_boundary_and_cache_history(equivalent_policy):
    """@spec PORT-PREC-011, PORT-PREC-012: FP32 history and language boundary survive FP16."""
    core = _core()
    if equivalent_policy:
        core.policy = PrecisionPolicy({"*": "fp32", "encoder_compute": "fp16"})
    caches = StreamingCaches(n_layers=2, batch=3, left_context=8, d_model=32, conv_kernel=5, device=torch.device("cpu"))
    caches.channel.normal_()
    caches.time.normal_()
    old_channel, old_time = caches.channel.clone(), caches.time.clone()
    channel_ptr, time_ptr = caches.channel.data_ptr(), caches.time.data_ptr()
    mel = torch.randn(3, 16, 33)
    original_mel = mel.clone()
    lengths = torch.tensor([0, 1, 4])
    seen = []
    handle = core.lid.register_forward_pre_hook(lambda _module, args: seen.append(args[0]))
    try:
        encoded, conditioned = execute_encoder_transition(
            core,
            mel,
            caches,
            torch.zeros(3, dtype=torch.long),
            lengths,
            4,
            torch.tensor([0, 1, 2]),
        )
    finally:
        handle.remove()
    assert encoded is seen[0]
    assert encoded.dtype == conditioned.dtype == torch.float32
    assert torch.isfinite(encoded).all() and torch.isfinite(conditioned).all()
    assert torch.count_nonzero(encoded[0]) == torch.count_nonzero(conditioned[0]) == 0
    torch.testing.assert_close(mel, original_mel, rtol=0, atol=0)
    assert caches.channel.data_ptr() == channel_ptr and caches.time.data_ptr() == time_ptr
    assert caches.channel.dtype == caches.time.dtype == torch.float32
    torch.testing.assert_close(caches.valid, lengths)
    for row, length in enumerate(lengths.tolist()):
        torch.testing.assert_close(caches.channel[:, row, : 8 - length], old_channel[:, row, length:], rtol=0, atol=0)
        torch.testing.assert_close(caches.time[:, row, :, : 4 - length], old_time[:, row, :, length:], rtol=0, atol=0)


def test_mixed_encoder_lid_dtypes_bind_compiler_model_identity():
    """@spec PORT-PREC-013: compiler identity binds component parameter dtypes."""
    core = _core()
    assert _compiled_transition_runner_device(core, activation_dtype=torch.float16) == torch.device("cpu")
    before = _compiled_transition_model_state(core)
    assert {state.dtype for name, state in before.parameters if name.startswith("encoder.")} == {"torch.float16"}
    assert {state.dtype for name, state in before.parameters if name.startswith("lid.")} == {"torch.float32"}
    core.encoder.float()
    assert _compiled_transition_model_state(core) != before
    with pytest.raises(ValueError, match="parameter dtype differs"):
        _compiled_transition_runner_device(core, activation_dtype=torch.float16)


@pytest.mark.parametrize("selection", [None, "float32", "float16", "legacy_bf16_policy"])
def test_apply_policy_dtypes_preserves_module_boundaries(selection):
    """@spec PORT-PREC-010: frontend, conditioner and decoder retain their policy."""
    from vllm_omni.model_executor.models.nemotron_asr.nemotron_asr import apply_policy_dtypes
    from vllm_omni.model_executor.models.nemotron_asr.precision import BF16_COMPUTE

    core = _core()
    core.policy = (
        BF16_COMPUTE
        if selection == "legacy_bf16_policy"
        else resolve_precision_policy(SimpleNamespace(experimental_encoder_compute_dtype=selection))
    )
    core.encoder.float()
    core.featurizer = torch.nn.Linear(2, 2)
    core.predictor = torch.nn.LSTM(2, 2)
    core.joint = torch.nn.Linear(2, 2)
    apply_policy_dtypes(core)
    assert {p.dtype for p in core.encoder.parameters()} == {core.policy.dtype_for("encoder_compute")}
    assert {p.dtype for p in core.featurizer.parameters()} == {torch.float32}
    for module in (core.lid, core.predictor, core.joint):
        assert {p.dtype for p in module.parameters()} == {core.policy.dtype_for("weights")}


def test_weight_loading_preserves_encoder_fp16_and_decoder_fp32():
    """@spec PORT-PREC-017: FP32 source tensors survive encoder load conversion."""
    from vllm_omni.model_executor.models.nemotron_asr.nemotron_asr import NemotronASRForRNNT

    tiny = _core()
    core = torch.nn.Module()
    core.encoder, core.lid = tiny.encoder, tiny.lid
    model = NemotronASRForRNNT.__new__(NemotronASRForRNNT)
    torch.nn.Module.__init__(model)
    model.core = core
    model._encoder_execution = SimpleNamespace(_model_state=None)
    weights = [(name, torch.full_like(value, 1.0001, dtype=torch.float32)) for name, value in core.state_dict().items()]
    source_snapshots = {name: value.clone() for name, value in weights}
    model.load_weights(weights)
    assert {p.dtype for p in core.encoder.parameters()} == {torch.float16}
    assert {p.dtype for p in core.lid.parameters()} == {torch.float32}
    for name, source in weights:
        assert source.dtype == torch.float32
        torch.testing.assert_close(source, source_snapshots[name], rtol=0, atol=0)
        target = core.state_dict()[name]
        torch.testing.assert_close(target, source.to(target.dtype), rtol=0, atol=0)
