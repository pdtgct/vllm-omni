# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""bf16 compute lane tests (PORT-PREC-001/003/005, GPU-free tier).

The beta2 finding made concrete: the compute modules must actually
consult the ``PrecisionPolicy`` — ``apply_policy_dtypes`` casts them,
the entry seams cast activations to follow, and every state class
keeps its own axis. These tests pin the delegation mechanics and the
mixed-dtype seams (encoder entry, cache read/write, joint inputs,
LSTM state) under ``BF16_COMPUTE``; parity vs goldens is the pod
tier's transcript-blocking gate, not asserted here.
"""

import pytest
import torch

from vllm_omni.model_executor.models.nemotron_asr.encoder import (
    FastConformerEncoder,
    StreamingCaches,
    stream_step,
)
from vllm_omni.model_executor.models.nemotron_asr.precision import (
    BF16_COMPUTE,
    FP32_BRINGUP,
    PrecisionPolicy,
)
from vllm_omni.model_executor.models.nemotron_asr.rnnt import (
    Joint,
    Predictor,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _tiny_encoder() -> FastConformerEncoder:
    torch.manual_seed(31)
    enc = FastConformerEncoder(
        feat_in=16,
        d_model=32,
        d_ff=64,
        n_layers=2,
        n_heads=4,
        conv_kernel=5,
        subsampling_channels=16,
        att_context=(8, 1),
    )
    enc.eval()
    return enc


def test_bf16_compute_policy_is_constructible_and_distinct():
    """State classes at fp32 pass the recurrent guard; the identifier
    keys a separate provenance lane from bring-up fp32.
    """
    assert BF16_COMPUTE.dtype_for("weights") == torch.bfloat16
    assert BF16_COMPUTE.dtype_for("lstm_state") == torch.float32
    assert BF16_COMPUTE.dtype_for("conv_state") == torch.float32
    assert BF16_COMPUTE.identifier != FP32_BRINGUP.identifier


def test_full_context_forward_runs_bf16_with_fp32_mel():
    """The encoder entry seam casts fp32 mel to the weight dtype."""
    enc = _tiny_encoder().to(torch.bfloat16)
    torch.manual_seed(7)
    mel = torch.randn(1, 16, 64)  # fp32, as the featurizer emits
    with torch.inference_mode():
        out, _ = enc(mel, torch.tensor([64]))
    assert out.dtype == torch.bfloat16
    assert torch.isfinite(out.float()).all()


def test_stream_step_bf16_keeps_fp32_caches():
    """attention_cache/conv_state stay their constructed (fp32) dtype
    across steps while compute runs bf16 — read-cast in, write-cast
    out (PORT-PREC-005's axis independence at the serve tier).
    """
    enc = _tiny_encoder().to(torch.bfloat16)
    caches = StreamingCaches(
        n_layers=2, batch=1, d_model=32, left_context=8,
        conv_kernel=5, device=torch.device("cpu"),
    )
    torch.manual_seed(7)
    with torch.inference_mode():
        for _ in range(3):
            out = stream_step(
                enc, torch.randn(1, 16, 32), caches, drop_extra=0
            )
    assert out.dtype == torch.bfloat16
    assert caches.channel.dtype == torch.float32
    assert caches.time.dtype == torch.float32
    assert torch.isfinite(out.float()).all()
    assert torch.isfinite(caches.channel).all()


def test_joint_bridges_fp32_pred_to_bf16_weights():
    """The joint seam: fp32 pred_out (recurrent dtype) + bf16 weights
    compute without error, logits in the weight dtype.
    """
    torch.manual_seed(5)
    joint = Joint(
        enc_hidden=32, pred_hidden=16, joint_hidden=24, vocab_size=11
    ).to(torch.bfloat16)
    logits = joint.logits(
        torch.randn(1, 32, dtype=torch.bfloat16),
        torch.randn(1, 16),  # fp32, from the fp32-state predictor
    )
    assert logits.dtype == torch.bfloat16
    assert logits.shape == (1, 12)


def test_predictor_state_stays_fp32_under_bf16_weights():
    """PORT-PREC-005: the manual cell up-casts bf16 weights to the
    state dtype per step; (h, c) accumulate at fp32 throughout.
    """
    torch.manual_seed(5)
    pred = Predictor(
        vocab_size=11, pred_hidden=16, pred_rnn_layers=2
    ).to(torch.bfloat16)
    h = torch.zeros(2, 1, 16)
    c = torch.zeros(2, 1, 16)
    with torch.inference_mode():
        for label in (3, 7, 11):
            out, (h, c) = pred.step(torch.tensor([label]), (h, c))
    assert h.dtype == torch.float32
    assert c.dtype == torch.float32
    assert torch.isfinite(out.float()).all()


def _stub_core(policy: PrecisionPolicy):
    """Duck-typed stand-in: apply_policy_dtypes touches only these
    attributes, and the full-size NemotronASRCore is too heavy for the
    GPU-free tier (its encoder is fixed at the 0.6B geometry).
    """
    from types import SimpleNamespace

    torch.manual_seed(5)
    return SimpleNamespace(
        policy=policy,
        encoder=_tiny_encoder(),
        lid=torch.nn.Linear(32, 32),
        predictor=Predictor(
            vocab_size=11, pred_hidden=16, pred_rnn_layers=2
        ),
        joint=Joint(
            enc_hidden=32, pred_hidden=16, joint_hidden=24, vocab_size=11
        ),
        featurizer=torch.nn.Linear(8, 8),
    )


def test_apply_policy_requires_matching_compute_classes():
    """This realization derives activations from the weight dtype at
    the entry seams; a policy splitting them must refuse, not lie.
    """
    from vllm_omni.model_executor.models.nemotron_asr.nemotron_asr import (
        apply_policy_dtypes,
    )

    core = _stub_core(
        PrecisionPolicy(
            {"weights": "fp32", "activations": "bf16", "*": "fp32"}
        )
    )
    with pytest.raises(ValueError, match="activations"):
        apply_policy_dtypes(core)


def test_apply_policy_casts_compute_not_featurizer():
    from vllm_omni.model_executor.models.nemotron_asr.nemotron_asr import (
        apply_policy_dtypes,
    )

    core = _stub_core(BF16_COMPUTE)
    apply_policy_dtypes(core)
    assert next(core.encoder.parameters()).dtype == torch.bfloat16
    assert next(core.joint.parameters()).dtype == torch.bfloat16
    assert next(core.predictor.parameters()).dtype == torch.bfloat16
    # The mel front-end stays fp32: upstream of the activations seam.
    assert next(core.featurizer.parameters()).dtype == torch.float32
