# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""FastConformer encoder structure tests (P2 full-context regime).

Numerical parity against the golden sixth cell lands with weight
conversion on-pod; these GPU-free tests pin the structural contracts:
shape/length arithmetic, strict causality at zero lookahead (through
subsampling, attention, AND the causal conv), intra-chunk lookahead at
nonzero right context, and padding invariance.
"""

import pytest
import torch

from vllm_omni.model_executor.models.nemotron_asr.encoder import (
    FastConformerEncoder,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _tiny(att_context=(8, 0)) -> FastConformerEncoder:
    torch.manual_seed(31)
    enc = FastConformerEncoder(
        feat_in=16,
        d_model=32,
        d_ff=64,
        n_layers=2,
        n_heads=4,
        conv_kernel=5,
        subsampling_channels=16,
        att_context=att_context,
    )
    enc.eval()
    return enc


def _mel(frames: int, feat: int = 16, seed: int = 7) -> torch.Tensor:
    torch.manual_seed(seed)
    return torch.randn(1, feat, frames)


def test_output_shapes_and_lengths():
    enc = _tiny()
    mel = _mel(160)
    out, lens = enc(mel, torch.tensor([160]))
    assert out.shape[0] == 1 and out.shape[2] == 32
    assert out.shape[1] == int(lens[0]) == enc.pre_encode.output_lengths(
        torch.tensor([160])
    )


def test_strictly_causal_at_zero_lookahead():
    # att_context (8, 0): chunk = 1 frame; with causal subsampling and
    # the causal conv, encoder frame t must be invariant to any change
    # in audio strictly after its receptive window.
    enc = _tiny(att_context=(8, 0))
    mel_a = _mel(160)
    mel_b = mel_a.clone()
    mel_b[:, :, 120:] += 3.0  # perturb the tail only
    with torch.no_grad():
        out_a, _ = enc(mel_a, torch.tensor([160]))
        out_b, _ = enc(mel_b, torch.tensor([160]))
    # 120 mel frames -> at least 120//8 - 1 clean encoder frames; use a
    # conservative margin for subsampling edge effects.
    safe = 120 // 8 - 2
    torch.testing.assert_close(out_a[:, :safe], out_b[:, :safe])


def test_lookahead_is_intra_chunk_only():
    # att_context (8, 3): chunk = 4 encoder frames. Frames in chunk 0
    # may see chunk 0 only; perturbing chunk 2's audio must not change
    # chunk 0's output, but should change chunk 1's neighbors' -- no:
    # chunk 1 cannot see chunk 2 either (lookahead is INTRA-chunk).
    enc = _tiny(att_context=(8, 3))
    mel_a = _mel(160)
    mel_b = mel_a.clone()
    mel_b[:, :, 96:] += 3.0  # perturb from encoder frame 12 == chunk 3
    with torch.no_grad():
        out_a, _ = enc(mel_a, torch.tensor([160]))
        out_b, _ = enc(mel_b, torch.tensor([160]))
    # chunks 0-1 (frames 0..7) sit >1 full chunk before the
    # perturbation; margin for subsampling receptive field.
    torch.testing.assert_close(out_a[:, :6], out_b[:, :6])


def test_padding_frames_do_not_change_valid_output():
    enc = _tiny()
    mel = _mel(160)
    padded = torch.cat([mel, torch.zeros(1, 16, 40)], dim=2)
    with torch.no_grad():
        out_a, lens_a = enc(mel, torch.tensor([160]))
        out_b, lens_b = enc(padded, torch.tensor([160]))
    n = int(lens_a[0])
    assert int(lens_b[0]) == n
    torch.testing.assert_close(out_a[:, :n], out_b[:, :n])


def test_weight_names_follow_nemo_layout():
    enc = _tiny()
    names = {name for name, _ in enc.named_parameters()}
    for expected in (
        "pre_encode.conv.0.weight",
        "pre_encode.out.weight",
        "layers.0.self_attn.linear_q.weight",
        "layers.0.self_attn.pos_bias_u",
        "layers.0.conv.depthwise_conv.weight",
        "layers.1.norm_out.weight",
        "layers.0.feed_forward1.linear1.weight",
    ):
        assert expected in names, expected
    # Bias-free per the checkpoint config (attention_bias /
    # convolution_bias false).
    assert "layers.0.self_attn.linear_q.bias" not in names
    assert "layers.0.conv.pointwise_conv1.bias" not in names
