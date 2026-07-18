# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Projected-K/V cache equivalence probe (alpha1 gating verification).

``StreamingCaches.channel`` stores pre-projection normed attention
inputs and ``_stream_attention`` re-projects the whole ``[cache | new]``
window every step. The engine-native page layout stores projected K/V
instead (the vLLM KV-page convention). Because ``linear_k``/``linear_v``
are bias-free position-wise maps, projecting a row at its own chunk step
and caching the result must equal caching the row and re-projecting it
later; these tests pin that equivalence — outputs AND the cache
recursion — over long streams and ragged per-session fill.
"""

import pytest
import torch

from vllm_omni.model_executor.models.nemotron_asr.encoder import (
    _LOG_BASE,
    FastConformerEncoder,
    _stream_attention,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _tiny(att_context=(8, 1)) -> FastConformerEncoder:
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


def _stream_attention_projected(
    attn: torch.nn.Module,
    x: torch.Tensor,
    *,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    valid: torch.Tensor,
    pos_emb: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """``_stream_attention`` with the K/V (not input) rows cached.

    ``k_cache``/``v_cache``: (B, C, h, d_k), right-aligned like the
    reference channel cache. New frames are projected once, here, and
    appended; the window keeps the last C projected rows.
    """
    batch, new_frames, _ = x.shape
    capacity = k_cache.shape[1]

    k_new = attn.linear_k(x).view(batch, new_frames, attn.h, attn.d_k)
    v_new = attn.linear_v(x).view(batch, new_frames, attn.h, attn.d_k)
    k = torch.cat([k_cache, k_new], dim=1).transpose(1, 2)
    v = torch.cat([v_cache, v_new], dim=1).transpose(1, 2)
    t2 = capacity + new_frames

    q = attn.linear_q(x).view(batch, new_frames, attn.h, attn.d_k)
    p = attn.linear_pos(pos_emb).view(
        pos_emb.size(0), -1, attn.h, attn.d_k
    ).transpose(1, 2)
    q_u = (q + attn.pos_bias_u).transpose(1, 2)
    q_v = (q + attn.pos_bias_v).transpose(1, 2)
    matrix_bd = attn._rel_shift(torch.matmul(q_v, p.transpose(-2, -1)))
    matrix_ac = torch.matmul(q_u, k.transpose(-2, -1))
    scores = (
        matrix_ac + matrix_bd[:, :, :, : matrix_ac.size(-1)]
    ) / attn.s_d_k
    row = torch.arange(capacity, device=x.device).unsqueeze(0)
    dead = row < (capacity - valid.unsqueeze(1))
    mask = torch.zeros(
        batch, 1, new_frames, t2, dtype=torch.bool, device=x.device
    )
    mask[:, :, :, :capacity] = dead.unsqueeze(1).unsqueeze(2)
    scores = scores.masked_fill(mask, -_LOG_BASE)
    weights = torch.softmax(scores, dim=-1).masked_fill(mask, 0.0)
    out = torch.matmul(weights, v)
    out = out.transpose(1, 2).reshape(batch, new_frames, attn.h * attn.d_k)
    new_k = torch.cat([k_cache, k_new], dim=1)[:, -capacity:]
    new_v = torch.cat([v_cache, v_new], dim=1)[:, -capacity:]
    return attn.linear_out(out), new_k, new_v


def _run_both(
    *,
    batch: int,
    chunks: int,
    frames_per_chunk: int,
    capacity: int,
    start_offsets: list[int] | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Drive reference and candidate over a stream; return stacked
    outputs (chunks, B, F, d) plus final projected caches of each side.
    """
    enc = _tiny()
    layer = enc.layers[0]
    attn = layer.self_attn
    d_model = 32
    torch.manual_seed(97)
    stream = torch.randn(chunks, batch, frames_per_chunk, d_model)
    if start_offsets is None:
        start_offsets = [0] * batch
    pos_emb = enc.pos_enc(
        torch.zeros(1, frames_per_chunk + capacity, d_model)
    )

    cache_ref = torch.zeros(batch, capacity, d_model)
    k_cache = torch.zeros(batch, capacity, attn.h, attn.d_k)
    v_cache = torch.zeros(batch, capacity, attn.h, attn.d_k)
    valid = torch.zeros(batch, dtype=torch.long)
    outs_ref, outs_prj = [], []
    with torch.inference_mode():
        for step in range(chunks):
            # Ragged sessions: element b joins at chunk start_offsets[b];
            # before that its input is zeros and its valid stays 0.
            active = torch.tensor(
                [step >= off for off in start_offsets]
            )
            x = torch.where(
                active.view(batch, 1, 1), stream[step], 0.0
            )
            out_ref, cache_ref = _stream_attention(
                layer, x, cache=cache_ref, valid=valid, pos_emb=pos_emb,
                # Uniform full-valid rows: the probe's raggedness is
                # carried by ``valid`` (dead cache rows), matching the
                # pre-length-aware reference semantics exactly.
                new_valid=torch.ones(
                    x.shape[0], x.shape[1], dtype=torch.bool
                ),
                new_lengths=torch.full(
                    (x.shape[0],), x.shape[1], dtype=torch.long
                ),
            )
            out_prj, k_cache, v_cache = _stream_attention_projected(
                attn, x,
                k_cache=k_cache, v_cache=v_cache,
                valid=valid, pos_emb=pos_emb,
            )
            outs_ref.append(out_ref)
            outs_prj.append(out_prj)
            valid = torch.clamp(
                valid + active.long() * frames_per_chunk, max=capacity
            )
        ref_k = attn.linear_k(cache_ref).view(
            batch, capacity, attn.h, attn.d_k
        )
    return (
        torch.stack(outs_ref),
        torch.stack(outs_prj),
        ref_k,
        k_cache,
    )


def test_projected_kv_matches_reference_single_stream():
    out_ref, out_prj, _, _ = _run_both(
        batch=1, chunks=12, frames_per_chunk=4, capacity=8
    )
    torch.testing.assert_close(out_prj, out_ref, rtol=0.0, atol=1e-6)


def test_projected_kv_no_drift_over_long_stream():
    """Drift check: the cache recursion must not accumulate error."""
    out_ref, out_prj, _, _ = _run_both(
        batch=1, chunks=64, frames_per_chunk=4, capacity=8
    )
    early = (out_prj[:8] - out_ref[:8]).abs().max()
    late = (out_prj[-8:] - out_ref[-8:]).abs().max()
    assert late <= 1e-6, f"late-stream divergence {late}"
    assert late <= max(float(early), 1e-7) * 4, (
        f"error grows along the stream: early {early} late {late}"
    )


def test_projected_kv_matches_reference_ragged_batch():
    """Sessions joining mid-stream (zero-at-admission semantics)."""
    out_ref, out_prj, _, _ = _run_both(
        batch=3, chunks=10, frames_per_chunk=4, capacity=8,
        start_offsets=[0, 3, 7],
    )
    torch.testing.assert_close(out_prj, out_ref, rtol=0.0, atol=1e-6)


def test_projected_cache_recursion_equals_projected_reference_cache():
    """The invariant page eviction relies on: candidate K cache ==
    projection of the reference channel cache, at every capacity fill.
    """
    _, _, ref_k, k_cache = _run_both(
        batch=2, chunks=9, frames_per_chunk=4, capacity=8,
        start_offsets=[0, 2],
    )
    torch.testing.assert_close(k_cache, ref_k, rtol=0.0, atol=1e-6)


def test_projected_kv_partial_fill_masks_dead_rows():
    """Below-capacity fill: dead (zero) cache rows must not leak into
    the softmax on either side — outputs equal from the first chunk.
    """
    out_ref, out_prj, _, _ = _run_both(
        batch=1, chunks=2, frames_per_chunk=3, capacity=8
    )
    torch.testing.assert_close(out_prj, out_ref, rtol=0.0, atol=1e-6)
