# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fully paged streaming encoder step (alpha2 seam swap, math tier).

``stream_step_paged`` is ``encoder.stream_step`` with every piece of
cross-chunk state moved out of ``StreamingCaches`` into pages — the
one seam where the served track and the engine-native track diverge:

- left-context attention -> projected-K/V blocks (``attention_pages``,
  sliding-window spec territory);
- depthwise-conv tails   -> per-layer ``ConvCachePage`` state pages
  (``MambaSpec`` territory), addressed by ``state_indices`` exactly as
  the engine hands ``MambaBase`` layers their page pool + per-request
  index tensor.

Sessions own page indices, never tensor rows: batch composition may
change every step (join/leave/reorder) and a session's state follows
its indices. The engine tier binds these arguments from the forward
context (mamba_attn metadata pattern); the math is already final here.
"""

import torch

from vllm_omni.model_executor.models.nemotron_asr.attention_pages import (
    paged_stream_attention,
)
from vllm_omni.model_executor.models.nemotron_asr.encoder import (
    FastConformerEncoder,
)


def stream_step_paged(
    encoder: FastConformerEncoder,
    chunk_mel: torch.Tensor,
    *,
    kv_pages: torch.Tensor,
    block_tables: torch.Tensor,
    seq_lens: torch.Tensor,
    conv_pages: torch.Tensor,
    state_indices: torch.Tensor,
    window: int,
    drop_extra: int,
) -> torch.Tensor:
    """One paged streaming encoder step (batch of sessions).

    ``chunk_mel``: (B, feat, mel) as ``stream_step``. ``kv_pages``:
    (L, 2, num_blocks, block_frames, h, d_k) attention page pool;
    ``block_tables``/``seq_lens`` as in ``attention_pages``.
    ``conv_pages``: (L, num_pages, d_model, kernel-1) conv-tail pool;
    ``state_indices``: (B,) page index per session (ConvCachePage
    binding). Returns the chunk's encoder frames (B, F, d) and
    advances pages in place.
    """
    lengths = torch.full(
        (chunk_mel.shape[0],), chunk_mel.shape[2], device=chunk_mel.device
    )
    x, _ = encoder.pre_encode(chunk_mel, lengths)
    if drop_extra:
        x = x[:, drop_extra:]
    pos_emb = encoder.pos_enc(
        torch.zeros(
            1, x.shape[1] + window, x.shape[2],
            device=x.device, dtype=x.dtype,
        )
    )
    for idx, layer in enumerate(encoder.layers):
        residual = x
        y = layer.norm_feed_forward1(x)
        residual = residual + 0.5 * layer.feed_forward1(y)
        y = layer.norm_self_att(residual)
        attn_out = paged_stream_attention(
            layer.self_attn, y,
            kv_pages=kv_pages[idx],
            block_tables=block_tables,
            seq_lens=seq_lens,
            window=window,
            pos_emb=pos_emb,
        )
        residual = residual + attn_out
        y = layer.norm_conv(residual)
        conv_out = _paged_conv(
            layer, y, pages=conv_pages[idx], state_indices=state_indices
        )
        residual = residual + conv_out
        y = layer.norm_feed_forward2(residual)
        residual = residual + 0.5 * layer.feed_forward2(y)
        x = layer.norm_out(residual)
    return x


def _paged_conv(
    layer: torch.nn.Module,
    x: torch.Tensor,
    *,
    pages: torch.Tensor,
    state_indices: torch.Tensor,
) -> torch.Tensor:
    """``_stream_conv`` with the tail read from / written to pages.

    ``pages``: (num_pages, d_model, kernel-1); rows are gathered by
    ``state_indices`` before the causal conv and the advanced tails are
    scattered back to the same rows afterwards.
    """
    conv = layer.conv
    y = x.transpose(1, 2)
    y = torch.nn.functional.glu(conv.pointwise_conv1(y), dim=1)
    tail = pages[state_indices]
    # conv_state axis: read-cast to compute dtype; the scatter below
    # casts back to the page dtype implicitly.
    padded = torch.cat([tail.to(y.dtype), y], dim=-1)
    pages[state_indices] = padded[:, :, -tail.shape[-1] :]
    y = conv.depthwise_conv(padded)
    y = conv.batch_norm(y.transpose(1, 2)).transpose(1, 2)
    y = torch.nn.functional.silu(y)
    return conv.pointwise_conv2(y).transpose(1, 2)
