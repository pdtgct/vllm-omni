# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Paged left-context attention (alpha1: gather-then-exact-math).

The engine-native form of ``_stream_attention``: the per-session
left-context window lives in projected-K/V pages (the vLLM KV-page
convention; equivalence to the proven pre-projection cache is pinned by
``test_projected_kv_probe``), one chunk per block (the whisper_causal
block-pooling geometry: block = ``frames_per_chunk`` KV rows, window =
``window // frames_per_chunk`` blocks, one scheduler token per chunk).

This module is deliberately backend-free: it owns the page read/write
and the exact Transformer-XL math over tiny shapes (queries <= F,
keys <= window + F), so parity is provable GPU-free; the attention-
backend subclass that feeds it engine metadata (slot mapping / block
tables, whisper_causal.py pattern) wires in at the engine tier.
FlexAttention ``score_mod`` is a later perf swap, not this layer.
"""

import torch

from vllm_omni.model_executor.models.nemotron_asr.encoder import (
    _LOG_BASE,
    RelPositionMHA,
)


def write_chunk_kv(
    attn: RelPositionMHA,
    x: torch.Tensor,
    *,
    kv_pages: torch.Tensor,
    block_tables: torch.Tensor,
    seq_lens: torch.Tensor,
) -> None:
    """Project new frames once and scatter K/V into their chunk block.

    ``x``: (B, F, d) normed attention inputs for the new chunk.
    ``kv_pages``: (2, num_blocks, block_frames, h, d_k) page pool.
    ``block_tables``: (B, max_blocks) session block ids, chunk-indexed.
    ``seq_lens``: (B,) frames already paged per session — must sit on a
    block boundary (one chunk per block; a short terminal chunk may
    close a stream but never precedes another write).
    """
    batch, new_frames, _ = x.shape
    block_frames = kv_pages.shape[2]
    if bool((seq_lens % block_frames != 0).any()):
        raise ValueError("chunk write must start on a block boundary")
    k_new = attn.linear_k(x).view(batch, new_frames, attn.h, attn.d_k)
    v_new = attn.linear_v(x).view(batch, new_frames, attn.h, attn.d_k)
    chunk_idx = seq_lens // block_frames
    for b in range(batch):
        block = block_tables[b, chunk_idx[b]]
        kv_pages[0, block, :new_frames] = k_new[b]
        kv_pages[1, block, :new_frames] = v_new[b]


def gather_window_kv(
    *,
    kv_pages: torch.Tensor,
    block_tables: torch.Tensor,
    seq_lens: torch.Tensor,
    new_frames: int,
    window: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Materialize each session's ``[window | new]`` K/V, right-aligned.

    Returns ``(k, v)`` of shape (B, window + new_frames, h, d_k) laid
    out exactly like the reference cache: the trailing ``new_frames``
    rows are the current chunk, the ``window`` region is right-aligned
    with dead (under-fill) rows zeroed — garbage in unallocated pool
    blocks can never leak a NaN through the masked softmax.
    """
    _, _, block_frames, h, d_k = kv_pages.shape
    batch = block_tables.shape[0]
    dev = kv_pages.device
    out = torch.zeros(
        2, batch, window + new_frames, h, d_k,
        device=dev, dtype=kv_pages.dtype,
    )
    for b in range(batch):
        total = int(seq_lens[b]) + new_frames
        live = min(int(seq_lens[b]), window)
        start = total - live - new_frames  # absolute first live frame
        for row in range(live + new_frames):
            f = start + row
            page = block_tables[b, f // block_frames]
            out[:, b, window + new_frames - (live + new_frames) + row] = (
                kv_pages[:, page, f % block_frames]
            )
    return out[0], out[1]


def paged_stream_attention(
    attn: RelPositionMHA,
    x: torch.Tensor,
    *,
    kv_pages: torch.Tensor,
    block_tables: torch.Tensor,
    seq_lens: torch.Tensor,
    window: int,
    pos_emb: torch.Tensor,
) -> torch.Tensor:
    """One paged streaming attention step (write, gather, exact math).

    Numerically the projected-K/V form of ``_stream_attention``:
    identical scores/mask/softmax over ``[window | new]`` with dead
    under-fill rows masked; the pages replace the in-module cache.
    Returns (B, F, d) outputs for the new frames.
    """
    batch, new_frames, _ = x.shape
    write_chunk_kv(
        attn, x,
        kv_pages=kv_pages,
        block_tables=block_tables,
        seq_lens=seq_lens,
    )
    k, v = gather_window_kv(
        kv_pages=kv_pages,
        block_tables=block_tables,
        seq_lens=seq_lens,
        new_frames=new_frames,
        window=window,
    )
    k = k.transpose(1, 2)
    v = v.transpose(1, 2)
    t2 = window + new_frames

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
    valid = torch.clamp(seq_lens, max=window)
    row = torch.arange(window, device=x.device).unsqueeze(0)
    dead = row < (window - valid.unsqueeze(1))
    mask = torch.zeros(
        batch, 1, new_frames, t2, dtype=torch.bool, device=x.device
    )
    mask[:, :, :, :window] = dead.unsqueeze(1).unsqueeze(2)
    scores = scores.masked_fill(mask, -_LOG_BASE)
    weights = torch.softmax(scores, dim=-1).masked_fill(mask, 0.0)
    out = torch.matmul(weights, v)
    out = out.transpose(1, 2).reshape(batch, new_frames, attn.h * attn.d_k)
    return attn.linear_out(out)
