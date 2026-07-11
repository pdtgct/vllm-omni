# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Paged left-context attention tests (alpha1, GPU-free tier).

Native-vs-reference parity of ``paged_stream_attention`` against the
golden-proven ``_stream_attention``, plus the page-machinery contracts
the engine relies on: out-of-window blocks are dead weight (freeing
them cannot change output), block-table indirection is layout-free,
and garbage in unallocated pool blocks — including NaN — never leaks
through the masked softmax (zero-at-admission analog,
PORT-STATE-003).
"""

import pytest
import torch

from vllm_omni.model_executor.models.nemotron_asr.attention_pages import (
    paged_stream_attention,
    write_chunk_kv,
)
from vllm_omni.model_executor.models.nemotron_asr.encoder import (
    FastConformerEncoder,
    _stream_attention,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

D_MODEL = 32


def _tiny(att_context=(8, 1)) -> FastConformerEncoder:
    torch.manual_seed(31)
    enc = FastConformerEncoder(
        feat_in=16,
        d_model=D_MODEL,
        d_ff=64,
        n_layers=2,
        n_heads=4,
        conv_kernel=5,
        subsampling_channels=16,
        att_context=att_context,
    )
    enc.eval()
    return enc


def _drive(
    *,
    batch: int,
    chunks: int,
    frames: int,
    window: int,
    start_offsets: list[int] | None = None,
    block_ids: list[list[int]] | None = None,
    num_blocks: int | None = None,
    pool_fill: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run reference and paged paths over one stream.

    Returns stacked (chunks, B, F, d) outputs of each side plus the
    final ``kv_pages`` and ``block_tables`` for follow-on assertions.
    """
    enc = _tiny()
    layer = enc.layers[0]
    attn = layer.self_attn
    if start_offsets is None:
        start_offsets = [0] * batch
    if num_blocks is None:
        num_blocks = batch * chunks + 1
    if block_ids is None:
        block_ids = [
            [b * chunks + c for c in range(chunks)] for b in range(batch)
        ]
    torch.manual_seed(97)
    stream = torch.randn(chunks, batch, frames, D_MODEL)
    pos_emb = enc.pos_enc(torch.zeros(1, frames + window, D_MODEL))

    kv_pages = torch.zeros(2, num_blocks, frames, attn.h, attn.d_k)
    if pool_fill is not None:
        kv_pages.fill_(pool_fill)
    block_tables = torch.tensor(block_ids, dtype=torch.long)
    seq_lens = torch.zeros(batch, dtype=torch.long)

    cache_ref = torch.zeros(batch, window, D_MODEL)
    valid = torch.zeros(batch, dtype=torch.long)
    outs_ref, outs_paged = [], []
    with torch.inference_mode():
        for step in range(chunks):
            active = torch.tensor([step >= off for off in start_offsets])
            x = torch.where(active.view(batch, 1, 1), stream[step], 0.0)
            out_ref, cache_ref = _stream_attention(
                layer, x, cache=cache_ref, valid=valid, pos_emb=pos_emb
            )
            out_paged = paged_stream_attention(
                attn, x,
                kv_pages=kv_pages,
                block_tables=block_tables,
                seq_lens=seq_lens,
                window=window,
                pos_emb=pos_emb,
            )
            outs_ref.append(out_ref)
            outs_paged.append(out_paged)
            valid = torch.clamp(
                valid + active.long() * frames, max=window
            )
            seq_lens = seq_lens + active.long() * frames
    return (
        torch.stack(outs_ref),
        torch.stack(outs_paged),
        kv_pages,
        block_tables,
    )


def test_paged_matches_reference_single_stream():
    out_ref, out_paged, _, _ = _drive(
        batch=1, chunks=12, frames=4, window=8
    )
    torch.testing.assert_close(out_paged, out_ref, rtol=0.0, atol=1e-6)


def test_paged_matches_reference_ragged_batch():
    out_ref, out_paged, _, _ = _drive(
        batch=3, chunks=10, frames=4, window=8,
        start_offsets=[0, 3, 7],
    )
    torch.testing.assert_close(out_paged, out_ref, rtol=0.0, atol=1e-6)


def test_out_of_window_blocks_are_free():
    """Poisoning every block older than the window must not change the
    next chunk's output — the engine may free those blocks at will.
    """
    enc = _tiny()
    layer = enc.layers[0]
    attn = layer.self_attn
    frames, window, chunks = 4, 8, 10
    torch.manual_seed(11)
    stream = torch.randn(chunks + 1, 1, frames, D_MODEL)
    pos_emb = enc.pos_enc(torch.zeros(1, frames + window, D_MODEL))
    kv_pages = torch.zeros(2, chunks + 1, frames, attn.h, attn.d_k)
    block_tables = torch.arange(chunks + 1, dtype=torch.long).unsqueeze(0)
    seq_lens = torch.zeros(1, dtype=torch.long)
    with torch.inference_mode():
        for step in range(chunks):
            paged_stream_attention(
                attn, stream[step],
                kv_pages=kv_pages, block_tables=block_tables,
                seq_lens=seq_lens, window=window, pos_emb=pos_emb,
            )
            seq_lens = seq_lens + frames
        clean = paged_stream_attention(
            attn, stream[chunks],
            kv_pages=kv_pages.clone(), block_tables=block_tables,
            seq_lens=seq_lens.clone(), window=window, pos_emb=pos_emb,
        )
        in_window_blocks = window // frames + 1  # window + current
        kv_pages[:, : chunks - in_window_blocks + 1] = 1e9
        poisoned = paged_stream_attention(
            attn, stream[chunks],
            kv_pages=kv_pages, block_tables=block_tables,
            seq_lens=seq_lens, window=window, pos_emb=pos_emb,
        )
    torch.testing.assert_close(poisoned, clean, rtol=0.0, atol=0.0)


def test_block_table_indirection_is_layout_free():
    """Scrambled, interleaved block ids give identical results —
    sessions own logical chunks, not physical page ranges.
    """
    out_a, out_b = (
        _drive(
            batch=2, chunks=6, frames=4, window=8,
            block_ids=ids, num_blocks=16,
        )[1]
        for ids in (
            [[0, 1, 2, 3, 4, 5], [6, 7, 8, 9, 10, 11]],
            [[11, 3, 7, 0, 9, 5], [2, 10, 4, 8, 1, 6]],
        )
    )
    torch.testing.assert_close(out_b, out_a, rtol=0.0, atol=0.0)


def test_unallocated_pool_garbage_cannot_leak():
    """NaN in every never-written pool slot: outputs stay finite and
    equal to the reference — dead under-fill rows are zeroed at
    gather, so freed-session residue can never poison a new session.
    """
    out_ref, out_paged, _, _ = _drive(
        batch=1, chunks=3, frames=4, window=8,
        num_blocks=32, pool_fill=float("nan"),
    )
    assert torch.isfinite(out_paged).all()
    torch.testing.assert_close(out_paged, out_ref, rtol=0.0, atol=1e-6)


def test_write_requires_block_boundary():
    enc = _tiny()
    attn = enc.layers[0].self_attn
    with pytest.raises(ValueError, match="block boundary"):
        write_chunk_kv(
            attn, torch.zeros(1, 4, D_MODEL),
            kv_pages=torch.zeros(2, 4, 4, attn.h, attn.d_k),
            block_tables=torch.zeros(1, 4, dtype=torch.long),
            seq_lens=torch.tensor([2]),
        )
