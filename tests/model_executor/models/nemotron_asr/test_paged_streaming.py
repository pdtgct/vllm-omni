# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Paged stream_step equivalence tests (alpha2 seam swap, GPU-free).

``stream_step_paged`` must be ``stream_step`` with the state relocated:
same outputs, same conv-tail recursion, and — the properties the
engine actually buys — state that lives ONLY in the pages (a restored
page pool replays a step bit-for-bit) and follows ``state_indices``
under batch recomposition (PORT-STATE-001/003; CORNER-004's
cross-stream isolation at the math tier).
"""

import pytest
import torch

from vllm_omni.model_executor.models.nemotron_asr.encoder import (
    FastConformerEncoder,
    StreamingCaches,
    stream_step,
)
from vllm_omni.model_executor.models.nemotron_asr.paged_streaming import (
    stream_step_paged,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

D_MODEL = 32
KERNEL = 5
N_LAYERS = 2
WINDOW = 8
MEL_CHUNK = 32


def _tiny() -> FastConformerEncoder:
    torch.manual_seed(31)
    enc = FastConformerEncoder(
        feat_in=16,
        d_model=D_MODEL,
        d_ff=64,
        n_layers=N_LAYERS,
        n_heads=4,
        conv_kernel=KERNEL,
        subsampling_channels=16,
        att_context=(WINDOW, 1),
    )
    enc.eval()
    return enc


class _Paged:
    """Page pools + per-session tables for a test stream."""

    def __init__(
        self, enc: FastConformerEncoder, *, batch: int, chunks: int,
        frames: int,
    ) -> None:
        attn = enc.layers[0].self_attn
        self.kv_pages = torch.zeros(
            N_LAYERS, 2, batch * chunks + 1, frames, attn.h, attn.d_k
        )
        self.conv_pages = torch.zeros(
            N_LAYERS, batch + 2, D_MODEL, KERNEL - 1
        )
        self.block_tables = torch.stack(
            [
                torch.arange(chunks, dtype=torch.long) + b * chunks
                for b in range(batch)
            ]
        )
        self.state_indices = torch.arange(batch, dtype=torch.long)
        self.seq_lens = torch.zeros(batch, dtype=torch.long)

    def clone_pools(self) -> tuple[torch.Tensor, torch.Tensor]:
        return self.kv_pages.clone(), self.conv_pages.clone()


def _frames_per_chunk(enc: FastConformerEncoder) -> int:
    mel = torch.zeros(1, 16, MEL_CHUNK)
    x, _ = enc.pre_encode(mel, torch.tensor([MEL_CHUNK]))
    return x.shape[1]


def _stream(chunks: int, batch: int, seed: int = 7) -> torch.Tensor:
    torch.manual_seed(seed)
    return torch.randn(chunks, batch, 16, MEL_CHUNK)


def _run_reference(
    enc: FastConformerEncoder, mel_stream: torch.Tensor
) -> tuple[torch.Tensor, StreamingCaches]:
    chunks, batch = mel_stream.shape[:2]
    caches = StreamingCaches(
        n_layers=N_LAYERS, batch=batch, d_model=D_MODEL,
        left_context=WINDOW, conv_kernel=KERNEL,
        device=torch.device("cpu"),
    )
    outs = []
    with torch.inference_mode():
        for step in range(chunks):
            outs.append(
                stream_step(enc, mel_stream[step], caches, drop_extra=0)
            )
    return torch.stack(outs), caches


def _run_paged(
    enc: FastConformerEncoder, mel_stream: torch.Tensor, paged: _Paged,
    frames: int,
) -> torch.Tensor:
    chunks = mel_stream.shape[0]
    outs = []
    with torch.inference_mode():
        for step in range(chunks):
            outs.append(
                stream_step_paged(
                    enc, mel_stream[step],
                    kv_pages=paged.kv_pages,
                    block_tables=paged.block_tables,
                    seq_lens=paged.seq_lens,
                    conv_pages=paged.conv_pages,
                    state_indices=paged.state_indices,
                    window=WINDOW,
                    drop_extra=0,
                )
            )
            paged.seq_lens = paged.seq_lens + frames
    return torch.stack(outs)


def test_paged_step_matches_stream_step_single():
    enc = _tiny()
    frames = _frames_per_chunk(enc)
    mel = _stream(chunks=10, batch=1)
    out_ref, _ = _run_reference(enc, mel)
    paged = _Paged(enc, batch=1, chunks=10, frames=frames)
    out_paged = _run_paged(enc, mel, paged, frames)
    torch.testing.assert_close(out_paged, out_ref, rtol=0.0, atol=2e-6)


def test_paged_step_matches_stream_step_batch():
    enc = _tiny()
    frames = _frames_per_chunk(enc)
    mel = _stream(chunks=8, batch=3)
    out_ref, _ = _run_reference(enc, mel)
    paged = _Paged(enc, batch=3, chunks=8, frames=frames)
    out_paged = _run_paged(enc, mel, paged, frames)
    torch.testing.assert_close(out_paged, out_ref, rtol=0.0, atol=2e-6)


def test_conv_page_recursion_matches_reference():
    enc = _tiny()
    frames = _frames_per_chunk(enc)
    mel = _stream(chunks=6, batch=2)
    _, caches = _run_reference(enc, mel)
    paged = _Paged(enc, batch=2, chunks=6, frames=frames)
    _run_paged(enc, mel, paged, frames)
    for layer in range(N_LAYERS):
        torch.testing.assert_close(
            paged.conv_pages[layer, paged.state_indices],
            caches.time[layer],
            rtol=0.0,
            atol=2e-6,
        )


def test_state_lives_only_in_pages():
    """Restoring the page pools replays a step bit-for-bit — no hidden
    cross-chunk state survives outside the pages (the property that
    makes sessions parkable/evictable at the engine tier).
    """
    enc = _tiny()
    frames = _frames_per_chunk(enc)
    mel = _stream(chunks=5, batch=1)
    paged = _Paged(enc, batch=1, chunks=5, frames=frames)
    _run_paged(enc, mel[:4], paged, frames)
    kv_snap, conv_snap = paged.clone_pools()
    seq_snap = paged.seq_lens.clone()
    with torch.inference_mode():
        first = stream_step_paged(
            enc, mel[4],
            kv_pages=paged.kv_pages, block_tables=paged.block_tables,
            seq_lens=paged.seq_lens, conv_pages=paged.conv_pages,
            state_indices=paged.state_indices, window=WINDOW,
            drop_extra=0,
        )
        paged.kv_pages.copy_(kv_snap)
        paged.conv_pages.copy_(conv_snap)
        replay = stream_step_paged(
            enc, mel[4],
            kv_pages=paged.kv_pages, block_tables=paged.block_tables,
            seq_lens=seq_snap, conv_pages=paged.conv_pages,
            state_indices=paged.state_indices, window=WINDOW,
            drop_extra=0,
        )
    torch.testing.assert_close(replay, first, rtol=0.0, atol=0.0)


def test_batch_recomposition_follows_state_indices():
    """Swapping two sessions' batch rows (with their tables/indices)
    swaps the outputs — state binds to the session, never to the batch
    slot (cross-stream isolation, CORNER-004 analog). Tolerance, not
    bitwise: permuting batch composition changes kernel tiling/reduction
    order, which torch does not promise to be row-order invariant;
    bitwise replay under IDENTICAL composition is pinned by
    ``test_state_lives_only_in_pages``.
    """
    enc = _tiny()
    frames = _frames_per_chunk(enc)
    chunks = 6
    mel = _stream(chunks=chunks, batch=2)
    control = _Paged(enc, batch=2, chunks=chunks, frames=frames)
    out_control = _run_paged(enc, mel, control, frames)

    swapped = _Paged(enc, batch=2, chunks=chunks, frames=frames)
    swap_at = 3
    _run_paged(enc, mel[:swap_at], swapped, frames)
    perm = torch.tensor([1, 0])
    swapped.block_tables = swapped.block_tables[perm]
    swapped.state_indices = swapped.state_indices[perm]
    swapped.seq_lens = swapped.seq_lens[perm]
    out_tail = _run_paged(enc, mel[swap_at:, :, :][:, perm], swapped, frames)
    torch.testing.assert_close(
        out_tail, out_control[swap_at:][:, perm], rtol=0.0, atol=2e-6
    )
