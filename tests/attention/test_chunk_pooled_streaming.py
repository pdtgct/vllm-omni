# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""α1 tests-first: chunk-pooled sliding-window streaming attention.

Specs: PORT-STATE-001 (left-context attention as paged sliding-window
KV, projected K/V), PORT-PREC-001 (dtypes via PrecisionPolicy);
consult dispositions D-α1a..e + OPEN-α1 (b) (ledger, 2026-07-13).
Shape follows tests/attention/test_fish_kvcache_attn.py: CPU tier on
fakes/duck configs; pod tier gpu-marked. The math-tier suite
(test_attention_pages.py) is the promoted reference oracle.
"""

from types import SimpleNamespace

import pytest
import torch
from vllm.v1.kv_cache_interface import FullAttentionSpec, SlidingWindowSpec

from vllm_omni.attention.chunk_pooled_streaming import (
    CHECKPOINT_WINDOW_FRAMES,
    ChunkPooledStreamingAttention,
    ChunkPooledStreamingBackend,
    ChunkPooledStreamingImpl,
    expected_alignment,
    pool_common_metadata,
    pooled_num_kv_heads,
    pooled_window,
)
from vllm_omni.model_executor.models.nemotron_asr.attention_pages import (
    paged_stream_attention,
    write_chunk_kv,
)
from vllm_omni.model_executor.models.nemotron_asr.encoder import (
    RelPositionMHA,
)
from vllm_omni.model_executor.models.nemotron_asr.precision import (
    FP32_BRINGUP,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

HEADS = 8
HEAD_SIZE = 128
#: The five chunk configs' encoder frames per chunk (bps).
BPS = (1, 2, 4, 7, 14)
#: The conv reference page (α2's state bundle): (1024, 8) fp32.
CONV_PAGE_BYTES = 1024 * 8 * 4
#: Per-frame attention bytes for this geometry (K+V, fp32).
FRAME_BYTES = 2 * HEADS * HEAD_SIZE * 4


def duck_cfg(block_size: int = 16) -> SimpleNamespace:
    return SimpleNamespace(
        cache_config=SimpleNamespace(block_size=block_size),
    )


def make_layer(bps: int) -> ChunkPooledStreamingAttention:
    return ChunkPooledStreamingAttention(
        num_heads=HEADS,
        head_size=HEAD_SIZE,
        frames_per_chunk=bps,
        policy=FP32_BRINGUP,
        prefix=f"encoder.layers.0.attn.bps{bps}",
    )


def tiny_mha(h: int = 2, d_k: int = 4) -> RelPositionMHA:
    torch.manual_seed(0)
    return RelPositionMHA(d_model=h * d_k, n_heads=h)


# ---- contract greens: the decided formulas -----------------------------------


def test_pooled_formulas_match_the_consult_table():
    # cdiv(56, bps) + 1 pooled tokens; heads x bps.
    assert [pooled_window(56, bps) for bps in BPS] == [57, 29, 15, 9, 5]
    assert [pooled_num_kv_heads(HEADS, bps) for bps in BPS] == [
        8,
        16,
        32,
        56,
        112,
    ]


def test_preferred_block_size_is_one_chunk():
    # Truthful preference (D-α1c); core's hook may still raise it for
    # small chunk configs — geometry stays core's decision.
    assert ChunkPooledStreamingBackend.get_preferred_block_size() == 1


# ---- spec emission (D-α1b: pooled units at the pin) ---------------------------


@pytest.mark.parametrize("bps", BPS)
def test_spec_is_pooled_sliding_window(bps):
    layer = make_layer(bps)
    spec = layer.get_kv_cache_spec(duck_cfg())
    assert isinstance(spec, SlidingWindowSpec)
    assert spec.num_kv_heads == pooled_num_kv_heads(HEADS, bps)
    assert spec.sliding_window == pooled_window(
        CHECKPOINT_WINDOW_FRAMES, bps
    )
    assert spec.dtype == torch.float32  # PrecisionPolicy attention cache
    assert spec.block_size == duck_cfg().cache_config.block_size


@pytest.mark.parametrize("bps", BPS)
def test_spec_page_matches_hook_reference(bps):
    # THE load-bearing regression (D-α1c): with pooled denomination,
    # the α2 alignment hook's per-token reference (FullAttentionSpec
    # at block 1, config geometry) equals our spec's per-token page —
    # pre-equalization holds by construction.
    layer = make_layer(bps)
    spec = layer.get_kv_cache_spec(duck_cfg(block_size=1))
    reference = FullAttentionSpec(
        block_size=1,
        num_kv_heads=pooled_num_kv_heads(HEADS, bps),
        head_size=HEAD_SIZE,
        dtype=torch.float32,
    )
    assert spec.page_size_bytes == reference.page_size_bytes
    assert spec.page_size_bytes == bps * FRAME_BYTES


def test_alignment_table_matches_consult_geometry():
    # The arithmetic contract the registry-driven α4 integration test
    # proves end-to-end: block size per chunk config, padded page ==
    # actual attention page (never the conv page short).
    expected = {
        1: (4, 32768),
        2: (2, 32768),
        4: (1, 32768),
        7: (1, 57344),
        14: (1, 114688),
    }
    for bps, want in expected.items():
        got = expected_alignment(
            bps,
            per_token_frame_bytes=FRAME_BYTES,
            reference_state_page_bytes=CONV_PAGE_BYTES,
        )
        assert got == want, bps
        block, padded = got
        assert padded >= CONV_PAGE_BYTES
        assert padded == block * bps * FRAME_BYTES


# ---- metadata pooling (the mirrored builder arithmetic) -----------------------


def synthetic_metadata() -> SimpleNamespace:
    return SimpleNamespace(
        query_start_loc=torch.tensor([0, 1, 3]),
        seq_lens=torch.tensor([5, 2]),
        num_actual_tokens=3,
        max_query_len=2,
        max_seq_len=5,
        slot_mapping=torch.tensor([40, 41, -1]),
    )


def test_builder_expands_metadata_by_frames_per_chunk():
    pooled = pool_common_metadata(synthetic_metadata(), 7)
    assert pooled.query_start_loc.tolist() == [0, 7, 21]
    assert pooled.seq_lens.tolist() == [35, 14]
    assert pooled.num_actual_tokens == 21
    assert pooled.max_query_len == 14
    assert pooled.max_seq_len == 35
    # slot*bps + arange(bps); padding slots (-1) preserved.
    assert pooled.slot_mapping[:7].tolist() == list(range(280, 287))
    assert pooled.slot_mapping[7:14].tolist() == list(range(287, 294))
    assert pooled.slot_mapping[14:].tolist() == [-1] * 7


def test_builder_leaves_the_original_untouched():
    original = synthetic_metadata()
    pool_common_metadata(original, 7)
    assert original.num_actual_tokens == 3
    assert original.slot_mapping.tolist() == [40, 41, -1]


def test_guard_raises_on_non_streaming_metadata():
    # Loud guard, never a silent fallback (the new-model posture).
    with pytest.raises(ValueError):
        pool_common_metadata(SimpleNamespace(), 7)
    with pytest.raises(ValueError):
        pool_common_metadata(synthetic_metadata(), 0)


# ---- the two-unit contract -----------------------------------------------------


def test_impl_masks_frames_spec_counts_pooled():
    bps = 7
    impl = ChunkPooledStreamingImpl(
        num_heads=HEADS, head_size=HEAD_SIZE, frames_per_chunk=bps
    )
    layer = make_layer(bps)
    spec = layer.get_kv_cache_spec(duck_cfg())
    assert impl.window_frames == CHECKPOINT_WINDOW_FRAMES  # frames
    assert spec.sliding_window == pooled_window(  # pooled tokens
        CHECKPOINT_WINDOW_FRAMES, bps
    )


# ---- impl vs the math-tier oracle ------------------------------------------------


def pages_and_tables(
    *, batch: int, blocks: int, block_frames: int, h: int, d_k: int
):
    kv_pages = torch.zeros(2, blocks, block_frames, h, d_k)
    block_tables = torch.arange(batch * (blocks // batch)).view(batch, -1)
    return kv_pages, block_tables


def test_impl_write_matches_math_tier_and_handles_mid_block():
    h, d_k, frames = 2, 4, 4
    attn = tiny_mha(h, d_k)
    x = torch.randn(1, frames, h * d_k)
    # Boundary-aligned: equivalent to the math tier's write_chunk_kv.
    ref_pages, tables = pages_and_tables(
        batch=1, blocks=4, block_frames=frames, h=h, d_k=d_k
    )
    write_chunk_kv(
        attn,
        x,
        kv_pages=ref_pages,
        block_tables=tables,
        seq_lens=torch.tensor([0]),
    )
    impl = ChunkPooledStreamingImpl(
        num_heads=h, head_size=d_k, frames_per_chunk=frames
    )
    impl_pages = torch.zeros_like(ref_pages)
    slots = torch.arange(frames)  # block 0, rows 0..3
    impl.write_kv(attn, x, kv_pages=impl_pages, slot_mapping=slots)
    torch.testing.assert_close(impl_pages, ref_pages)
    # Mid-block: the generalization the math tier's boundary assert
    # deliberately lacks — write rows 2..5 across a block boundary.
    impl_pages2 = torch.zeros_like(ref_pages)
    impl.write_kv(
        attn, x, kv_pages=impl_pages2, slot_mapping=torch.arange(2, 6)
    )
    assert impl_pages2[:, 0, 2:].abs().sum() > 0
    assert impl_pages2[:, 1, :2].abs().sum() > 0
    assert impl_pages2[:, 0, :2].abs().sum() == 0


def reference_step(attn, x, *, pages, tables, seq_lens, window, pos_emb):
    return paged_stream_attention(
        attn,
        x,
        kv_pages=pages,
        block_tables=tables,
        seq_lens=seq_lens,
        window=window,
        pos_emb=pos_emb,
    )


def test_backend_step_matches_paged_reference_ragged_batch():
    # Promotes the math-tier parity tests as the oracle: a ragged
    # 2-session batch, one mid-stream and one fresh, permuted order —
    # the block-table indirection makes recomposition layout-free.
    h, d_k, frames, window = 2, 4, 4, 8
    attn = tiny_mha(h, d_k)
    pos_emb = torch.randn(1, 2 * (window + frames) - 1, h * d_k)
    impl = ChunkPooledStreamingImpl(
        num_heads=h,
        head_size=d_k,
        frames_per_chunk=frames,
        window_frames=window,
    )
    pages_ref, tables = pages_and_tables(
        batch=2, blocks=8, block_frames=frames, h=h, d_k=d_k
    )
    pages_impl = pages_ref.clone()
    seq_lens = torch.tensor([8, 0])  # ragged: mid-stream vs fresh
    x = torch.randn(2, frames, h * d_k)
    want = reference_step(
        attn,
        x,
        pages=pages_ref,
        tables=tables,
        seq_lens=seq_lens,
        window=window,
        pos_emb=pos_emb,
    )
    got = impl.forward(
        attn,
        x,
        kv_pages=pages_impl,
        block_tables=tables,
        seq_lens=seq_lens,
        pos_emb=pos_emb,
    )
    torch.testing.assert_close(got, want)


def test_first_chunk_is_ordinary():
    # Q4: seq_lens=0 is just a step — no 8L+1 knowledge anywhere in
    # backend inputs (the asymmetry lives in mel windowing upstream).
    h, d_k, frames, window = 2, 4, 4, 8
    attn = tiny_mha(h, d_k)
    pos_emb = torch.randn(1, 2 * (window + frames) - 1, h * d_k)
    impl = ChunkPooledStreamingImpl(
        num_heads=h,
        head_size=d_k,
        frames_per_chunk=frames,
        window_frames=window,
    )
    pages, tables = pages_and_tables(
        batch=1, blocks=4, block_frames=frames, h=h, d_k=d_k
    )
    x = torch.randn(1, frames, h * d_k)
    want = reference_step(
        attn,
        x,
        pages=pages.clone(),
        tables=tables,
        seq_lens=torch.tensor([0]),
        window=window,
        pos_emb=pos_emb,
    )
    got = impl.forward(
        attn,
        x,
        kv_pages=pages,
        block_tables=tables,
        seq_lens=torch.tensor([0]),
        pos_emb=pos_emb,
    )
    torch.testing.assert_close(got, want)


def test_out_of_window_blocks_unreferenced():
    # Eviction safety, promoted from the math tier: poison every pool
    # block older than the pooled window; outputs must not move.
    h, d_k, frames, window = 2, 4, 2, 4
    attn = tiny_mha(h, d_k)
    pos_emb = torch.randn(1, 2 * (window + frames) - 1, h * d_k)
    impl = ChunkPooledStreamingImpl(
        num_heads=h,
        head_size=d_k,
        frames_per_chunk=frames,
        window_frames=window,
    )
    pages, tables = pages_and_tables(
        batch=1, blocks=8, block_frames=frames, h=h, d_k=d_k
    )
    seq_lens = torch.tensor([10])  # 5 chunks in; window keeps 2 + new
    x = torch.randn(1, frames, h * d_k)
    baseline = impl.forward(
        attn,
        x,
        kv_pages=pages.clone(),
        block_tables=tables,
        seq_lens=seq_lens,
        pos_emb=pos_emb,
    )
    poisoned = pages.clone()
    poisoned[:, :2] = float("nan")  # chunks 0-1: out of window
    again = impl.forward(
        attn,
        x,
        kv_pages=poisoned,
        block_tables=tables,
        seq_lens=seq_lens,
        pos_emb=pos_emb,
    )
    torch.testing.assert_close(again, baseline)


# ---- OPEN-α1 (b): terminal short chunk — dead rows masked ------------------------


def test_terminal_short_chunk_masks_dead_rows():
    # Decided (b), Pete 2026-07-13: never silence-pad; the impl masks
    # dead rows so live outputs are bit-identical to a reference run
    # on the truncated chunk, and dead rows never contaminate them.
    h, d_k, frames, window = 2, 4, 4, 8
    live = 3  # terminal chunk carries 3 of 4 frames
    attn = tiny_mha(h, d_k)
    # pos_emb is sized for the LIVE computation (window + live): the
    # serving layer computes pos_emb from actual sizes each step, and
    # the impl contract requires the live-sized tensor when
    # live_frames is set.
    pos_emb_live = torch.randn(1, 2 * (window + live) - 1, h * d_k)
    impl = ChunkPooledStreamingImpl(
        num_heads=h,
        head_size=d_k,
        frames_per_chunk=frames,
        window_frames=window,
    )
    pages, tables = pages_and_tables(
        batch=1, blocks=4, block_frames=frames, h=h, d_k=d_k
    )
    x = torch.randn(1, frames, h * d_k)
    want_live = reference_step(
        attn,
        x[:, :live],
        pages=pages.clone(),
        tables=tables,
        seq_lens=torch.tensor([4]),
        window=window,
        pos_emb=pos_emb_live,
    )
    got = impl.forward(
        attn,
        x,
        kv_pages=pages,
        block_tables=tables,
        seq_lens=torch.tensor([4]),
        pos_emb=pos_emb_live,
        live_frames=torch.tensor([live]),
    )
    torch.testing.assert_close(got[:, :live], want_live)
    assert got[:, live:].abs().sum() == 0  # dead rows are zero
    assert not torch.isnan(got).any()


# ---- pod tier (gpu-marked; the α1 code slice + pod round activate these) ---------


@pytest.mark.gpu
def test_native_vs_goldens_all_chunk_sizes():
    # CI-blocking parity oracle: the backend path vs the pinned
    # NeMo-simulator goldens, all five chunk configs (EVAL doctrine).
    raise NotImplementedError("activates with the α1 code slice pod round")


@pytest.mark.gpu
def test_workspace_reuse_across_steps():
    # No allocation growth across steps (the fish workspace lane).
    raise NotImplementedError("activates with the α1 code slice pod round")


@pytest.mark.gpu
def test_cudagraph_capture_rejected_loudly():
    # α1 claims no capture support: the fish-style NEGATIVE pin —
    # capture is rejected loudly, never wrong-answers silently.
    raise NotImplementedError("activates with the α1 code slice pod round")
