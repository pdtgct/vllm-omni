# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Chunk-pooled sliding-window streaming attention (α1 engine tier).

The capability, named for what it is (the nemotron-asr streaming
encoder is its first user): each scheduler token is one audio chunk
carrying ``frames_per_chunk`` (bps) KV rows, the per-layer left
context is a sliding window over those rows, and the window lives in
paged projected-K/V blocks. MIRRORS the whisper_causal block-pooling
pattern rather than subclassing it (consult D-α1a: the core class's
``attn_backend`` injection parameter is dead and its backend whitelist
cannot express this model's Transformer-XL relative-position math) —
what is reused for real is the base ``Attention`` spec machinery, the
×bps metadata-expansion arithmetic, and the pooled-``num_kv_heads``
spec channel. The exact math underneath is the proven
``attention_pages`` gather-then-exact-math tier; FlexAttention
``score_mod`` is a later perf swap.

Two-unit contract (D-α1b): the KV-cache spec speaks POOLED scheduler
tokens (``pooled_window`` — the v0.25 semantics, emitted here at the
v0.24 pin so the bump is citation-only), while the impl masks in
FRAMES (the checkpoint's 56-frame window). Pooled denomination
(D-α1c): every per-token quantity the engine accounts is denominated
in chunks — the config presents ``pooled_num_kv_heads`` and this
backend prefers block size 1 (truthful: one chunk per block is the
kernel's eviction granularity; core's alignment hook still raises the
block size for small chunk configs so the attention page covers the
conv page — geometry stays core's decision).

Terminal short chunk (OPEN-α1 decided (b), Pete 2026-07-13): a final
chunk with fewer than bps frames occupies one scheduler token; the
impl masks the dead query/KV rows — audio is never silence-padded, so
the engine path stays bit-identical to the golden-proven math tier.
"""

import copy
from typing import Any

import torch

from vllm_omni.model_executor.models.nemotron_asr.attention_pages import (
    paged_stream_attention,
)
from vllm_omni.model_executor.models.nemotron_asr.precision import (
    PrecisionPolicy,
)

#: The checkpoint's per-layer left-context window, in encoder frames.
CHECKPOINT_WINDOW_FRAMES = 56


def pooled_window(window_frames: int, frames_per_chunk: int) -> int:
    """The spec's sliding window in POOLED scheduler tokens.

    ``cdiv(window, bps) + 1`` — the +1 is the eviction margin for the
    block currently being written (the v0.25 whisper_causal formula,
    emitted at the pin per D-α1b).
    """
    return -(-window_frames // frames_per_chunk) + 1


def pooled_num_kv_heads(num_kv_heads: int, frames_per_chunk: int) -> int:
    """The spec's KV-head count under pooling.

    ``heads × bps`` — the fiction that makes per-token page bytes
    account for the bps frames each scheduler token carries (the core
    class's own spec channel, and D-α1c's denomination applied at the
    config surface the alignment hook reads).
    """
    return num_kv_heads * frames_per_chunk


def expected_alignment(
    frames_per_chunk: int,
    *,
    per_token_frame_bytes: int,
    reference_state_page_bytes: int,
) -> tuple[int, int]:
    """The alignment the α2 hook converges to for this geometry.

    Returns ``(attn_block_size, mamba_page_size_padded)`` — the
    consult's table: pooled per-token bytes = frames-per-chunk ×
    per-frame bytes; block size = the smallest count of pooled tokens
    whose page covers the reference state page; padded = that
    attention page. Pins the arithmetic contract the registry-driven
    integration test proves end-to-end at the α4 rung.
    """
    if frames_per_chunk < 1:
        raise ValueError(
            f"frames_per_chunk must be >= 1, got {frames_per_chunk}"
        )
    per_token = frames_per_chunk * per_token_frame_bytes
    block_size = max(1, -(-reference_state_page_bytes // per_token))
    return block_size, block_size * per_token


def pool_common_metadata(common: Any, frames_per_chunk: int) -> Any:
    """Expand scheduler-token metadata to frame rows (×bps).

    The whisper_causal builder arithmetic, mirrored: deep-copy the
    metadata; multiply ``query_start_loc``, ``seq_lens``,
    ``num_actual_tokens``, ``max_query_len``, ``max_seq_len`` by bps;
    expand ``slot_mapping`` as ``slot*bps + arange(bps)`` with padding
    slots (−1) preserved. Uniform for every chunk, first included —
    the 8L+1 first-chunk asymmetry never reaches this layer (consult
    Q4). Malformed metadata raises ``ValueError`` — a loud guard,
    never a silent fallback (the new-model posture).
    """
    if frames_per_chunk < 1:
        raise ValueError(
            f"frames_per_chunk must be >= 1, got {frames_per_chunk}"
        )
    required = (
        "query_start_loc",
        "seq_lens",
        "num_actual_tokens",
        "max_query_len",
        "max_seq_len",
        "slot_mapping",
    )
    missing = [name for name in required if not hasattr(common, name)]
    if missing:
        raise ValueError(
            f"metadata is not streaming-shaped; missing {missing}"
        )
    pooled = copy.deepcopy(common)
    bps = frames_per_chunk
    pooled.query_start_loc = common.query_start_loc * bps
    pooled.seq_lens = common.seq_lens * bps
    pooled.num_actual_tokens = common.num_actual_tokens * bps
    pooled.max_query_len = common.max_query_len * bps
    pooled.max_seq_len = common.max_seq_len * bps
    slots = common.slot_mapping
    expanded = slots.unsqueeze(-1) * bps + torch.arange(
        bps, device=slots.device, dtype=slots.dtype
    )
    # Padding slots (-1) stay -1, never a live-looking index.
    pooled.slot_mapping = expanded.clamp(min=-1).reshape(-1)
    return pooled


class _PooledMetadataBuilder:
    """The thin builder face over :func:`pool_common_metadata`."""

    def __init__(self, frames_per_chunk: int) -> None:
        if frames_per_chunk < 1:
            raise ValueError(
                f"frames_per_chunk must be >= 1, got {frames_per_chunk}"
            )
        self.frames_per_chunk = frames_per_chunk

    def build(self, common: Any) -> Any:
        return pool_common_metadata(common, self.frames_per_chunk)


class ChunkPooledStreamingBackend:
    """The backend face core sees (factory-built, D-α1a mirror)."""

    @staticmethod
    def get_name() -> str:
        return "CHUNK_POOLED_STREAMING"

    @classmethod
    def get_preferred_block_size(cls) -> int:
        """One chunk per block — this kernel's eviction granularity.

        Core's alignment hook may still raise the block size (small
        chunk configs, D-α1c); preference is truthful, geometry stays
        core's decision.
        """
        return 1

    @classmethod
    def get_builder_cls(cls) -> Any:
        """The metadata builder (wraps :func:`pool_common_metadata`)."""
        return _PooledMetadataBuilder


class ChunkPooledStreamingImpl:
    """The impl: page write + exact TXL math, frame-unit masking.

    ``forward_includes_kv_cache_update`` — projected K/V scatter from
    the pooled slot mapping (the math tier's ``write_chunk_kv``
    generalized to mid-block writes for bps < block frames), then the
    ``paged_stream_attention`` math over ``[window | new]``. Masks in
    FRAMES; dead rows of a terminal short chunk are masked, never
    silence-padded (OPEN-α1 (b)).
    """

    def __init__(
        self,
        *,
        num_heads: int,
        head_size: int,
        frames_per_chunk: int,
        window_frames: int = CHECKPOINT_WINDOW_FRAMES,
    ) -> None:
        if num_heads < 1 or head_size < 1:
            raise ValueError("num_heads and head_size must be >= 1")
        if frames_per_chunk < 1:
            raise ValueError(
                f"frames_per_chunk must be >= 1, got {frames_per_chunk}"
            )
        if window_frames < 1:
            raise ValueError(
                f"window_frames must be >= 1, got {window_frames}"
            )
        self.num_heads = num_heads
        self.head_size = head_size
        self.frames_per_chunk = frames_per_chunk
        self.window_frames = window_frames

    def write_kv(
        self,
        attn: Any,
        x: torch.Tensor,
        *,
        kv_pages: torch.Tensor,
        slot_mapping: torch.Tensor,
        live_frames: torch.Tensor | None = None,
    ) -> None:
        """Projected-K/V scatter from pooled slots (mid-block safe)."""
        batch, frames, _ = x.shape
        h = attn.h
        d_k = attn.d_k
        k = attn.linear_k(x).view(batch, frames, h, d_k)
        v = attn.linear_v(x).view(batch, frames, h, d_k)
        block_frames = kv_pages.shape[2]
        flat_k = k.reshape(batch * frames, h, d_k)
        flat_v = v.reshape(batch * frames, h, d_k)
        slots = slot_mapping.reshape(-1)
        for i in range(slots.numel()):
            slot = int(slots[i])
            if slot < 0:
                continue  # padding rows never write
            if live_frames is not None and (
                (i % frames) >= int(live_frames[i // frames])
            ):
                continue  # dead rows of a terminal short chunk
            page, row = divmod(slot, block_frames)
            kv_pages[0, page, row] = flat_k[i]
            kv_pages[1, page, row] = flat_v[i]

    def forward(
        self,
        attn: Any,
        x: torch.Tensor,
        *,
        kv_pages: torch.Tensor,
        block_tables: torch.Tensor,
        seq_lens: torch.Tensor,
        pos_emb: torch.Tensor,
        live_frames: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """One streaming step ≡ ``paged_stream_attention`` (ragged OK).

        Without ``live_frames`` this IS the math tier — one call, no
        re-implementation. With ``live_frames`` (a terminal short
        chunk, OPEN-α1 (b)): the live rows compute exactly as a
        truncated chunk would — never silence-padded — and the dead
        output rows are zero; ``pos_emb`` must then be sized for
        ``window + live`` (the serving layer computes pos_emb from
        actual sizes each step). v1 masks uniformly per batch: a batch
        mixing different live counts raises rather than guessing.
        """
        if live_frames is None:
            return paged_stream_attention(
                attn,
                x,
                kv_pages=kv_pages,
                block_tables=block_tables,
                seq_lens=seq_lens,
                window=self.window_frames,
                pos_emb=pos_emb,
            )
        live_values = {int(n) for n in live_frames}
        if len(live_values) != 1:
            raise ValueError(
                "v1 terminal-short-chunk masking is uniform per batch; "
                f"got live counts {sorted(live_values)}"
            )
        live = live_values.pop()
        batch, frames, d_model = x.shape
        if not 0 < live <= frames:
            raise ValueError(
                f"live_frames must be in (0, {frames}], got {live}"
            )
        out_live = paged_stream_attention(
            attn,
            x[:, :live],
            kv_pages=kv_pages,
            block_tables=block_tables,
            seq_lens=seq_lens,
            window=self.window_frames,
            pos_emb=pos_emb,
        )
        out = torch.zeros(
            batch, frames, d_model, device=x.device, dtype=out_live.dtype
        )
        out[:, :live] = out_live
        return out


class ChunkPooledStreamingAttention:
    """The ``Attention``-subclass face (spec emission, D-α1b/c).

    Constructed with frame-unit geometry (window 56); emits a
    ``SlidingWindowSpec`` in pooled units with pooled KV heads, dtype
    from the ``PrecisionPolicy``'s attention-cache class. The code
    slice wires this onto vLLM's ``Attention`` base so the standard
    spec walk discovers it.
    """

    def __init__(
        self,
        *,
        num_heads: int,
        head_size: int,
        frames_per_chunk: int,
        window_frames: int = CHECKPOINT_WINDOW_FRAMES,
        policy: PrecisionPolicy,
        prefix: str,
    ) -> None:
        if frames_per_chunk < 1:
            raise ValueError(
                f"frames_per_chunk must be >= 1, got {frames_per_chunk}"
            )
        self.num_heads = num_heads
        self.head_size = head_size
        self.frames_per_chunk = frames_per_chunk
        self.window_frames = window_frames
        self._policy = policy
        self.prefix = prefix

    def get_kv_cache_spec(self, vllm_config: Any) -> Any:
        """``SlidingWindowSpec`` in pooled units (D-α1b)."""
        from vllm.v1.kv_cache_interface import SlidingWindowSpec

        return SlidingWindowSpec(
            block_size=vllm_config.cache_config.block_size,
            num_kv_heads=pooled_num_kv_heads(
                self.num_heads, self.frames_per_chunk
            ),
            head_size=self.head_size,
            dtype=self._policy.dtype_for("attention_cache"),
            sliding_window=pooled_window(
                self.window_frames, self.frames_per_chunk
            ),
        )
