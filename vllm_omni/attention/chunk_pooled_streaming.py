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

from typing import Any

import torch

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
    raise NotImplementedError


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
    raise NotImplementedError


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
        raise NotImplementedError


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
        raise NotImplementedError

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
        raise NotImplementedError

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

        ``live_frames`` marks a terminal short chunk's real rows; dead
        rows are masked out of scores and produce zero output rows
        that never contaminate live ones.
        """
        raise NotImplementedError


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
        raise NotImplementedError

    def get_kv_cache_spec(self, vllm_config: Any) -> Any:
        """``SlidingWindowSpec`` in pooled units (D-α1b)."""
        raise NotImplementedError
