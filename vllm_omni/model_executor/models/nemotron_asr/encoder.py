# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Cache-aware FastConformer encoder — P2 full-context regime.

Op-for-op translation of the pinned NeMo modules for the shipped
checkpoint's configuration (HF config.json, fetched 2026-07-11:
d_model 1024, 8 heads, FF 4096, 24 layers, conv kernel 9, causal
dw-striding 8x subsampling at 256 channels, attention/convolution bias
FALSE, activation silu, scale_input FALSE):

- dw-striding causal subsampling (subsampling.py:184-257,421-478)
- Transformer-XL relative-position MHA with per-layer pos biases
  (multi_head_attention.py:212-354; mask polarity True = masked)
- causal depthwise conv module (conformer_modules.py:236-340,
  causal_convs.py:73-151; left pad 8, right 0)
- macaron conformer layer (conformer_modules.py:160-230)
- centered rel-pos encoding (multi_head_attention.py:1056-1100)

Streaming (per-chunk cache threading over spec pages) is the P3
re-plumb; this regime runs one window over the whole utterance with
cross-chunk state dormant (PORT-REGIME-001/002). All source refs
@ NeMo de242add.
"""

import math

import torch
from torch import nn

from vllm_omni.model_executor.models.nemotron_asr.masks import (
    chunked_limited_mask,
)

_LOG_BASE = 10000.0


def _conv_out_len(length: int, *, pad: int, kernel: int, stride: int) -> int:
    return (length + pad - kernel) // stride + 1


class CausalConv2dSub(nn.Conv2d):
    """CausalConv2D: (k-1, s-1) asymmetric zero pad on BOTH axes."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, padding=0, **kwargs)
        k = self.kernel_size[0]
        s = self.stride[0]
        self._pad = (k - 1, s - 1, k - 1, s - 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return super().forward(nn.functional.pad(x, self._pad))


class SubsamplingDwStriding(nn.Module):
    """Causal dw-striding ConvSubsampling, 8x (three stride-2 stages)."""

    def __init__(
        self,
        *,
        feat_in: int,
        d_model: int,
        conv_channels: int,
        kernel: int = 3,
        stride: int = 2,
        stages: int = 3,
    ) -> None:
        super().__init__()
        self.kernel = kernel
        self.stride = stride
        self.stages = stages
        layers: list[nn.Module] = [
            CausalConv2dSub(1, conv_channels, kernel, stride),
            nn.ReLU(),
        ]
        for _ in range(stages - 1):
            layers.append(
                CausalConv2dSub(
                    conv_channels,
                    conv_channels,
                    kernel,
                    stride,
                    groups=conv_channels,
                )
            )
            layers.append(nn.Conv2d(conv_channels, conv_channels, 1))
            layers.append(nn.ReLU())
        self.conv = nn.Sequential(*layers)
        freq = feat_in
        pad = (kernel - 1) + (stride - 1)
        for _ in range(stages):
            freq = _conv_out_len(freq, pad=pad, kernel=kernel, stride=stride)
        self.out = nn.Linear(conv_channels * freq, d_model)

    def output_lengths(self, lengths: torch.Tensor) -> torch.Tensor:
        pad = (self.kernel - 1) + (self.stride - 1)
        out = lengths
        for _ in range(self.stages):
            out = torch.div(
                out + pad - self.kernel, self.stride, rounding_mode="floor"
            ) + 1
        return out.to(torch.int64)

    def forward(
        self, mel: torch.Tensor, lengths: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """``(B, feat_in, T)`` mel -> ``(B, T//8, d_model)``."""
        # Activations follow the policy-cast weights from this seam on;
        # the fp32 mel front-end sits upstream of it (PORT-PREC-001).
        mel = mel.to(next(self.parameters()).dtype)
        x = mel.transpose(1, 2).unsqueeze(1)  # (B, 1, T, F)
        x = self.conv(x)
        b, c, t, f = x.size()
        x = self.out(x.transpose(1, 2).reshape(b, t, c * f))
        return x, self.output_lengths(lengths)


class RelPositionalEncoding(nn.Module):
    """Centered TXL relative positions L-1 .. -(L-1); no input scale."""

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.d_model = d_model
        self.register_buffer("pe", torch.zeros(1, 0, d_model), persistent=False)

    def _extend(self, length: int, ref: torch.Tensor) -> None:
        if self.pe.size(1) >= 2 * length - 1:
            return
        positions = torch.arange(
            length - 1, -length, -1, dtype=torch.float32, device=ref.device
        ).unsqueeze(1)
        pe = torch.zeros(positions.size(0), self.d_model, device=ref.device)
        div_term = torch.exp(
            torch.arange(
                0, self.d_model, 2, dtype=torch.float32, device=ref.device
            )
            * -(math.log(_LOG_BASE) / self.d_model)
        )
        pe[:, 0::2] = torch.sin(positions * div_term)
        pe[:, 1::2] = torch.cos(positions * div_term)
        self.pe = pe.unsqueeze(0).to(ref.dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return pos_emb ``(1, 2T-1, d)`` for input ``(B, T, d)``."""
        self._extend(x.size(1), x)
        length = x.size(1)
        center = self.pe.size(1) // 2 + 1
        return self.pe[:, center - length : center + length - 1]


class RelPositionMHA(nn.Module):
    """Transformer-XL rel-pos attention (per-layer pos biases)."""

    def __init__(self, *, d_model: int, n_heads: int) -> None:
        super().__init__()
        self.h = n_heads
        self.d_k = d_model // n_heads
        self.s_d_k = math.sqrt(self.d_k)
        self.linear_q = nn.Linear(d_model, d_model, bias=False)
        self.linear_k = nn.Linear(d_model, d_model, bias=False)
        self.linear_v = nn.Linear(d_model, d_model, bias=False)
        self.linear_out = nn.Linear(d_model, d_model, bias=False)
        self.linear_pos = nn.Linear(d_model, d_model, bias=False)
        self.pos_bias_u = nn.Parameter(torch.zeros(n_heads, self.d_k))
        self.pos_bias_v = nn.Parameter(torch.zeros(n_heads, self.d_k))

    @staticmethod
    def _rel_shift(x: torch.Tensor) -> torch.Tensor:
        b, h, qlen, pos_len = x.size()
        x = nn.functional.pad(x, pad=(1, 0))
        x = x.view(b, h, -1, qlen)
        return x[:, :, 1:].view(b, h, qlen, pos_len)

    def forward(
        self,
        x: torch.Tensor,
        *,
        pos_emb: torch.Tensor,
        masked: torch.Tensor,
    ) -> torch.Tensor:
        """Self-attention; ``masked`` is (B, T, T) True = MAY NOT attend."""
        b, t, _ = x.shape
        q = self.linear_q(x).view(b, t, self.h, self.d_k)
        k = self.linear_k(x).view(b, t, self.h, self.d_k).transpose(1, 2)
        v = self.linear_v(x).view(b, t, self.h, self.d_k).transpose(1, 2)
        p = self.linear_pos(pos_emb).view(
            pos_emb.size(0), -1, self.h, self.d_k
        ).transpose(1, 2)

        q_u = (q + self.pos_bias_u).transpose(1, 2)
        q_v = (q + self.pos_bias_v).transpose(1, 2)
        matrix_bd = self._rel_shift(torch.matmul(q_v, p.transpose(-2, -1)))
        matrix_ac = torch.matmul(q_u, k.transpose(-2, -1))
        scores = (
            matrix_ac + matrix_bd[:, :, :, : matrix_ac.size(-1)]
        ) / self.s_d_k
        mask = masked.unsqueeze(1)
        scores = scores.masked_fill(mask, -_LOG_BASE)
        attn = torch.softmax(scores, dim=-1).masked_fill(mask, 0.0)
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).reshape(b, t, self.h * self.d_k)
        return self.linear_out(out)


class ConformerConv(nn.Module):
    """Pointwise->GLU->causal depthwise->norm->silu->pointwise."""

    def __init__(
        self, *, d_model: int, kernel: int, norm_type: str
    ) -> None:
        super().__init__()
        self.norm_type = norm_type
        self.left_pad = kernel - 1
        self.pointwise_conv1 = nn.Conv1d(
            d_model, d_model * 2, 1, bias=False
        )
        self.depthwise_conv = nn.Conv1d(
            d_model, d_model, kernel, groups=d_model, bias=False
        )
        if norm_type == "layer_norm":
            self.batch_norm: nn.Module = nn.LayerNorm(d_model)
        elif norm_type == "batch_norm":
            self.batch_norm = nn.BatchNorm1d(d_model)
        else:
            raise ValueError(f"unsupported conv norm: {norm_type}")
        self.pointwise_conv2 = nn.Conv1d(d_model, d_model, 1, bias=False)

    def forward(
        self, x: torch.Tensor, pad_zero: torch.Tensor | None
    ) -> torch.Tensor:
        """``(B, T, d)`` -> ``(B, T, d)``; ``pad_zero`` True = padding."""
        x = x.transpose(1, 2)
        x = nn.functional.glu(self.pointwise_conv1(x), dim=1)
        if pad_zero is not None:
            x = x.masked_fill(pad_zero.unsqueeze(1), 0.0)
        x = self.depthwise_conv(
            nn.functional.pad(x, (self.left_pad, 0))
        )
        if self.norm_type == "layer_norm":
            x = self.batch_norm(x.transpose(1, 2)).transpose(1, 2)
        else:
            x = self.batch_norm(x)
        x = nn.functional.silu(x)
        return self.pointwise_conv2(x).transpose(1, 2)


class FeedForward(nn.Module):
    """Linear -> silu -> Linear (dropout is eval-noop, omitted).

    Bias-free: the checkpoint's encoder ``use_bias: False`` covers every
    Linear/Conv in the layer (confirmed from the restored model config).
    """

    def __init__(self, *, d_model: int, d_ff: int) -> None:
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff, bias=False)
        self.linear2 = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(nn.functional.silu(self.linear1(x)))


class ConformerLayer(nn.Module):
    """Macaron block: 0.5FF -> MHA -> conv -> 0.5FF -> norm_out."""

    def __init__(
        self,
        *,
        d_model: int,
        d_ff: int,
        n_heads: int,
        conv_kernel: int,
        conv_norm_type: str,
    ) -> None:
        super().__init__()
        self.norm_feed_forward1 = nn.LayerNorm(d_model)
        self.feed_forward1 = FeedForward(d_model=d_model, d_ff=d_ff)
        self.norm_self_att = nn.LayerNorm(d_model)
        self.self_attn = RelPositionMHA(d_model=d_model, n_heads=n_heads)
        self.norm_conv = nn.LayerNorm(d_model)
        self.conv = ConformerConv(
            d_model=d_model, kernel=conv_kernel, norm_type=conv_norm_type
        )
        self.norm_feed_forward2 = nn.LayerNorm(d_model)
        self.feed_forward2 = FeedForward(d_model=d_model, d_ff=d_ff)
        self.norm_out = nn.LayerNorm(d_model)

    def forward(
        self,
        x: torch.Tensor,
        *,
        pos_emb: torch.Tensor,
        masked: torch.Tensor,
        pad_zero: torch.Tensor | None,
    ) -> torch.Tensor:
        x = x + 0.5 * self.feed_forward1(self.norm_feed_forward1(x))
        x = x + self.self_attn(
            self.norm_self_att(x), pos_emb=pos_emb, masked=masked
        )
        x = x + self.conv(self.norm_conv(x), pad_zero)
        x = x + 0.5 * self.feed_forward2(self.norm_feed_forward2(x))
        return self.norm_out(x)


class FastConformerEncoder(nn.Module):
    """Full-context regime encoder (PORT-REGIME-001/002).

    One attention window over the whole utterance at the configured
    ``att_context_size``; cross-chunk state pages dormant. The P3
    streaming path re-plumbs the same layers over paged caches.
    """

    def __init__(
        self,
        *,
        feat_in: int = 128,
        d_model: int = 1024,
        d_ff: int = 4096,
        n_layers: int = 24,
        n_heads: int = 8,
        conv_kernel: int = 9,
        conv_norm_type: str = "layer_norm",
        subsampling_channels: int = 256,
        att_context: tuple[int, int] = (56, 13),
    ) -> None:
        super().__init__()
        self.att_context = att_context
        self.pre_encode = SubsamplingDwStriding(
            feat_in=feat_in,
            d_model=d_model,
            conv_channels=subsampling_channels,
        )
        self.pos_enc = RelPositionalEncoding(d_model)
        self.layers = nn.ModuleList(
            ConformerLayer(
                d_model=d_model,
                d_ff=d_ff,
                n_heads=n_heads,
                conv_kernel=conv_kernel,
                conv_norm_type=conv_norm_type,
            )
            for _ in range(n_layers)
        )

    def forward(
        self, mel: torch.Tensor, lengths: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """``(B, feat_in, T_mel)`` -> ``(B, T, d_model)`` encoder_raw.

        Output is (B, T, D) — NeMo transposes to (B, D, T) at its
        encoder boundary; the port keeps time-major throughout (layout,
        not math; the capture-hook comparison transposes accordingly).
        """
        x, out_lens = self.pre_encode(mel, lengths)
        pos_emb = self.pos_enc(x)
        t = x.size(1)
        may_attend = chunked_limited_mask(
            t, self.att_context, device=x.device
        )
        valid = torch.arange(t, device=x.device).unsqueeze(
            0
        ) < out_lens.unsqueeze(1)
        pair_valid = valid.unsqueeze(2) & valid.unsqueeze(1)
        masked = ~(may_attend.unsqueeze(0) & pair_valid)
        pad_zero = ~valid
        for layer in self.layers:
            x = layer(x, pos_emb=pos_emb, masked=masked, pad_zero=pad_zero)
        return x, out_lens


class StreamingCaches:
    """Per-session encoder caches (the spec pages' in-module form).

    ``channel``: per-layer normed attention inputs, ``(L, B, 56, d)``
    (NeMo ``cache_last_channel``). ``time``: per-layer conv tails,
    ``(L, B, d, kernel-1)`` (``cache_last_time``). ``valid``: filled
    cache rows per batch element. The 56-slot capacity IS the attention
    window (8 chunks x 7 frames at every published config), so the
    streaming mask reduces to 'all valid cached rows + full intra-chunk'.
    """

    def __init__(
        self,
        *,
        n_layers: int,
        batch: int,
        d_model: int,
        left_context: int,
        conv_kernel: int,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        self.left_context = left_context
        self.channel = torch.zeros(
            n_layers, batch, left_context, d_model, device=device, dtype=dtype
        )
        self.time = torch.zeros(
            n_layers, batch, d_model, conv_kernel - 1, device=device,
            dtype=dtype,
        )
        self.valid = torch.zeros(batch, dtype=torch.long, device=device)


def _stream_attention(
    layer: ConformerLayer,
    x: torch.Tensor,
    *,
    cache: torch.Tensor,
    valid: torch.Tensor,
    pos_emb: torch.Tensor,
    new_valid: torch.Tensor,
    new_lengths: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """One layer's attention over [cache | new] keys (NeMo update_cache).

    ``x`` is the normed attention input for the NEW frames (B, F, d);
    ``cache`` holds the previous normed inputs (B, C, d). ``new_valid``
    is the (B, F) per-row frame-validity mask and ``new_lengths`` the
    (B,) logical frame counts: padded new frames are masked as keys AND
    queries (a fully masked query row softmaxes uniform then zeroes,
    so its output is exactly 0 — never NaN), and the cache advances by
    each row's LOGICAL length via a per-row gather over [cache | new].
    Returns the attention output for the new frames and the advanced
    cache.
    """
    attn = layer.self_attn
    batch, new_frames, _ = x.shape
    capacity = cache.shape[1]
    # The cache keeps its own policy axis (attention_cache); compute
    # runs in the activations dtype, so read-cast here, write-cast on
    # advance (PORT-PREC-001/005 — state dtype never follows compute).
    keys = torch.cat([cache.to(x.dtype), x], dim=1)  # (B, C+F, d)

    b, t2 = batch, keys.shape[1]
    q = attn.linear_q(x).view(b, new_frames, attn.h, attn.d_k)
    k = attn.linear_k(keys).view(b, t2, attn.h, attn.d_k).transpose(1, 2)
    v = attn.linear_v(keys).view(b, t2, attn.h, attn.d_k).transpose(1, 2)
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
    # Mask: cache rows beyond each element's valid count are dead;
    # padded new frames are dead keys; padded queries mask fully.
    row = torch.arange(capacity, device=x.device).unsqueeze(0)
    dead = row < (capacity - valid.unsqueeze(1))  # (B, C) True = dead
    mask = torch.zeros(
        batch, 1, new_frames, t2, dtype=torch.bool, device=x.device
    )
    mask[:, :, :, :capacity] = dead.unsqueeze(1).unsqueeze(2)
    mask[:, :, :, capacity:] = (~new_valid).unsqueeze(1).unsqueeze(2)
    mask = mask | (~new_valid).unsqueeze(1).unsqueeze(-1)
    scores = scores.masked_fill(mask, -_LOG_BASE)
    weights = torch.softmax(scores, dim=-1).masked_fill(mask, 0.0)
    out = torch.matmul(weights, v)
    out = out.transpose(1, 2).reshape(batch, new_frames, attn.h * attn.d_k)
    # Advance cache by each row's logical length: slot j of the new
    # cache is [cache | x][j + F_b] — F_b = 0 leaves the row's cache
    # bit-identical; gather indices never touch padded frames.
    aidx = (
        new_lengths.view(-1, 1)
        + torch.arange(capacity, device=x.device).unsqueeze(0)
    ).unsqueeze(-1).expand(b, capacity, keys.shape[2])
    new_cache = (
        torch.cat([cache, x.to(cache.dtype)], dim=1).gather(1, aidx)
    )
    return attn.linear_out(out), new_cache


def _stream_conv(
    layer: ConformerLayer,
    x: torch.Tensor,
    cache: torch.Tensor,
    *,
    new_lengths: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Conv module over [time_cache | new] (CausalConv1D.update_cache).

    The causal depthwise conv makes every valid output independent of
    padded columns (and the norm is per-position), so only the cache
    tail needs length awareness: slot j of the new tail gathers
    [cache | glu][j + F_b], each row's own logical append.
    """
    conv = layer.conv
    y = x.transpose(1, 2)
    y = torch.nn.functional.glu(conv.pointwise_conv1(y), dim=1)
    # conv_state axis: read-cast to compute dtype, write-cast back.
    padded = torch.cat([cache.to(y.dtype), y], dim=-1)
    cw = cache.shape[-1]
    aidx = (
        new_lengths.view(-1, 1)
        + torch.arange(cw, device=y.device).unsqueeze(0)
    ).unsqueeze(1).expand(y.shape[0], y.shape[1], cw)
    new_cache = padded.gather(2, aidx).to(cache.dtype)
    y = conv.depthwise_conv(padded)
    y = conv.batch_norm(y.transpose(1, 2)).transpose(1, 2)
    y = torch.nn.functional.silu(y)
    return conv.pointwise_conv2(y).transpose(1, 2), new_cache


def stream_step(
    encoder: FastConformerEncoder,
    chunk_mel: torch.Tensor,
    caches: StreamingCaches,
    *,
    drop_extra: int | None = None,
    out_offsets: torch.Tensor | None = None,
    out_lengths: torch.Tensor | None = None,
    out_width: int | None = None,
) -> torch.Tensor:
    """One cached streaming encoder step (batch of sessions).

    ``chunk_mel``: (B, feat, mel [+9-mel pre-encode context for
    non-first chunks]). Advances the caches in place — numerically the
    cached form of the prefix computation the P3 probes proved
    golden-exact.

    Length-aware form (PORT-ADV-004): ``out_offsets`` (B,) is each
    row's pre-encode drop, ``out_lengths`` (B,) its logical encoder
    frame count, and ``out_width`` the fixed host-derived output
    width. Rows are realigned per-row after subsampling (the causal
    conv makes valid outputs independent of trailing padding),
    attention masks padded keys AND queries, caches append by logical
    lengths, ``window_valid`` accumulates logical lengths, and output
    frames at or past a row's length are exactly zero. A zero-length
    row leaves its caches bit-identical. No tensor value determines a
    shape; the call issues no host/device synchronization.

    ``drop_extra`` is the uniform legacy adapter (``run_forward_step``
    and the P3/P4 probes): equivalent to offsets = ``drop_extra``,
    lengths = full width, over the same single algorithm.
    """
    b = chunk_mel.shape[0]
    device = chunk_mel.device
    lengths = torch.full((b,), chunk_mel.shape[2], device=device)
    x, _ = encoder.pre_encode(chunk_mel, lengths)
    if drop_extra is not None:
        if out_offsets is not None or out_lengths is not None:
            raise ValueError(
                "pass either drop_extra (uniform legacy) or the "
                "per-row out_offsets/out_lengths/out_width form"
            )
        out_width = x.shape[1] - drop_extra
        out_offsets = torch.full(
            (b,), drop_extra, dtype=torch.long, device=device
        )
        out_lengths = torch.full(
            (b,), out_width, dtype=torch.long, device=device
        )
    assert (
        out_offsets is not None
        and out_lengths is not None
        and out_width is not None
    )
    # Per-row realignment: row b's encoder frames start at its own
    # pre-encode drop. Clamp keeps padded columns in-bounds; their
    # values are masked everywhere below.
    gidx = (
        out_offsets.view(-1, 1)
        + torch.arange(out_width, device=device).unsqueeze(0)
    ).clamp(max=max(x.shape[1] - 1, 0))
    x = x.gather(1, gidx.unsqueeze(-1).expand(b, out_width, x.shape[2]))
    new_valid = (
        torch.arange(out_width, device=device).unsqueeze(0)
        < out_lengths.view(-1, 1)
    )
    cache_len = caches.channel.shape[2]
    pos_emb = encoder.pos_enc(
        torch.zeros(
            1, out_width + cache_len, x.shape[2],
            device=device, dtype=x.dtype,
        )
    )
    for idx, layer in enumerate(encoder.layers):
        residual = x
        y = layer.norm_feed_forward1(x)
        residual = residual + 0.5 * layer.feed_forward1(y)
        y = layer.norm_self_att(residual)
        attn_out, caches.channel[idx] = _stream_attention(
            layer, y,
            cache=caches.channel[idx],
            valid=caches.valid,
            pos_emb=pos_emb,
            new_valid=new_valid,
            new_lengths=out_lengths,
        )
        residual = residual + attn_out
        y = layer.norm_conv(residual)
        conv_out, caches.time[idx] = _stream_conv(
            layer, y, caches.time[idx], new_lengths=out_lengths
        )
        residual = residual + conv_out
        y = layer.norm_feed_forward2(residual)
        residual = residual + 0.5 * layer.feed_forward2(y)
        x = layer.norm_out(residual)
    caches.valid = torch.clamp(
        caches.valid + out_lengths.to(caches.valid.dtype),
        max=caches.left_context,
    )
    # Padded-output zeroing (PORT-ADV-004): frames at or past each
    # row's logical length are exactly zero.
    return torch.where(new_valid.unsqueeze(-1), x, x.new_zeros(()))
