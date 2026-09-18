# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Packed hybrid K/V cache stepper for streaming attention.

Always projects new frames with the injected ``canonical_project`` and
advances shadow hidden/K/V with the same per-row gather. Attention K/V
are either those shadow rows plus the new projection, or a fresh
``linear_k`` / ``linear_v`` on the contiguous full hidden sequence.

``x`` is FP32 ``(B, F, D)`` and ``cache`` is FP32 ``(B, C, 3D)`` packed
``[hidden | K | V]``. ``new_lengths`` is the existing validated int64
``(B,)`` logical-length vector. The injected canonical projector returns
FP32 ``(B, F, D)`` and is assumed batch/row invariant. The static Python
``use_projected_history`` selects the experimental fast geometry; initial dead
history stays masked by the caller's normal attention mask.

Return path
-----------
``attention_k, attention_v, next_cache`` with static shapes
``(B, C+F, D)``, ``(B, C+F, D)``, ``(B, C, 3*D)``.

* Always: ``new_k/new_v = canonical_project(x, linear_*.weight/bias)``;
  ``next_cache`` gathers ``cat(cache, packed_new)`` at ``j + new_lengths``
  on all three planes. ``new_lengths == 0`` is an identity gather on
  that row (bitwise preserve).
* If ``use_projected_history``: attention is
  ``cat(shadow_k, new_k)`` / ``cat(shadow_v, new_v)``.
* Else: attention is ``linear_k/linear_v(cat(hidden_hist, x).contiguous())``
  while shadow still advances as above.

Graph: ``cat`` / ``gather`` / ``arange`` / ``reshape`` /
module forward / injected project only. No config selector.
"""

from __future__ import annotations

from collections.abc import Callable

import torch

CanonicalProject = Callable[[torch.Tensor, torch.Tensor, torch.Tensor | None], torch.Tensor]


def hybrid_kv(
    x: torch.Tensor,
    cache: torch.Tensor,
    new_lengths: torch.Tensor,
    canonical_project: CanonicalProject,
    linear_k: torch.nn.Linear,
    linear_v: torch.nn.Linear,
    use_projected_history: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    _check_static(x, cache, new_lengths, linear_k, linear_v)
    if not isinstance(use_projected_history, bool):
        raise ValueError("use_projected_history must be a static boolean")

    b, f, d = x.shape
    c = cache.shape[1]
    hidden_hist = cache[:, :, :d]
    shadow_k = cache[:, :, d : 2 * d]
    shadow_v = cache[:, :, 2 * d : 3 * d]

    new_k = canonical_project(x, linear_k.weight, linear_k.bias)
    new_v = canonical_project(x, linear_v.weight, linear_v.bias)
    if new_k.shape != x.shape or new_v.shape != x.shape:
        raise ValueError(
            f"canonical_project must return (B, F, D) matching x; got {tuple(new_k.shape)} and {tuple(new_v.shape)}"
        )
    if new_k.dtype != torch.float32 or new_v.dtype != torch.float32:
        raise ValueError("canonical_project outputs must be FP32")
    if new_k.device != x.device or new_v.device != x.device:
        raise ValueError("canonical_project outputs must match x.device")

    if use_projected_history:
        attn_k = torch.cat((shadow_k, new_k), dim=1)
        attn_v = torch.cat((shadow_v, new_v), dim=1)
    else:
        full_hidden = torch.cat((hidden_hist, x), dim=1).contiguous()
        attn_k = linear_k(full_hidden)
        attn_v = linear_v(full_hidden)

    packed_new = torch.cat((x, new_k, new_v), dim=-1)
    next_cache = _advance_packed(cache, packed_new, new_lengths, b, c)
    return attn_k, attn_v, next_cache


def _advance_packed(
    cache: torch.Tensor,
    packed_new: torch.Tensor,
    new_lengths: torch.Tensor,
    b: int,
    c: int,
) -> torch.Tensor:
    # next[b, j] = cat(cache, packed_new)[b, j + new_lengths[b]]
    # new_lengths == 0 => identity gather (all three planes).
    lengths = new_lengths
    ext = torch.cat((cache, packed_new), dim=1)
    j = torch.arange(c, device=cache.device, dtype=torch.long)
    idx = j.unsqueeze(0) + lengths.unsqueeze(1)
    planes = ext.shape[-1]
    return ext.gather(1, idx.unsqueeze(-1).expand(b, c, planes))


def _check_static(
    x: torch.Tensor,
    cache: torch.Tensor,
    new_lengths: torch.Tensor,
    linear_k: torch.nn.Linear,
    linear_v: torch.nn.Linear,
) -> None:
    if x.ndim != 3:
        raise ValueError(f"x must be (B, F, D), got {tuple(x.shape)}")
    if cache.ndim != 3:
        raise ValueError(f"cache must be (B, C, 3D), got {tuple(cache.shape)}")
    b, _f, d = x.shape
    if cache.shape[0] != b or cache.shape[-1] != 3 * d:
        raise ValueError(f"cache must be ({b}, C, {3 * d}), got {tuple(cache.shape)}")
    if new_lengths.shape != (b,) or new_lengths.dtype != torch.int64:
        raise ValueError(f"new_lengths must be int64 ({b},), got {new_lengths.dtype} {tuple(new_lengths.shape)}")
    if x.dtype != torch.float32 or cache.dtype != torch.float32:
        raise ValueError("x and cache must be FP32")
    if x.device != cache.device or new_lengths.device != x.device:
        raise ValueError("x, cache, and new_lengths must share a device")
    for name, mod in (("linear_k", linear_k), ("linear_v", linear_v)):
        weight = getattr(mod, "weight", None)
        if weight is None or weight.ndim != 2:
            raise ValueError(f"{name}.weight must be 2-D")
        if weight.shape != (d, d):
            raise ValueError(f"{name}.weight must be ({d}, {d}), got {tuple(weight.shape)}")
        if weight.dtype != torch.float32:
            raise ValueError(f"{name}.weight must be FP32")
        if weight.device != x.device:
            raise ValueError(f"{name}.weight must match x.device")
        bias = getattr(mod, "bias", None)
        if bias is not None:
            if bias.shape != (d,):
                raise ValueError(f"{name}.bias must be ({d},), got {tuple(bias.shape)}")
            if bias.dtype != torch.float32:
                raise ValueError(f"{name}.bias must be FP32")
            if bias.device != x.device:
                raise ValueError(f"{name}.bias must match x.device")
