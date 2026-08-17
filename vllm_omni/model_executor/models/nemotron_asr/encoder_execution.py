# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Feature-gated execution for the static streaming encoder transition.

Both arms execute the same encoder, language-conditioning, and padded-row
zeroing function. The candidate changes only the execution mechanism: it
lets Inductor specialize one full graph for each exact host-derived tensor
shape. Compiler CUDA graphs stay disabled because session cache scratch is
mutable and its addresses are not a stable graph-owned input contract.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol

import torch

from vllm_omni.model_executor.models.nemotron_asr.encoder import stream_step

if TYPE_CHECKING:
    from vllm_omni.model_executor.models.nemotron_asr.nemotron_asr import (
        NemotronASRCore,
    )


class EncoderCaches(Protocol):
    """Structural cache surface consumed by :func:`stream_step`."""

    channel: Any
    time: Any
    valid: torch.Tensor
    left_context: int


EncoderTransition = Callable[
    [
        torch.Tensor,
        EncoderCaches,
        torch.Tensor,
        torch.Tensor,
        int,
        torch.Tensor,
    ],
    tuple[torch.Tensor, torch.Tensor],
]


@dataclass(frozen=True)
class ResolvedEncoderExecution:
    """One immutable startup selection and its bound transition."""

    arm: str
    transition: EncoderTransition


def execute_encoder_transition(
    core: NemotronASRCore,
    mel: torch.Tensor,
    caches: EncoderCaches,
    out_offsets: torch.Tensor,
    out_lengths: torch.Tensor,
    out_width: int,
    prompt_index: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run the exact cache-aware encoder and conditioning transition."""
    encoded = stream_step(
        core.encoder,
        mel,
        caches,
        out_offsets=out_offsets,
        out_lengths=out_lengths,
        out_width=out_width,
    )
    conditioned = core.lid(encoded, prompt_index=prompt_index)
    frame = torch.arange(out_width, device=mel.device).view(1, -1, 1)
    conditioned = torch.where(
        frame < out_lengths.view(-1, 1, 1),
        conditioned,
        conditioned.new_zeros(()),
    )
    return encoded, conditioned


def build_encoder_execution(
    core: NemotronASRCore,
    hf_config: Any,
) -> ResolvedEncoderExecution:
    """Resolve the analysis-only encoder execution arm at startup.

    ``eager`` is the compatibility baseline when the field is absent.
    ``compiled-static`` uses a full, static graph and deliberately has no
    eager fallback: a graph break or compilation failure invalidates that
    experimental arm. ``dynamic=False`` may cache multiple exact-shape
    specializations; the profiling warmup must cover the measured shapes.
    """
    arm = getattr(hf_config, "encoder_execution_arm", None) or "eager"
    if arm not in {"eager", "compiled-static"}:
        raise ValueError(f"unknown encoder_execution_arm {arm!r} (known: ['compiled-static', 'eager'])")

    def transition(
        mel: torch.Tensor,
        caches: EncoderCaches,
        out_offsets: torch.Tensor,
        out_lengths: torch.Tensor,
        out_width: int,
        prompt_index: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return execute_encoder_transition(
            core,
            mel,
            caches,
            out_offsets,
            out_lengths,
            out_width,
            prompt_index,
        )

    if arm == "eager":
        return ResolvedEncoderExecution(arm=arm, transition=transition)
    compiled = torch.compile(
        transition,
        fullgraph=True,
        dynamic=False,
        options={"triton.cudagraphs": False},
    )
    return ResolvedEncoderExecution(arm=arm, transition=compiled)


__all__ = [
    "EncoderCaches",
    "EncoderTransition",
    "ResolvedEncoderExecution",
    "build_encoder_execution",
    "execute_encoder_transition",
]
