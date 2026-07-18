# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Language-ID prompt conditioning (PORT-LID-001/002/004).

Stateless per chunk, post-encoder: the session's prompt index (resolved
once at admission from the checkpoint's ``prompt_dictionary``; ``auto``
is dictionary entry, not a detection algorithm) is scattered into a
one-hot vector, concatenated with every encoder frame, and projected
through ``prompt_kernel`` — op-for-op NeMo's
``_apply_prompt_to_encoded`` (mixins.py:971-1001 @ de242add) in
(B, T, D)-native layout. Only the integer index persists per session.
"""

import torch
from torch import nn


def resolve_prompt_index(
    prompt_dictionary: dict[str, int], target_lang: str
) -> int:
    """Resolve a locale to its prompt index at session admission.

    Raises:
        ValueError: For a locale absent from the dictionary, naming the
            available keys (PORT-LID-001).
    """
    if target_lang not in prompt_dictionary:
        available = sorted(prompt_dictionary)
        raise ValueError(
            f"unknown target_lang {target_lang!r}; available: {available}"
        )
    return prompt_dictionary[target_lang]


class PromptConditioner(nn.Module):
    """``prompt_kernel`` projection over one-hot-conditioned frames.

    Weight names mirror NeMo's ``prompt_kernel`` Sequential (``0`` and
    ``2`` Linear layers) so checkpoint weights map directly. Dense
    layers migrate to vLLM's ``quant_config`` linear primitives when the
    model class assembles (PORT-PREC-004); the math here is frozen by
    the reference test either way.
    """

    def __init__(self, *, enc_hidden: int, num_prompts: int) -> None:
        super().__init__()
        self.enc_hidden = enc_hidden
        self.num_prompts = num_prompts
        self.prompt_kernel = nn.Sequential(
            nn.Linear(enc_hidden + num_prompts, enc_hidden * 2),
            nn.ReLU(),
            nn.Linear(enc_hidden * 2, enc_hidden),
        )

    def forward(
        self, encoded: torch.Tensor, *, prompt_index: int | torch.Tensor
    ) -> torch.Tensor:
        """Condition encoder frames on the session's language prompt.

        Args:
            encoded: ``(batch, time, enc_hidden)`` encoder output
                (``encoder_raw`` in capture-hook terms).
            prompt_index: the admitted prompt index — a scalar for
                one session, or a ``(batch,)`` long tensor for
                ROW-WISE conditioning (one call, no per-prompt
                fragmentation; PORT-PERF-001).

        Returns:
            ``(batch, time, enc_hidden)`` conditioned frames
            (``encoder_conditioned``), in the input dtype.
        """
        batch, time, _ = encoded.shape
        prompt = torch.zeros(
            batch,
            time,
            self.num_prompts,
            dtype=encoded.dtype,
            device=encoded.device,
        )
        if isinstance(prompt_index, torch.Tensor):
            prompt[
                torch.arange(batch, device=encoded.device),
                :,
                prompt_index.to(encoded.device),
            ] = 1.0
        else:
            prompt[:, :, prompt_index] = 1.0
        out_dtype = encoded.dtype
        conditioned: torch.Tensor = self.prompt_kernel(
            torch.cat([encoded, prompt], dim=-1)
        ).to(out_dtype)
        return conditioned
