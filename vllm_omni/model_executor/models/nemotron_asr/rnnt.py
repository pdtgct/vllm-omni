# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""RNN-T predictor, joint, and greedy label-looping chunk decode.

Semantics match NeMo's greedy label-looping computer (PORT-DEC-001):
``max_symbols_per_step = 10`` with hard-forced frame advance, blank
advances the frame cursor, blank-as-pad (SOS = blank, embedding pads at
the blank index), predictor state and last label carried across chunk
boundaries with no SOS re-injection. The joint follows NeMo's split
form (rnnt.py:1677-1724 @ de242add): per-side projections into the
joint hidden, broadcast sum, activation, final linear over ``V + 1``
(blank last). Greedy argmax needs no softmax at temperature 1.

This module is the single boundary the decode choice lives behind
(D-b chunk-loop + replay is the target; the loop here is the semantic
reference the fused/batched implementation is held to by tests).
"""

from dataclasses import dataclass

import torch
from torch import nn

from vllm_omni.model_executor.models.nemotron_asr.rnnt_cell import (
    ManualLSTM,
)

MAX_SYMBOLS_PER_STEP = 10


@dataclass
class DecodeState:
    """Per-session decode state carried across chunks (page-backed).

    ``h``/``c``: predictor LSTM state ``(layers, batch, hidden)``.
    ``last_label``: ``(batch,)`` int64; blank at session start (SOS).
    """

    h: torch.Tensor
    c: torch.Tensor
    last_label: torch.Tensor


class Predictor(nn.Module):
    """Embedding + LSTM prediction network (blank-as-pad)."""

    def __init__(
        self,
        *,
        vocab_size: int,
        pred_hidden: int,
        pred_rnn_layers: int,
    ) -> None:
        super().__init__()
        self.blank_id = vocab_size  # blank is the last index (V)
        self.embed = nn.Embedding(
            vocab_size + 1, pred_hidden, padding_idx=self.blank_id
        )
        self.rnn = ManualLSTM(
            input_size=pred_hidden,
            hidden_size=pred_hidden,
            num_layers=pred_rnn_layers,
        )

    def step(
        self,
        labels: torch.Tensor,
        state: tuple[torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """One prediction step from the previous label."""
        return self.rnn.step(self.embed(labels), state)


class Joint(nn.Module):
    """NeMo split joint: side projections, sum, activation, final."""

    def __init__(
        self,
        *,
        enc_hidden: int,
        pred_hidden: int,
        joint_hidden: int,
        vocab_size: int,
    ) -> None:
        super().__init__()
        self.enc = nn.Linear(enc_hidden, joint_hidden)
        self.pred = nn.Linear(pred_hidden, joint_hidden)
        self.joint_net = nn.Sequential(
            nn.ReLU(), nn.Linear(joint_hidden, vocab_size + 1)
        )

    def logits(
        self, enc_frame: torch.Tensor, pred_out: torch.Tensor
    ) -> torch.Tensor:
        """Joint logits for one (frame, prediction) pair, ``(B, V+1)``."""
        return self.joint_net(self.enc(enc_frame) + self.pred(pred_out))


def greedy_decode_chunk(
    enc_frames: torch.Tensor,
    predictor: Predictor,
    joint: Joint,
    state: DecodeState,
    *,
    max_symbols: int = MAX_SYMBOLS_PER_STEP,
) -> tuple[list[int], DecodeState]:
    """Greedy label-looping decode over one chunk's encoder frames.

    Reference (single-stream) semantics of NeMo's batched computer:
    per frame, emit up to ``max_symbols`` non-blank labels — each
    advancing the predictor — then advance the frame on blank or on the
    hard cap. State flows in and out untouched by chunk boundaries
    (no SOS re-injection, PORT-DEC-001).

    Args:
        enc_frames: ``(time, enc_hidden)`` conditioned encoder frames.
        predictor: The prediction network.
        joint: The joint network.
        state: Carried decode state; mutated copy returned.
        max_symbols: Per-frame emission cap (checkpoint value 10).

    Returns:
        Emitted label ids (the replay-queue content for this chunk,
        PORT-DEC-002) and the advanced state.
    """
    h, c = state.h, state.c
    last_label = state.last_label
    emitted: list[int] = []
    pred_out, pred_state = predictor.step(last_label, (h, c))
    for t in range(enc_frames.shape[0]):
        frame = enc_frames[t].unsqueeze(0)
        for _ in range(max_symbols):
            logits = joint.logits(frame, pred_out)
            label = int(logits.argmax(dim=-1))
            if label == predictor.blank_id:
                break
            emitted.append(label)
            last_label = torch.tensor(
                [label], dtype=torch.long, device=enc_frames.device
            )
            h, c = pred_state
            pred_out, pred_state = predictor.step(last_label, (h, c))
    # Blank does not advance the predictor: state reflects the last
    # non-blank emission only.
    return emitted, DecodeState(h=h, c=c, last_label=last_label)
