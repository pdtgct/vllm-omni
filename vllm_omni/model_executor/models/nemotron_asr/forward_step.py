# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The forward step over page-backed session state (bring-up BU-c1).

``run_forward_step`` is the pure compute the model's ``forward`` runs
once the engine hands it the bound page pools and per-row
``state_indices``: it is loader-testable off the engine (synthetic
pools + a fake metadata bundle), while ``forward`` itself stays a thin
wrapper that reads ``get_forward_context()`` and calls this (that
extraction is verified on the pod, BU-c2).

The pipeline is fixed-order (D-BUc-1; the four ordering hazards are
enforced here): classify rows by token id → echo-guard replay rows
against the PRE-advance queue head → session-first-zero chunk rows
(the window len-slot == 0 signal, before the encoder reads the views)
→ ``stream_step`` over the window/conv page views → LID → the
chunk-decode fills the replay+LSTM pages → one ``replay_step`` over all
rows → the emitted id is written into every row's hidden output as the
decision carrier (``compute_logits`` reads it back by row position).
"""

import torch

from vllm_omni.model_executor.models.nemotron_asr.forward_ops import (
    ROLE_CHUNK,
    ROLE_REPLAY,
)


def run_forward_step(
    core: torch.nn.Module,
    input_ids: torch.Tensor,
    inputs_embeds: torch.Tensor,
    *,
    channel_pools: list[torch.Tensor],
    time_pools: list[torch.Tensor],
    len_pools: list[torch.Tensor],
    h_pool: torch.Tensor,
    c_pool: torch.Tensor,
    queue_pool: torch.Tensor,
    book_pool: torch.Tensor,
    state_indices: torch.Tensor,
    placeholder_id: int,
    park_id: int,
    feat: int,
    drop_extra: int,
    prompt_index: int,
) -> torch.Tensor:
    """One forward step over page-backed state; returns hidden carriers.

    ``core``: the ``NemotronASRCore`` (encoder / lid / predictor /
    joint). ``input_ids`` / ``inputs_embeds``: ``(N,)`` / ``(N, H)`` —
    chunk rows carry a packed mel row (``pack_audio_carrier``), replay/
    flush rows carry zeros (their id rides ``input_ids``).
    ``*_pool``: engine-bound per-state pools (one entry per encoder
    layer for the channel/time/len window+conv pools). ``state_indices``:
    ``(N,)`` the page block index per row (a session follows its index).

    Returns ``(N, H)`` where each row's decision carrier holds the id
    that row emits this step (a chunk row's first burst label, a replay
    row's next queued label, or the park id).

    Raises:
        ValueError: On a replay-echo mismatch (a corrupted session),
            via ``verify_replay_echo``.
    """
    raise NotImplementedError("BU-c1 code phase")
