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

from typing import TYPE_CHECKING

import torch

from vllm_omni.model_executor.models.nemotron_asr.encoder import stream_step
from vllm_omni.model_executor.models.nemotron_asr.forward_ops import (
    ROLE_CHUNK,
    ROLE_REPLAY,
    classify_step_rows,
    unpack_audio_carrier,
)
from vllm_omni.model_executor.models.nemotron_asr.rnnt import (
    QUEUE_HEAD,
    QUEUE_LAST_LABEL,
    QUEUE_LEN,
    QUEUE_PROMPT,
    decode_chunk_paged,
    replay_step,
    verify_replay_echo,
    write_decision_carrier,
)
from vllm_omni.model_executor.models.nemotron_asr.state_layers import (
    PagedStreamingCaches,
)

if TYPE_CHECKING:
    from vllm_omni.model_executor.models.nemotron_asr.nemotron_asr import (
        NemotronASRCore,
    )


def run_forward_step(
    core: "NemotronASRCore",
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
    n_rows = int(input_ids.shape[0])
    hidden = int(inputs_embeds.shape[1])
    n_layers = len(channel_pools)
    window = int(channel_pools[0].shape[1])
    idx = state_indices.long()

    heads = book_pool[idx, QUEUE_HEAD].long()
    lens = book_pool[idx, QUEUE_LEN].long()
    roles = classify_step_rows(
        input_ids, placeholder_id=placeholder_id, queue_lengths=lens - heads
    )

    # Hazard 2: echo-guard replay rows against the PRE-advance head.
    replay_mask = roles == ROLE_REPLAY
    if bool(replay_mask.any()):
        r_idx = idx[replay_mask]
        r_head = heads[replay_mask]
        forced = queue_pool[r_idx, r_head - 1].long()
        verify_replay_echo(input_ids[replay_mask], forced)

    # Chunk rows: session-first-zero (hazard 1), then stream_step → LID →
    # decode fills the queue (hazard 3: before the drain below).
    chunk_rows = (roles == ROLE_CHUNK).nonzero(as_tuple=True)[0].tolist()
    with torch.no_grad():
        for row in chunk_rows:
            block = int(idx[row])
            session_first = int(len_pools[0][block, 0]) == 0
            if session_first:
                for layer in range(n_layers):
                    channel_pools[layer][block].zero_()
                    time_pools[layer][block].zero_()
                    len_pools[layer][block].zero_()
                h_pool[block].zero_()
                c_pool[block].zero_()
                book_pool[block, QUEUE_LAST_LABEL] = float(core.blank_id)
            caches = PagedStreamingCaches(
                channel_views=[channel_pools[la][block] for la in range(n_layers)],
                time_views=[time_pools[la][block] for la in range(n_layers)],
                len_slots=[len_pools[la][block] for la in range(n_layers)],
                left_context=window,
            )
            mel = unpack_audio_carrier(
                inputs_embeds[row], feat=feat
            ).unsqueeze(0)
            enc = stream_step(
                # PagedStreamingCaches is StreamingCaches' structural twin
                # (migration-proven bit-for-bit); stream_step reads only
                # the shared .channel/.time/.valid surface.
                core.encoder, mel, caches,  # type: ignore[arg-type]
                drop_extra=0 if session_first else drop_extra,
            )
            prompt_index = int(book_pool[block, QUEUE_PROMPT])
            conditioned = core.lid(enc, prompt_index=prompt_index)
            decode_chunk_paged(
                conditioned, core.predictor, core.joint,
                h_pool=h_pool, c_pool=c_pool, queue_pool=queue_pool,
                book_pool=book_pool, state_indices=torch.tensor([block]),
            )

    # One replay/drain over all rows; the emitted id is the decision.
    emitted = replay_step(
        queue_pool, book_pool, state_indices=idx, park_id=park_id
    )
    hidden_out = torch.zeros(
        n_rows, hidden, dtype=inputs_embeds.dtype, device=inputs_embeds.device
    )
    write_decision_carrier(hidden_out, emitted)
    return hidden_out
