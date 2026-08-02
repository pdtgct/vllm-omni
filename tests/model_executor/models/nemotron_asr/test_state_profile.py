# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Aggregate-state profile projections for the Nemotron consumer."""

from __future__ import annotations

from types import SimpleNamespace

import torch

from vllm_omni.model_executor.models.nemotron_asr.state_profile import (
    build_nemotron_persistent_state_spec,
    project_nemotron_state_pools,
)
from vllm_omni.model_executor.persistent_state.storage import (
    allocate_persistent_state_storage,
)


def _config() -> SimpleNamespace:
    return SimpleNamespace(
        n_layers=24,
        att_context_left=56,
        d_model=1024,
        conv_kernel=9,
        pred_rnn_layers=2,
        pred_hidden=640,
        n_mels=128,
        num_asr_labels=13_087,
        endpoint_history_capacity_frames=12,
    )


def test_manifest_builds_one_exact_aggregate_and_zero_copy_pool_views() -> None:
    # @spec PORT-STATE-001 / PORT-STATE-002 / PORT-STATE-009
    spec = build_nemotron_persistent_state_spec(_config())
    storage = allocate_persistent_state_storage(spec, 2, torch.device("cpu"))
    storage.initialize_fresh_state_slot(1, generation=4)
    pools = project_nemotron_state_pools(storage, n_layers=24)

    assert len(spec.descriptors) == 99
    assert len(pools.channel) == len(pools.convolution) == 24
    assert pools.replay_book.shape == (2, 7)
    assert pools.frontend_counters.shape == (2, 8)
    assert pools.endpoint_book.shape == (2, 6)
    assert pools.replay_book[1, 2].item() == 13_087

    pools.replay_book[1, 4] = 9
    pools.frontend_counters[1, 7] = 1
    pools.endpoint_book[1, 3] = 6
    assert storage.views["decode.layers.0.replay.book.geometry"][1].item() == 9
    assert storage.views["frontend.counters.finalized"][1].item() == 1
    assert storage.views["endpoint.book.segment_generation"][1].item() == 6


def test_fresh_generation_erases_every_byte_and_reinitializes_blank() -> None:
    # @spec PORT-STATE-003 / PORT-STATE-009
    spec = build_nemotron_persistent_state_spec(_config())
    storage = allocate_persistent_state_storage(spec, 1, torch.device("cpu"))
    storage.initialize_fresh_state_slot(0, generation=1)
    storage.raw.fill_(0x7F)
    storage.initialize_fresh_state_slot(0, generation=2)
    pools = project_nemotron_state_pools(storage, n_layers=24)

    assert pools.replay_book[0, 2].item() == 13_087
    pools.replay_book[0, 2] = 0
    assert torch.count_nonzero(storage.raw).item() == 0
