# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Pod-tier coverage for the model/runner row-plan hook.

The pure resolver matrix lives in ``test_plan_local.py``. This file pins the
real ``NemotronASRForRNNT.prepare_row_plan_context`` integration with the
vLLM request-feature shape that exposed the concurrent encoder-cache hit.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

PLACEHOLDER_ID = 13_089
HEADER = (2.0, 1_280.0, 0.0, 0.0, 0.0, 0.0, 10_000.0)


def _feature() -> SimpleNamespace:
    return SimpleNamespace(
        data={"audio": SimpleNamespace(data=torch.tensor(HEADER, dtype=torch.float32))},
        modality="audio",
        identifier="same-complete-envelope",
        mm_position=SimpleNamespace(offset=0, length=1),
    )


def _request(block_id: int) -> SimpleNamespace:
    return SimpleNamespace(block_ids=([block_id],), mm_features=[_feature()])


def test_two_concurrent_identical_chunks_bind_on_cache_hit() -> None:
    # @spec PORT-ADV-003 / PORT-INT-003
    # Request ``second`` is absent from scheduled_encoder_inputs because
    # vLLM reused ``first``'s identical encoder output. Both placeholder
    # rows nevertheless have a semantic feature and must mint independent
    # request/page bindings.
    pytest.importorskip("vllm.multimodal")
    from vllm_omni.model_executor.models.nemotron_asr.nemotron_asr import (
        NemotronASRForRNNT,
    )
    from vllm_omni.model_executor.models.nemotron_asr.plan import (
        PlanContextSlot,
        SessionRegistry,
    )

    model = object.__new__(NemotronASRForRNNT)
    model._commit_sink = None
    model._registry = SessionRegistry()
    model._plan_slot = PlanContextSlot()
    model._plan_step = 0
    model.config = SimpleNamespace(audio_chunk_token_id=PLACEHOLDER_ID)
    model.core = SimpleNamespace(lid=SimpleNamespace(num_prompts=4))
    requests = {
        "first": _request(1),
        "second": _request(2),
    }

    model.prepare_row_plan_context(
        req_ids=["first", "second"],
        token_ids_cpu=torch.tensor([[PLACEHOLDER_ID], [PLACEHOLDER_ID]], dtype=torch.long),
        num_computed_tokens_cpu=torch.tensor([0, 0], dtype=torch.long),
        num_scheduled_tokens=np.array([1, 1], dtype=np.int32),
        requests=requests,
        scheduled_encoder_inputs={"first": [0]},
    )

    context = model._plan_slot.consume()
    assert context.request_ids == ("first", "second")
    assert context.is_chunk.tolist() == [True, True]
    assert context.block_ids.tolist() == [1, 2]
    assert context.admission_generation.tolist() == [1, 2]


def test_encoder_cache_identifier_covers_every_envelope_value() -> None:
    # @spec PORT-INT-003
    # The encoder output may be shared only when the complete raw carrier
    # is identical. Pin the vLLM identifier dependency for every header
    # slot and for sample content.
    pytest.importorskip("vllm.multimodal")
    from vllm.multimodal.hasher import MultiModalHasher

    base = np.asarray((*HEADER, 0.25, -0.5), dtype=np.float32)
    baseline = MultiModalHasher.hash_kwargs(
        model_id="nemotron-asr-cache-identity",
        audio=base,
    )
    for index in range(base.shape[0]):
        changed = base.copy()
        changed[index] += np.float32(1.0)
        assert (
            MultiModalHasher.hash_kwargs(
                model_id="nemotron-asr-cache-identity",
                audio=changed,
            )
            != baseline
        )
