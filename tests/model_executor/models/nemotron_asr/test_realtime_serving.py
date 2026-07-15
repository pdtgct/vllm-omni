# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""α4 CPU-tier: buffer contract, echo guard, row-alignment contract.

Specs: PORT-SESS-001/002/003 (one yield per admitted chunk; fixed
segmenter; hold-until-park via the park id echoing on input_stream;
NeMo tail rules on finalize), PORT-DEC-007 (the replay-echo guard —
the model-owned active half of the pin composition), PORT-DEC-002
(compute_logits reads decision rows by BATCH POSITION, never request
id). Consult: D-α4b/e + Q5.

Loader-runnable (no vllm imports at module level); the registry/
allocator halves live in test_omni_serving_binding.py (pod tier).
"""

import asyncio
from types import SimpleNamespace

import pytest
import torch

from vllm_omni.model_executor.models.nemotron_asr.rnnt import (
    verify_replay_echo,
    write_decision_carrier,
)
from vllm_omni.model_executor.models.nemotron_asr.streaming import (
    buffer_stream,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

PARK_ID = 13089  # placeholder eos in these tests; real value from config


def _run(coro):
    return asyncio.get_event_loop_policy().new_event_loop().run_until_complete(
        coro
    )


async def _collect_yields(chunks_of_samples, park_after_each=True):
    """Drive buffer_realtime_audio with synthetic PCM.

    Feeds each ndarray into audio_stream; echoes the park id on
    input_stream after every yield (the drained signal) so the hold
    never deadlocks the test.
    """
    import numpy as np

    async def audio_stream():
        for n in chunks_of_samples:
            yield np.zeros(n, dtype=np.float32)

    input_stream: asyncio.Queue = asyncio.Queue()
    model_config = None  # the α4 code phase binds the real config read
    yields = []
    agen = buffer_stream(
        audio_stream(), input_stream, model_config
    )
    async for prompt in agen:
        yields.append(prompt)
        if park_after_each:
            input_stream.put_nowait([PARK_ID])
    return yields


# ---- one yield per admitted chunk (PORT-SESS-001/002) --------------------------


def test_one_yield_per_chunk_at_the_admitted_config():
    # 560 ms chunks at 16 kHz = 8960 samples per chunk: feeding exactly
    # three chunks' worth of audio yields exactly three prompts —
    # never a fourth from padding (PORT-SESS-003's never-zero-pad).
    yields = _run(_collect_yields([8960, 8960, 8960]))
    assert len(yields) == 3


def test_ragged_appends_rechunk_to_the_admitted_size():
    # The wire cadence is the client's; the yield cadence is the
    # admitted chunk's. 2 × 13440 samples = exactly 3 × 8960.
    yields = _run(_collect_yields([13440, 13440]))
    assert len(yields) == 3


def test_subchunk_tail_is_processed_as_is_never_padded():
    # Finalize with a 4480-sample residual (half a chunk): the tail
    # yields as-is (partial tails processed, PORT-SESS-003); a
    # sub-8-mel-frame remainder (under 1280 samples) is dropped.
    assert len(_run(_collect_yields([8960, 4480]))) == 2
    assert len(_run(_collect_yields([8960, 1279]))) == 1


def test_next_chunk_holds_until_park_echo():
    # With no park echo on input_stream, the generator must not yield
    # chunk 2 (buffer-until-drained) — bounded wait, then assert only
    # chunk 1 emerged.
    import numpy as np

    async def scenario():
        async def audio_stream():
            yield np.zeros(8960, dtype=np.float32)
            yield np.zeros(8960, dtype=np.float32)

        input_stream: asyncio.Queue = asyncio.Queue()
        agen = buffer_stream(
            audio_stream(), input_stream, None
        )
        first = await asyncio.wait_for(agen.__anext__(), timeout=2)
        assert first is not None
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(agen.__anext__(), timeout=0.2)

    _run(scenario())


# ---- the yield shape (PORT-INT-003 / D-BU-1) -----------------------------------


def test_yield_carries_the_placeholder_token():
    # The yield must be TokensPrompt-shaped — prompt_token_ids=[
    # placeholder] + multi_modal_data audio — not a bare
    # multi_modal_data dict (invalid on the real render path). One
    # placeholder token per chunk makes PORT-POOL-001 true by
    # construction; the id comes from the config.
    import numpy as np

    async def scenario():
        async def audio_stream():
            yield np.zeros(8960, dtype=np.float32)

        input_stream: asyncio.Queue = asyncio.Queue()
        model_config = SimpleNamespace(audio_chunk_token_id=13089)
        agen = buffer_stream(
            audio_stream(), input_stream, model_config
        )
        prompt = await agen.__anext__()
        assert prompt["prompt_token_ids"] == [13089]
        assert "audio" in prompt["multi_modal_data"]

    _run(scenario())


# ---- the replay-echo guard (PORT-DEC-007's active half) ------------------------


def test_echo_guard_passes_on_faithful_feedback():
    ids = torch.tensor([5, 7, 13088], dtype=torch.long)
    verify_replay_echo(ids.clone(), ids)  # no raise


def test_echo_guard_aborts_loudly_on_corruption():
    # An exclusion mask (or any hostile param) that redirects the
    # argmax shows up as observed != forced on the NEXT step — the one
    # place the model has authority to reject (the α3 sampler negative
    # is the why; this is the what).
    forced = torch.tensor([5, 7], dtype=torch.long)
    observed = torch.tensor([5, 3], dtype=torch.long)
    with pytest.raises(ValueError):
        verify_replay_echo(observed, forced)


# ---- compute_logits row alignment (PORT-DEC-002, Q5) ---------------------------


def test_compute_logits_reads_rows_by_batch_position():
    # The engine gathers last-scheduled-token rows in batch order and
    # hands them to compute_logits — the carrier is read by ROW
    # POSITION. Permuting the rows must permute the outputs
    # identically: no hidden per-request bookkeeping.
    # ``compute_logits`` is a method on the (now vLLM-coupled) model
    # class, so this seam is pod/engine-covered; it skips off-engine.
    pytest.importorskip("vllm.multimodal")
    from vllm_omni.model_executor.models.nemotron_asr.nemotron_asr import (
        NemotronASRForRNNT,
    )

    model = object.__new__(NemotronASRForRNNT)  # seam only; no __init__
    hidden = torch.zeros(3, 32, dtype=torch.float32)
    ids = torch.tensor([11, 22, 33], dtype=torch.long)
    write_decision_carrier(hidden, ids)
    logits = model.compute_logits(hidden)
    assert torch.equal(logits.argmax(dim=-1), ids)
    perm = torch.tensor([2, 0, 1])
    logits_perm = model.compute_logits(hidden[perm])
    assert torch.equal(logits_perm.argmax(dim=-1), ids[perm])
