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
import importlib.util
import sys
import types
from collections.abc import AsyncIterator, Coroutine
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch

_PKG = (
    Path(__file__).resolve().parents[4]
    / "vllm_omni/model_executor/models/nemotron_asr"
)
_BASE = "vllm_omni.model_executor.models.nemotron_asr"


def _load_chain() -> dict[str, Any]:
    for name in (
        "vllm_omni",
        "vllm_omni.model_executor",
        "vllm_omni.model_executor.models",
        _BASE,
    ):
        if name not in sys.modules:
            sys.modules[name] = types.ModuleType(name)
    loaded: dict[str, Any] = {}
    for mod in (
        "precision", "masks", "featurizer", "encoder", "lid",
        "manifests", "frontend", "rnnt_cell", "rnnt",
        "configuration_nemotron_asr", "session", "streaming",
    ):
        spec = importlib.util.spec_from_file_location(
            f"{_BASE}.{mod}", _PKG / f"{mod}.py"
        )
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[f"{_BASE}.{mod}"] = module
        spec.loader.exec_module(module)
        loaded[mod] = module
    return loaded


_MODULES = _load_chain()
verify_replay_echo = _MODULES["rnnt"].verify_replay_echo
write_decision_carrier = _MODULES["rnnt"].write_decision_carrier
buffer_stream = _MODULES["streaming"].buffer_stream
NemotronRealtimeSession = _MODULES["session"].NemotronRealtimeSession

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

PARK_ID = 13088  # the config's eos_token_id in these tests
PLACEHOLDER_ID = 13089
#: Locales chosen so the stamped rows read as the literal row numbers
#: the PORT-LID-001 assertions below name.
PROMPTS = {"auto": 0, "en-US": 2, "de-DE": 7, "fr-FR": 3}


def _session(**kwargs: Any) -> Any:
    """A typed realtime session over a bare config (PORT-RTC-001).

    Every session control the segmenter reads is now a typed attribute
    of this object; the retired ``getattr``-with-default reads had no
    writer on the standard serving path.
    """
    hf = SimpleNamespace(
        eos_token_id=PARK_ID,
        audio_chunk_token_id=PLACEHOLDER_ID,
        prompt_dictionary=dict(PROMPTS),
        num_prompts=128,
    )
    return NemotronRealtimeSession.from_model_config(hf, **kwargs)


def _run(coro: Coroutine[Any, Any, Any]) -> Any:
    loop = asyncio.get_event_loop_policy().new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.run_until_complete(loop.shutdown_asyncgens())
        loop.close()


async def _collect_yields(
    chunks_of_samples: list[int], park_after_each: bool = True
) -> list[dict[str, Any]]:
    """Drive buffer_realtime_audio with synthetic PCM.

    Feeds each ndarray into audio_stream; echoes the park id on
    input_stream after every yield (the drained signal) so the hold
    never deadlocks the test.
    """
    import numpy as np

    async def audio_stream() -> AsyncIterator[Any]:
        for n in chunks_of_samples:
            yield np.zeros(n, dtype=np.float32)

    input_stream: asyncio.Queue = asyncio.Queue()
    yields = []
    agen = buffer_stream(
        audio_stream(), input_stream, _session()
    )
    async for prompt in agen:
        yields.append(prompt)
        if park_after_each:
            input_stream.put_nowait([PARK_ID])
    return yields


# ---- one yield per admitted chunk (PORT-SESS-001/002) --------------------------


def test_one_yield_per_chunk_at_the_admitted_config() -> None:
    # 560 ms chunks at 16 kHz = 8960 samples per chunk: feeding exactly
    # three chunks' worth of audio yields three cadence CHUNKs plus the
    # explicit zero-sample final-tail transaction (PORT-SESS-003).
    yields = _run(_collect_yields([8960, 8960, 8960]))
    assert len(yields) == 4
    tail = yields[-1]["multi_modal_data"]["audio"]
    assert tail[1] == 0
    assert tail[3] == 1
    assert tail[5] == 3


def test_ragged_appends_rechunk_to_the_admitted_size() -> None:
    # The wire cadence is the client's; the yield cadence is the
    # admitted chunk's. 2 × 13440 samples = 3 × 8960 plus the explicit
    # zero-sample final-tail transaction.
    yields = _run(_collect_yields([13440, 13440]))
    assert len(yields) == 4


def test_subchunk_tail_is_processed_as_is_never_padded() -> None:
    # Finalize with a 4480-sample residual (half a chunk): the tail
    # yields as-is. Even a sub-8-mel-frame residual is represented by
    # an explicit final-tail transaction; the frontend may commit zero
    # frames while finalization still advances atomically.
    assert len(_run(_collect_yields([8960, 4480]))) == 2
    yields = _run(_collect_yields([8960, 1279]))
    assert len(yields) == 2
    tail = yields[-1]["multi_modal_data"]["audio"]
    assert tail[1] == 1279
    assert tail[3] == 1


def test_zero_audio_still_emits_one_final_tail_transaction() -> None:
    yields = _run(_collect_yields([]))
    assert len(yields) == 1
    envelope = yields[0]["multi_modal_data"]["audio"]
    assert envelope[1] == 0
    assert envelope[3] == 1
    assert envelope.shape[0] == len(
        _MODULES["manifests"].ENVELOPE_HEADER_FIELDS
    )


def test_next_chunk_holds_until_park_echo() -> None:
    # With no park echo on input_stream, the generator must not yield
    # chunk 2 (buffer-until-drained) — bounded wait, then assert only
    # chunk 1 emerged.
    import numpy as np

    async def scenario() -> None:
        async def audio_stream() -> AsyncIterator[Any]:
            yield np.zeros(8960, dtype=np.float32)
            yield np.zeros(8960, dtype=np.float32)

        input_stream: asyncio.Queue = asyncio.Queue()
        agen = buffer_stream(
            audio_stream(), input_stream, _session()
        )
        first = await asyncio.wait_for(agen.__anext__(), timeout=2)
        assert first is not None
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(agen.__anext__(), timeout=0.2)

    _run(scenario())


def test_locale_update_is_stamped_at_the_next_mint() -> None:
    # PORT-LID-001: the last valid update ordered before CHUNK-carrier
    # mint is stamped into that immutable carrier. A live per-session
    # config view (ING-VEH-007) mutates the selection between chunks;
    # the change appears at the NEXT mint, never retroactively. The
    # selection now moves through the session's own validated selector
    # (PORT-RTC-001), not an untyped attribute write.
    import numpy as np

    async def scenario() -> None:
        session = _session(locale="en-US")

        async def audio_stream() -> AsyncIterator[Any]:
            yield np.zeros(8960, dtype=np.float32)
            yield np.zeros(8960, dtype=np.float32)

        input_stream: asyncio.Queue = asyncio.Queue()
        agen = buffer_stream(audio_stream(), input_stream, session)
        first = (await agen.__anext__())["multi_modal_data"]["audio"]
        assert first[4] == 2.0
        session.select_prompt("de-DE")
        input_stream.put_nowait([PARK_ID])
        second = (await agen.__anext__())["multi_modal_data"]["audio"]
        assert second[4] == 7.0
        assert first[4] == 2.0  # the earlier carrier is immutable
        session.select_prompt("fr-FR")
        input_stream.put_nowait([PARK_ID])
        tail = (await agen.__anext__())["multi_modal_data"]["audio"]
        assert tail[3] == 1.0
        assert tail[4] == 3.0  # final-tail takes the current selection

    _run(scenario())


# ---- the yield shape (PORT-INT-003 / D-BU-1) -----------------------------------


def test_yield_carries_the_placeholder_token() -> None:
    # The yield must be TokensPrompt-shaped — prompt_token_ids=[
    # placeholder] + multi_modal_data audio — not a bare
    # multi_modal_data dict (invalid on the real render path). One
    # placeholder token per chunk makes PORT-POOL-001 true by
    # construction; the id comes from the config.
    import numpy as np

    async def scenario() -> None:
        async def audio_stream() -> AsyncIterator[Any]:
            yield np.zeros(8960, dtype=np.float32)

        input_stream: asyncio.Queue = asyncio.Queue()
        agen = buffer_stream(
            audio_stream(), input_stream, _session()
        )
        prompt = await agen.__anext__()
        assert prompt["prompt_token_ids"] == [PLACEHOLDER_ID]
        assert "audio" in prompt["multi_modal_data"]

    _run(scenario())


# ---- the replay-echo guard (PORT-DEC-007's active half) ------------------------


def test_echo_guard_passes_on_faithful_feedback() -> None:
    ids = torch.tensor([5, 7, 13088], dtype=torch.long)
    verify_replay_echo(ids.clone(), ids)  # no raise


def test_echo_guard_aborts_loudly_on_corruption() -> None:
    # An exclusion mask (or any hostile param) that redirects the
    # argmax shows up as observed != forced on the NEXT step — the one
    # place the model has authority to reject (the α3 sampler negative
    # is the why; this is the what).
    forced = torch.tensor([5, 7], dtype=torch.long)
    observed = torch.tensor([5, 3], dtype=torch.long)
    with pytest.raises(ValueError):
        verify_replay_echo(observed, forced)


# ---- compute_logits row alignment (PORT-DEC-002, Q5) ---------------------------


def test_compute_logits_reads_rows_by_batch_position() -> None:
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
