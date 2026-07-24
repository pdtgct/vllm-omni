# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The model-local realtime session contract + receipt ledger (RFC-1 W2a).

Specs: PORT-RTC-001 (typed session built only through its factory from
the validated HF configuration; immutable geometry/park/placeholder;
fail-closed construction; no untyped attribute fallback in the
segmenter), PORT-RTC-002 (one carrier ticket per cadence minted BEFORE
that carrier's prompt is yielded, piece receipts naming exactly the
tickets a frame minted, FIFO completion, model-owned finite backlog),
PORT-LID-001 (mid-session selection through the session's own validated
selector; unknown locale rejects the update and preserves the prior
prompt).

Loader-runnable (no vllm imports at module level), same importlib chain
as test_realtime_serving.py — extended with configuration_nemotron_asr
and session.
"""

import asyncio
import dataclasses
import importlib.util
import math
import sys
import types
from collections.abc import AsyncIterator, Coroutine
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

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
        "manifests", "configuration_nemotron_asr", "session", "streaming",
    ):
        dotted = f"{_BASE}.{mod}"
        # Reuse-if-present: other test files chain-load these canonical
        # names at collection time, and lazily-importing modules under
        # test resolve through sys.modules at call time — overwriting
        # here would split class identity across chains.
        existing = sys.modules.get(dotted)
        if existing is not None and getattr(existing, "__file__", None):
            loaded[mod] = existing
            continue
        spec = importlib.util.spec_from_file_location(dotted, _PKG / f"{mod}.py")
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[dotted] = module
        spec.loader.exec_module(module)
        loaded[mod] = module
    return loaded


_MODULES = _load_chain()
_SESSION = _MODULES["session"]
NemotronRealtimeSession = _SESSION.NemotronRealtimeSession
AdmittedGeometry = _SESSION.AdmittedGeometry
ReceiptLedger = _SESSION.ReceiptLedger
DEFAULT_CADENCE = _SESSION.DEFAULT_CADENCE
DEFAULT_LOCALE = _SESSION.DEFAULT_LOCALE
LEDGER_BACKLOG_S = _SESSION.LEDGER_BACKLOG_S
buffer_stream = _MODULES["streaming"].buffer_stream
CADENCES = _MODULES["manifests"].CADENCES
RAW_SAMPLES_PER_CHUNK = _MODULES["manifests"].RAW_SAMPLES_PER_CHUNK

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

PARK_ID = 13088
PLACEHOLDER_ID = 13089
PROMPTS = {"auto": 101, "en-US": 2, "de-DE": 7, "fr-FR": 3}


def _hf(**overrides: Any) -> SimpleNamespace:
    """A bare NemotronASRConfig stand-in (the test path the factory
    accepts alongside the vLLM ModelConfig wrapper)."""
    fields: dict[str, Any] = {
        "eos_token_id": PARK_ID,
        "audio_chunk_token_id": PLACEHOLDER_ID,
        "prompt_dictionary": dict(PROMPTS),
        "num_prompts": 128,
    }
    fields.update(overrides)
    return SimpleNamespace(**fields)


def _session(**kwargs: Any) -> Any:
    return NemotronRealtimeSession.from_model_config(_hf(), **kwargs)


def _run(coro: Coroutine[Any, Any, Any]) -> Any:
    loop = asyncio.get_event_loop_policy().new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.run_until_complete(loop.shutdown_asyncgens())
        loop.close()


def _audio(*sizes: int) -> AsyncIterator[Any]:
    async def stream() -> AsyncIterator[Any]:
        for size in sizes:
            yield np.zeros(size, dtype=np.float32)

    return stream()


# ---- factory resolution + fail-closed construction (PORT-RTC-001) -------------


# @spec PORT-RTC-001, PORT-SESS-002
def test_factory_derives_geometry_from_the_manifests() -> None:
    for cadence, samples in RAW_SAMPLES_PER_CHUNK.items():
        session = _session(cadence=cadence, locale="en-US")
        assert session.geometry.cadence == cadence
        assert session.geometry.chunk_samples == samples
        assert session.geometry.geometry_id == list(CADENCES).index(cadence)


# @spec PORT-RTC-001
def test_factory_defaults_are_the_published_module_constants() -> None:
    assert DEFAULT_CADENCE == "560ms"
    assert DEFAULT_LOCALE == "auto"
    session = _session()
    assert session.geometry.chunk_samples == 8_960
    assert session.prompt_index == PROMPTS["auto"]
    assert session.ledger is None


# @spec PORT-RTC-001
def test_factory_unwraps_the_model_config_wrapper() -> None:
    wrapper = SimpleNamespace(hf_config=_hf(), nemotron_prompt_index=0)
    session = NemotronRealtimeSession.from_model_config(wrapper)
    assert session.park_token_id == PARK_ID
    assert session.audio_chunk_token_id == PLACEHOLDER_ID


# @spec PORT-RTC-001
@pytest.mark.parametrize(
    "overrides",
    [
        {"eos_token_id": None},
        {"audio_chunk_token_id": None},
        {"eos_token_id": -1},
        {"audio_chunk_token_id": "13089"},
    ],
)
def test_factory_fails_closed_on_a_missing_or_invalid_token_id(
    overrides: dict[str, Any],
) -> None:
    with pytest.raises(ValueError):
        NemotronRealtimeSession.from_model_config(_hf(**overrides))


# @spec PORT-RTC-001, PORT-SESS-002
def test_factory_fails_closed_on_an_unknown_cadence() -> None:
    # Strictly earlier than the retired mid-generator rejection: no
    # session exists at an un-admitted geometry.
    with pytest.raises(ValueError):
        _session(cadence="240ms")


# @spec PORT-RTC-001, PORT-LID-001
def test_factory_fails_closed_on_an_unknown_locale() -> None:
    with pytest.raises(ValueError):
        _session(locale="xx-XX")


# @spec PORT-RTC-001, PORT-LID-001, PORT-REGIME-003
def test_factory_resolves_unambiguous_iso_code_to_checkpoint_locale() -> None:
    session = _session(locale="en")
    assert session.prompt_index == PROMPTS["en-US"]


# @spec PORT-RTC-001, PORT-LID-001, PORT-REGIME-003
def test_factory_normalizes_checkpoint_locale_casing() -> None:
    session = _session(locale="EN_us")
    assert session.prompt_index == PROMPTS["en-US"]


# @spec PORT-RTC-001, PORT-LID-001, PORT-REGIME-003
def test_exact_checkpoint_locale_wins_over_iso_expansion() -> None:
    prompts = {
        "auto": 0,
        "en": 1,
        "en-US": 2,
        "en-GB": 3,
        "EN_us": 4,
    }
    session = NemotronRealtimeSession.from_model_config(
        _hf(prompt_dictionary=prompts), locale="EN_us"
    )
    assert session.prompt_index == prompts["EN_us"]


# @spec PORT-RTC-001, PORT-LID-001, PORT-REGIME-003
def test_ambiguous_iso_code_requires_explicit_checkpoint_locale() -> None:
    prompts = {"auto": 0, "en-US": 2, "en-GB": 3}
    with pytest.raises(ValueError, match="ambiguous.*explicit"):
        NemotronRealtimeSession.from_model_config(
            _hf(prompt_dictionary=prompts), locale="en"
        )


# @spec PORT-RTC-001, PORT-LID-001
def test_factory_reuses_the_published_prompt_dictionary_validator() -> None:
    # Admission validation IS configuration_nemotron_asr's validator:
    # a dictionary it rejects can never reach a session.
    with pytest.raises(ValueError):
        NemotronRealtimeSession.from_model_config(_hf(num_prompts=0))
    with pytest.raises(ValueError):
        NemotronRealtimeSession.from_model_config(
            _hf(prompt_dictionary={"auto": 999})
        )
    with pytest.raises(ValueError):
        NemotronRealtimeSession.from_model_config(_hf(prompt_dictionary={}))


# ---- immutability + the validated selector (PORT-RTC-001 / PORT-LID-001) ------


# @spec PORT-RTC-001, PORT-SESS-002
def test_geometry_park_and_placeholder_are_immutable() -> None:
    session = _session()
    for attribute in ("geometry", "park_token_id", "audio_chunk_token_id"):
        with pytest.raises(AttributeError):
            setattr(session, attribute, 0)
    with pytest.raises(dataclasses.FrozenInstanceError):
        session.geometry.chunk_samples = 1_280


# @spec PORT-RTC-001, PORT-LID-001
def test_prompt_index_is_written_only_through_select_prompt() -> None:
    session = _session()
    with pytest.raises(AttributeError):
        session.prompt_index = 4
    assert session.select_prompt("de-DE") == PROMPTS["de-DE"]
    assert session.prompt_index == PROMPTS["de-DE"]
    assert session.select_prompt("EN") == PROMPTS["en-US"]
    assert session.prompt_index == PROMPTS["en-US"]


# @spec PORT-LID-001
def test_select_prompt_rejects_unknown_locale_and_preserves_prior() -> None:
    session = _session(locale="en-US")
    with pytest.raises(ValueError):
        session.select_prompt("xx-XX")
    assert session.prompt_index == PROMPTS["en-US"]


# ---- the receipt ledger (PORT-RTC-002) ---------------------------------------


# @spec PORT-RTC-002
def test_tickets_are_minted_in_carrier_order() -> None:
    async def scenario() -> None:
        session = _session(with_ledger=True)
        ledger = session.ledger
        queue: asyncio.Queue = asyncio.Queue()
        agen = buffer_stream(_audio(8_960 * 3), queue, session)
        sequences = []
        async for prompt in agen:
            envelope = prompt["multi_modal_data"]["audio"]
            sequences.append(int(envelope[5]))
            queue.put_nowait([PARK_ID])
        assert sequences == [0, 1, 2, 3]
        minted = [t.sequence for t in ledger.pending]
        assert minted == [0, 1, 2, 3]
        assert [t.final_tail for t in ledger.pending] == [
            False, False, False, True
        ]

    _run(scenario())


# @spec PORT-RTC-002
def test_a_ticket_exists_before_its_prompt_is_yielded() -> None:
    # THE RACE PIN (round-3 finding 1): the completion handle must
    # already exist when the prompt leaves the segmenter, so a park
    # returned immediately after the prompt can never be lost.
    async def scenario() -> None:
        session = _session(with_ledger=True)
        ledger = session.ledger
        queue: asyncio.Queue = asyncio.Queue()
        agen = buffer_stream(_audio(8_960 * 2), queue, session)
        first = await agen.__anext__()
        envelope = first["multi_modal_data"]["audio"]
        # Both cadences of the frame were stamped in one synchronous
        # pass, so both tickets exist while carrier 0 is in flight.
        assert len(ledger.pending) == 2
        assert ledger.pending[0].sequence == int(envelope[5])
        assert not ledger.pending[0].done.done()
        await agen.aclose()

    _run(scenario())


# @spec PORT-RTC-002
def test_piece_receipt_names_exactly_the_tickets_that_frame_minted() -> None:
    async def scenario() -> None:
        session = _session(with_ledger=True)
        ledger = session.ledger
        queue: asyncio.Queue = asyncio.Queue()
        agen = buffer_stream(_audio(4_480, 13_440, 8_960), queue, session)
        await agen.__anext__()  # drives frames 1-2; frame 1 mints nothing
        first = await ledger.next_piece()
        assert first.tickets == ()
        assert first.samples_consumed == 4_480
        second = await ledger.next_piece()
        assert [t.sequence for t in second.tickets] == [0, 1]
        assert second.samples_consumed == 4_480 + 13_440
        await agen.aclose()

    _run(scenario())


# @spec PORT-RTC-002
def test_complete_next_completes_the_oldest_pending_ticket() -> None:
    async def scenario() -> None:
        ledger = ReceiptLedger(max_pending_carriers=8)
        first = ledger.mint(final_tail=False, admission_ms_mod=1)
        second = ledger.mint(final_tail=False, admission_ms_mod=2)
        ledger.complete_next("a")
        assert await first.done == "a"
        assert not second.done.done()
        ledger.complete_next("b")
        assert await second.done == "b"
        assert ledger.pending == ()

    _run(scenario())


# @spec PORT-RTC-002
def test_complete_next_without_a_pending_ticket_is_loud() -> None:
    async def scenario() -> None:
        ledger = ReceiptLedger(max_pending_carriers=8)
        with pytest.raises(RuntimeError):
            ledger.complete_next("orphan park")
        ticket = ledger.mint(final_tail=True, admission_ms_mod=0)
        ledger.complete_next("ok")
        assert ticket.done.done()
        with pytest.raises(RuntimeError):
            ledger.complete_next("orphan park")

    _run(scenario())


# @spec PORT-RTC-002
def test_the_backlog_cap_raises_at_mint() -> None:
    async def scenario() -> None:
        ledger = ReceiptLedger(max_pending_carriers=2)
        ledger.mint(final_tail=False, admission_ms_mod=0)
        ledger.mint(final_tail=False, admission_ms_mod=1)
        with pytest.raises(RuntimeError):
            ledger.mint(final_tail=False, admission_ms_mod=2)
        # Never a dropped ticket: the record still holds exactly the
        # two admitted carriers, and completing one reopens the cap.
        assert [t.sequence for t in ledger.pending] == [0, 1]
        ledger.complete_next(None)
        assert ledger.mint(final_tail=False, admission_ms_mod=3).sequence == 2

    _run(scenario())


# @spec PORT-RTC-002
def test_the_backlog_cap_fails_the_segmenter_before_the_yield() -> None:
    async def scenario() -> None:
        session = _session(with_ledger=True, max_pending_carriers=2)
        queue: asyncio.Queue = asyncio.Queue()
        agen = buffer_stream(_audio(8_960 * 3), queue, session)
        with pytest.raises(RuntimeError):
            await agen.__anext__()

    _run(scenario())


# @spec PORT-RTC-002
def test_the_default_backlog_derives_from_the_admitted_cadence() -> None:
    assert LEDGER_BACKLOG_S == 30.0
    for cadence, samples in RAW_SAMPLES_PER_CHUNK.items():
        session = _session(cadence=cadence, with_ledger=True)
        expected = math.ceil(LEDGER_BACKLOG_S / (samples / 16_000))
        assert session.ledger.max_pending_carriers == expected
    widest = _session(cadence="1120ms", with_ledger=True)
    assert widest.ledger.max_pending_carriers == 27


# @spec PORT-RTC-002
def test_geometry_seconds_backs_the_backlog_derivation() -> None:
    geometry = AdmittedGeometry.from_cadence("560ms")
    assert geometry.seconds == pytest.approx(0.56)


# ---- ledger terminal failure propagation (PORT-RTC-002; W3 needs this) ---------


def test_fail_unblocks_a_next_piece_waiter_before_acknowledgement() -> None:
    # @spec PORT-RTC-002
    # Engine died before consuming a frame: a consumer blocked in
    # next_piece() must be woken with the failure, not hang forever.
    async def scenario() -> None:
        ledger = ReceiptLedger(max_pending_carriers=4)
        waiter = asyncio.ensure_future(ledger.next_piece())
        await asyncio.sleep(0)  # let it park on the waiter
        assert not waiter.done()
        boom = RuntimeError("engine died mid-generation")
        ledger.fail(boom)
        with pytest.raises(RuntimeError, match="engine died"):
            await waiter

    _run(scenario())


def test_fail_rejects_a_pending_ticket_between_mint_and_park() -> None:
    # @spec PORT-RTC-002
    # Engine died after a carrier was minted but before its park: the
    # ticket's done future must fail, not block the consumer forever.
    async def scenario() -> None:
        ledger = ReceiptLedger(max_pending_carriers=4)
        ticket = ledger.mint(final_tail=False, admission_ms_mod=7)
        boom = RuntimeError("engine died after mint")
        ledger.fail(boom)
        with pytest.raises(RuntimeError, match="after mint"):
            await ticket.done

    _run(scenario())


def test_fail_makes_later_operations_reject() -> None:
    # @spec PORT-RTC-002
    ledger = ReceiptLedger(max_pending_carriers=4)
    ledger.fail(RuntimeError("dead"))
    with pytest.raises(RuntimeError, match="terminally failed"):
        ledger.mint(final_tail=False, admission_ms_mod=1)
    with pytest.raises(RuntimeError, match="terminally failed"):
        ledger.acknowledge_piece(160)
    with pytest.raises(RuntimeError, match="terminally failed"):
        ledger.complete_next("x")
    # next_piece() raises the ORIGINAL failure, not the wrapper: a
    # terminal failure must dominate, and the consumer sees exactly what
    # killed the engine.
    with pytest.raises(RuntimeError, match="dead"):
        _run(ledger.next_piece())


def test_fail_is_idempotent() -> None:
    # @spec PORT-RTC-002
    ledger = ReceiptLedger(max_pending_carriers=4)
    ledger.fail(RuntimeError("first"))
    ledger.fail(RuntimeError("second"))  # no raise, no state churn
    with pytest.raises(RuntimeError, match="terminally failed"):
        ledger.mint(final_tail=False, admission_ms_mod=1)


def test_fail_dominates_a_previously_queued_receipt() -> None:
    # @spec PORT-RTC-002
    # A sub-cadence frame queues a receipt with no ticket. If the engine
    # then dies, that queued receipt must NOT be handed to a consumer as
    # a successful empty hypothesis: the terminal failure dominates, and
    # next_piece() raises the ORIGINAL error fail() was given (not the
    # generic "terminally failed" wrapper).
    async def scenario() -> None:
        ledger = ReceiptLedger(max_pending_carriers=4)
        ledger.acknowledge_piece(160)  # a sub-cadence piece queues a receipt
        boom = RuntimeError("engine died")
        ledger.fail(boom)
        with pytest.raises(RuntimeError, match="engine died") as excinfo:
            await ledger.next_piece()
        assert excinfo.value is boom

    _run(scenario())
