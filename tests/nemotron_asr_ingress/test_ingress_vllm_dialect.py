"""The β server's vLLM ``/v1/realtime`` dialect binding (ING-1).

Wire shapes per ``speech_to_text/realtime/protocol.py`` @ the PIN
(``ee0da84ab``): client ``session.update`` / ``input_audio_buffer.
append`` (base64 PCM16 @ 16 kHz) / ``input_audio_buffer.commit
{final}``; server ``session.created`` / ``transcription.delta`` /
``transcription.done`` / ``error {error, code}``. Pin-native codes are
kept by name; the pin's silently-ignored second commit is NOT
inherited (ING-ERR-004).
"""

import base64
from collections.abc import Mapping
from typing import Any

import numpy as np
import pytest
from ingress_helpers import (
    CHUNK_SAMPLES,
    PROVENANCE,
    FakeTranscriber,
    make_core,
    make_gate,
    make_values,
)

from nemotron_asr_ingress.core import (
    IngressValues,
    InProcessGate,
    SessionCore,
)
from nemotron_asr_ingress.events import AdmissionOutcome, Configure
from nemotron_asr_ingress.vllm_dialect import VllmRealtimeAdapter

MODEL = "nemotron-asr"


def make_adapter(
    gate: InProcessGate | None = None,
    values: IngressValues | None = None,
) -> tuple[VllmRealtimeAdapter, FakeTranscriber, InProcessGate]:
    core, fake, the_gate = make_core(gate=gate, values=values)
    return VllmRealtimeAdapter(core, model_name=MODEL), fake, the_gate


def update_event(**extra: Any) -> dict[str, Any]:
    return {
        "type": "session.update",
        "model": MODEL,
        "chunk_ms": 80,
        "target_lang": "en-US",
        **extra,
    }


def append_event(samples: "np.typing.NDArray[np.int16]") -> dict[str, Any]:
    payload = base64.b64encode(samples.tobytes()).decode("ascii")
    return {"type": "input_audio_buffer.append", "audio": payload}


def pcm_chunk(value: int = 16384, n: int = CHUNK_SAMPLES) -> Any:
    return np.full((n,), value, dtype=np.int16)


def error_codes(events: list[dict[str, Any]]) -> list[str]:
    return [e["code"] for e in events if e.get("type") == "error"]


async def admit(
    adapter: VllmRealtimeAdapter, now: float = 0.0
) -> list[dict[str, Any]]:
    """Configure (admission answers the update) and stream one chunk."""
    configured = await adapter.on_event(update_event(), now=now)
    return configured + await adapter.on_event(
        append_event(pcm_chunk()), now=now
    )


# @spec ING-CLI-001
def test_session_created_is_sent_on_connect() -> None:
    adapter, _, _ = make_adapter()
    (created,) = adapter.on_connect()
    assert created["type"] == "session.created"
    assert created["id"]


# @spec ING-LIFE-001
async def test_commit_before_session_update_is_model_not_validated() -> None:
    adapter, _, _ = make_adapter()
    events = await adapter.on_event(
        {"type": "input_audio_buffer.commit", "final": False}, now=0.0
    )
    assert error_codes(events) == ["model_not_validated"]


# @spec ING-ERR-001
async def test_session_update_without_model_is_invalid_event() -> None:
    adapter, _, _ = make_adapter()
    events = await adapter.on_event({"type": "session.update"}, now=0.0)
    assert error_codes(events) == ["invalid_event"]


# @spec ING-ERR-001
async def test_session_update_with_wrong_model_is_model_not_found() -> None:
    adapter, _, _ = make_adapter()
    events = await adapter.on_event(update_event(model="whisper-large"), now=0.0)
    assert error_codes(events) == ["model_not_found"]


# @spec ING-ERR-003
async def test_invalid_chunk_ms_extension_names_the_field() -> None:
    adapter, _, _ = make_adapter()
    events = await adapter.on_event(update_event(chunk_ms=397), now=0.0)
    assert error_codes(events) == ["invalid_config_field"]
    (event,) = events
    assert "chunk_ms" in event["error"]


# @spec ING-ADM-001
async def test_watermark_full_answers_busy_at_session_update() -> None:
    # The additive fields complete the config at session.update, so
    # the admission outcome answers the update itself (ADM-001), never
    # the first append (Pete, PR #13 review).
    values = make_values(watermark=1, admission_queue=0)
    gate = make_gate(values)
    occupant, _, _ = make_core(gate=gate, values=values)
    occupant.configure(Configure(), now=0.0)

    adapter, fake, _ = make_adapter(gate=gate, values=values)
    events = await adapter.on_event(update_event(), now=0.0)
    assert error_codes(events) == ["busy"]
    assert fake.steps == []


# @spec ING-CORE-004, ING-ADM-001
async def test_session_update_is_answered_by_session_admitted() -> None:
    # session.created stays the connect-time event; session.admitted
    # is the distinct admission answer to session.update, carrying
    # provenance (Pete, PR #13 review).
    adapter, _, _ = make_adapter()
    events = await adapter.on_event(update_event(), now=0.0)
    admitted = [e for e in events if e["type"] == "session.admitted"]
    assert len(admitted) == 1
    provenance = admitted[0]["provenance"]
    assert provenance["precision_policy_id"]
    assert "fingerprint" in provenance
    # Appends never re-answer admission.
    more = await adapter.on_event(append_event(pcm_chunk()), now=0.1)
    assert [e for e in more if e["type"] == "session.admitted"] == []


# @spec ING-ADM-005
async def test_pre_roll_overflow_is_fatal_to_the_unadmitted_attempt() -> None:
    # Pre-admission overflow: fatal to the attempt, session never
    # admitted, no audio processed, safe to retry with backoff; avoid
    # by waiting for the admission answer before streaming ahead of it.
    values = make_values(watermark=1, admission_queue=1, pre_roll_bytes=1024)
    gate = make_gate(values)
    occupant, _, _ = make_core(gate=gate, values=values)
    occupant.configure(Configure(), now=0.0)

    adapter, fake, _ = make_adapter(gate=gate, values=values)
    assert await adapter.on_event(update_event(), now=0.0) == []  # queued
    events = await adapter.on_event(append_event(pcm_chunk()), now=0.1)
    assert error_codes(events) == ["buffer_overflow"]
    assert fake.steps == []


# @spec ING-FE-006
async def test_append_decodes_base64_pcm16_to_float32() -> None:
    adapter, fake, _ = make_adapter()
    await admit(adapter)  # one full chunk of 16384
    assert len(fake.steps) == 1
    np.testing.assert_allclose(fake.steps[0], np.float32(0.5))


# @spec ING-CORE-005
async def test_deltas_are_projected_from_consecutive_cumulatives() -> None:
    adapter, _, _ = make_adapter()
    first = await admit(adapter)
    second = await adapter.on_event(append_event(pcm_chunk()), now=0.1)
    deltas = [
        e["delta"] for e in first + second if e["type"] == "transcription.delta"
    ]
    # Cumulative "hey" then "hey there" project to incremental deltas.
    assert deltas == ["hey", " there"]


# @spec ING-ERR-001
async def test_undecodable_audio_is_invalid_audio() -> None:
    adapter, _, _ = make_adapter()
    await adapter.on_event(update_event(), now=0.0)
    events = await adapter.on_event(
        {"type": "input_audio_buffer.append", "audio": "@@not-base64@@"},
        now=0.0,
    )
    assert error_codes(events) == ["invalid_audio"]


# @spec ING-LIFE-002
async def test_final_commit_produces_transcription_done() -> None:
    adapter, fake, _ = make_adapter()
    await admit(adapter)
    events = await adapter.on_event(
        {"type": "input_audio_buffer.commit", "final": True}, now=0.1
    )
    done = [e for e in events if e["type"] == "transcription.done"]
    assert len(done) == 1
    assert done[0]["text"] == fake.cumulative()


# @spec ING-ERR-004
async def test_append_after_final_is_session_terminal_not_silent() -> None:
    adapter, _, _ = make_adapter()
    await admit(adapter)
    await adapter.on_event(
        {"type": "input_audio_buffer.commit", "final": True}, now=0.1
    )
    events = await adapter.on_event(append_event(pcm_chunk()), now=0.2)
    assert error_codes(events) == ["session_terminal"]


# @spec ING-ERR-004
async def test_second_final_commit_is_session_terminal() -> None:
    adapter, _, _ = make_adapter()
    await admit(adapter)
    await adapter.on_event(
        {"type": "input_audio_buffer.commit", "final": True}, now=0.1
    )
    events = await adapter.on_event(
        {"type": "input_audio_buffer.commit", "final": True}, now=0.2
    )
    assert error_codes(events) == ["session_terminal"]


# @spec ING-ERR-004
async def test_first_nonfinal_commit_is_accepted_quietly() -> None:
    # Pin mechanics: the first non-final commit starts generation; the
    # shared client sends it against stock upstream, so β accepts it.
    adapter, _, _ = make_adapter()
    await admit(adapter)
    events = await adapter.on_event(
        {"type": "input_audio_buffer.commit", "final": False}, now=0.1
    )
    assert error_codes(events) == []


# @spec ING-ERR-004
async def test_second_nonfinal_commit_surfaces_an_error_not_a_log_line() -> None:
    # connection.py:173-176 @ the PIN ignores this with a warning log;
    # downstream never inherits the silent drop.
    adapter, _, _ = make_adapter()
    await admit(adapter)
    await adapter.on_event(
        {"type": "input_audio_buffer.commit", "final": False}, now=0.1
    )
    events = await adapter.on_event(
        {"type": "input_audio_buffer.commit", "final": False}, now=0.2
    )
    assert error_codes(events) == ["protocol_order"]


# @spec ING-ERR-001
async def test_unknown_event_type_answers_unknown_event() -> None:
    adapter, _, _ = make_adapter()
    events = await adapter.on_event({"type": "session.get"}, now=0.0)
    assert error_codes(events) == ["unknown_event"]


# @spec ING-CORE-003
def test_adapter_rejects_the_reserved_queued_outcome() -> None:
    adapter, _, _ = make_adapter()
    with pytest.raises(ValueError):
        adapter.project_admission(AdmissionOutcome.QUEUED, None)


# @spec ING-CORE-002
async def test_session_update_extensions_default_to_upstream_semantics() -> None:
    # Without the additive fields the admitted config is the LLD's
    # surface-1 default: 560 ms chunks, target_lang auto.
    adapter, fake, _ = make_adapter()
    await adapter.on_event({"type": "session.update", "model": MODEL}, now=0.0)
    chunk_560ms = 560 * 16
    await adapter.on_event(append_event(pcm_chunk(n=chunk_560ms)), now=0.1)
    assert [len(chunk) for chunk in fake.steps] == [chunk_560ms]


# ---- admission-at-session.update ordering (Phase-6 note, Pete) -----------


def queue_behind_occupant(
    **value_overrides: Any,
) -> tuple[VllmRealtimeAdapter, FakeTranscriber, Any]:
    """An adapter queued behind one admitted occupant (W=1, queue=1)."""
    values = make_values(watermark=1, admission_queue=1, **value_overrides)
    gate = make_gate(values)
    occupant, _, _ = make_core(gate=gate, values=values)
    occupant.configure(Configure(), now=0.0)
    adapter, fake, _ = make_adapter(gate=gate, values=values)
    return adapter, fake, occupant


# @spec ING-ADM-001, ING-CORE-004
async def test_queued_admission_defers_the_answer_to_poll() -> None:
    # The queued path defers the answer rather than racing it: nothing
    # answers the update, nothing answers an append, and the admitted
    # event arrives on poll alone once the slot frees.
    adapter, _, occupant = queue_behind_occupant()
    assert await adapter.on_event(update_event(), now=0.0) == []
    assert await adapter.on_event(append_event(pcm_chunk()), now=0.1) == []
    assert await adapter.poll(now=0.2) == []

    await occupant.close(now=0.3)
    events = await adapter.poll(now=0.4)
    assert events
    assert events[0]["type"] == "session.admitted"


# @spec ING-CORE-004, ING-ADM-005
async def test_admitted_precedes_every_delta_on_the_queued_release() -> None:
    # session.admitted rides the resolving poll first; the pre-roll
    # drains behind it, so provenance always precedes the first delta.
    adapter, fake, occupant = queue_behind_occupant()
    await adapter.on_event(update_event(), now=0.0)
    await adapter.on_event(append_event(pcm_chunk()), now=0.1)
    await occupant.close(now=0.3)

    events = await adapter.poll(now=0.4)
    types = [e["type"] for e in events]
    assert types.index("session.admitted") == 0
    assert "transcription.delta" in types
    assert [len(chunk) for chunk in fake.steps] == [CHUNK_SAMPLES]


# @spec ING-ADM-005
async def test_held_finalize_replays_after_admission_in_order() -> None:
    adapter, fake, occupant = queue_behind_occupant()
    await adapter.on_event(update_event(), now=0.0)
    await adapter.on_event(append_event(pcm_chunk()), now=0.1)
    await adapter.on_event(
        {"type": "input_audio_buffer.commit", "final": True}, now=0.2
    )
    assert fake.steps == []  # nothing processed pre-admission
    await occupant.close(now=0.3)

    events = await adapter.poll(now=0.4)
    types = [e["type"] for e in events]
    assert types[0] == "session.admitted"
    assert types[-1] == "transcription.done"
    assert types.index("transcription.delta") < types.index(
        "transcription.done"
    )


# @spec ING-ADM-002, ING-ADM-005
async def test_wait_timeout_discards_the_pre_roll() -> None:
    adapter, fake, _ = queue_behind_occupant(admission_wait_s=5.0)
    await adapter.on_event(update_event(), now=0.0)
    await adapter.on_event(append_event(pcm_chunk()), now=0.1)

    events = await adapter.poll(now=5.2)
    assert error_codes(events) == ["admission_wait_timeout"]
    assert fake.steps == []


class SilentTranscriber:
    """A kernel hearing only silence: the cumulative stays empty."""

    def __init__(self) -> None:
        self.flush_called = False

    async def step(self, chunk: "np.typing.NDArray[np.float32]") -> str:
        return ""

    async def flush(self, residual: "np.typing.NDArray[np.float32]") -> str:
        self.flush_called = True
        return ""

    async def update_locale(self, target_lang: str) -> None:
        return None

    async def abort(self) -> None:
        return None


# @spec ING-CORE-005
async def test_silent_chunks_emit_no_empty_deltas() -> None:
    # A silent stream finalizes to "" with ZERO delta events: an
    # empty-payload delta is dialect noise the pin itself never sends
    # (no tokens, no delta). Pod-gate regression, 2026-07-12.
    values = make_values()
    gate = make_gate(values)
    core = SessionCore(
        gate=gate,
        transcriber=SilentTranscriber(),
        values=values,
        provenance=PROVENANCE,
    )
    adapter = VllmRealtimeAdapter(core, model_name=MODEL)
    await adapter.on_event(update_event(), now=0.0)
    events = await adapter.on_event(append_event(pcm_chunk(value=0)), now=0.1)
    events += await adapter.on_event(append_event(pcm_chunk(value=0)), now=0.2)
    assert [e for e in events if e["type"] == "transcription.delta"] == []
    done = await adapter.on_event(
        {"type": "input_audio_buffer.commit", "final": True}, now=0.3
    )
    assert [e["type"] for e in done] == ["transcription.done"]
    assert done[0]["text"] == ""


# ---- transport-drop mapping (β convergence) -------------------------------


# @spec ING-LIFE-004
async def test_on_disconnect_frees_the_slot_and_aborts() -> None:
    adapter, fake, gate = make_adapter()
    await adapter.on_event(update_event(), now=0.0)
    assert gate.active == 1
    await adapter.on_disconnect(now=0.1)
    assert gate.active == 0
    assert fake.aborted
    assert adapter.terminal


# @spec ING-LIFE-004, ING-ADM-005
async def test_on_disconnect_while_queued_frees_the_queue_slot() -> None:
    values = make_values(watermark=1, admission_queue=1)
    gate = make_gate(values)
    occupant, _, _ = make_core(gate=gate, values=values)
    occupant.configure(Configure(), now=0.0)

    adapter, fake, _ = make_adapter(gate=gate, values=values)
    assert await adapter.on_event(update_event(), now=0.0) == []
    await adapter.on_event(append_event(pcm_chunk()), now=0.1)  # pre-rolled
    await adapter.on_disconnect(now=0.2)
    assert fake.steps == []
    # The queue slot freed: the next attempt queues rather than busy.
    late, _, _ = make_core(gate=gate, values=values)
    assert late.configure(Configure(), now=0.3) == []


async def test_terminal_property_mirrors_the_core() -> None:
    adapter, _, _ = make_adapter()
    await adapter.on_event(update_event(), now=0.0)
    assert not adapter.terminal
    await adapter.on_event(
        {"type": "input_audio_buffer.commit", "final": True}, now=0.1
    )
    assert adapter.terminal


def _mapping_is_wire_dict(event: Mapping[str, Any]) -> bool:
    return isinstance(event.get("type"), str)


# @spec ING-CORE-001
async def test_every_outbound_event_is_a_wire_dict_with_a_type() -> None:
    adapter, _, _ = make_adapter()
    events = adapter.on_connect() + await admit(adapter)
    assert events
    assert all(_mapping_is_wire_dict(event) for event in events)
