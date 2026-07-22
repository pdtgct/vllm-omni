"""Admission: watermark, bounded queue, wait timeout (ING-ADM-001..005)."""

from ingress_helpers import audio, make_core, make_gate, make_values

from nemotron_asr_ingress import errors
from nemotron_asr_ingress.core import PreRollBuffer
from nemotron_asr_ingress.events import (
    Admitted,
    Busy,
    Configure,
    SessionError,
)


# @spec ING-ADM-001
def test_busy_answers_the_configure_itself() -> None:
    values = make_values(watermark=1, admission_queue=0)
    gate = make_gate(values)
    first, _, _ = make_core(gate=gate, values=values)
    assert isinstance(first.configure(Configure(), now=0.0)[0], Admitted)

    second, _, _ = make_core(gate=gate, values=values)
    answer = second.configure(Configure(), now=0.0)
    assert len(answer) == 1
    assert isinstance(answer[0], Busy)


# @spec ING-ADM-001, ING-CORE-004
def test_admitted_answer_carries_provenance() -> None:
    core, _, _ = make_core()
    (admitted,) = core.configure(Configure(), now=0.0)
    assert isinstance(admitted, Admitted)
    assert admitted.session_id
    assert admitted.provenance["precision_policy_id"]
    assert "fingerprint" in admitted.provenance


# @spec ING-ADM-002
async def test_queued_configure_is_answered_when_a_slot_frees() -> None:
    values = make_values(watermark=1, admission_queue=1)
    gate = make_gate(values)
    first, _, _ = make_core(gate=gate, values=values)
    first.configure(Configure(), now=0.0)

    queued, _, _ = make_core(gate=gate, values=values)
    assert queued.configure(Configure(), now=0.0) == []
    assert await queued.poll(now=1.0) == []

    await first.close(now=2.0)
    answer = await queued.poll(now=2.1)
    assert len(answer) == 1
    assert isinstance(answer[0], Admitted)


# @spec ING-ADM-002
async def test_admission_wait_timeout_releases_a_queued_session() -> None:
    values = make_values(watermark=1, admission_queue=1, admission_wait_s=5.0)
    gate = make_gate(values)
    first, _, _ = make_core(gate=gate, values=values)
    first.configure(Configure(), now=0.0)

    queued, _, _ = make_core(gate=gate, values=values)
    queued.configure(Configure(), now=0.0)
    (released,) = await queued.poll(now=5.1)
    assert isinstance(released, SessionError)
    assert released.code == errors.ADMISSION_WAIT_TIMEOUT
    assert queued.terminal


# @spec ING-ADM-002
async def test_idle_timer_never_runs_against_a_queued_session() -> None:
    values = make_values(
        watermark=1,
        admission_queue=1,
        admission_wait_s=100.0,
        idle_ttl_s=1.0,
    )
    gate = make_gate(values)
    first, _, _ = make_core(gate=gate, values=values)
    first.configure(Configure(), now=0.0)

    queued, _, _ = make_core(gate=gate, values=values)
    queued.configure(Configure(), now=0.0)
    # Far past the idle TTL, well inside the admission wait: a queued
    # session is released by the wait timeout only, never the idle
    # timer (PORT-SESS-005 carve-out).
    assert await queued.poll(now=50.0) == []


# @spec ING-ADM-001, ING-ADM-002
def test_full_queue_answers_busy() -> None:
    values = make_values(watermark=1, admission_queue=1)
    gate = make_gate(values)
    first, _, _ = make_core(gate=gate, values=values)
    first.configure(Configure(), now=0.0)
    queued, _, _ = make_core(gate=gate, values=values)
    queued.configure(Configure(), now=0.0)

    third, _, _ = make_core(gate=gate, values=values)
    answer = third.configure(Configure(), now=0.0)
    assert len(answer) == 1
    assert isinstance(answer[0], Busy)


# @spec ING-ADM-004
async def test_watermark_counts_parked_sessions() -> None:
    values = make_values(watermark=1, admission_queue=0, idle_ttl_s=60.0)
    gate = make_gate(values)
    parked, _, _ = make_core(gate=gate, values=values)
    parked.configure(Configure(chunk_ms=80), now=0.0)
    await parked.receive_audio(audio(1280), now=0.1)
    # No audio since 0.1: idle-parked between chunks, well inside the
    # TTL. Its GPU residency still occupies the watermark.
    late, _, _ = make_core(gate=gate, values=values)
    answer = late.configure(Configure(), now=30.0)
    assert isinstance(answer[0], Busy)


# @spec ING-ADM-005
def test_pre_roll_releases_everything_in_arrival_order() -> None:
    pre_roll = PreRollBuffer(max_bytes=1024)
    assert pre_roll.hold_audio(b"one")
    assert pre_roll.hold_audio(b"two")
    pre_roll.hold_update({"target_lang": "es-US"})
    pre_roll.hold_finalize()
    kinds = [kind for kind, _ in pre_roll.release()]
    assert kinds == ["audio", "audio", "update", "finalize"]


# @spec ING-ADM-005
def test_pre_roll_preserves_payloads_raw() -> None:
    pre_roll = PreRollBuffer(max_bytes=1024)
    pre_roll.hold_audio(b"\x01\x02")
    pre_roll.hold_update({"target_lang": "es-US"})
    released = pre_roll.release()
    assert released[0] == ("audio", b"\x01\x02")
    assert released[1] == ("update", {"target_lang": "es-US"})


# @spec ING-ADM-005
def test_pre_roll_discards_whole_on_busy() -> None:
    pre_roll = PreRollBuffer(max_bytes=1024)
    pre_roll.hold_audio(b"audio")
    pre_roll.hold_finalize()
    pre_roll.discard()
    assert pre_roll.release() == []


# @spec ING-ADM-005
def test_pre_roll_is_bounded() -> None:
    pre_roll = PreRollBuffer(max_bytes=4)
    assert pre_roll.hold_audio(b"1234")
    assert not pre_roll.hold_audio(b"5")
