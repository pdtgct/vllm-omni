"""Session-core lifecycle, chunking, and terminal semantics.

ING-CORE-005, ING-ERR-004, ING-LIFE-002..009, ING-FE-005/006 — all at
the sans-IO seam, driven with explicit time.
"""

import numpy as np
import pytest
from ingress_helpers import (
    CHUNK_SAMPLES,
    audio,
    make_core,
    make_values,
)

from nemotron_asr_ingress import errors
from nemotron_asr_ingress.core import ChunkBuffer
from nemotron_asr_ingress.events import (
    Configure,
    Final,
    Partial,
    SessionError,
    UpdateAck,
)

CFG = Configure(chunk_ms=80, target_lang="en-US")


# @spec ING-FE-006, ING-CORE-001
def test_transcriber_receives_exact_admitted_chunks() -> None:
    core, fake, _ = make_core()
    core.configure(CFG, now=0.0)
    # Odd-size pieces spanning 3 chunk boundaries plus a residual.
    total = 3 * CHUNK_SAMPLES + CHUNK_SAMPLES // 2
    for start in range(0, total, 700):
        n = min(700, total - start)
        core.receive_audio(audio(n, start=start), now=0.1)
    assert [len(chunk) for chunk in fake.steps] == [CHUNK_SAMPLES] * 3


# @spec ING-CORE-005
def test_partials_carry_the_cumulative_hypothesis() -> None:
    core, fake, _ = make_core()
    core.configure(CFG, now=0.0)
    events = core.receive_audio(audio(2 * CHUNK_SAMPLES), now=0.1)
    partials = [e for e in events if isinstance(e, Partial)]
    assert [p.cumulative for p in partials] == ["hey", "hey there"]
    assert [p.chunk_index for p in partials] == [0, 1]


# @spec ING-FE-006
def test_finalize_hands_the_residual_raw_and_unpadded() -> None:
    core, fake, _ = make_core()
    core.configure(CFG, now=0.0)
    residual_len = CHUNK_SAMPLES // 2
    clip = audio(CHUNK_SAMPLES + residual_len)
    core.receive_audio(clip, now=0.1)
    core.finalize(now=0.2)
    assert fake.flush_called
    assert fake.flushed is not None
    # Exactly the leftover samples, exactly their values, no padding:
    # the tail transform is PORT-SESS-003's, never the front-end's.
    assert len(fake.flushed) == residual_len
    np.testing.assert_array_equal(fake.flushed, clip[CHUNK_SAMPLES:])


# @spec ING-LIFE-002
def test_exactly_one_final_per_session() -> None:
    core, fake, _ = make_core()
    core.configure(CFG, now=0.0)
    core.receive_audio(audio(CHUNK_SAMPLES), now=0.1)
    events = core.finalize(now=0.2)
    finals = [e for e in events if isinstance(e, Final)]
    assert len(finals) == 1
    assert core.terminal


# @spec ING-ERR-004, ING-LIFE-002
def test_audio_after_finalize_answers_session_terminal() -> None:
    core, fake, _ = make_core()
    core.configure(CFG, now=0.0)
    core.finalize(now=0.1)
    answer = core.receive_audio(audio(CHUNK_SAMPLES), now=0.2)
    assert len(answer) == 1
    assert isinstance(answer[0], SessionError)
    assert answer[0].code == errors.SESSION_TERMINAL
    # Nothing reached the engine after finalize acceptance.
    assert fake.steps == []


# @spec ING-ERR-004
def test_second_finalize_answers_session_terminal_never_silence() -> None:
    core, _, _ = make_core()
    core.configure(CFG, now=0.0)
    core.finalize(now=0.1)
    answer = core.finalize(now=0.2)
    assert [type(e) for e in answer] == [SessionError]
    assert isinstance(answer[0], SessionError)
    assert answer[0].code == errors.SESSION_TERMINAL


# @spec ING-LIFE-003
def test_zero_audio_finalize_is_a_valid_degenerate_session() -> None:
    core, fake, _ = make_core()
    core.configure(CFG, now=0.0)
    events = core.finalize(now=0.1)
    assert events == [Final(transcript="")]
    # Skip-flush: PORT's tail transform is never invoked with nothing.
    assert not fake.flush_called
    assert core.terminal


# @spec ING-LIFE-004
def test_close_frees_the_slot_and_aborts_immediately() -> None:
    core, fake, gate = make_core()
    core.configure(CFG, now=0.0)
    assert gate.active == 1
    core.close(now=0.1)
    assert gate.active == 0
    assert fake.aborted


# @spec ING-LIFE-005, ING-LIFE-006
def test_idle_ttl_backstop_aborts_with_idle_timeout() -> None:
    values = make_values(idle_ttl_s=60.0)
    core, fake, gate = make_core(values=values)
    core.configure(CFG, now=0.0)
    (abort,) = core.poll(now=60.1)
    assert isinstance(abort, SessionError)
    assert abort.code == errors.IDLE_TIMEOUT
    assert gate.active == 0
    assert fake.aborted


# @spec ING-LIFE-005
def test_receive_cadence_resets_the_idle_clock() -> None:
    values = make_values(idle_ttl_s=60.0)
    core, _, _ = make_core(values=values)
    core.configure(CFG, now=0.0)
    core.receive_audio(audio(100), now=50.0)
    assert core.poll(now=100.0) == []
    assert core.poll(now=110.5) != []


# @spec ING-LIFE-007
def test_valid_locale_update_is_acked_and_forwarded() -> None:
    core, fake, _ = make_core()
    core.configure(CFG, now=0.0)
    events = core.update({"target_lang": "es-US"}, now=0.1)
    assert events == [UpdateAck(honored={"target_lang": "es-US"})]
    assert fake.locales == ["es-US"]


# @spec ING-LIFE-007
def test_unknown_locale_update_names_the_field_and_continues() -> None:
    core, fake, _ = make_core()
    core.configure(CFG, now=0.0)
    (err,) = core.update({"target_lang": "xx-XX"}, now=0.1)
    assert isinstance(err, SessionError)
    assert err.code == errors.UNKNOWN_LOCALE
    assert "target_lang" in err.fields
    assert fake.locales == []
    # The session continues at the prior locale.
    events = core.receive_audio(audio(CHUNK_SAMPLES), now=0.2)
    assert any(isinstance(e, Partial) for e in events)


# @spec ING-LIFE-008
def test_admission_fixed_field_update_is_rejected_with_continuation() -> None:
    core, fake, _ = make_core()
    core.configure(CFG, now=0.0)
    (err,) = core.update({"chunk_ms": 320}, now=0.1)
    assert isinstance(err, SessionError)
    assert err.code == errors.CONFIG_CHANGE_REJECTED
    assert "chunk_ms" in err.fields
    # Still chunking at the admitted config.
    core.receive_audio(audio(CHUNK_SAMPLES), now=0.2)
    assert [len(chunk) for chunk in fake.steps] == [CHUNK_SAMPLES]


# @spec ING-LIFE-009
def test_mixed_update_rejects_first_then_acks_the_honored_subset() -> None:
    core, _, _ = make_core()
    core.configure(CFG, now=0.0)
    events = core.update({"target_lang": "es-US", "chunk_ms": 320}, now=0.1)
    assert len(events) == 2
    rejection, ack = events
    assert isinstance(rejection, SessionError)
    assert rejection.code == errors.CONFIG_CHANGE_REJECTED
    assert "chunk_ms" in rejection.fields
    assert isinstance(ack, UpdateAck)
    assert dict(ack.honored) == {"target_lang": "es-US"}


# @spec ING-FE-005
def test_oversized_burst_answers_buffer_overflow() -> None:
    values = make_values(chunk_buffer_s=0.25)  # 4000 samples at 16 kHz
    core, _, _ = make_core(values=values)
    core.configure(CFG, now=0.0)
    events = core.receive_audio(audio(8000), now=0.1)
    codes = [e.code for e in events if isinstance(e, SessionError)]
    assert errors.BUFFER_OVERFLOW in codes


# @spec ING-LIFE-007, ING-ADM-001
def test_unknown_locale_at_admission_rejects_without_a_slot() -> None:
    core, _, gate = make_core()
    answer = core.configure(
        Configure(chunk_ms=80, target_lang="xx-XX"), now=0.0
    )
    assert len(answer) == 1
    assert isinstance(answer[0], SessionError)
    assert answer[0].code == errors.UNKNOWN_LOCALE
    assert "target_lang" in answer[0].fields
    assert gate.active == 0


# @spec ING-FE-005, ING-FE-006
def test_chunk_buffer_pops_exact_chunks_across_odd_splits() -> None:
    buffer = ChunkBuffer(chunk_samples=CHUNK_SAMPLES, max_seconds=10.0)
    total = 2 * CHUNK_SAMPLES + 37
    clip = audio(total)
    for start in range(0, total, 501):
        assert buffer.append(clip[start : start + 501])
    chunks = buffer.pop_chunks()
    assert [len(chunk) for chunk in chunks] == [CHUNK_SAMPLES] * 2
    np.testing.assert_array_equal(
        np.concatenate(chunks + [buffer.residual()]), clip
    )


# @spec ING-FE-005
def test_chunk_buffer_refuses_growth_past_its_bound() -> None:
    buffer = ChunkBuffer(chunk_samples=CHUNK_SAMPLES, max_seconds=0.25)
    assert buffer.append(audio(4000))
    assert not buffer.append(audio(1))


# @spec ING-LIFE-002
def test_no_input_after_final_reaches_the_engine() -> None:
    core, fake, _ = make_core()
    core.configure(CFG, now=0.0)
    core.receive_audio(audio(CHUNK_SAMPLES), now=0.1)
    core.finalize(now=0.2)
    steps_at_final = len(fake.steps)
    core.receive_audio(audio(CHUNK_SAMPLES), now=0.3)
    core.update({"target_lang": "es-US"}, now=0.4)
    assert len(fake.steps) == steps_at_final
    assert fake.locales == []


# @spec ING-LIFE-003
def test_zero_audio_finalize_still_frees_the_slot() -> None:
    core, _, gate = make_core()
    core.configure(CFG, now=0.0)
    core.finalize(now=0.1)
    core.close(now=0.2)
    assert gate.active == 0


def test_chunk_buffer_bound_must_be_positive() -> None:
    with pytest.raises(ValueError):
        ChunkBuffer(chunk_samples=CHUNK_SAMPLES, max_seconds=0.0)


def test_ingress_core_is_the_durable_home_for_shared_values() -> None:
    # The durable package owns the shared values; serve imports from
    # ingress, never the reverse (β convergence dependency direction).
    from nemotron_asr_ingress.core import (
        SAMPLE_RATE,
        VALID_CHUNK_MS,
        IdleClock,
    )

    assert SAMPLE_RATE == 16000
    assert VALID_CHUNK_MS == (80, 160, 320, 560, 1120)
    clock = IdleClock(ttl_s=1.0, now=0.0)
    assert not clock.expired(0.5)
    assert clock.expired(1.5)


# @spec ING-ADM-004
def test_gate_holds_the_residency_cap_under_concurrent_configures() -> None:
    # β convergence moves gate calls onto per-connection executor
    # threads; the residency cap must hold under concurrent configures
    # and releases, never just under event-loop serialization.
    import threading

    from nemotron_asr_ingress.core import InProcessGate
    from nemotron_asr_ingress.events import AdmissionOutcome

    gate = InProcessGate(watermark=4, queue_len=0, wait_s=5.0)
    violations: list[int] = []

    def hammer(worker: int) -> None:
        for i in range(300):
            session_id = f"s{worker}-{i}"
            if gate.request(session_id, 0.0) is AdmissionOutcome.ADMITTED:
                active = gate.active
                if active > 4:
                    violations.append(active)
                gate.release(session_id)

    threads = [
        threading.Thread(target=hammer, args=(worker,)) for worker in range(8)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert violations == []
    assert gate.active == 0
