"""The shared vLLM-dialect session client's event core (ING-CLI-001).

One client, three roles (remote provider, endpoint-tier parity,
``benchmarks/asr`` driver). The folding contract pins the verified
omni behavior: ``RealtimeConnection`` keeps the transcription events
and always sends a terminal ``response.audio.done {has_audio: false}``
for text-only models (@ ``d4a869fe``) — the client must tolerate and
ignore every ``response.audio.*`` event.
"""

from nemotron_asr_ingress.client import (
    LatencyRecorder,
    RealtimeOutcome,
    fold_server_event,
)


def fold_all(*events: dict[str, object]) -> RealtimeOutcome:
    outcome = RealtimeOutcome()
    for event in events:
        fold_server_event(outcome, event)
    return outcome


# @spec ING-CLI-001
def test_session_created_carries_the_session_id() -> None:
    outcome = fold_all({"type": "session.created", "id": "sess-42"})
    assert outcome.session_id == "sess-42"
    assert not outcome.done


# @spec ING-CLI-001, ING-CORE-004
def test_session_admitted_carries_provenance_when_present() -> None:
    outcome = fold_all(
        {
            "type": "session.admitted",
            "provenance": {"precision_policy_id": "pp-x"},
        }
    )
    assert outcome.provenance is not None
    assert outcome.provenance["precision_policy_id"] == "pp-x"


# @spec ING-CLI-001
def test_stock_upstream_never_sends_admitted_and_that_is_fine() -> None:
    outcome = fold_all(
        {"type": "session.created", "id": "sess-1"},
        {"type": "transcription.delta", "delta": "hey"},
        {"type": "transcription.done", "text": "hey there"},
    )
    assert outcome.provenance is None
    assert outcome.final == "hey there"
    assert outcome.done


# @spec ING-CLI-001
def test_deltas_accumulate_in_order() -> None:
    outcome = fold_all(
        {"type": "transcription.delta", "delta": "hey"},
        {"type": "transcription.delta", "delta": " there"},
    )
    assert outcome.deltas == ["hey", " there"]
    assert outcome.final is None
    assert not outcome.done


# @spec ING-CLI-001
def test_transcription_done_is_the_terminal_transcript() -> None:
    outcome = fold_all(
        {"type": "transcription.delta", "delta": "hey"},
        {"type": "transcription.done", "text": "hey there friend"},
    )
    assert outcome.final == "hey there friend"
    assert outcome.done


# @spec ING-CLI-001
def test_response_audio_done_is_tolerated_and_ignored() -> None:
    # The omni finally-block guarantee: this arrives even for
    # text-only ASR models, before or after transcription.done.
    outcome = fold_all(
        {"type": "transcription.done", "text": "hey"},
        {"type": "response.audio.done", "has_audio": False},
    )
    assert outcome.errors == []
    assert outcome.done
    assert outcome.final == "hey"


# @spec ING-CLI-001
def test_response_audio_events_never_terminate_the_transcript() -> None:
    outcome = fold_all(
        {"type": "response.audio.delta", "audio": "AAAA"},
        {"type": "response.audio.done", "has_audio": True},
    )
    assert not outcome.done
    assert outcome.final is None
    assert outcome.errors == []


# @spec ING-CLI-001
def test_error_events_are_recorded_never_raised() -> None:
    outcome = fold_all({"type": "error", "error": "busy", "code": "busy"})
    assert len(outcome.errors) == 1
    assert outcome.errors[0]["code"] == "busy"


# @spec ING-CLI-001
def test_unknown_event_types_are_noted_never_fatal() -> None:
    outcome = fold_all(
        {"type": "conversation.created"},
        {"type": "transcription.done", "text": "hey"},
    )
    assert "conversation.created" in outcome.ignored
    assert outcome.errors == []
    assert outcome.done


# ---- chunk-to-partial latency at the client edge (β convergence) ----------


# @spec ING-CLI-001
def test_latency_gap_is_delta_arrival_minus_latest_send() -> None:
    outcome = RealtimeOutcome()
    recorder = LatencyRecorder(outcome)
    recorder.note_audio_sent(now=1.0)
    recorder.fold({"type": "transcription.delta", "delta": "hey"}, now=1.25)
    recorder.note_audio_sent(now=2.0)
    recorder.note_audio_sent(now=3.0)
    recorder.fold({"type": "transcription.delta", "delta": " there"}, now=3.5)
    assert recorder.gaps == [0.25, 0.5]
    assert outcome.deltas == ["hey", " there"]


def test_non_delta_events_fold_without_recording_gaps() -> None:
    outcome = RealtimeOutcome()
    recorder = LatencyRecorder(outcome)
    recorder.note_audio_sent(now=1.0)
    recorder.fold({"type": "session.created", "id": "sess-1"}, now=1.1)
    recorder.fold({"type": "transcription.done", "text": "hey"}, now=1.2)
    assert recorder.gaps == []
    assert outcome.done
    assert outcome.session_id == "sess-1"
