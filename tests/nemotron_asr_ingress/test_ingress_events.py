"""Session-core event vocabulary contract (ING-CORE-003/004)."""

from nemotron_asr_ingress.events import (
    AdmissionOutcome,
    Admitted,
    SessionError,
)


# @spec ING-CORE-003
def test_admission_outcome_carries_exactly_the_reserved_vocabulary() -> None:
    assert {member.name for member in AdmissionOutcome} == {
        "ADMITTED",
        "BUSY",
        "QUEUED",
    }
    assert AdmissionOutcome.ADMITTED.value == "admitted"
    assert AdmissionOutcome.BUSY.value == "busy"
    assert AdmissionOutcome.QUEUED.value == "queued"


# @spec ING-CORE-004
def test_admitted_event_requires_provenance() -> None:
    event = Admitted(
        session_id="sess-1",
        provenance={"precision_policy_id": "pp-x", "fingerprint": {}},
    )
    assert event.provenance["precision_policy_id"] == "pp-x"
    assert "fingerprint" in event.provenance


# @spec ING-ERR-003
def test_session_error_names_offending_fields() -> None:
    err = SessionError(
        code="config_change_rejected", fields=("chunk_ms",), detail="fixed"
    )
    assert err.fields == ("chunk_ms",)
