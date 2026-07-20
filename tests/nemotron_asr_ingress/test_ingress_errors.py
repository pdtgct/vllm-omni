"""The error catalog and its dialect projections (ING-ERR-001/002)."""

from nemotron_asr_ingress import errors
from nemotron_asr_ingress.errors import catalog

ALL_CODES = (
    errors.BUSY,
    errors.ADMISSION_WAIT_TIMEOUT,
    errors.IDLE_TIMEOUT,
    errors.PROTOCOL_ORDER,
    errors.INVALID_CONFIG_FIELD,
    errors.UNSUPPORTED_CAPABILITY,
    errors.UNKNOWN_LOCALE,
    errors.CONFIG_CHANGE_REJECTED,
    errors.UNSUPPORTED_FORMAT,
    errors.INVALID_AUDIO,
    errors.BUFFER_OVERFLOW,
    errors.SESSION_TERMINAL,
    errors.INTERNAL,
)


# @spec ING-ERR-001
def test_catalog_covers_every_code_with_every_dialect_column() -> None:
    table = catalog()
    assert set(table) == set(ALL_CODES)
    for code, projection in table.items():
        assert projection.grpc_status, code
        assert projection.nim, code
        # The vLLM dialect is fixed pcm16 @ 16 kHz: unsupported_format
        # is the one condition it cannot express.
        if code == errors.UNSUPPORTED_FORMAT:
            assert projection.vllm is None
        else:
            assert projection.vllm, code


# @spec ING-ERR-002
def test_capacity_codes_are_three_distinct_codes() -> None:
    three = {
        errors.BUSY,
        errors.ADMISSION_WAIT_TIMEOUT,
        errors.IDLE_TIMEOUT,
    }
    assert len(three) == 3
    table = catalog()
    assert table[errors.BUSY].grpc_status == "RESOURCE_EXHAUSTED"
    assert (
        table[errors.ADMISSION_WAIT_TIMEOUT].grpc_status == "DEADLINE_EXCEEDED"
    )
    assert table[errors.IDLE_TIMEOUT].grpc_status == "ABORTED"


# @spec ING-ERR-001
def test_pin_inherited_vllm_codes_project_by_name() -> None:
    table = catalog()
    assert table[errors.PROTOCOL_ORDER].vllm == "model_not_validated"
    assert table[errors.INVALID_AUDIO].vllm == "invalid_audio"
    assert table[errors.INTERNAL].vllm == "processing_error"


# @spec ING-ERR-001
def test_grpc_projections_match_the_catalog_table() -> None:
    table = catalog()
    expected = {
        errors.PROTOCOL_ORDER: "FAILED_PRECONDITION",
        errors.INVALID_CONFIG_FIELD: "INVALID_ARGUMENT",
        errors.UNSUPPORTED_CAPABILITY: "UNIMPLEMENTED",
        errors.UNKNOWN_LOCALE: "INVALID_ARGUMENT",
        errors.UNSUPPORTED_FORMAT: "INVALID_ARGUMENT",
        errors.BUFFER_OVERFLOW: "RESOURCE_EXHAUSTED",
        errors.INTERNAL: "INTERNAL",
    }
    for code, status in expected.items():
        assert table[code].grpc_status == status, code
