"""NIM-shaped HTTP shim: aux endpoints, our provenance (ING-SHIM-001..004).

Sans-IO direct-call tests over the response builders and the router;
shapes are pinned to the public NIM ASR HTTP reference. The shim
contains no inference code — its transcription path is an injected
delegate — and its ``/v1/metadata`` reports this deployment's own
provenance, never NIM model-registry identity.
"""

import inspect
import json
from typing import Any

import pytest

from nemotron_asr_ingress import nim_http
from nemotron_asr_ingress.nim_http import (
    AUX_PATHS,
    NimHttpShim,
    api_validation_error,
    health_live,
    health_ready,
    metadata_response,
    models_response,
    request_parse_error,
    version_response,
)

MODEL = "nemotron-asr"

PROVENANCE: dict[str, str] = {
    "image": "docker.io/example/nemotron-omni-port:dev-v0.24.0",
    "pin": "vllm-omni v0.24.0",
    "precision_policy_id": "pp-d479361445b4",
}


def make_shim(
    ready: bool = True, transcript: str = "hello world"
) -> tuple[NimHttpShim, list[tuple[bytes, str]]]:
    calls: list[tuple[bytes, str]] = []

    def delegate(audio: bytes, language: str) -> str:
        calls.append((audio, language))
        return transcript

    shim = NimHttpShim(
        model_name=MODEL,
        release="0.1.0",
        api="3.1.0",
        provenance=PROVENANCE,
        ready=lambda: ready,
        transcriptions=delegate,
    )
    return shim, calls


# ---- contract greens ---------------------------------------------------------


# @spec ING-SHIM-001
def test_aux_paths_are_exactly_the_five_of_the_spec() -> None:
    assert {
        "/v1/health/ready",
        "/v1/health/live",
        "/v1/models",
        "/v1/version",
        "/v1/metadata",
    } == AUX_PATHS


# @spec ING-SHIM-004
def test_shim_module_imports_no_inference_or_serving_code() -> None:
    source = inspect.getsource(nim_http)
    for forbidden in (
        "vllm_omni",
        "vllm.",
        "import torch",
        "import grpc",
    ):
        assert forbidden not in source


# ---- ING-SHIM-001: the aux endpoint shapes -----------------------------------


# @spec ING-SHIM-001
def test_health_ready_matches_the_reference_body() -> None:
    status, body = health_ready(True)
    assert status == 200
    assert body == {
        "object": "health.response",
        "message": "ready",
        "status": "ready",
    }


# @spec ING-SHIM-001
def test_health_not_ready_is_503_with_a_status_object() -> None:
    status, body = health_ready(False)
    assert status == 503
    # The sidecar-probe lesson: the body is a JSON object carrying a
    # string status, never a bare boolean.
    assert isinstance(body, dict)
    assert isinstance(body["status"], str)
    assert body["status"] != "ready"


# @spec ING-SHIM-001
def test_health_live_matches_the_reference_body() -> None:
    status, body = health_live(True)
    assert status == 200
    assert body == {
        "object": "health.response",
        "message": "live",
        "status": "live",
    }
    not_live_status, not_live_body = health_live(False)
    assert not_live_status == 503
    assert isinstance(not_live_body["status"], str)


# @spec ING-SHIM-001
def test_models_is_the_openai_list_shape_with_our_model() -> None:
    body = models_response(MODEL)
    assert body["object"] == "list"
    (entry,) = body["data"]
    assert entry["id"] == MODEL  # honest where the reference says "unknown"
    assert entry["object"] == "model"
    assert set(entry) >= {"id", "object", "created", "owned_by"}


# @spec ING-SHIM-001
def test_version_carries_release_and_api() -> None:
    body = version_response("0.1.0", "3.1.0")
    assert body == {"release": "0.1.0", "api": "3.1.0"}


# ---- ING-SHIM-002: metadata is ours ------------------------------------------


# @spec ING-SHIM-002
def test_metadata_reports_our_provenance() -> None:
    body = metadata_response(MODEL, "0.1.0", PROVENANCE)
    serialized = json.dumps(body)
    assert PROVENANCE["image"] in serialized
    assert PROVENANCE["precision_policy_id"] in serialized
    (model_info,) = body["modelInfo"]
    assert MODEL in model_info["shortName"]


# @spec ING-SHIM-002
def test_metadata_never_reproduces_nim_registry_identity() -> None:
    body = metadata_response(MODEL, "0.1.0", PROVENANCE)
    serialized = json.dumps(body)
    assert "ngc://" not in serialized
    assert "selectedModelProfileId" not in body


# ---- ING-SHIM-003: the two error-body shapes ---------------------------------


# @spec ING-SHIM-003
def test_api_validation_error_is_the_detail_shape() -> None:
    status, body = api_validation_error("Bad Request, need model or language")
    assert status == 400
    assert body == {"detail": "Bad Request, need model or language"}


# @spec ING-SHIM-003
def test_request_parse_error_is_the_nested_error_shape() -> None:
    status, body = request_parse_error(
        "file: Field required", "BadRequestError", 400
    )
    assert status == 400
    assert body == {
        "error": {
            "message": "file: Field required",
            "type": "BadRequestError",
            "code": 400,
        }
    }


# ---- the router --------------------------------------------------------------


# @spec ING-SHIM-001
def test_router_serves_every_aux_path() -> None:
    shim, _calls = make_shim(ready=True)
    for path in AUX_PATHS:
        result = shim.handle_get(path)
        assert result is not None, path
        status, body = result
        assert status == 200
        assert isinstance(body, dict)


# @spec ING-SHIM-001
def test_router_returns_none_for_paths_it_does_not_own() -> None:
    shim, _calls = make_shim()
    assert shim.handle_get("/v1/audio/transcriptions") is None
    assert shim.handle_get("/v2/anything") is None


# @spec ING-SHIM-001
def test_router_ready_follows_the_readiness_seam() -> None:
    shim, _calls = make_shim(ready=False)
    result = shim.handle_get("/v1/health/ready")
    assert result is not None
    status, _body = result
    assert status == 503
    live = shim.handle_get("/v1/health/live")
    assert live is not None
    assert live[0] == 200  # liveness is process-level, not model-level


# ---- ING-SHIM-004: delegated inference ---------------------------------------


# @spec ING-SHIM-004
def test_transcriptions_delegates_and_shapes_json() -> None:
    shim, calls = make_shim(transcript="What is natural language processing?")
    status, body = shim.handle_transcriptions(
        {"file": b"\x00\x01", "language": "en-US"}
    )
    assert status == 200
    assert body == {"text": "What is natural language processing?"}
    assert calls == [(b"\x00\x01", "en-US")]


# @spec ING-SHIM-004
def test_transcriptions_text_format_returns_the_bare_transcript() -> None:
    shim, _calls = make_shim(transcript="hello world")
    status, body = shim.handle_transcriptions(
        {"file": b"\x00\x01", "language": "en-US", "response_format": "text"}
    )
    assert status == 200
    assert body == "hello world"


# @spec ING-SHIM-003, ING-SHIM-004
def test_missing_file_is_the_parse_error_shape() -> None:
    shim, calls = make_shim()
    status, body = shim.handle_transcriptions({"language": "en-US"})
    assert status == 400
    assert isinstance(body, dict)
    assert "file" in body["error"]["message"]
    assert calls == []  # never delegated


# @spec ING-SHIM-003, ING-SHIM-004
def test_missing_language_and_model_is_the_validation_error_shape() -> None:
    shim, calls = make_shim()
    status, body = shim.handle_transcriptions({"file": b"\x00\x01"})
    assert status == 400
    assert isinstance(body, dict)
    assert set(body) == {"detail"}
    assert calls == []


# @spec ING-SHIM-004
@pytest.mark.parametrize("key", ["language", "model"])
def test_language_or_model_alone_satisfies_validation(key: str) -> None:
    # The reference's own error text: "need model or language" —
    # either one admits the request.
    shim, calls = make_shim(transcript="ok")
    form: dict[str, Any] = {"file": b"\x00\x01", key: "en-US"}
    status, _body = shim.handle_transcriptions(form)
    assert status == 200
    assert len(calls) == 1
