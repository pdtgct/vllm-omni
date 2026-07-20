"""The NIM-shaped HTTP shim: NIM's auxiliary surface, our provenance.

A drop-in *target* for NIM HTTP client code paths, never a
counterfeit (ING-SHIM-002): the five auxiliary endpoints answer with
NIM's response shapes — the public NIM ASR HTTP reference
(docs.nvidia.com/nim/speech — ASR HTTP REST API Reference) — while
``/v1/metadata`` reports this deployment's own provenance (image,
PIN, PrecisionPolicy identifier) and never reproduces NIM
model-registry identity (``ngc://`` URLs, NIM profile hashes).

Sans-IO like the ingress adapters: response builders return
``(status, body)`` pairs and the router maps paths — the serving
shell owns sockets and HTTP framing. The shim contains no inference
code (ING-SHIM-004): its transcription path delegates to an injected
handler — the in-process ``/v1/audio/transcriptions`` equivalent or
a remote endpoint — and owns only the NIM shapes around it, including
the reference's two documented error-body formats keyed by error
source (ING-SHIM-003): API-level validation errors are
``{"detail": …}``; request parsing errors are
``{"error": {"message", "type", "code"}}``.
"""

from collections.abc import Callable, Mapping
from typing import Any

#: The five auxiliary paths the shim serves (ING-SHIM-001).
AUX_PATHS: frozenset[str] = frozenset(
    {
        "/v1/health/ready",
        "/v1/health/live",
        "/v1/models",
        "/v1/version",
        "/v1/metadata",
    }
)


# @spec ING-SHIM-001
def health_ready(ready: bool) -> tuple[int, dict[str, str]]:
    """``GET /v1/health/ready``: 200 when ready, 503 when not.

    The body is always a JSON object carrying ``status`` — never a
    bare boolean (the sidecar-probe lesson, ING-SHIM-001); the ready
    body matches the reference exactly
    (``{"object": "health.response", "message": "ready",
    "status": "ready"}``).
    """
    if ready:
        return (
            200,
            {
                "object": "health.response",
                "message": "ready",
                "status": "ready",
            },
        )
    return (
        503,
        {
            "object": "health.response",
            "message": "not ready",
            "status": "not_ready",
        },
    )


# @spec ING-SHIM-001
def health_live(live: bool) -> tuple[int, dict[str, str]]:
    """``GET /v1/health/live``: the liveness analog of ready."""
    if live:
        return (
            200,
            {
                "object": "health.response",
                "message": "live",
                "status": "live",
            },
        )
    return (
        503,
        {
            "object": "health.response",
            "message": "not live",
            "status": "not_live",
        },
    )


# @spec ING-SHIM-001
def models_response(model_name: str) -> dict[str, Any]:
    """``GET /v1/models``: the OpenAI-compatible model list.

    One entry whose ``id`` is the served model's name — honest where
    the reference example says ``"unknown"``.
    """
    return {
        "object": "list",
        "data": [
            {
                "id": model_name,
                "object": "model",
                "created": 0,
                "owned_by": "system",
            }
        ],
    }


# @spec ING-SHIM-001
def version_response(release: str, api: str) -> dict[str, str]:
    """``GET /v1/version``: ``{"release", "api"}`` with our values."""
    return {"release": release, "api": api}


# @spec ING-SHIM-002
def metadata_response(
    model_name: str, release: str, provenance: Mapping[str, Any]
) -> dict[str, Any]:
    """``GET /v1/metadata``: NIM's shape, this deployment's identity.

    ``modelInfo`` names the served model without an ``ngc://`` URL;
    the provenance mapping (image, PIN, PrecisionPolicy identifier)
    rides the body; no NIM profile hash appears anywhere
    (ING-SHIM-002).
    """
    return {
        "version": release,
        "modelInfo": [{"modelUrl": "", "shortName": model_name}],
        "repository_override": "",
        "assetInfo": [],
        "licenseInfo": {},
        "provenance": dict(provenance),
    }


# @spec ING-SHIM-003
def api_validation_error(detail: str) -> tuple[int, dict[str, str]]:
    """An API-level validation error: ``{"detail": …}`` @ 400.

    The reference's first documented error-body shape (bad parameter
    value, missing ``language``/``model``).
    """
    return (400, {"detail": detail})


# @spec ING-SHIM-003
def request_parse_error(
    message: str, error_type: str, status: int
) -> tuple[int, dict[str, Any]]:
    """A request parsing error: the nested ``error`` object shape.

    The reference's second documented error-body shape (missing
    required ``file`` field): ``{"error": {"message", "type",
    "code"}}`` with the HTTP status repeated as ``code``.
    """
    return (
        status,
        {"error": {"message": message, "type": error_type, "code": status}},
    )


# @spec ING-SHIM-001, ING-SHIM-004
class NimHttpShim:
    """The shim's sans-IO router: aux endpoints + delegated inference.

    ``ready`` is the readiness probe seam (the serving shell knows
    whether the kernel is standing); ``transcriptions`` is the
    inference delegate (ING-SHIM-004) — it receives the decoded form
    fields and returns the transcript text; the shim owns request
    validation, the NIM response shapes, and the two error-body
    formats (ING-SHIM-003). This module imports no inference or
    serving code.
    """

    def __init__(
        self,
        model_name: str,
        release: str,
        api: str,
        provenance: Mapping[str, Any],
        ready: Callable[[], bool],
        transcriptions: Callable[[bytes, str], str],
    ) -> None:
        """Bind identity values, the readiness seam, and the delegate."""
        self._model_name = model_name
        self._release = release
        self._api = api
        self._provenance = provenance
        self._ready = ready
        self._transcriptions = transcriptions

    # @spec ING-SHIM-001
    def handle_get(self, path: str) -> tuple[int, dict[str, Any]] | None:
        """Route one GET; ``None`` for paths the shim does not own."""
        if path == "/v1/health/ready":
            return health_ready(self._ready())
        if path == "/v1/health/live":
            # Liveness is process-level: reachable means live; model
            # readiness is the ready endpoint's question.
            return health_live(True)
        if path == "/v1/models":
            return (200, models_response(self._model_name))
        if path == "/v1/version":
            return (200, version_response(self._release, self._api))
        if path == "/v1/metadata":
            return (
                200,
                metadata_response(
                    self._model_name, self._release, self._provenance
                ),
            )
        return None

    # @spec ING-SHIM-003, ING-SHIM-004
    def handle_transcriptions(
        self, form: Mapping[str, Any]
    ) -> tuple[int, dict[str, Any] | str]:
        """``POST /v1/audio/transcriptions``: validate, delegate, shape.

        Missing ``file`` -> the parsing-error shape @ 400; neither
        ``language`` nor ``model`` -> the validation-error shape @ 400
        (both per the reference's own examples); otherwise the
        delegate transcribes and the response follows
        ``response_format`` — ``{"text": …}`` for ``json`` (the
        default), the bare transcript string for ``text``.
        """
        if "file" not in form:
            return request_parse_error(
                "file: Field required", "BadRequestError", 400
            )
        if not form.get("language") and not form.get("model"):
            return api_validation_error("Bad Request, need model or language")
        transcript = self._transcriptions(
            form["file"], str(form.get("language") or "")
        )
        if form.get("response_format") == "text":
            return (200, transcript)
        return (200, {"text": transcript})
