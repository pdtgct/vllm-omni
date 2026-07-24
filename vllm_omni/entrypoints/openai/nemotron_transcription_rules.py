# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Pure decision rules for the Nemotron transcription adapter.

The import-light sibling of ``serving_nemotron_transcription.py``
(RFC-1 brief §C): every decision the HTTP adapter makes that does NOT
need the engine — the honest-subset rejection table, language
normalization, and the orchestrator-error -> HTTP mapping — lives here
as pure functions over plain values, so the contracts run GPU-free by
file-path loading while the vllm-coupled adapter module stays
pod-gated. Imports are stdlib plus the engine-free orchestrator module
(``ephemeral_session``) only; adding a vllm/torch/numpy import here
breaks the GPU-free tier by construction.

The honest-subset posture (brief §C): a request control the delegated
path cannot honor draws a NAMED rejection — parameter, message, and
status — never silent acceptance. The subset is deliberately small:
``json``/``text`` responses of a non-streamed, greedy, single-result
transcription; everything else names itself in the error.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from http import HTTPStatus

from vllm_omni.entrypoints.ephemeral_session import (
    AdmissionBusyError,
    FinalizationTimeoutError,
)

#: The response formats the delegated path can honor. ``verbose_json``
#: needs segments/timestamps the RNN-T session does not produce in this
#: slice; ``srt``/``vtt`` are not supported by the stock path either.
SUPPORTED_RESPONSE_FORMATS: frozenset[str] = frozenset({"json", "text"})

#: The adapter's locale for an absent request language (brief §C:
#: ``None -> "auto"``). "auto" is the checkpoint's own
#: language-identification locale; the model's ``validate_language``
#: gate confirms membership against the served prompt dictionary.
AUTO_LOCALE = "auto"

#: Sampling knobs the pinned greedy decode cannot honor, with the one
#: value (their schema default) that means "unset". Anything else draws
#: a named rejection: the RNN-T decode is greedy in-model
#: (PORT-DEC-005/007), so a knob that cannot take effect must not be
#: silently accepted.
FIXED_DECODE_DEFAULTS: tuple[tuple[str, float | None], ...] = (
    ("temperature", 0.0),
    ("top_p", None),
    ("top_k", None),
    ("min_p", None),
    ("seed", None),
    ("frequency_penalty", 0.0),
    ("repetition_penalty", None),
    ("presence_penalty", 0.0),
    ("max_completion_tokens", None),
)


# @spec ING-FE-005
def parse_positive_finite_env(*, name: str, raw: str) -> float:
    """Parse one positive-finite ENV tuning value.

    Args:
        name: Environment-variable name, for diagnostics.
        raw: Raw environment value.

    Returns:
        The parsed positive finite float.

    Raises:
        ValueError: If ``raw`` is not numeric, finite, and positive.
    """
    try:
        value = float(raw)
    except ValueError as exc:
        raise ValueError(f"{name}={raw!r} is not a number") from exc
    if not math.isfinite(value) or value <= 0:
        raise ValueError(
            f"{name} must be finite and positive, got {value}"
        )
    return value


@dataclass(frozen=True)
class RequestRejection:
    """One named rejection of an unsupported request control.

    Attributes:
        param: The request parameter being rejected (the error's
            ``param`` field — the "named" in "named error").
        message: Why the delegated path cannot honor it.
        status_code: HTTP status (400 for every request-shape reject).
        err_type: The vllm error-body ``type`` string.
    """

    param: str
    message: str
    status_code: int = HTTPStatus.BAD_REQUEST.value
    err_type: str = "BadRequestError"


@dataclass(frozen=True)
class MappedError:
    """One orchestrator failure projected to its HTTP error identity."""

    status_code: int
    err_type: str
    message: str


# @spec PORT-INT-006
def first_rejection(
    *,
    response_format: str = "json",
    stream: bool = False,
    timestamp_granularities: Sequence[str] | None = None,
    use_beam_search: bool = False,
    n: int = 1,
    length_penalty: float = 1.0,
    include_stop_str_in_output: bool = False,
    vllm_xargs: Mapping[str, str | int | float | bool] | None = None,
    hotwords: str | None = None,
    prompt: str = "",
    to_language: str | None = None,
    temperature: float = 0.0,
    top_p: float | None = None,
    top_k: int | None = None,
    min_p: float | None = None,
    seed: int | None = None,
    frequency_penalty: float | None = 0.0,
    repetition_penalty: float | None = None,
    presence_penalty: float | None = 0.0,
    max_completion_tokens: int | None = None,
) -> RequestRejection | None:
    """Return the first named rejection for an unsupported control.

    Defaults mirror the ``TranscriptionRequest`` schema defaults, so a
    request at defaults passes the whole table and callers override
    only what they set. Check order is fixed but not contractual — any
    single unsupported control suffices to reject.

    Args:
        response_format: Requested output format.
        stream: SSE streaming flag (no streaming in this slice).
        timestamp_granularities: Requested timestamp granularities.
        use_beam_search: Beam-search opt-in.
        n: Beam count (only meaningful with beam search; 1 is default).
        length_penalty: Beam length penalty (unsupported).
        include_stop_str_in_output: Stop-string rendering control
            (unsupported).
        vllm_xargs: Custom extension arguments (unsupported).
        hotwords: Bias phrases the model does not support.
        prompt: Style/continuation prompt (empty string means unset).
        to_language: Target language (a translation control).
        temperature: Sampling knob; the decode is pinned greedy.
        top_p: Sampling knob; the decode is pinned greedy.
        top_k: Sampling knob; the decode is pinned greedy.
        min_p: Sampling knob; the decode is pinned greedy.
        seed: Sampling knob; the decode is deterministic.
        frequency_penalty: Sampling knob; the decode is pinned greedy.
        repetition_penalty: Sampling knob; the decode is pinned greedy.
        presence_penalty: Sampling knob; the decode is pinned greedy.
        max_completion_tokens: Generation cap the session ignores.

    Returns:
        The first :class:`RequestRejection`, or ``None`` when every
        control is within the honest subset.
    """
    if response_format not in SUPPORTED_RESPONSE_FORMATS:
        return RequestRejection(
            param="response_format",
            message=(
                f"response_format {response_format!r} is not supported; "
                f"supported: {sorted(SUPPORTED_RESPONSE_FORMATS)}"
            ),
        )
    if stream:
        return RequestRejection(
            param="stream",
            message=(
                "stream=true is not supported: this endpoint returns "
                "exactly one final transcription (no streaming in this "
                "slice)"
            ),
        )
    if timestamp_granularities:
        return RequestRejection(
            param="timestamp_granularities",
            message=(
                "timestamp_granularities are not supported: the model "
                "produces no word/segment timestamps"
            ),
        )
    if use_beam_search:
        return RequestRejection(
            param="use_beam_search",
            message=(
                "beam search is not supported: the RNN-T decode is a "
                "pinned greedy decode"
            ),
        )
    if n != 1:
        return RequestRejection(
            param="n",
            message=(
                f"n={n} is not supported: the greedy decode returns "
                "exactly one hypothesis"
            ),
        )
    if length_penalty != 1.0:
        return RequestRejection(
            param="length_penalty",
            message=(
                f"length_penalty={length_penalty!r} is not supported: "
                "the RNN-T decode is pinned greedy"
            ),
        )
    if include_stop_str_in_output:
        return RequestRejection(
            param="include_stop_str_in_output",
            message=(
                "include_stop_str_in_output is not supported: the "
                "session returns an already-final RNN-T transcript"
            ),
        )
    if vllm_xargs:
        return RequestRejection(
            param="vllm_xargs",
            message=(
                "vllm_xargs are not supported by the delegated "
                "Nemotron transcription path"
            ),
        )
    if hotwords:
        return RequestRejection(
            param="hotwords",
            message="hotwords are not supported by this model",
        )
    if prompt:
        return RequestRejection(
            param="prompt",
            message=(
                "prompt is not supported: the session's conditioning is "
                "the checkpoint's own locale prompt, selected via "
                "'language'"
            ),
        )
    if to_language is not None:
        return RequestRejection(
            param="to_language",
            message=(
                "to_language is a translation control; this model does "
                "not translate"
            ),
        )
    provided: dict[str, float | int | None] = {
        "temperature": temperature,
        "top_p": top_p,
        "top_k": top_k,
        "min_p": min_p,
        "seed": seed,
        "frequency_penalty": frequency_penalty,
        "repetition_penalty": repetition_penalty,
        "presence_penalty": presence_penalty,
        "max_completion_tokens": max_completion_tokens,
    }
    for knob, default in FIXED_DECODE_DEFAULTS:
        value = provided[knob]
        # ``None`` always means "unset"; otherwise only the schema
        # default is acceptable — the greedy decode honors nothing else.
        if value is not None and value != default:
            return RequestRejection(
                param=knob,
                message=(
                    f"{knob}={value!r} is not supported: the RNN-T "
                    "decode is pinned greedy and deterministic "
                    "(sampling controls have no effect)"
                ),
            )
    return None


# @spec PORT-LID-001, PORT-REGIME-003
def normalize_language(language: str | None) -> str:
    """Map the OpenAI request ``language`` to a candidate locale.

    ``None``/empty/whitespace -> :data:`AUTO_LOCALE` (brief §C);
    anything else is stripped but otherwise preserved for the model's
    ``validate_language`` gate. The model resolves exact checkpoint
    keys first, then normalized casing and unique ISO-639-1 matches;
    preserving the raw tag here is what lets an exact checkpoint key
    win. This function performs NO membership check: the checkpoint
    prompt dictionary is the model's authority, not the adapter's
    (PORT-LID-001).

    Args:
        language: The raw request field.

    Returns:
        The candidate locale string.
    """
    if language is None or not language.strip():
        return AUTO_LOCALE
    return language.strip()


def map_orchestrator_error(error: BaseException) -> MappedError | None:
    """Project an orchestrator failure to its named HTTP error.

    - :class:`AdmissionBusyError` -> 429 with the SHARED named capacity
      error (PORT-STATE-004: realtime and transcription project the
      same busy condition; the adapter must not invent its own status).
    - :class:`FinalizationTimeoutError` -> 504 (the drain exceeded the
      configured finalization limit; engine state was aborted).
    - Anything else -> ``None``: not this table's to map — the caller
      lets it propagate to the app's generic error handling.

    Args:
        error: The exception raised by ``transcribe_ephemeral``.

    Returns:
        The :class:`MappedError`, or ``None`` when unmapped.
    """
    if isinstance(error, AdmissionBusyError):
        return MappedError(
            status_code=HTTPStatus.TOO_MANY_REQUESTS.value,
            err_type="TooManyRequestsError",
            message=f"transcription capacity reached: {error}",
        )
    if isinstance(error, FinalizationTimeoutError):
        return MappedError(
            status_code=HTTPStatus.GATEWAY_TIMEOUT.value,
            err_type="GatewayTimeoutError",
            message=str(error),
        )
    return None
