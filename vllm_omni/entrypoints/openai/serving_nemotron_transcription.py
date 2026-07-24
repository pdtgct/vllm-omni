# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The Nemotron ``/v1/audio/transcriptions`` adapter (PORT-INT-006).

Subclasses core's ``OpenAIServingTranscription`` and replaces ONLY the
execution leg: instead of the stock prompt-splitting +
``engine_client.generate`` path, the request delegates to the canonical
ephemeral orchestrator ``transcribe_ephemeral`` with the concrete
``NemotronSessionFactory`` (RFC-1 brief §C) — one streaming session at
the model's canonical geometry, never a full-context pass
(PORT-REGIME-002).

**Override point: ``create_transcription`` (the narrowest method
boundary).** The base's "execution leg" is not a separable method — it
is inline in ``_create_speech_to_text`` between preprocessing and
response assembly — so overriding lower would mean re-implementing that
whole body anyway. Base behavior is kept DELIBERATELY, per the brief's
checklist, by re-invoking the base's own machinery:

- upload limits: the router's ``read_upload_with_limit`` bounds the
  upload before this handler runs, and the base's
  ``max_audio_filesize_mb`` check is repeated here verbatim;
- decoding/resampling: ``self._decode_and_chunk_speech_async`` (the
  base's executor-backed decode; the model's speech-to-text config
  disables window chunking, so it yields ONE resampled clip);
- request IDs and metadata: ``self._base_request_id`` plus the base's
  ``RequestResponseMetadata`` stamping;
- model check / engine-dead check: the base's ``_check_model`` and
  ``engine_client.errored`` guards, in the base's order.

The eager base ``__init__`` runs un-bypassed: it calls the model's
``get_speech_to_text_config`` (the classmethod the model now provides),
which this adapter genuinely needs — ``asr_config.sample_rate`` sizes
the pre-submit bound and drives the decode resample.

What the stock path had that this adapter consciously REJECTS (named
errors via the pure rules sibling ``nemotron_transcription_rules``,
never silent): streaming, ``verbose_json``/``srt``/``vtt``, timestamp
granularities, beam search, hotwords, non-empty ``prompt``,
``to_language``, and non-default sampling knobs (the RNN-T decode is
pinned greedy in-model, PORT-DEC-005/007 — ``post_process_output`` and
inter-chunk separators are likewise not applicable: the session's
transcript is already final).

Deadlines: an HTTP transcription request carries NO transport deadline
of its own — client disconnect arrives as CANCELLATION, not a deadline
(brief §C round-5 note). The adapter passes ``deadline=None`` and the
configured finalization limit governs; a cancellation propagates into
``transcribe_ephemeral``, whose cleanup takes the orchestrator's abort
path (abort then release) before re-raising.

Capacity: ``AdmissionBusyError`` maps to the SHARED named capacity
error (HTTP 429, PORT-STATE-004) — the limiter behind it is load
shedding only, never residency (see ``ServingConcurrencyLimiter``'s
disclaimers). ``FinalizationTimeoutError`` maps to 504.

This module imports vllm and is therefore pod-gated (ruff/mypy +
importorskip'd tests on CPU); every PURE decision lives in the
GPU-free rules sibling. Tenet 3 holds: nothing here imports
``nemotron_asr_ingress``.
"""

from __future__ import annotations

import math
import os
from http import HTTPStatus
from typing import TYPE_CHECKING, Any, cast

from fastapi import Request
from vllm.engine.protocol import EngineClient
from vllm.entrypoints.openai.engine.protocol import RequestResponseMetadata
from vllm.entrypoints.openai.models.serving import OpenAIServingModels
from vllm.entrypoints.serve.utils.request_logger import RequestLogger
from vllm.entrypoints.speech_to_text.transcription.protocol import (
    TranscriptionRequest,
    TranscriptionResponse,
)
from vllm.entrypoints.speech_to_text.transcription.serving import (
    OpenAIServingTranscription,
)
from vllm.exceptions import VLLMValidationError
from vllm.logger import init_logger

from vllm_omni.entrypoints.ephemeral_session import transcribe_ephemeral
from vllm_omni.entrypoints.nemotron_session import (
    NemotronSessionFactory,
    ServingConcurrencyLimiter,
)
from vllm_omni.entrypoints.openai.nemotron_transcription_rules import (
    first_rejection,
    map_orchestrator_error,
    normalize_language,
    parse_positive_finite_env,
)

if TYPE_CHECKING:
    from vllm_omni.entrypoints.ephemeral_session import SessionFactory

logger = init_logger(__name__)

#: The mandatory finite finalization bound (ING-LIFE-005) — how long the
#: final-tail drain (flush + finish) may take before the session is
#: aborted and the request answers 504. ENV-tunable, like every tunable.
FINALIZATION_TIMEOUT_ENV = "VLLM_OMNI_NEMOTRON_FINALIZATION_TIMEOUT_S"
_FINALIZATION_TIMEOUT_S_DEFAULT = 30.0

#: The serving layer's pre-submit bound in SECONDS of audio per ``feed``
#: call (ING-FE-005). A bound on one submission's size — deliberately
#: NOT a cadence: the model's segmenter owns all cadence arithmetic
#: (ING-FE-006). Samples are derived from the model's own
#: ``asr_config.sample_rate`` at construction.
SUBMIT_BOUND_ENV = "VLLM_OMNI_NEMOTRON_SUBMIT_BOUND_S"
_SUBMIT_BOUND_S_DEFAULT = 10.0


def _finalization_timeout_s() -> float:
    """The configured finalization bound, validated positive-finite."""
    return parse_positive_finite_env(
        name=FINALIZATION_TIMEOUT_ENV,
        raw=os.getenv(
            FINALIZATION_TIMEOUT_ENV, str(_FINALIZATION_TIMEOUT_S_DEFAULT)
        ),
    )


# @spec ING-FE-005
def _submit_bound_s() -> float:
    """The configured pre-submit bound, validated positive-finite."""
    return parse_positive_finite_env(
        name=SUBMIT_BOUND_ENV,
        raw=os.getenv(SUBMIT_BOUND_ENV, str(_SUBMIT_BOUND_S_DEFAULT)),
    )


# @spec PORT-INT-006, PORT-REGIME-002, ING-FE-005
class NemotronServingTranscription(OpenAIServingTranscription):
    """``OpenAIServingTranscription`` with a delegated execution leg.

    See the module docstring for the kept-vs-rejected base behavior and
    the override-point justification. One instance is constructed by
    the flag-guarded api_server wiring (never by the capability
    classvar — the task stays off by default, brief §D) and holds the
    concrete factory bound to the SHARED app-scope limiter.
    """

    def __init__(
        self,
        engine_client: EngineClient,
        models: OpenAIServingModels,
        *,
        request_logger: RequestLogger | None,
        limiter: ServingConcurrencyLimiter,
        factory: SessionFactory | None = None,
        return_tokens_as_token_ids: bool = False,
        enable_force_include_usage: bool = False,
    ) -> None:
        """Construct over the eager base, then bind the delegation seam.

        Args:
            engine_client: The ``AsyncOmni`` handle; also the engine the
                concrete factory drives.
            models: The serving-models registry (base plumbing).
            request_logger: The base's request logger.
            limiter: The SHARED app-scope ``ServingConcurrencyLimiter``
                (one instance across every transport, injected at app
                init — never constructed per-transport).
            factory: Session-factory override for tests; ``None`` builds
                the real ``NemotronSessionFactory`` over ``limiter``.
            return_tokens_as_token_ids: Base passthrough.
            enable_force_include_usage: Base passthrough.
        """
        # The eager base __init__ is NOT bypassed: it resolves
        # asr_config via the model's get_speech_to_text_config, which
        # sizes the resample target and the pre-submit bound below.
        super().__init__(
            engine_client,
            models,
            request_logger=request_logger,
            return_tokens_as_token_ids=return_tokens_as_token_ids,
            enable_force_include_usage=enable_force_include_usage,
        )
        self._factory: SessionFactory = (
            factory
            if factory is not None
            else NemotronSessionFactory(engine=engine_client, limiter=limiter)
        )
        self._finalization_timeout_s = _finalization_timeout_s()
        self._submit_bound_samples = max(
            1, int(self.asr_config.sample_rate * _submit_bound_s())
        )

    async def create_transcription(
        self,
        audio_data: bytes,
        request: TranscriptionRequest,
        raw_request: Request | None = None,
    ) -> Any:
        """One clip in, exactly one final transcription out.

        The base's guard order is preserved (honest-subset rejection
        first, then model check, engine-dead check, filesize, decode),
        then the execution leg delegates to ``transcribe_ephemeral``.

        Args:
            audio_data: The uploaded clip (already bounded by the
                router's ``read_upload_with_limit``).
            request: The parsed OpenAI transcription request.
            raw_request: The FastAPI request, when serving HTTP.

        Returns:
            A ``TranscriptionResponse`` (json/text formats share the
            base's JSON body shape) or a named ``ErrorResponse``.
        """
        rejection = first_rejection(
            response_format=request.response_format,
            stream=bool(request.stream),
            timestamp_granularities=request.timestamp_granularities,
            use_beam_search=request.use_beam_search,
            n=request.n,
            length_penalty=request.length_penalty,
            include_stop_str_in_output=request.include_stop_str_in_output,
            vllm_xargs=request.vllm_xargs,
            hotwords=request.hotwords,
            prompt=request.prompt,
            to_language=request.to_language,
            temperature=request.temperature,
            top_p=request.top_p,
            top_k=request.top_k,
            min_p=request.min_p,
            seed=request.seed,
            frequency_penalty=request.frequency_penalty,
            repetition_penalty=request.repetition_penalty,
            presence_penalty=request.presence_penalty,
            max_completion_tokens=request.max_completion_tokens,
        )
        if rejection is not None:
            return self.create_error_response(
                rejection.message,
                err_type=rejection.err_type,
                status_code=HTTPStatus(rejection.status_code),
                param=rejection.param,
            )

        error_check_ret = await self._check_model(request)
        if error_check_ret is not None:
            return error_check_ret
        if not request.model:
            request.model = self.models.model_name()
        if self.engine_client.errored:
            raise self.engine_client.dead_error

        # None -> "auto", then the model's checkpoint-locale gate: the
        # prompt dictionary is the ONE authority (PORT-LID-001). The
        # model_config kwarg is the Nemotron extension of the protocol
        # signature, so the call goes through Any.
        try:
            locale: str = cast(Any, self.model_cls).validate_language(
                normalize_language(request.language),
                model_config=self.model_config,
            )
        except ValueError as exc:
            return self.create_error_response(
                str(exc),
                err_type="BadRequestError",
                status_code=HTTPStatus.BAD_REQUEST,
                param="language",
            )

        # The base's own filesize guard, kept verbatim (the router's
        # read_upload_with_limit already bounds HTTP uploads; this keeps
        # programmatic callers bounded too).
        if len(audio_data) / 1024**2 > self.max_audio_filesize_mb:
            raise VLLMValidationError(
                "Maximum file size exceeded",
                parameter="audio_filesize_mb",
                value=len(audio_data) / 1024**2,
            )

        request_id = f"{self.task_type}-{self._base_request_id(raw_request)}"
        request_metadata = RequestResponseMetadata(request_id=request_id)
        if raw_request:
            raw_request.state.request_metadata = request_metadata

        # The base's decode/resample, kept deliberately: the model's
        # speech-to-text config disables window chunking, so this yields
        # one clip resampled to the model's sample rate.
        chunks, duration_s = await self._decode_and_chunk_speech_async(
            audio_data
        )

        try:
            result = await transcribe_ephemeral(
                chunks,
                factory=self._factory,
                locale=locale,
                finalization_timeout_s=self._finalization_timeout_s,
                submit_bound_samples=self._submit_bound_samples,
                # HTTP carries no transport deadline; client disconnect
                # arrives as cancellation and takes the orchestrator's
                # abort path (brief §C).
                deadline=None,
            )
        except Exception as error:
            mapped = map_orchestrator_error(error)
            if mapped is None:
                raise
            logger.info(
                "Transcription request %s rejected: %s",
                request_id,
                mapped.message,
            )
            return self.create_error_response(
                mapped.message,
                err_type=mapped.err_type,
                status_code=HTTPStatus(mapped.status_code),
            )

        # The base's usage shape: duration seconds, rounded up per the
        # OpenAI spec. json and text share the base's JSON body.
        usage = {
            "type": "duration",
            "seconds": int(math.ceil(duration_s)),
        }
        return TranscriptionResponse(text=result.text, usage=usage)


__all__ = [
    "FINALIZATION_TIMEOUT_ENV",
    "NemotronServingTranscription",
    "SUBMIT_BOUND_ENV",
]
