from __future__ import annotations

import asyncio
import functools
import inspect
from collections.abc import AsyncGenerator
from typing import Any, cast

import numpy as np
from vllm.engine.protocol import StreamingInput
from vllm.entrypoints.speech_to_text.realtime.serving import OpenAIServingRealtime
from vllm.inputs import PromptType
from vllm.renderers.inputs.preprocess import parse_model_prompt


def _resolve_park_token_id(hf_config: Any) -> int:
    """Resolve the park-token id from the model package's own authority.

    A module-level indirection (not an inline import in the property) so
    the resolution seam is patchable and the Nemotron model package is
    imported only when a recognized streaming model is actually served.
    """
    from vllm_omni.model_executor.models.nemotron_asr.rnnt import (
        park_token_id,
    )

    return park_token_id(hf_config)


#: Architecture identities whose park semantics this serving owns.
#: Park-token resolution is gated on MODEL IDENTITY, never on signature
#: shape (review round 2026-07-28, F5): any future realtime model may
#: declare the same three generic observer params for call-shape
#: compatibility without inheriting Nemotron's RNN-T park semantics.
_NEMOTRON_STREAMING_ARCHITECTURES = frozenset({"Nemotron3_5AsrForRNNT"})


class NemotronServingRealtime(OpenAIServingRealtime):
    """Threads the installed PORT-OBS-003 observer and the PORT-SESS-001
    accepted-audio budget into the natively (ledgerless) constructed
    session.

    Phase-6 round 2, Q2 (lead-decided): no vLLM-core change and no
    contextvar side channel. This fork-owned subclass overrides
    ``transcribe_realtime`` with the SAME body as upstream
    ``OpenAIServingRealtime.transcribe_realtime`` — pinned vLLM v0.24.0
    @ ee0da84ab,
    ``vllm/entrypoints/speech_to_text/realtime/serving.py`` lines 55-89
    — except it passes ``observer``/``accepted_audio_budget_s`` through
    to ``buffer_realtime_audio``'s widened optional keyword-only params.
    The fork owns both this subclass and the model classmethod's
    widened signature, so no vLLM core file changes.

    PIN-DRIFT GUARD: re-diff this override's body against the upstream
    ``transcribe_realtime`` method at every vLLM pin bump (env-design.md
    §Version pin governance) — an upstream change to that method this
    override doesn't pick up would silently regress native-path
    observation without failing loudly.
    """

    def __init__(
        self,
        *args: Any,
        observer: Any = None,
        accepted_audio_budget_s: float | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        # PORT-OBS-003/PORT-SESS-001: resolved once at construction
        # (after install — see api_server.py's construction site) and
        # threaded through on every call; absent (``None``) reproduces
        # today's inert behavior exactly.
        self._observer = observer
        self._accepted_audio_budget_s = accepted_audio_budget_s

    @functools.cached_property
    def _model_declares_widened_buffer_kwargs(self) -> bool:
        """Whether ``model_cls.buffer_realtime_audio`` declares the
        fork-widened ``observer``/``accepted_audio_budget_s`` params.

        Only the fork's Nemotron classmethod is widened; every other
        realtime-capable model at the pin (e.g. ``qwen3_omni``'s, and
        upstream's ``SupportsRealtime`` implementations) carries the
        un-widened upstream signature and must receive the *exact*
        upstream call — passing the kwargs unconditionally is a
        ``TypeError`` that breaks a model the pre-metrics base served
        fine. Explicit-declaration check on purpose: a ``**kwargs``
        catch-all does not opt a model into observability it never
        implemented. Resolved once per serving instance (``model_cls``
        is itself cached on the base class).
        """
        params = inspect.signature(self.model_cls.buffer_realtime_audio).parameters
        return (
            "observer" in params
            and "accepted_audio_budget_s" in params
            and "session_key" in params
        )

    @functools.cached_property
    def _is_recognized_streaming_model(self) -> bool:
        """Model-identity gate for park semantics (review F5): the
        architecture list is the identity authority; the widened
        signature remains a call-shape check only."""
        hf_config = getattr(self.model_config, "hf_config", None)
        architectures = getattr(hf_config, "architectures", None) or ()
        return any(arch in _NEMOTRON_STREAMING_ARCHITECTURES for arch in architectures)

    @functools.cached_property
    def park_token_id(self) -> int | None:
        """The model's park-token id, or ``None`` for models this fork
        does not observe.

        Gated on architecture identity, never signature shape: the
        recognized Nemotron streaming model resolves through the model
        package's own authority and FAILS LOUDLY if the authority cannot
        produce a token — a silent ``None`` here would disable every
        connection-layer park observation invisibly, the A27 GPU-round
        defect class. Any other realtime model — including a future one
        that declares the same widened observer params for call-shape
        compatibility — resolves ``None`` and stays fully inert
        (PORT-OBS-003: only park detection depends on the token; cleanup
        and lifecycle completion depend on the observer/session key).
        """
        if not self._is_recognized_streaming_model:
            return None
        return _resolve_park_token_id(self.model_config.hf_config)

    async def transcribe_realtime(
        self,
        audio_stream: AsyncGenerator[np.ndarray, None],
        input_stream: asyncio.Queue[list[int]],
        *,
        session_key: str | None = None,
    ) -> AsyncGenerator[StreamingInput, None]:
        """Transform audio stream into StreamingInput for engine.generate().

        Identical to upstream except for the ``observer``/
        ``accepted_audio_budget_s``/``session_key`` kwargs passed to
        ``buffer_realtime_audio`` (see the class docstring's pin-drift
        guard). ``session_key`` is the per-generation correlation key —
        the engine request id the fork connection mints in
        ``start_generation`` (PORT-OBS-003 as amended); keyword-only and
        optional, so upstream-shaped callers remain valid.
        """
        model_config = self.model_config
        renderer = self.renderer

        if self._model_declares_widened_buffer_kwargs:
            buffered = self.model_cls.buffer_realtime_audio(
                audio_stream,
                input_stream,
                model_config,
                observer=self._observer,
                accepted_audio_budget_s=self._accepted_audio_budget_s,
                session_key=session_key,
            )
        else:
            # The exact upstream call shape — un-widened models keep
            # pre-metrics base behavior, unobserved.
            buffered = self.model_cls.buffer_realtime_audio(
                audio_stream, input_stream, model_config
            )
        stream_input_iter = cast(AsyncGenerator[PromptType, None], buffered)

        async for prompt in stream_input_iter:
            parsed_prompt = parse_model_prompt(model_config, prompt)
            (engine_input,) = await renderer.render_cmpl_async([parsed_prompt])

            yield StreamingInput(prompt=engine_input)
