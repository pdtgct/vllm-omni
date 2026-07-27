from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator
from typing import Any, cast

import numpy as np
from vllm.engine.protocol import StreamingInput
from vllm.entrypoints.speech_to_text.realtime.serving import OpenAIServingRealtime
from vllm.inputs import PromptType
from vllm.renderers.inputs.preprocess import parse_model_prompt


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

    async def transcribe_realtime(
        self,
        audio_stream: AsyncGenerator[np.ndarray, None],
        input_stream: asyncio.Queue[list[int]],
    ) -> AsyncGenerator[StreamingInput, None]:
        """Transform audio stream into StreamingInput for engine.generate().

        Identical to upstream except for the ``observer``/
        ``accepted_audio_budget_s`` kwargs passed to
        ``buffer_realtime_audio`` (see the class docstring's pin-drift
        guard).
        """
        model_config = self.model_config
        renderer = self.renderer

        stream_input_iter = cast(
            AsyncGenerator[PromptType, None],
            self.model_cls.buffer_realtime_audio(
                audio_stream,
                input_stream,
                model_config,
                observer=self._observer,
                accepted_audio_budget_s=self._accepted_audio_budget_s,
            ),
        )

        async for prompt in stream_input_iter:
            parsed_prompt = parse_model_prompt(model_config, prompt)
            (engine_input,) = await renderer.render_cmpl_async([parsed_prompt])

            yield StreamingInput(prompt=engine_input)
