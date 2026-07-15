# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The realtime input segmenter — pure, engine-free (PORT-SESS-001/002/003).

The chunking behaviour that backs ``buffer_realtime_audio``: fixed
segmenter, hold-until-park backpressure, and the NeMo tail rules. It
carries no vLLM coupling (numpy + async generators only), so it stays on
the macOS loader path and CPU-tested while the model class that exposes
it — via the thin ``SupportsRealtime.buffer_realtime_audio`` classmethod
— is engine/pod-gated (see ``static-analysis-for-vllm-coupled-code``).
"""

from __future__ import annotations

from collections.abc import AsyncGenerator
from typing import Any

import numpy as np

#: Default admitted chunk width (samples) when the session omits it.
_DEFAULT_CHUNK_SAMPLES = 8960
#: Minted carrier placeholder id fallback (config's audio_chunk_token_id).
_DEFAULT_PLACEHOLDER_ID = 13089
#: 8 mel frames * 160-sample hop: the shortest tail worth decoding.
_MIN_TAIL_SAMPLES = 1280


async def buffer_stream(
    audio_stream: Any,
    input_stream: Any,
    model_config: Any,
) -> AsyncGenerator[dict[str, Any], None]:
    """Chunk client audio: one yield = one StreamingUpdate.

    Fixed segmenter built once at generator start from the session's
    admitted chunk config (PORT-SESS-002); holds chunk N+1 until the
    park token id for chunk N appears on ``input_stream``
    (buffer-until-drained, PORT-SESS-001 — defense in depth over core's
    park-time queue consumption); applies the NeMo tail rules on
    finalize (PORT-SESS-003: partial tails as-is, sub-8-mel-frame
    remainders dropped, never zero-padded).
    """
    chunk_samples = getattr(
        model_config, "nemotron_chunk_samples", _DEFAULT_CHUNK_SAMPLES
    )
    park_id = getattr(model_config, "park_token_id", None)
    placeholder_id = getattr(
        model_config, "audio_chunk_token_id", _DEFAULT_PLACEHOLDER_ID
    )

    async def hold_until_park() -> None:
        while True:
            ids = await input_stream.get()
            if park_id is None or park_id in ids:
                return

    def prompt(chunk: np.ndarray) -> dict[str, Any]:
        # TokensPrompt shape: one placeholder token per chunk
        # (PORT-INT-003 / D-BU-1) — a bare multi_modal_data dict is
        # invalid on the real render path.
        return {
            "prompt_token_ids": [placeholder_id],
            "multi_modal_data": {"audio": chunk},
        }

    buffer = np.zeros(0, dtype=np.float32)
    yielded = False
    async for frame in audio_stream:
        buffer = np.concatenate([buffer, frame])
        while buffer.shape[0] >= chunk_samples:
            chunk, buffer = buffer[:chunk_samples], buffer[chunk_samples:]
            if yielded:
                await hold_until_park()
            yield prompt(chunk)
            yielded = True
    if buffer.shape[0] >= _MIN_TAIL_SAMPLES:
        if yielded:
            await hold_until_park()
        yield prompt(buffer)
