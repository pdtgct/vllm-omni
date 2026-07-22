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

import time
from collections import deque
from collections.abc import AsyncGenerator
from typing import Any

import numpy as np

from vllm_omni.model_executor.models.nemotron_asr.manifests import (
    ADMISSION_EPOCH_MODULUS_MS,
    CADENCES,
    ENVELOPE_HEADER_FIELDS,
    RAW_SAMPLES_PER_CHUNK,
)

#: Default admitted chunk width (samples) when the session omits it.
_DEFAULT_CHUNK_SAMPLES = 8960
#: Minted carrier placeholder id fallback (config's audio_chunk_token_id).
_DEFAULT_PLACEHOLDER_ID = 13089
#: Geometry ids follow manifests.CADENCES order; the admitted geometry
#: is derived from the session's chunk width — an unknown width fails
#: closed (PORT-SESS-002: geometry is admission-selected, never
#: invented).
_GEOMETRY_BY_SAMPLES = {
    RAW_SAMPLES_PER_CHUNK[label]: index
    for index, label in enumerate(CADENCES)
}
_ENVELOPE_VERSION = 2.0
_HEADER_SLOTS = len(ENVELOPE_HEADER_FIELDS)


def _admission_ms_mod() -> int:
    """The acceptance wall-clock stamp (design §Ingress-deadline
    plumbing): milliseconds since epoch, modulo
    ``manifests.ADMISSION_EPOCH_MODULUS_MS`` so it stays FP32-exact
    when it rides the envelope header. Wall-clock, not monotonic —
    this value crosses from the frontend process into the worker
    process, and only wall-clock is comparable across that boundary
    (the same basis vLLM's own scheduler uses for cross-process
    ``Request.arrival_time`` ordering)."""
    return int(time.time() * 1000) % ADMISSION_EPOCH_MODULUS_MS


def mint_envelope(
    samples: np.ndarray,
    *,
    geometry_id: int,
    final_tail: bool,
    prompt_index: int,
    chunk_sequence: int,
    admission_ms_mod: int,
) -> np.ndarray:
    """Mint one CHUNK envelope: the versioned header + raw samples.

    This is the serving tier's twin admission record (design §Phase-6c
    transaction seams): the worker-side plan provider reads the same
    header host-side to stamp the session registry, and the device path
    re-validates it against that authority. Header integers must stay
    exactly FP32-representable (PORT-INT-004). ``admission_ms_mod`` is
    the caller's job to capture at true cadence-completion time, not at
    mint time — see :func:`buffer_stream`'s ready-queue.
    """
    header = np.zeros(_HEADER_SLOTS, dtype=np.float32)
    header[0] = _ENVELOPE_VERSION
    header[1] = float(samples.shape[0])
    header[2] = float(geometry_id)
    header[3] = 1.0 if final_tail else 0.0
    header[4] = float(prompt_index)
    header[5] = float(chunk_sequence)
    header[6] = float(admission_ms_mod)
    return np.concatenate([header, samples.astype(np.float32, copy=False)])


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
    finalize (PORT-SESS-003: exactly one actual-residual final-tail,
    including an explicit zero-sample transaction; never zero-padded).
    """
    chunk_samples = getattr(
        model_config, "nemotron_chunk_samples", _DEFAULT_CHUNK_SAMPLES
    )
    park_id = getattr(model_config, "park_token_id", None)
    placeholder_id = getattr(
        model_config, "audio_chunk_token_id", _DEFAULT_PLACEHOLDER_ID
    )
    geometry_id = _GEOMETRY_BY_SAMPLES.get(int(chunk_samples))
    if geometry_id is None:
        raise ValueError(
            f"chunk width {chunk_samples} is not an admitted cadence "
            f"({sorted(_GEOMETRY_BY_SAMPLES)}); geometry is selected at "
            "admission, never invented (PORT-SESS-002)"
        )

    async def hold_until_park() -> None:
        while True:
            ids = await input_stream.get()
            if park_id is None or park_id in ids:
                return

    sequence = 0

    def prompt(
        chunk: np.ndarray, *, final_tail: bool, admission_ms_mod: int
    ) -> dict[str, Any]:
        # TokensPrompt shape: one placeholder token per chunk
        # (PORT-INT-003 / D-BU-1) — a bare multi_modal_data dict is
        # invalid on the real render path. The mm payload is the minted
        # ENVELOPE (header + raw samples), the serving tier's twin
        # admission record (design §Phase-6c transaction seams).
        nonlocal sequence
        # The session-control prompt is re-read at each mint so the
        # last valid update ordered before mint is the one stamped
        # (PORT-LID-001); a live per-session config view (ING-VEH-007)
        # makes mid-session locale updates visible exactly here.
        envelope = mint_envelope(
            chunk,
            geometry_id=geometry_id,
            final_tail=final_tail,
            prompt_index=int(
                getattr(model_config, "nemotron_prompt_index", 0)
            ),
            chunk_sequence=sequence,
            admission_ms_mod=admission_ms_mod,
        )
        sequence += 1
        return {
            "prompt_token_ids": [placeholder_id],
            "multi_modal_data": {"audio": envelope},
        }

    buffer = np.zeros(0, dtype=np.float32)
    # Detection (stamping a completed cadence's admission time) and
    # delivery (yielding behind hold_until_park backpressure) are
    # deliberately DECOUPLED: PORT-SESS-001 requires a ready unit's
    # timestamp to reflect when its audio truly completed, "even
    # behind an in-flight CHUNK" — so every chunk completable from
    # the buffer is stamped in one synchronous pass (no ``await``
    # between completion and stamping), then drained through the
    # hold in FIFO order. Without this, a second chunk completed in
    # the same burst would only be stamped when the generator resumes
    # after the first chunk's hold — understating its true queuing
    # delay exactly in the case the LLD calls out.
    ready: deque[tuple[np.ndarray, int]] = deque()
    yielded = False
    async for frame in audio_stream:
        buffer = np.concatenate([buffer, frame])
        while buffer.shape[0] >= chunk_samples:
            chunk, buffer = buffer[:chunk_samples], buffer[chunk_samples:]
            ready.append((chunk, _admission_ms_mod()))
        while ready:
            chunk, admission_ms_mod = ready.popleft()
            if yielded:
                await hold_until_park()
            yield prompt(
                chunk, final_tail=False, admission_ms_mod=admission_ms_mod
            )
            yielded = True
    # Finalization is an explicit protocol transaction even when the
    # residual is shorter than the frontend's minimum commit or is
    # exactly zero. The frontend owns the zero-frame decision; the
    # session transition still needs the final marker (PORT-SESS-003).
    if yielded:
        await hold_until_park()
    yield prompt(
        buffer, final_tail=True, admission_ms_mod=_admission_ms_mod()
    )
