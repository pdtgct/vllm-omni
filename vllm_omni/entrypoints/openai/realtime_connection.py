from __future__ import annotations

import asyncio
import base64
import inspect
import json
import time
from collections.abc import AsyncGenerator, Mapping
from typing import Any, cast
from uuid import uuid4

import numpy as np
from vllm.entrypoints.openai.engine.protocol import ErrorResponse, UsageInfo
from vllm.entrypoints.speech_to_text.realtime.connection import RealtimeConnection as VllmRealtimeConnection
from vllm.entrypoints.speech_to_text.realtime.protocol import TranscriptionDelta, TranscriptionDone
from vllm.logger import init_logger

from vllm_omni.entrypoints.async_omni import AsyncOmni
from vllm_omni.entrypoints.utils import coerce_param_message_types
from vllm_omni.metrics.streaming_transport import observe_safely

logger = init_logger(__name__)


class RealtimeConnection(VllmRealtimeConnection):
    """Omni realtime connection with audio-only server events.

    Reuses upstream vLLM websocket/session lifecycle and only customizes
    generation output handling to emit audio deltas.
    """

    def __init__(self, *args, observer: Any = None, park_token_id: int | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.engine = cast(AsyncOmni, self.serving.engine_client)
        self._realtime_audio_ref: np.ndarray | None = None
        # PORT-OBS-003: the connection adapter observes chunk terminal
        # disposition (`_run_generation`) and connection-layer open
        # rejections (`_check_model`) at this layer, but it does NOT own
        # session-open accounting — that fires at observer-bearing
        # NemotronRealtimeSession construction (PORT-OBS-006: never
        # WebSocket acceptance), which this `__init__` deliberately does
        # not call.
        self._observer = observer
        # PORT-OBS-003: generic (never model-specific) park-token id.
        # `None` is inert — no park detection is attempted. A committed
        # park is inferred from park-token identity PLUS the single
        # in-flight handle (`observer.complete_inflight`), never token
        # identity plus text emptiness — so a carrierless scheduler park
        # echo or a FLUSH park (neither has an in-flight handle) is
        # correctly ignored rather than double-counted.
        self._park_token_id = park_token_id

    async def start_generation(self):
        """Fork override of upstream ``start_generation`` (pinned vLLM
        v0.24.0, ``realtime/connection.py``): same body, except the
        engine request id is minted HERE — once, before the serving
        input stream is constructed — and threaded as both the
        serving's ``session_key`` and the generation's ``request_id``
        (PORT-OBS-003 as amended: one per-generation correlation
        identity, never two). Re-diff against upstream at every pin
        bump, same as ``NemotronServingRealtime.transcribe_realtime``.
        """
        if self.generation_task is not None and not self.generation_task.done():
            logger.warning("Generation already in progress, ignoring commit")
            return

        request_id = f"rt-{self.connection_id}-{uuid4()}"

        audio_stream = self.audio_stream_generator()
        input_stream = asyncio.Queue[list[int]]()

        # The fork serving accepts the key keyword-only; an
        # upstream-shaped serving (no such param) gets the exact
        # upstream call and simply goes unkeyed/unobserved.
        transcribe = self.serving.transcribe_realtime
        if "session_key" in inspect.signature(transcribe).parameters:
            streaming_input_gen = transcribe(audio_stream, input_stream, session_key=request_id)
        else:
            streaming_input_gen = transcribe(audio_stream, input_stream)

        self.generation_task = asyncio.create_task(
            self._run_generation(streaming_input_gen, input_stream, request_id=request_id)
        )

    def _check_model(self, model: str | None) -> None | ErrorResponse:
        """Wrap the inherited model-validation gate with PORT-OBS-007.

        A denied session open is diagnostic-only, never a scaling signal,
        and carries no cadence label (rejection precedes admission).
        """
        err = super()._check_model(model)
        if err is not None and self._observer is not None:
            observe_safely(self._observer.session_open_rejected, reason="model")
        return err

    @staticmethod
    def _tensor_to_numpy(value) -> np.ndarray | None:
        if value is None:
            return None
        if isinstance(value, np.ndarray):
            arr = value
        elif hasattr(value, "detach"):
            arr = value.detach().float().cpu().numpy()
        else:
            try:
                arr = np.asarray(value)
            except Exception:
                return None
        if arr.ndim > 1:
            arr = arr.reshape(-1)
        return arr.astype(np.float32, copy=False)

    @staticmethod
    def _numpy_audio_prefix_match(prev: np.ndarray, curr: np.ndarray) -> bool:
        n = prev.shape[0]
        if n == 0:
            return True
        if curr.shape[0] < n:
            return False
        return bool(np.allclose(curr[:n], prev, rtol=1e-3, atol=2e-4))

    def _raw_waveform_to_deltas(self, arr: np.ndarray) -> list[np.ndarray]:
        """Convert one streaming PCM f32 chunk into incremental piece(s) for the client.

        Some engine paths emit a growing cumulative waveform each step; others emit
        true per-step deltas. We support both without duplicating audio on the client.
        """
        if arr.size == 0:
            return []
        ref = self._realtime_audio_ref
        if ref is None:
            self._realtime_audio_ref = arr.copy()
            return [arr]
        if self._numpy_audio_prefix_match(ref, arr):
            delta = arr[ref.shape[0] :]
            self._realtime_audio_ref = arr.copy()
            return [delta] if delta.size > 0 else []
        # True per-step delta (not a prefix extension of what we have seen).
        self._realtime_audio_ref = np.concatenate([ref, arr])
        return [arr]

    def _extract_audio_chunks(self, output) -> tuple[list[np.ndarray], int]:
        mm = getattr(output, "multimodal_output", None)
        if mm is None:
            return [], 24000
        # Support both MultimodalPayload and plain dict
        if not isinstance(mm, Mapping):
            return [], 24000

        sr = mm.get("sr") or mm.get("sample_rate") or mm.get("audio_sample_rate") or 24000
        if isinstance(sr, (list, tuple)) and sr:
            sr = sr[-1]
        if hasattr(sr, "item"):
            sr = sr.item()
        sample_rate_hz = int(sr)
        key = "audio" if "audio" in mm else ("model_outputs" if "model_outputs" in mm else None)
        if key is None:
            return [], sample_rate_hz

        raw_audio = mm.get(key)
        chunks: list[np.ndarray] = []
        if isinstance(raw_audio, (list, tuple)):
            if len(raw_audio) > 0:
                arr = self._tensor_to_numpy(raw_audio[-1])
                if arr is not None and arr.size > 0:
                    chunks.extend(self._raw_waveform_to_deltas(arr))
        else:
            arr = self._tensor_to_numpy(raw_audio)
            if arr is not None and arr.size > 0:
                chunks.extend(self._raw_waveform_to_deltas(arr))
        return chunks, sample_rate_hz

    @staticmethod
    def _pcm16_b64(audio_f32: np.ndarray) -> str:
        clipped = np.clip(audio_f32, -1.0, 1.0)
        pcm16 = (clipped * 32767.0).astype(np.int16)
        return base64.b64encode(pcm16.tobytes()).decode("utf-8")

    # @spec ING-LIFE-011, PORT-OBS-006
    async def _run_generation(
        self,
        streaming_input_gen: AsyncGenerator,
        input_stream: asyncio.Queue[list[int]],
        *,
        request_id: str | None = None,
    ) -> None:
        sent_audio = False
        audio_done_sent = False
        full_text = ""
        prompt_token_ids_len = 0
        completion_tokens_len = 0
        self._realtime_audio_ref = None
        cancelled: asyncio.CancelledError | None = None
        disconnected = False
        model_finalized = False
        # ``getattr`` defensive: some fixtures across this test suite
        # construct a connection via ``RealtimeConnection.__new__``,
        # bypassing ``__init__`` (and hence these two attributes)
        # entirely — inert observation, not an AttributeError, is the
        # correct behavior for such a connection.
        observer = getattr(self, "_observer", None)
        park_token_id = getattr(self, "_park_token_id", None)
        # PORT-OBS-003 (amended): the per-generation correlation key is
        # minted in start_generation and passed in. With an observer
        # installed, a keyless invocation must NOT mint a second
        # identity and observe under it (that mismatch is exactly the
        # D6 defect class) — observation for THIS generation is disabled
        # loudly instead; keyless compatibility remains only on the
        # unobserved path.
        if request_id is None:
            if observer is not None:
                logger.warning(
                    "generation started without the threaded correlation "
                    "id — observation disabled for this generation "
                    "(PORT-OBS-003)"
                )
                observer = None
            request_id = f"rt-{self.connection_id}-{uuid4()}"

        # Coerce cumulative outputs to delta outputs; this ensures
        # we don't emit redundant MM data & drain after emitting.
        sampling_params_list = list(self.engine.default_sampling_params_list)
        sampling_params_list = coerce_param_message_types(
            sampling_params_list,
            is_streaming=True,
        )

        try:
            result_gen = self.engine.generate(
                prompt=streaming_input_gen,
                request_id=request_id,
                sampling_params_list=sampling_params_list,
            )

            async for output in result_gen:
                stage_id = getattr(output, "stage_id", None)
                if stage_id == 0 and output.outputs:
                    first_output = output.outputs[0]
                    new_token_ids = list(first_output.token_ids)
                    if new_token_ids:
                        input_stream.put_nowait(new_token_ids)
                    # PORT-OBS-003: the single-in-flight-handle
                    # correlation authority — resolves to the CHUNK this
                    # park completes, or ``None`` for a carrierless
                    # scheduler park echo or FLUSH's ticketless park
                    # (correctly ignored, every waiting-ready unit for
                    # this session left outstanding).
                    if observer is not None and park_token_id is not None and park_token_id in new_token_ids:
                        handle = observe_safely(observer.complete_inflight, request_id)
                        if handle is not None:
                            observe_safely(observer.unit_parked, handle, park_stamp_s=time.monotonic())

                    if output.prompt_token_ids:
                        prompt_token_ids_len = max(
                            prompt_token_ids_len,
                            len(output.prompt_token_ids),
                        )

                    delta_text = first_output.text or ""
                    full_text += delta_text
                    completion_tokens_len += len(new_token_ids)

                    if delta_text:
                        await self.send(TranscriptionDelta(delta=delta_text))

                audio_chunks, sample_rate = self._extract_audio_chunks(output)

                for chunk in audio_chunks:
                    sent_audio = True
                    await self.send_json(
                        {
                            "type": "response.audio.delta",
                            "audio": self._pcm16_b64(chunk),
                            "format": "pcm16",
                            "sample_rate_hz": sample_rate,
                        }
                    )

                if not self._is_connected:
                    disconnected = True
                    break
            else:
                # The engine generator exhausted normally — on this path
                # the request finishes only through finalization/FLUSH,
                # so MODEL completion is decided here, before any
                # terminal send: transport delivery is not part of model
                # completion (PORT-OBS-006; review round F3).
                model_finalized = True

            if self._is_connected:
                usage = UsageInfo(
                    prompt_tokens=prompt_token_ids_len,
                    completion_tokens=completion_tokens_len,
                    total_tokens=prompt_token_ids_len + completion_tokens_len,
                )
                await self.send(TranscriptionDone(text=full_text, usage=usage))

                if sent_audio:
                    await self.send_json(
                        {
                            "type": "response.audio.done",
                            "has_audio": True,
                        }
                    )
                    audio_done_sent = True
        except asyncio.CancelledError as e:
            # Amendment 4 (PORT-OBS-006): cancellation is an ABORT —
            # recorded in the single terminal section below, then
            # RE-RAISED so task-cancellation semantics stay intact.
            cancelled = e
        except Exception as e:
            logger.exception("Error in generation: %s", e)
            if self._is_connected:
                await self.send_error(str(e), "processing_error")
        finally:
            # Always send terminal event so clients don't hang forever.
            if self._is_connected and not audio_done_sent:
                try:
                    await self.send_json({"type": "response.audio.done", "has_audio": sent_audio})
                except Exception:
                    logger.exception("Failed to send response.audio.done")
            while not self.audio_queue.empty():
                self.audio_queue.get_nowait()
            # PORT-OBS-003/006: the native path's single observer
            # terminal section. Gated on the observer/session key ONLY —
            # never the park token (only park DETECTION needs the token;
            # a token-less connection still clears and finishes).
            #
            # Reason taxonomy (PORT-OBS-006):
            #   completed — result generation exhausted after the
            #     model's FLUSH park: on this path the engine finishes
            #     the request only through finalization, so normal
            #     exhaustion WITH zero outstanding units is completion.
            #   aborted   — disconnect, cancellation.
            #   error     — engine/protocol failure, or an ostensibly
            #     normal end that still had outstanding units
            #     (lifecycle divergence): the remaining work is cleared
            #     as error and the session finishes as error.
            if observer is not None:
                if cancelled is not None or disconnected:
                    outcome, reason = "aborted", "aborted"
                elif model_finalized:
                    # Classified from finalization state, NOT from
                    # generation_error: a terminal-send failure after
                    # the model finished is a transport event and must
                    # not turn completion into error (PORT-OBS-006).
                    outcome, reason = "error", "completed"
                else:
                    outcome, reason = "error", "error"
                remaining = observe_safely(
                    observer.clear_all_outstanding, request_id, outcome=outcome
                )
                if reason == "completed" and remaining:
                    logger.warning(
                        "generation ended normally with %s undisposed unit(s) "
                        "— lifecycle divergence, session finishes as error",
                        remaining,
                    )
                    reason = "error"
                observe_safely(observer.session_finished, session_key=request_id, reason=reason)
            if cancelled is not None:
                raise cancelled

    async def send_json(self, payload: dict):
        await self.websocket.send_text(json.dumps(payload))
