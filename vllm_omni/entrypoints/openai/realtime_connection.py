from __future__ import annotations

import asyncio
import base64
import json
from collections.abc import AsyncGenerator, Mapping
from typing import cast
from uuid import uuid4

import numpy as np
from vllm.entrypoints.openai.engine.protocol import UsageInfo
from vllm.entrypoints.speech_to_text.realtime.connection import RealtimeConnection as VllmRealtimeConnection
from vllm.entrypoints.speech_to_text.realtime.protocol import TranscriptionDelta, TranscriptionDone
from vllm.logger import init_logger

from vllm_omni.engine.persistent_state_service import (
    PersistentStateCapacityExhausted,
    PersistentStateIndeterminate,
    PersistentStateServiceUnavailable,
    StateLease,
)
from vllm_omni.entrypoints.async_omni import AsyncOmni
from vllm_omni.entrypoints.utils import coerce_param_message_types
from vllm_omni.model_executor.models.nemotron_asr.endpointing import (
    EndpointPolicy,
)
from vllm_omni.model_executor.models.nemotron_asr.session import (
    NemotronRealtimeSession,
)
from vllm_omni.model_executor.models.nemotron_asr.transcript import (
    BoundedTranscript,
    OutputCapacityExceeded,
    SegmentCompletion,
)

logger = init_logger(__name__)


class RealtimeConnection(VllmRealtimeConnection):
    """Omni realtime connection with audio-only server events.

    Reuses upstream vLLM websocket/session lifecycle and only customizes
    generation output handling to emit audio deltas.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.engine = cast(AsyncOmni, self.serving.engine_client)
        self._realtime_audio_ref: np.ndarray | None = None
        try:
            self._persistent_state_service = (
                self.engine.get_persistent_state_service()
            )
        except (AttributeError, RuntimeError):
            self._persistent_state_service = None
        self._state_lease: StateLease | object | None = None
        self._state_operation_id: str | None = None
        self._state_session_key: str | None = None
        self._state_release_done = False
        self._nemotron_session: NemotronRealtimeSession | None = None
        self._native_fifo_event = asyncio.Event()
        self._native_finalized = False
        self._segment_generation = 0
        self.session_configuration_timeout = float(
            getattr(self.serving, "session_configuration_timeout_s", 30.0)
        )
        if self.session_configuration_timeout <= 0:
            raise ValueError("session_configuration_timeout must be positive")
        self._configuration_timeout_task: asyncio.Task[None] | None = None

    async def handle_connection(self):
        """Arm model-admission lifetime separately from transport lifetime."""
        self._configuration_timeout_task = asyncio.create_task(
            self._configuration_timeout(),
            name=f"realtime-config-{self.connection_id}",
        )
        try:
            return await super().handle_connection()
        finally:
            if self._configuration_timeout_task is not None:
                self._configuration_timeout_task.cancel()

    async def _configuration_timeout(self) -> None:
        await asyncio.sleep(self.session_configuration_timeout)
        if self._is_model_validated:
            return
        await self.send_error(
            "Model session configuration timed out.",
            "model_not_validated",
        )
        self._is_connected = False
        close = getattr(self.websocket, "close", None)
        if close is not None:
            await close(code=1008)

    async def _native_audio_controls(self) -> AsyncGenerator[np.ndarray, None]:
        """Wake the admitted FIFO without duplicating accepted audio.

        Audio pieces are copied into ``AcceptedAudioAuthority`` at append
        time and every append wakes its cadence-ready FIFO with a zero-sample
        control.  A nonfinal commit first appends its ordered forced-EOU
        barrier; a final commit ends this generator so ``buffer_stream``
        drains the already-minted final tail and FLUSHes the request.
        """

        while True:
            await self._native_fifo_event.wait()
            self._native_fifo_event.clear()
            if self._native_finalized:
                return
            yield np.empty(0, dtype=np.float32)

    async def _transcribe_nemotron_realtime(
        self,
        audio_stream: AsyncGenerator[np.ndarray, None],
        input_stream: asyncio.Queue[list[int]],
    ) -> AsyncGenerator:
        """Render the exact admitted session instead of rebuilding one."""

        from vllm.engine.protocol import StreamingInput
        from vllm.renderers.inputs.preprocess import parse_model_prompt

        session = getattr(self, "_nemotron_session", None)
        if session is None:
            raise RuntimeError("Nemotron session is not configured")
        stream = self.serving.model_cls.buffer_realtime_audio(
            audio_stream,
            input_stream,
            session,
        )
        async for prompt in stream:
            parsed = parse_model_prompt(self.serving.model_config, prompt)
            (engine_input,) = await self.serving.renderer.render_cmpl_async(
                [parsed]
            )
            yield StreamingInput(prompt=engine_input)

    async def start_generation(self):
        if self._nemotron_session is None:
            await super().start_generation()
            return
        if self.generation_task is not None and not self.generation_task.done():
            return
        input_stream: asyncio.Queue[list[int]] = asyncio.Queue()
        streaming_input = self._transcribe_nemotron_realtime(
            self._native_audio_controls(),
            input_stream,
        )
        self.generation_task = asyncio.create_task(
            self._run_generation(streaming_input, input_stream)
        )

    async def handle_event(self, event: dict):
        """Route Nemotron audio through its bounded session authority."""

        session = getattr(self, "_nemotron_session", None)
        event_type = event.get("type")
        if (
            getattr(self, "_persistent_state_service", None) is not None
            and event_type in {
                "input_audio_buffer.append",
                "input_audio_buffer.commit",
            }
            and not self._is_model_validated
        ):
            await self.send_error(
                "Model not validated. Send session.update first.",
                "model_not_validated",
            )
            return
        if event_type == "session.update" and self._persistent_state_service is not None:
            # Definitive capacity_exhausted/service_unavailable denials leave
            # transport connected and _is_model_validated=False.  The helper
            # sets _is_model_validated = True only after the same operation_id
            # is reconciled and model-session construction succeeds.
            await self._configure_persistent_session(event)
            return
        if session is None or event_type == "session.update":
            return await super().handle_event(event)
        if event_type == "input_audio_buffer.append":
            if not self._is_model_validated:
                await self.send_error(
                    "Model not validated. Send session.update first.",
                    "model_not_validated",
                )
                return
            try:
                audio_bytes = base64.b64decode(event["audio"], validate=True)
                audio_array = (
                    np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32)
                    / 32768.0
                )
                if audio_array.size == 0:
                    raise ValueError("empty audio")
                if audio_array.size / 1024**2 > self._max_audio_filesize_mb:
                    raise ValueError("maximum audio size exceeded")
                session.accept_audio(audio_array)
            except Exception:
                logger.exception("Failed to accept realtime audio")
                await self.send_error("Invalid audio data", "invalid_audio")
                return
            self._native_fifo_event.set()
            if self.generation_task is None or self.generation_task.done():
                await self.start_generation()
            return
        if event_type == "input_audio_buffer.commit":
            if not self._is_model_validated:
                await self.send_error(
                    "Model not validated. Send session.update first.",
                    "model_not_validated",
                )
                return
            if bool(event.get("final", False)):
                session.begin_finalize()
                self._native_finalized = True
            else:
                session.force_segment()
            self._native_fifo_event.set()
            if self.generation_task is None or self.generation_task.done():
                await self.start_generation()
            return
        return await super().handle_event(event)

    async def _configure_persistent_session(self, event: dict) -> None:
        """Reserve state, construct the session, then publish model admission."""
        model = event.get("model")
        if model is None:
            await self.send_error("Missing required field: model", "invalid_event")
            return
        err = self._check_model(model)
        if err is not None:
            await self.send_error(err.error.message, "model_not_found")
            return
        if self._is_model_validated:
            # Model identity and state ownership are immutable after admission.
            return

        service = self._persistent_state_service
        assert service is not None
        check_health = getattr(service, "check_health", None)
        if check_health is not None:
            await check_health()
        inventory = getattr(service, "inventory", None) or {}
        schema_id = str(inventory.get("schema_id", "state-manifest-v1"))
        profile_id = str(inventory.get("profile_id", "default"))
        reserve_operation_id = uuid4().hex
        session_key = f"rt-{self.connection_id}-{uuid4().hex}"
        try:
            lease = await service.reserve(
                operation_id=reserve_operation_id,
                session_key=session_key,
                schema_id=schema_id,
                profile_id=profile_id,
            )
        except PersistentStateCapacityExhausted:
            await self.send_error(
                "Persistent-state capacity is exhausted.",
                "capacity_exhausted",
            )
            return
        except PersistentStateIndeterminate:
            # The service reconciles the same operation.  The connection does
            # not mint a parallel reserve attempt while its result is unknown.
            await self.send_error(
                "Persistent-state admission is indeterminate.",
                "service_unavailable",
            )
            return
        except PersistentStateServiceUnavailable:
            await self.send_error(
                "Persistent-state service is unavailable.",
                "service_unavailable",
            )
            return

        release_operation_id = uuid4().hex
        self._state_lease = lease
        self._state_operation_id = release_operation_id
        self._state_session_key = session_key
        try:
            model_config = getattr(self.serving, "model_config", None)
            if model_config is not None:
                endpoint = event.get("endpointing") or {}
                hf = getattr(model_config, "hf_config", model_config)
                endpoint_policy = EndpointPolicy.resolve(
                    mode=str(endpoint.get("mode", "greedy_blank")),
                    stop_history_ms=endpoint.get("stop_history_ms", 800),
                    residue_frames=int(endpoint.get("residue_frames", 2)),
                    frame_stride_ms=80,
                    history_capacity_frames=int(
                        getattr(hf, "endpoint_history_capacity_frames", 12)
                    ),
                )
                generation = int(getattr(lease, "generation"))
                engine_epoch = str(getattr(lease, "engine_epoch"))
                self._nemotron_session = NemotronRealtimeSession.from_model_config(
                    model_config,
                    cadence=str(event.get("cadence", "560ms")),
                    locale=str(event.get("locale", "auto")),
                    endpoint_policy=endpoint_policy,
                    request_id=session_key,
                    engine_epoch=engine_epoch,
                    lease_generation=generation,
                )
        except Exception:
            await service.release(
                operation_id=release_operation_id,
                lease=lease,
                reason="configuration_error",
            )
            self._state_release_done = True
            self._state_lease = None
            self._state_operation_id = None
            self._state_session_key = None
            raise
        self._is_model_validated = True
        if self._configuration_timeout_task is not None:
            self._configuration_timeout_task.cancel()

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

    def _completion_from_tokens(
        self,
        output: object,
        token_ids: list[int],
        in_flight: object | None,
    ) -> object | None:
        """Project the committed EOU control into a serving completion."""

        completion = getattr(output, "segment_completion", None)
        if completion is not None:
            self._segment_generation = max(
                getattr(self, "_segment_generation", 0),
                int(completion.generation),
            )
            return completion
        session = getattr(self, "_nemotron_session", None)
        if session is None or session.eou_token_id not in token_ids:
            return None
        generation = getattr(self, "_segment_generation", 0) + 1
        self._segment_generation = generation
        reason = (
            "forced"
            if getattr(in_flight, "kind", None) == "forced_eou"
            else "model"
        )
        return SegmentCompletion(
            generation=generation,
            text="",
            reason=reason,
        )

    # @spec ING-LIFE-011
    async def _run_generation(
        self,
        streaming_input_gen: AsyncGenerator,
        input_stream: asyncio.Queue[list[int]],
    ) -> None:
        request_id = getattr(self, "_state_session_key", None) or (
            f"rt-{self.connection_id}-{uuid4()}"
        )
        sent_audio = False
        audio_done_sent = False
        prompt_token_ids_len = 0
        completion_tokens_len = 0
        self._realtime_audio_ref = None
        transcript = BoundedTranscript(
            max_retained_bytes=int(
                getattr(self, "max_retained_transcript_bytes", 1 << 20)
            ),
            fragment_overhead_bytes=0,
            terminal_headroom_bytes=0,
        )

        # Coerce cumulative outputs to delta outputs; this ensures
        # we don't emit redundant MM data & drain after emitting.
        sampling_params_list = list(self.engine.default_sampling_params_list)
        sampling_params_list = coerce_param_message_types(
            sampling_params_list,
            is_streaming=True,
        )

        try:
            if getattr(self, "_state_lease", None) is not None:
                # _bind_state_lease adds additional_information containing
                # the acknowledged persistent_state_binding to each turn.
                streaming_input_gen = self._bind_state_lease(
                    streaming_input_gen,
                    self._state_lease,
                )
            result_gen = self.engine.generate(
                prompt=streaming_input_gen,
                request_id=request_id,
                sampling_params_list=sampling_params_list,
                request_id_already_unique=(
                    getattr(self, "_state_lease", None) is not None
                ),
            )

            async for output in result_gen:
                stage_id = getattr(output, "stage_id", None)
                if stage_id == 0 and output.outputs:
                    first_output = output.outputs[0]
                    new_token_ids = list(first_output.token_ids)
                    session = getattr(self, "_nemotron_session", None)
                    in_flight = (
                        session.accepted_audio.in_flight_unit
                        if session is not None
                        else None
                    )
                    if new_token_ids:
                        input_stream.put_nowait(new_token_ids)

                    if output.prompt_token_ids:
                        prompt_token_ids_len = max(
                            prompt_token_ids_len,
                            len(output.prompt_token_ids),
                        )

                    delta_text = first_output.text or ""
                    projection = transcript.commit_result(delta_text)
                    completion_tokens_len += len(new_token_ids)

                    if projection.delta:
                        await self.send(
                            TranscriptionDelta(delta=projection.delta)
                        )

                    completion = self._completion_from_tokens(
                        output,
                        new_token_ids,
                        in_flight,
                    )
                    if completion is not None:
                        segment = transcript.complete_segment(
                            generation=int(completion.generation),
                            reason=completion.reason,
                        )
                        if segment is not None:
                            await self.send_json(
                                {
                                    "type": "transcription.segment.done",
                                    "generation": segment.generation,
                                    "text": segment.text,
                                    "reason": segment.reason,
                                    "usage": {
                                        "completion_tokens": len(
                                            new_token_ids
                                        ),
                                    },
                                }
                            )

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
                    break

            if self._is_connected:
                usage = UsageInfo(
                    prompt_tokens=prompt_token_ids_len,
                    completion_tokens=completion_tokens_len,
                    total_tokens=prompt_token_ids_len + completion_tokens_len,
                )
                terminal = transcript.finish_terminal()
                await self.send(
                    TranscriptionDone(
                        text=terminal.complete_text,
                        usage=usage,
                        terminal=True,
                    )
                )

                if sent_audio:
                    await self.send_json(
                        {
                            "type": "response.audio.done",
                            "has_audio": True,
                        }
                    )
                    audio_done_sent = True
        except OutputCapacityExceeded:
            logger.exception("Realtime transcript capacity exhausted")
            if self._is_connected:
                await self.send_error(
                    "output_capacity_exceeded",
                    "output_capacity_exceeded",
                )
        except Exception as e:
            logger.exception("Error in generation: %s", e)
            if self._is_connected:
                await self.send_error(str(e), "processing_error")
        finally:
            # Always send terminal event so clients don't hang forever.
            if self._is_connected and sent_audio and not audio_done_sent:
                try:
                    await self.send_json({"type": "response.audio.done", "has_audio": sent_audio})
                except Exception:
                    logger.exception("Failed to send response.audio.done")
            while not self.audio_queue.empty():
                self.audio_queue.get_nowait()

    async def _bind_state_lease(
        self,
        streaming_input_gen: AsyncGenerator,
        lease: object,
    ) -> AsyncGenerator:
        """Attach the acknowledged logical binding to every streaming turn."""
        binding = {
            "engine_epoch": getattr(lease, "engine_epoch"),
            "session_key": getattr(lease, "session_key"),
            "generation": getattr(lease, "generation"),
            "schema_id": getattr(lease, "schema_id"),
            "profile_id": getattr(lease, "profile_id"),
            "binding_token": getattr(lease, "binding_token"),
        }
        session = self._nemotron_session
        endpoint_policy = None
        if session is not None:
            policy = session.endpoint_policy
            endpoint_policy = {
                "mode": policy.mode,
                "threshold_frames": policy.threshold_frames,
                "residue_frames": policy.residue_frames,
            }
        async for streaming_input in streaming_input_gen:
            prompt = dict(streaming_input.prompt)
            information = dict(prompt.get("additional_information") or {})
            information["persistent_state_binding"] = binding
            if endpoint_policy is not None:
                information["endpoint_policy"] = endpoint_policy
            prompt["additional_information"] = information
            streaming_input.prompt = prompt
            yield streaming_input

    async def cleanup(self):
        """Release resident state exactly once after transport cleanup begins."""
        await super().cleanup()
        if self._configuration_timeout_task is not None:
            self._configuration_timeout_task.cancel()
        if (
            self._persistent_state_service is not None
            and self._state_lease is not None
            and self._state_operation_id is not None
            and not self._state_release_done
        ):
            self._state_release_done = True
            await self._persistent_state_service.release(
                operation_id=self._state_operation_id,
                lease=self._state_lease,
                reason="connection_cleanup",
            )

    async def send_json(self, payload: dict):
        await self.websocket.send_text(json.dumps(payload))
