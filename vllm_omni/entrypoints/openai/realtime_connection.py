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
from vllm.entrypoints.speech_to_text.realtime.protocol import (
    ErrorEvent,
    TranscriptionDelta,
    TranscriptionDone,
)
from vllm.logger import init_logger

from vllm_omni.engine.persistent_state_service import (
    PersistentStateBackpressure,
    PersistentStateCapacityExhausted,
    PersistentStateIndeterminate,
    PersistentStateServiceUnavailable,
    PersistentStateUnsupportedServiceInterval,
    StateLease,
)
from vllm_omni.entrypoints.async_omni import AsyncOmni
from vllm_omni.entrypoints.session_lifecycle import (
    SessionLifecycleDeadline,
    SessionLifecycleKind,
)
from vllm_omni.entrypoints.utils import coerce_param_message_types
from vllm_omni.metrics.streaming_transport import observe_safely
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

    def __init__(self, *args, observer: Any = None, park_token_id: int | None = None, **kwargs):
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
        self._state_cleanup_lock = asyncio.Lock()
        self._pending_claim_task: asyncio.Task[None] | None = None
        self._configuration_timed_out = False
        self._nemotron_session: NemotronRealtimeSession | None = None
        self._native_fifo_event = asyncio.Event()
        self._native_finalized = False
        self._segment_generation = 0
        self._streaming_session_finished = False
        self._streaming_finish_reason = "aborted"
        self._terminal_code: str | None = None
        runtime = getattr(self.serving, "runtime_config", None)
        service_runtime = getattr(
            self._persistent_state_service,
            "runtime_config",
            None,
        )
        if self._persistent_state_service is not None and runtime is None:
            raise ValueError(
                "persistent-state serving requires its resolved runtime envelope"
            )
        if service_runtime is not None and runtime is not service_runtime:
            raise ValueError(
                "persistent-state serving and service runtime envelopes differ"
            )
        self._runtime_config = runtime
        configured_timeout = (
            runtime.session_configuration_timeout_s
            if runtime is not None
            else None
        )
        self.session_configuration_timeout = float(
            30.0 if configured_timeout is None else configured_timeout
        )
        if self.session_configuration_timeout <= 0:
            raise ValueError("session_configuration_timeout must be positive")
        self._configuration_timeout_task: asyncio.Task[None] | None = None
        self._unadmitted_timeout_task: asyncio.Task[None] | None = None
        self._unadmitted_deadline_ns: int | None = None
        self._unadmitted_timed_out = False
        self._pre_admission_timeouts_started = False
        self._admission_lock = asyncio.Lock()
        self._next_admission_retry_ns = 0
        self._effective_session: dict[str, Any] | None = None
        self._session_lifecycle = SessionLifecycleDeadline(
            idle_timeout_s=(
                runtime.session_idle_timeout_s
                if runtime is not None
                else None
            ),
            finalization_timeout_s=(
                runtime.session_finalization_timeout_s
                if runtime is not None
                else None
            ),
            on_expire=self._expire_session_lifecycle,
        )
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

    async def handle_connection(self):
        """Arm model-admission lifetime separately from transport lifetime."""
        self._start_pre_admission_timeouts()
        try:
            return await super().handle_connection()
        finally:
            await self._cancel_pre_admission_timeouts()

    # @spec PORT-SESS-012
    def _start_pre_admission_timeouts(self) -> None:
        """Arm distinct configuration and total-unadmitted clocks once."""
        if self._pre_admission_timeouts_started:
            return
        self._pre_admission_timeouts_started = True
        runtime = self._runtime_config
        configuration_timeout_s = float(
            getattr(
                runtime,
                "session_configuration_timeout_s",
                self.session_configuration_timeout,
            )
        )
        self.session_configuration_timeout = configuration_timeout_s
        total_timeout_s = float(
            getattr(
                runtime,
                "unadmitted_connection_timeout_s",
                configuration_timeout_s,
            )
        )
        loop = asyncio.get_running_loop()
        self._unadmitted_deadline_ns = time.monotonic_ns() + int(
            total_timeout_s * 1_000_000_000
        )
        self._configuration_timeout_task = loop.create_task(
            self._configuration_timeout(),
            name=f"realtime-config-{self.connection_id}",
        )
        self._unadmitted_timeout_task = loop.create_task(
            self._unadmitted_timeout(total_timeout_s),
            name=f"realtime-unadmitted-{self.connection_id}",
        )

    async def _cancel_task(self, task: asyncio.Task[None] | None) -> None:
        if task is None or task is asyncio.current_task():
            return
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)

    async def _cancel_configuration_timeout(self) -> None:
        task = self._configuration_timeout_task
        self._configuration_timeout_task = None
        await self._cancel_task(task)

    async def _cancel_pre_admission_timeouts(self) -> None:
        await self._cancel_configuration_timeout()
        task = self._unadmitted_timeout_task
        self._unadmitted_timeout_task = None
        await self._cancel_task(task)

    async def _configuration_timeout(self) -> None:
        await asyncio.sleep(self.session_configuration_timeout)
        if self._is_model_validated:
            return
        self._configuration_timed_out = True
        await self.send_error(
            "Model session configuration timed out.",
            "model_not_validated",
        )
        self._is_connected = False
        close = getattr(self.websocket, "close", None)
        if close is not None:
            await close(code=1008)

    async def _unadmitted_timeout(self, timeout_s: float) -> None:
        await asyncio.sleep(timeout_s)
        if self._is_model_validated:
            return
        self._unadmitted_timed_out = True
        await self.send_error(
            "Persistent-state admission lifetime expired.",
            "capacity_exhausted",
        )
        self._is_connected = False
        close = getattr(self.websocket, "close", None)
        if close is not None:
            await close(code=1013)

    async def _native_audio_controls(self) -> AsyncGenerator[np.ndarray, None]:
        """Wake the admitted FIFO without duplicating accepted audio.

        Audio pieces are copied into ``AcceptedAudioAuthority`` at append
        time and every append wakes its cadence-ready FIFO with a zero-sample
        control.  The first nonfinal commit starts generation without
        inventing a segment boundary; a later nonfinal commit appends its
        ordered forced-EOU barrier.  A final commit ends this generator so
        ``buffer_stream`` drains the already-minted final tail and FLUSHes
        the request.
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
        """Start one generation with a single pre-minted correlation id."""
        if self.generation_task is not None and not self.generation_task.done():
            logger.warning("Generation already in progress, ignoring commit")
            return
        if getattr(self, "_nemotron_session", None) is None:
            request_id = f"rt-{self.connection_id}-{uuid4()}"
            audio_stream = self.audio_stream_generator()
            input_stream: asyncio.Queue[list[int]] = asyncio.Queue()
            transcribe = self.serving.transcribe_realtime
            if "session_key" in inspect.signature(transcribe).parameters:
                streaming_input = transcribe(
                    audio_stream,
                    input_stream,
                    session_key=request_id,
                )
            else:
                streaming_input = transcribe(audio_stream, input_stream)
            self.generation_task = asyncio.create_task(
                self._run_generation(
                    streaming_input,
                    input_stream,
                    request_id=request_id,
                )
            )
            return
        request_id = self._state_session_key
        if request_id is None:
            raise RuntimeError(
                "admitted Nemotron session has no correlation identity"
            )
        input_stream: asyncio.Queue[list[int]] = asyncio.Queue()
        streaming_input = self._transcribe_nemotron_realtime(
            self._native_audio_controls(),
            input_stream,
        )
        self.generation_task = asyncio.create_task(
            self._run_generation(
                streaming_input,
                input_stream,
                request_id=request_id,
            )
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
        if (
            event_type == "session.update"
            and getattr(self, "_persistent_state_service", None) is not None
        ):
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
            if self._session_lifecycle.expired:
                await self.send_error(
                    "Session lifecycle has expired.",
                    "session_expired",
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
            self._arm_session_lifecycle_timeout("idle")
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
            generation_live = (
                self.generation_task is not None
                and not self.generation_task.done()
            )
            if bool(event.get("final", False)):
                if self._session_lifecycle.expired:
                    await self.send_error(
                        "Session lifecycle has expired.",
                        "session_expired",
                    )
                    return
                session.begin_finalize()
                self._native_finalized = True
                self._arm_session_lifecycle_timeout("finalization")
            elif generation_live:
                session.force_segment()
            self._native_fifo_event.set()
            if not generation_live:
                await self.start_generation()
            return
        return await super().handle_event(event)

    async def _configure_persistent_session(self, event: dict) -> None:
        """Reserve state, construct the session, then publish model admission."""
        async with self._admission_lock:
            await self._configure_persistent_session_locked(event)

    async def _configure_persistent_session_locked(self, event: dict) -> None:
        """Linearize one configuration/admission attempt per connection."""
        if not self._is_connected or self._unadmitted_timed_out:
            return
        model = event.get("model")
        if model is None:
            await self.send_error("Missing required field: model", "invalid_event")
            return
        err = self._check_model(model)
        if err is not None:
            await self.send_error(err.error.message, "model_not_found")
            return
        if self._is_model_validated:
            locale = event.get("locale")
            if locale is not None and self._nemotron_session is not None:
                await self._nemotron_session.update_locale(str(locale))
                assert self._effective_session is not None
                self._effective_session["locale"] = str(locale)
                await self._send_session_updated()
            return

        await self._cancel_configuration_timeout()
        now_ns = time.monotonic_ns()
        if now_ns < self._next_admission_retry_ns:
            retry_after_ms = max(
                1,
                (self._next_admission_retry_ns - now_ns + 999_999) // 1_000_000,
            )
            await self.send_error(
                "Persistent-state admission retry floor is active.",
                "capacity_exhausted",
                retry_after_ms=retry_after_ms,
            )
            return

        service = self._persistent_state_service
        assert service is not None
        check_health = getattr(
            service,
            "check_admission",
            getattr(service, "check_health", None),
        )
        reserve_operation_id = uuid4().hex
        session_key = f"rt-{self.connection_id}-{uuid4().hex}"
        cadence = str(event.get("cadence", "560ms"))
        try:
            service_interval_ms = int(cadence.removesuffix("ms"))
        except ValueError:
            await self.send_error("Invalid cadence.", "invalid_event")
            return
        try:
            if check_health is not None:
                await check_health()
            inventory = getattr(service, "inventory", None) or {}
            schema_id = str(inventory.get("schema_id", "state-manifest-v1"))
            profile_id = str(inventory.get("profile_id", "default"))
            lease = await service.reserve(
                operation_id=reserve_operation_id,
                session_key=session_key,
                schema_id=schema_id,
                profile_id=profile_id,
                service_interval_ms=service_interval_ms,
            )
        except PersistentStateUnsupportedServiceInterval as error:
            await self.send_error(
                str(error),
                "unsupported_service_interval",
            )
            return
        except (PersistentStateCapacityExhausted, PersistentStateBackpressure) as error:
            retry_after_ms = getattr(error, "retry_after_ms", None)
            if retry_after_ms is None:
                runtime = self._runtime_config
                retry_after_ms = int(
                    getattr(runtime, "admission_retry_floor_ms", 0)
                )
            await self.send_error(
                str(error),
                "capacity_exhausted",
                retry_after_ms=retry_after_ms,
            )
            self._next_admission_retry_ns = time.monotonic_ns() + int(
                retry_after_ms * 1_000_000
            )
            deadline = self._unadmitted_deadline_ns
            if (
                deadline is not None
                and self._next_admission_retry_ns >= deadline
            ):
                self._is_connected = False
                close = getattr(self.websocket, "close", None)
                if close is not None:
                    await close(code=1013)
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
        self._arm_pending_claim_timeout()
        if self._unadmitted_timed_out or not self._is_connected:
            await self._release_state_lease("unadmitted_connection_timeout")
            return
        if self._configuration_timed_out:
            await self._release_state_lease("configuration_timeout")
            return
        endpoint = event.get("endpointing") or {}
        resolved_endpoint = {
            "mode": str(endpoint.get("mode", "greedy_blank")),
            "stop_history_ms": int(endpoint.get("stop_history_ms", 800)),
            "residue_frames": int(endpoint.get("residue_frames", 2)),
        }
        try:
            model_config = getattr(self.serving, "model_config", None)
            if model_config is not None:
                runtime = self._runtime_config
                if runtime is None:
                    raise RuntimeError(
                        "admitted persistent-state session has no runtime envelope"
                    )
                hf = getattr(model_config, "hf_config", model_config)
                endpoint_policy = EndpointPolicy.resolve(
                    mode=resolved_endpoint["mode"],
                    stop_history_ms=resolved_endpoint["stop_history_ms"],
                    residue_frames=resolved_endpoint["residue_frames"],
                    frame_stride_ms=80,
                    history_capacity_frames=int(
                        getattr(hf, "endpoint_history_capacity_frames", 12)
                    ),
                )
                generation = int(getattr(lease, "generation"))
                engine_epoch = str(getattr(lease, "engine_epoch"))
                self._nemotron_session = NemotronRealtimeSession.from_model_config(
                    model_config,
                    cadence=cadence,
                    locale=str(event.get("locale", "auto")),
                    endpoint_policy=endpoint_policy,
                    request_id=session_key,
                    engine_epoch=engine_epoch,
                    lease_generation=generation,
                    observer=self._observer,
                    accepted_audio_budget_s=float(
                        runtime.accepted_audio_budget_s
                    ),
                    accepted_audio_capacity_samples=int(
                        runtime.accepted_audio_capacity_samples
                    ),
                    max_retained_transcript_bytes=int(
                        runtime.max_retained_transcript_bytes
                    ),
                    max_session_samples=runtime.max_session_samples,
                )
                self._park_token_id = self._nemotron_session.park_token_id
        except Exception:
            await self._release_state_lease("configuration_error")
            raise
        self._is_model_validated = True
        self._effective_session = {
            "model": str(model),
            "cadence": cadence,
            "locale": str(event.get("locale", "auto")),
            "endpointing": resolved_endpoint,
        }
        self._arm_session_lifecycle_timeout("idle")
        task = self._unadmitted_timeout_task
        self._unadmitted_timeout_task = None
        await self._cancel_task(task)
        await self._send_session_updated()

    async def _send_session_updated(self) -> None:
        """Emit the standard acknowledgement after configuration commits."""
        if self._effective_session is None:
            return
        await self.send_json(
            {
                "type": "session.updated",
                "session": self._effective_session,
            }
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
        prompt_token_ids_len = 0
        completion_tokens_len = 0
        self._realtime_audio_ref = None
        runtime = getattr(self, "_runtime_config", None)
        transcript_limit = (
            runtime.max_retained_transcript_bytes
            if runtime is not None
            else getattr(self, "max_retained_transcript_bytes", None)
        )
        transcript = BoundedTranscript(
            max_retained_bytes=int(
                (1 << 20) if transcript_limit is None else transcript_limit
            ),
            fragment_overhead_bytes=0,
            terminal_headroom_bytes=0,
        )
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
                    disconnected = True
                    break
            else:
                # The engine generator exhausted normally — on this path
                # the request finishes only through finalization/FLUSH,
                # so MODEL completion is decided here, before any
                # terminal send: transport delivery is not part of model
                # completion (PORT-OBS-006; review round F3).
                model_finalized = True
                await self._cancel_session_lifecycle_timeout()

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
                self._streaming_finish_reason = "completed"

                if sent_audio:
                    await self.send_json(
                        {
                            "type": "response.audio.done",
                            "has_audio": True,
                        }
                    )
                    audio_done_sent = True
        except asyncio.CancelledError as error:
            cancelled = error
        except OutputCapacityExceeded:
            self._streaming_finish_reason = "error"
            logger.exception("Realtime transcript capacity exhausted")
            if self._is_connected:
                await self.send_error(
                    "output_capacity_exceeded",
                    "output_capacity_exceeded",
                )
        except Exception as e:
            self._streaming_finish_reason = "error"
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
                if not getattr(self, "_streaming_session_finished", False):
                    observe_safely(
                        observer.session_finished,
                        session_key=request_id,
                        reason=reason,
                    )
                    self._streaming_session_finished = True
            if cancelled is not None:
                raise cancelled

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

    def _arm_pending_claim_timeout(self) -> None:
        service = self._persistent_state_service
        if service is None or self._state_lease is None:
            return
        timeout_s = float(service.pending_claim_timeout_s)
        self._pending_claim_task = asyncio.create_task(
            self._expire_pending_claim(timeout_s),
            name=f"persistent-state-claim-{self.connection_id}",
        )

    async def _cancel_session_lifecycle_timeout(self) -> None:
        await self._session_lifecycle.close()

    def _arm_session_lifecycle_timeout(
        self,
        kind: SessionLifecycleKind,
    ) -> None:
        self._session_lifecycle.arm(kind)

    async def _expire_session_lifecycle(
        self,
        kind: SessionLifecycleKind,
    ) -> None:
        code = "idle_timeout" if kind == "idle" else "finalization_timeout"
        self._streaming_finish_reason = (
            "aborted" if kind == "idle" else "error"
        )
        self._terminal_code = code
        if self._is_connected:
            await self.send_error(code, code)
        await self.cleanup()
        self._is_connected = False
        close = getattr(self.websocket, "close", None)
        if close is not None:
            await close(code=1008)

    def _finish_streaming_session_once(self, reason: str) -> None:
        session = self._nemotron_session
        if (
            session is not None
            and self._observer is not None
            and not getattr(self, "_streaming_session_finished", False)
        ):
            outcome = "aborted" if reason == "aborted" else "error"
            observe_safely(
                self._observer.clear_all_outstanding,
                session.session_key,
                outcome=outcome,
            )
            self._streaming_session_finished = True
            observe_safely(
                self._observer.session_finished,
                session_key=session.session_key,
                reason=reason,
            )

    async def _release_state_lease(
        self,
        reason: str,
    ) -> bool:
        service = self._persistent_state_service
        lease = self._state_lease
        operation_id = self._state_operation_id
        if (
            service is None
            or lease is None
            or operation_id is None
            or self._state_release_done
        ):
            return False
        async with self._state_cleanup_lock:
            if self._state_release_done:
                return False
            task = self._pending_claim_task
            if task is not None and task is not asyncio.current_task():
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)
            self._finish_streaming_session_once(
                "error"
                if reason in {
                    "configuration_error",
                    "pending_claim_timeout",
                }
                else self._streaming_finish_reason
            )
            await service.release(
                operation_id=operation_id,
                lease=lease,
                reason=reason,
            )
            self._state_release_done = True
            return True

    async def _expire_pending_claim(self, timeout_s: float) -> None:
        await asyncio.sleep(timeout_s)
        service = self._persistent_state_service
        lease = self._state_lease
        if service is None or lease is None or self._state_release_done:
            return
        async with self._state_cleanup_lock:
            if self._state_release_done:
                return
            won = await service.claim_pending_cleanup(lease)
            if not won:
                return
            generation_task = self.generation_task
            await super().cleanup()
            if generation_task is not None:
                await asyncio.gather(
                    generation_task,
                    return_exceptions=True,
                )
            self._finish_streaming_session_once("error")
            assert self._state_operation_id is not None
            await service.release(
                operation_id=self._state_operation_id,
                lease=lease,
                reason="pending_claim_timeout",
            )
            self._state_release_done = True

    async def cleanup(self):
        """Release resident state exactly once after transport cleanup begins."""
        await self._cancel_session_lifecycle_timeout()
        generation_task = self.generation_task
        await super().cleanup()
        if generation_task is not None:
            await asyncio.gather(generation_task, return_exceptions=True)
        await self._cancel_pre_admission_timeouts()
        await self._release_state_lease(
            getattr(self, "_terminal_code", None) or "connection_cleanup"
        )

    async def send_json(self, payload: dict):
        await self.websocket.send_text(json.dumps(payload))

    async def send_error(
        self,
        message: str,
        code: str | None = None,
        *,
        retry_after_ms: int | None = None,
    ) -> None:
        """Send the inherited error shape with optional Omni retry metadata."""
        error = ErrorEvent(
            error=message,
            code=code,
            retry_after_ms=retry_after_ms,
        )
        await self.websocket.send_text(
            error.model_dump_json(exclude_none=True)
        )
