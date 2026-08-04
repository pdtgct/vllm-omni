# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Nemotron streaming scheduler policy.

The generic Omni AR scheduler appends streaming updates.  Nemotron's
checkpoint state is the history authority, so this scheduler replaces the
completed request envelope at each legal park while retaining the exact
persistent-state binding.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Protocol

from vllm.logger import init_logger
from vllm.v1.engine import EngineCoreEventType
from vllm.v1.request import Request, RequestStatus, StreamingUpdate

from vllm_omni.core.sched.omni_ar_scheduler import OmniARScheduler
from vllm_omni.core.sched.output import OmniSchedulerOutput
from vllm_omni.model_executor.persistent_state.manager import StateBinding

logger = init_logger(__name__)

_MIN_STREAMING_TRANSACTION_TOKENS = 142


class _PersistentStateRegistry(Protocol):
    def claim_pending_lease(self, **kwargs: Any) -> StateBinding: ...

    def mark_terminal(self, binding: StateBinding) -> None: ...


def _additional_information(value: object) -> Mapping[str, Any] | None:
    """Return a mapping view for the pin's two additional-info shapes."""

    if isinstance(value, Mapping):
        return value
    entries = getattr(value, "entries", None)
    return entries if isinstance(entries, Mapping) else None


def _binding_payload(request: Request) -> Mapping[str, Any] | None:
    information = _additional_information(
        getattr(request, "additional_information", None)
    )
    if information is None:
        return None
    binding = information.get("persistent_state_binding")
    return binding if isinstance(binding, Mapping) else None


class NemotronASRScheduler(OmniARScheduler):  # type: ignore[misc]
    """Scheduler policy for bounded, no-recompute Nemotron streams."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._claimed_state_bindings: dict[str, StateBinding] = {}
        self._validate_streaming_model_len(self.max_model_len)

    @staticmethod
    def _validate_streaming_model_len(max_model_len: int) -> None:
        if max_model_len < _MIN_STREAMING_TRANSACTION_TOKENS:
            raise ValueError(
                "max_model_len must contain one current streaming transaction "
                f"({_MIN_STREAMING_TRANSACTION_TOKENS} tokens)"
            )

    def _state_registry(self) -> _PersistentStateRegistry | None:
        return getattr(self, "persistent_state_registry", None)

    def _claim_initial_request(self, request: Request) -> bool:
        """Join an initial ADD to its already committed logical lease."""

        payload = _binding_payload(request)
        registry = self._state_registry()
        if payload is None or registry is None:
            request.status = RequestStatus.FINISHED_ERROR
            logger.error(
                "persistent-state lifecycle invariant failed for request %s",
                request.request_id,
            )
            return False
        try:
            binding = registry.claim_pending_lease(
                engine_epoch=payload["engine_epoch"],
                session_key=payload["session_key"],
                generation=payload["generation"],
                schema_id=payload["schema_id"],
                profile_id=payload["profile_id"],
                binding_token=payload.get("binding_token"),
            )
        except (KeyError, RuntimeError, ValueError):
            request.status = RequestStatus.FINISHED_ERROR
            logger.exception(
                "persistent-state lifecycle claim failed for request %s",
                request.request_id,
            )
            return False
        if not isinstance(binding, StateBinding):
            request.status = RequestStatus.FINISHED_ERROR
            logger.error(
                "persistent-state lifecycle claim returned an invalid binding"
            )
            return False
        self._claimed_state_bindings[request.request_id] = binding
        return True

    def _verify_streaming_readd(self, request: Request) -> StateBinding:
        """Verify that a parked update still names the claimed generation."""

        payload = _binding_payload(request)
        binding = self._claimed_state_bindings.get(request.request_id)
        if payload is None or binding is None:
            raise RuntimeError("streaming re-add lacks a claimed state binding")
        if payload.get("generation") != binding.generation:
            raise RuntimeError("streaming re-add generation mismatch")
        current_blocks = self.kv_cache_manager.get_blocks(request.request_id)
        if binding.slot_id not in {
            block_id
            for group in current_blocks.get_block_ids()
            for block_id in group
        }:
            raise RuntimeError("streaming re-add block binding mismatch")
        return binding

    def schedule(self, throttle_prefills: bool = False) -> OmniSchedulerOutput:
        """Claim logical state before the base scheduler admits initial work."""

        claim_pending_lease = self._claim_initial_request
        for request in tuple(self.waiting):
            payload = _binding_payload(request)
            if payload is None:
                continue
            # Keep the complete lease identity at this join point.  The helper
            # performs the atomic check; this validation makes an incomplete
            # API-side binding fail before base scheduling can allocate work.
            for field in (
                "engine_epoch",
                "session_key",
                "generation",
                "schema_id",
                "profile_id",
                "binding_token",
            ):
                if field not in payload:
                    request.status = RequestStatus.FINISHED_ERROR
                    break
            if request.request_id in self._claimed_state_bindings:
                self._verify_streaming_readd(request)
                continue
            if not claim_pending_lease(request):
                self.finish_requests(
                    request.request_id,
                    RequestStatus.FINISHED_ERROR,
                )

        output = super().schedule(throttle_prefills)
        output.persistent_state_bindings.update(
            {
                request_id: binding
                for request_id, binding in self._claimed_state_bindings.items()
                if request_id in output.num_scheduled_tokens
            }
        )
        return output

    def _preempt_request(self, request: Request, timestamp: float) -> None:
        """Turn an upstream PREEMPTED attempt into a no-recompute error."""

        del timestamp
        request.status = RequestStatus.FINISHED_ERROR
        logger.error(
            "persistent-state request %s cannot be recomputed after preemption",
            request.request_id,
        )
        self._free_request(request)

    def update_from_output(self, *args: Any, **kwargs: Any) -> Any:
        """Publish terminal identity without owning physical cleanup."""

        result = super().update_from_output(*args, **kwargs)
        registry = self._state_registry()
        if registry is not None:
            for request_id in tuple(self.finished_req_ids):
                binding = self._claimed_state_bindings.pop(request_id, None)
                if binding is not None:
                    registry.mark_terminal(binding)
        return result

    def _update_request_as_session(
        self,
        session: Request,
        update: StreamingUpdate,
    ) -> None:
        """Replace one legally parked transaction envelope in place.

        Core replaces on two legal paths: an update arriving for a
        session already parked (``WAITING_FOR_STREAMING_REQ``), or an
        update already queued when the park stop is processed — core's
        ``_handle_stopped_request`` then replaces atomically at the stop
        boundary, before any parked status is set, with the session
        still in its stop status (``FINISHED_STOPPED``). Both are legal
        parks; anything else (a running session, a length-capped or
        aborted stop) is the invariant violation this guard exists for.
        """

        if session.status not in (
            RequestStatus.WAITING_FOR_STREAMING_REQ,
            RequestStatus.FINISHED_STOPPED,
        ):
            raise RuntimeError(
                "streaming session must be at a legal park (parked "
                "waiting, or at its park-stop boundary) before "
                "replacement"
            )

        prompt_token_ids = list(update.prompt_token_ids or ())
        for feature in update.mm_features or ():
            position = feature.mm_position
            if position.offset < 0 or position.offset >= max(1, len(prompt_token_ids)):
                raise ValueError(
                    "multimodal feature position is outside the current prompt "
                    "for the replacement prompt"
                )

        request_id = session.request_id
        self._new_prompt_len_snapshot[request_id] = len(prompt_token_ids)
        if session.num_output_placeholders > 0:
            session.async_tokens_to_discard = 1
        session.num_output_placeholders = 0
        session.spec_token_ids = []

        original_information = getattr(session, "additional_information", None)
        original_mapping = _additional_information(original_information)
        original_binding = (
            original_mapping.get("persistent_state_binding")
            if original_mapping is not None
            else None
        )
        replacement_information = getattr(update, "additional_information", None)
        replacement_mapping = _additional_information(replacement_information)
        if isinstance(original_information, dict):
            merged_information = dict(replacement_mapping or {})
            if original_binding is not None:
                merged_information["persistent_state_binding"] = original_binding
            session.additional_information = merged_information
        elif replacement_information is not None:
            session.additional_information = replacement_information

        session.prompt_token_ids = prompt_token_ids
        session._all_token_ids[:] = prompt_token_ids
        session._output_token_ids.clear()
        session.num_prompt_tokens = len(prompt_token_ids)
        setattr(session, "num_computed_tokens", 0)
        session.prompt_embeds = None
        session.prompt_is_token_ids = None
        session.mm_features = list(update.mm_features or ())
        session.block_hashes.clear()
        session.update_block_hashes()
        session.arrival_time = update.arrival_time
        session.sampling_params = update.sampling_params
        session.max_tokens = update.max_tokens
        # Mirror the base scheduler: only a session counted into the
        # waiting-for-input population leaves it. On the stop-boundary
        # path the session was never parked, so there is nothing to
        # decrement.
        if session.status == RequestStatus.WAITING_FOR_STREAMING_REQ:
            self.num_waiting_for_streaming_input -= 1
        session.status = RequestStatus.WAITING

        if self.log_stats:
            session.record_event(EngineCoreEventType.QUEUED)
