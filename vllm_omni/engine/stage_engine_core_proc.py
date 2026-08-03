"""
Stage Core Process for vLLM-Omni V1 architecture.

StageEngineCoreProc inherits from vLLM's EngineCoreProc and runs the engine core
busy loop in a subprocess, communicating with StageEngineCoreClient via ZMQ.
"""

from __future__ import annotations

import contextlib
import os
import signal
import time
from typing import Any
from uuid import uuid4

import vllm.v1.engine.core as _vllm_engine_core_module
from vllm.logger import init_logger
from vllm.transformers_utils.config import (
    maybe_register_config_serialize_by_value,
)
from vllm.utils.system_utils import (
    decorate_logs,
    set_process_title,
)
from vllm.v1.engine import EngineCoreRequest, EngineCoreRequestType
from vllm.v1.engine.core import EngineCoreProc, EngineShutdownState
from vllm.v1.engine.utils import (
    EngineZmqAddresses,
    SignalCallback,
)

from vllm_omni.distributed.omni_coordinator import create_stage_coord_client
from vllm_omni.engine import OmniEngineCoreRequest
from vllm_omni.engine.persistent_state_config import (
    PersistentStateRuntimeConfig,
)
from vllm_omni.engine.serialization import deserialize_additional_information
from vllm_omni.engine.stage_init_utils import set_death_signal

logger = init_logger(__name__)


_SIGNAL_EXIT_BASE = 128


def _signal_exit_code(signum: int) -> int:
    """Return the conventional process exit code for signal-driven exits."""
    return _SIGNAL_EXIT_BASE + signum


def _install_omni_platform_for_stage_core() -> None:
    """Install Omni's platform before core cache-spec registration."""

    from vllm import platforms as vllm_platforms

    from vllm_omni.platforms import current_omni_platform

    if current_omni_platform.is_unspecified():
        raise RuntimeError("stage core requires a resolved Omni platform")
    vllm_platforms.current_platform = current_omni_platform


class StageEngineCoreProc(EngineCoreProc):
    """Stage-specific engine core process for vLLM-Omni.

    Inherits from EngineCoreProc and provides its own ``run_stage_core``
    entry point for launching in a subprocess.  Does **not** delegate to
    ``EngineCoreProc.run_engine_core()``.
    """

    def preprocess_add_request(
        self,
        request: EngineCoreRequest,
    ) -> tuple[Any, int]:
        """Restore Omni metadata after core constructs its base request."""

        scheduled, request_wave = super().preprocess_add_request(request)
        payload = getattr(request, "additional_information", None)
        if payload is not None:
            setattr(
                scheduled,
                "additional_information",
                deserialize_additional_information(payload),
            )
        return scheduled, request_wave

    def _persistent_state_manager(self) -> Any:
        from vllm_omni.model_executor.persistent_state.manager import (
            PersistentStateManager,
        )

        managers = self.scheduler.kv_cache_manager.coordinator.single_type_managers
        matches = [
            manager
            for manager in managers
            if isinstance(manager, PersistentStateManager)
        ]
        if len(matches) != 1:
            raise RuntimeError(
                "persistent_state requires exactly one resident manager"
            )
        manager = matches[0]
        runtime = PersistentStateRuntimeConfig.from_vllm_config(
            self.vllm_config
        )
        manager.configure_capacity(
            safety_reserve_slots=runtime.safety_reserve_slots,
            max_resident_sessions=runtime.max_resident_sessions,
        )
        return manager

    def _persistent_state_control(self) -> dict[str, Any]:
        control = getattr(self, "_persistent_state_control_state", None)
        if control is None:
            manager = self._persistent_state_manager()
            runtime = PersistentStateRuntimeConfig.from_vllm_config(
                self.vllm_config
            )
            if runtime.max_tombstones < 2 * manager.effective_capacity:
                raise ValueError(
                    "persistent_state_max_tombstones must reserve one "
                    "reserve and one release record per effective slot"
                )
            control = {
                "engine_epoch": manager.engine_epoch,
                "revision": 0,
                "operations": {},
                "pending": {},
                "claimed": {},
                "cleanup": {},
                "tombstone_ttl_s": runtime.tombstone_ttl_s,
                "max_tombstones": runtime.max_tombstones,
            }
            self._persistent_state_control_state = control
            self.scheduler.persistent_state_registry = self
        return control

    @staticmethod
    def _prune_persistent_state_operations(control: dict[str, Any]) -> None:
        """Expire completed operation records without evicting live ones."""

        now = time.monotonic()
        expired = [
            operation_id
            for operation_id, record in control["operations"].items()
            if float(record["expires_at"]) <= now
        ]
        for operation_id in expired:
            control["operations"].pop(operation_id)

    def _persistent_state_operation_result(
        self,
        control: dict[str, Any],
        operation_id: str,
    ) -> dict[str, Any] | None:
        self._prune_persistent_state_operations(control)
        record = control["operations"].get(operation_id)
        return None if record is None else record["result"]

    @staticmethod
    def _record_persistent_state_operation(
        control: dict[str, Any],
        operation_id: str,
        result: dict[str, Any],
    ) -> None:
        if len(control["operations"]) >= int(control["max_tombstones"]):
            raise RuntimeError(
                "persistent-state operation tombstone invariant exceeded"
            )
        control["operations"][operation_id] = {
            "result": result,
            "expires_at": time.monotonic()
            + float(control["tombstone_ttl_s"]),
        }

    def persistent_state_snapshot(self) -> dict[str, Any]:
        """Return the resident capability inventory without state content."""
        manager = self._persistent_state_manager()
        control = self._persistent_state_control()
        self._prune_persistent_state_operations(control)
        return {
            "engine_epoch": control["engine_epoch"],
            "manager_revision": control["revision"],
            "resident_count": len(manager._bindings),
            "physical_capacity": manager.physical_capacity,
            "safety_reserve": manager.safety_reserve_slots,
            "configured_limit": manager.configured_limit,
            "effective_capacity": manager.effective_capacity,
            "stage": manager.stage,
            "replica": manager.replica,
            "capabilities": ["resident"],
            "schema_id": manager.persistent_state_spec.schema_id,
            "profile_id": manager.profile_id,
            "persistent_state_tombstone_ttl_s": control[
                "tombstone_ttl_s"
            ],
            "persistent_state_max_tombstones": control["max_tombstones"],
        }

    def persistent_state_reserve(
        self,
        operation_id: str,
        session_key: str,
        schema_id: str,
        profile_id: str,
    ) -> dict[str, Any]:
        """Commit one pending persistent_state lease before audio admission."""
        manager = self._persistent_state_manager()
        control = self._persistent_state_control()
        previous = self._persistent_state_operation_result(
            control, operation_id
        )
        if previous is not None:
            return previous
        if (
            len(control["operations"]) + len(manager._bindings) + 2
            > int(control["max_tombstones"])
        ):
            raise RuntimeError(
                "persistent-state operation tombstone horizon is full"
            )
        if schema_id != manager.persistent_state_spec.schema_id:
            raise ValueError("persistent_state schema mismatch")
        if profile_id != manager.profile_id:
            raise ValueError("persistent_state profile mismatch")
        manager.allocate_new_blocks(session_key, 1, 1)
        binding = manager.get_state_binding(session_key)
        if binding is None:
            raise RuntimeError("persistent_state reservation lacks binding")
        binding_token = uuid4().hex
        lease = {
            "engine_epoch": binding.engine_epoch,
            "session_key": session_key,
            "generation": binding.generation,
            "schema_id": binding.schema_id,
            "profile_id": binding.profile_id,
            "location": "resident",
            "binding_token": binding_token,
        }
        control["pending"][binding_token] = binding
        control["revision"] += 1
        result = {
            "operation_id": operation_id,
            "manager_revision": control["revision"],
            "resident_count": len(manager._bindings),
            "lease": lease,
            "location_event": {
                "engine_epoch": binding.engine_epoch,
                "session_key": session_key,
                "generation": binding.generation,
                "location": "resident",
                "transition": "reserved",
            },
        }
        self._record_persistent_state_operation(
            control, operation_id, result
        )
        return result

    def claim_pending_lease(
        self,
        *,
        engine_epoch: str,
        session_key: str,
        generation: int,
        schema_id: str,
        profile_id: str,
        binding_token: str | None = None,
    ) -> Any:
        """Atomically join the initial scheduler ADD to a pending lease."""
        control = self._persistent_state_control()
        if binding_token is None:
            candidates = list(control["pending"].items())
            matches = [
                item
                for item in candidates
                if item[1].request_id == session_key
            ]
            if len(matches) != 1:
                raise ValueError("persistent_state pending claim is ambiguous")
            binding_token, binding = matches[0]
        else:
            binding = control["pending"].get(binding_token)
        expected = (
            engine_epoch,
            session_key,
            generation,
            schema_id,
            profile_id,
        )
        actual = (
            binding.engine_epoch,
            binding.request_id,
            binding.generation,
            binding.schema_id,
            binding.profile_id,
        ) if binding is not None else None
        if actual != expected:
            raise ValueError("persistent_state pending claim mismatch")
        control["pending"].pop(binding_token)
        control["claimed"][binding_token] = binding
        return binding

    def persistent_state_begin_pending_cleanup(
        self,
        lease: dict[str, Any],
    ) -> bool:
        """Atomically win cleanup against the initial scheduler claim.

        Returning ``False`` means the scheduler already claimed the exact
        generation. Returning ``True`` moves a still-pending binding into a
        cleanup-only state that the scheduler can no longer claim; physical
        release remains owned by the API cleanup operation.
        """

        control = self._persistent_state_control()
        binding_token = str(lease["binding_token"])
        expected = (
            str(lease["engine_epoch"]),
            str(lease["session_key"]),
            int(lease["generation"]),
            str(lease["schema_id"]),
            str(lease["profile_id"]),
        )
        binding = control["pending"].get(binding_token)
        if binding is not None:
            actual = (
                binding.engine_epoch,
                binding.request_id,
                binding.generation,
                binding.schema_id,
                binding.profile_id,
            )
            if actual != expected:
                raise ValueError("persistent_state pending cleanup mismatch")
            control["pending"].pop(binding_token)
            control["cleanup"][binding_token] = binding
            return True

        cleanup_binding = control["cleanup"].get(binding_token)
        if cleanup_binding is not None:
            actual = (
                cleanup_binding.engine_epoch,
                cleanup_binding.request_id,
                cleanup_binding.generation,
                cleanup_binding.schema_id,
                cleanup_binding.profile_id,
            )
            if actual != expected:
                raise ValueError("persistent_state cleanup binding mismatch")
            return True

        claimed = control["claimed"].get(binding_token)
        if claimed is not None:
            actual = (
                claimed.engine_epoch,
                claimed.request_id,
                claimed.generation,
                claimed.schema_id,
                claimed.profile_id,
            )
            if actual != expected:
                raise ValueError("persistent_state claimed cleanup mismatch")
            return False
        raise ValueError("persistent_state pending cleanup lease is stale")

    def mark_terminal(self, binding: Any) -> None:
        """Record model terminality while API cleanup retains ownership."""
        self._persistent_state_manager().mark_terminal(binding.request_id)

    def persistent_state_release(
        self,
        operation_id: str,
        lease: dict[str, Any],
        reason: str,
    ) -> dict[str, Any]:
        """Commit one idempotent persistent_state physical cleanup."""
        del reason
        manager = self._persistent_state_manager()
        control = self._persistent_state_control()
        previous = self._persistent_state_operation_result(
            control, operation_id
        )
        if previous is not None:
            return previous
        if (
            len(control["operations"]) + len(manager._bindings)
            > int(control["max_tombstones"])
        ):
            raise RuntimeError(
                "persistent-state cleanup tombstone headroom was not reserved"
            )
        binding_token = lease["binding_token"]
        binding = control["pending"].get(binding_token)
        owner = "pending"
        if binding is None:
            binding = control["cleanup"].get(binding_token)
            owner = "cleanup"
        if binding is None:
            claimed = control["claimed"].get(binding_token)
            if claimed is not None and not manager.is_terminal(
                claimed.request_id
            ):
                raise RuntimeError(
                    "persistent-state claimed lease is still running"
                )
            binding = claimed
            owner = "claimed"
        if binding is None:
            raise ValueError("persistent-state release lease is stale")
        expected = (
            str(lease["engine_epoch"]),
            str(lease["session_key"]),
            int(lease["generation"]),
            str(lease["schema_id"]),
            str(lease["profile_id"]),
        )
        actual = (
            binding.engine_epoch,
            binding.request_id,
            binding.generation,
            binding.schema_id,
            binding.profile_id,
        )
        if actual != expected:
            raise ValueError("persistent-state release lease mismatch")
        control[owner].pop(binding_token)
        manager.drop_lease(binding.request_id)
        generation = binding.generation
        session_key = binding.request_id
        control["revision"] += 1
        result = {
            "operation_id": operation_id,
            "manager_revision": control["revision"],
            "resident_count": len(manager._bindings),
            "location_event": {
                "engine_epoch": control["engine_epoch"],
                "session_key": session_key,
                "generation": generation,
                "location": "absent",
                "transition": "released",
            },
        }
        self._record_persistent_state_operation(
            control, operation_id, result
        )
        return result

    @staticmethod
    def run_stage_core(
        *args: Any,
        dp_rank: int = 0,
        local_dp_rank: int = 0,
        omni_coordinator_address: str | None = None,
        omni_stage_id: int | None = None,
        omni_replica_id: int = 0,
        **kwargs: Any,
    ) -> None:
        """Launch StageEngineCoreProc busy loop in background process.

        Omni-specific kwargs:
          - ``omni_coordinator_address``: ROUTER address of the head-side
            :class:`OmniCoordinator`. When provided, this subprocess
            instantiates an :class:`OmniCoordClientForStage` after the
            HELLO/INIT/READY handshake completes and reports its status +
            queue length via heartbeats. The hook is wired so each
            heartbeat refreshes ``queue_length`` from the live scheduler.
          - ``omni_stage_id``: logical stage id this replica belongs to.
            Required when ``omni_coordinator_address`` is provided.
          - ``omni_replica_id``: cluster-unique replica id within the
            stage (assigned by :class:`OmniMasterServer`). Used for
            logging / metrics only.
        """
        signal_callback: SignalCallback | None = None
        maybe_register_config_serialize_by_value()
        _install_omni_platform_for_stage_core()

        # Register vllm-omni reasoning parsers (e.g. step_audio) in this
        # subprocess so they are available when the engine core resolves
        # ``--reasoning-parser``.  The main process already registered them
        # at import time, but the forked subprocess starts with a fresh
        # ReasoningParserManager.
        try:
            import vllm_omni.reasoning  # noqa: F401
        except ImportError:
            logger.warning(
                "Failed to import vllm_omni.reasoning in subprocess; "
                "custom reasoning parsers (e.g. step_audio) will not be "
                "available."
            )

        engine_core: StageEngineCoreProc | None = None
        coord_client = None
        try:
            # NOTE: previous revisions hardcoded data_parallel_size=1 here
            # (TODO referencing issue #984). The hardcoding has been removed
            # so the DP fields propagate through from the caller exactly
            # like upstream vLLM.

            stage_label = f"stage{omni_stage_id}" if omni_stage_id is not None else "noid"
            set_death_signal(signal.SIGTERM)
            set_process_title(f"StageEngineCoreProc_{stage_label}_replica{omni_replica_id}_DP{dp_rank}")
            decorate_logs()
            # Workaround for flashinfer/jit-cache version mismatch in CI.
            # The parent process handles this gracefully via ring_globals.py,
            # but the subprocess hits an unprotected import in TopKTopPSampler.
            # Setting this env var allows the same graceful fallback to work.
            os.environ.setdefault("FLASHINFER_DISABLE_VERSION_CHECK", "1")
            os.environ["VLLM_OMNI_REPLICA_ID"] = str(max(int(omni_replica_id), 0))

            # Patch the decoder type so process_input_sockets (started
            # during __init__) decodes OmniEngineCoreRequest (which
            # carries additional_information) instead of the base
            # EngineCoreRequest.  Must happen BEFORE __init__ because
            # the IO thread creates MsgpackDecoder(EngineCoreRequest)
            # during __init__.
            _vllm_engine_core_module.EngineCoreRequest = OmniEngineCoreRequest
            logger.debug(
                "[StageEngineCoreProc] Patched EngineCoreRequest -> OmniEngineCoreRequest: %s",
                _vllm_engine_core_module.EngineCoreRequest,
            )

            engine_core = StageEngineCoreProc(
                *args,
                engine_index=dp_rank,
                **kwargs,
            )

            # Each subprocess corresponds to exactly one omni replica with
            # its own OmniMasterServer allocation, so the heartbeat client
            # runs unconditionally — there is no dp_rank-based gating.
            if omni_coordinator_address is not None:
                if omni_stage_id is None:
                    raise ValueError("omni_stage_id must be provided when omni_coordinator_address is set")
                addresses: EngineZmqAddresses = engine_core.addresses
                if not addresses.inputs or not addresses.outputs:
                    raise RuntimeError(
                        "EngineCore handshake did not populate input/output addresses; "
                        "cannot start OmniCoordClientForStage"
                    )
                scheduler = getattr(engine_core, "scheduler", None)
                if scheduler is None:
                    raise RuntimeError("EngineCore scheduler is not initialized")
                coord_client = create_stage_coord_client(
                    coord_zmq_addr=omni_coordinator_address,
                    input_addr=addresses.inputs[0],
                    output_addr=addresses.outputs[0],
                    stage_id=int(omni_stage_id),
                    queue_length_getter=scheduler.get_num_unfinished_requests,
                )

            def wakeup_engine() -> None:
                engine_core.input_queue.put_nowait((EngineCoreRequestType.WAKEUP, None))

            signal_callback = SignalCallback(wakeup_engine)

            def signal_handler(signum: int, frame: Any) -> None:
                engine_core.shutdown_state = EngineShutdownState.REQUESTED
                signal_callback.trigger()
                raise SystemExit(_signal_exit_code(signum))

            signal.signal(signal.SIGTERM, signal_handler)
            signal.signal(signal.SIGINT, signal_handler)

            engine_core.run_busy_loop()

        except SystemExit:
            logger.debug("StageEngineCoreProc exiting.")
            raise
        except Exception:
            if engine_core is None:
                logger.exception("StageEngineCoreProc failed to start.")
            else:
                logger.exception("StageEngineCoreProc encountered a fatal error.")
                engine_core._send_engine_dead()
            raise
        finally:
            signal.signal(signal.SIGTERM, signal.SIG_DFL)
            signal.signal(signal.SIGINT, signal.SIG_DFL)
            if signal_callback is not None:
                signal_callback.stop()
            if coord_client is not None:
                with contextlib.suppress(RuntimeError):
                    coord_client.close()
            if engine_core is not None:
                engine_core.shutdown()
