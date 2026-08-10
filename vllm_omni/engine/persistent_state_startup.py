# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Typed cold-start composition for persistent-state serving."""

from __future__ import annotations

import asyncio
from fractions import Fraction
from typing import Any

from vllm_omni.engine.persistent_state_admission import (
    AdmissionControllerConfig,
)
from vllm_omni.engine.persistent_state_capacity import (
    ServiceRoundExecution,
    compile_provisional_service_profile,
)
from vllm_omni.engine.persistent_state_priming import (
    run_service_priming_round,
)
from vllm_omni.engine.persistent_state_service import PersistentStateService

_REQUIRED_STARTUP_INVENTORY = frozenset(
    {
        "configured_limit",
        "effective_capacity",
        "execution_claim_ceiling",
        "execution_environment_key",
        "physical_capacity",
        "precision_policy",
        "profile_id",
        "schema_id",
        "slot_bytes",
    }
)


# @spec PORT-STATE-027 / ENV-MIG-011
def derive_admission_controller_config(
    runtime_config: Any,
    *,
    supported_intervals_ms: tuple[int, ...],
) -> AdmissionControllerConfig:
    """Project the validated process envelope into controller-native units."""

    return AdmissionControllerConfig(
        waiter_capacity=int(runtime_config.admission_waiter_capacity),
        max_inflight_reserves=int(runtime_config.admission_max_inflight_reserves),
        dispatch_budget=int(runtime_config.admission_dispatch_budget),
        aging_threshold_ns=int(float(runtime_config.admission_aging_threshold_s) * 1_000_000_000),
        admission_wait_timeout_s=float(runtime_config.admission_wait_timeout_s),
        retry_floor_ms=int(runtime_config.admission_retry_floor_ms),
        retry_jitter_ms=int(runtime_config.admission_retry_jitter_ms),
        recovery_backoff_s=tuple(runtime_config.recovery_backoff_s),
        release_convergence_timeout_s=float(runtime_config.release_convergence_timeout_s),
        supported_intervals_ms=supported_intervals_ms,
    )


def _service_executions(observations: list[Any]) -> list[Any]:
    """Convert real priming observations to the arithmetic compiler input."""

    converted: list[Any] = []
    for observation in observations:
        required = (
            "tier_id",
            "active_population",
            "elapsed_ns",
            "service_interval_ms",
            "geometry_id",
            "completed_legal_parks",
            "completed_model_rows",
            "post_jit",
            "continuously_loaded",
            "dummy_run",
            "is_profile",
        )
        if not all(hasattr(observation, name) for name in required):
            return observations
        converted.append(
            ServiceRoundExecution(
                tier_id=str(observation.tier_id),
                active_population=int(observation.active_population),
                elapsed_ns=int(observation.elapsed_ns),
                service_interval_ms=int(observation.service_interval_ms),
                geometry_id=int(observation.geometry_id),
                completed_legal_parks=int(observation.completed_legal_parks),
                completed_model_rows=(
                    None if observation.completed_model_rows is None else int(observation.completed_model_rows)
                ),
                post_jit=bool(observation.post_jit),
                continuously_loaded=bool(observation.continuously_loaded),
                dummy_run=bool(observation.dummy_run),
                is_profile=bool(observation.is_profile),
                scenario_id=str(getattr(observation, "scenario_id", "ordinary")),
            )
        )
    return converted


# @spec ENV-MIG-012 / PORT-PERF-005 / PORT-PERF-006 / PORT-INT-013
async def prepare_persistent_state_service(
    *,
    engine_client: Any,
    stage_client: Any,
    runtime_config: Any,
    startup_provider: Any,
    host_fatal_callback: Any,
) -> PersistentStateService:
    """Bootstrap, prime, compile, and seal one selected state service."""

    service = PersistentStateService(
        stage_client,
        reserve_queue_capacity=int(runtime_config.reserve_queue_capacity),
        cleanup_queue_capacity=int(getattr(runtime_config, "cleanup_queue_capacity", 256)),
        operation_timeout_s=float(getattr(runtime_config, "operation_timeout_s", 10.0)),
        reconciliation_timeout_s=float(getattr(runtime_config, "reconciliation_timeout_s", 30.0)),
        tombstone_ttl_s=float(getattr(runtime_config, "tombstone_ttl_s", 3600.0)),
        max_tombstones=int(getattr(runtime_config, "max_tombstones", 4096)),
        pending_claim_timeout_s=float(getattr(runtime_config, "pending_claim_timeout_s", 30.0)),
        runtime_config=runtime_config,
        host_fatal_callback=host_fatal_callback,
    )
    try:
        inventory = await service.bootstrap_handshake()
        if inventory.get("resident_state_scatter_warmup_complete") is not True:
            raise RuntimeError("persistent-state resident scatter warmup attestation is absent")
        missing_inventory = sorted(_REQUIRED_STARTUP_INVENTORY - inventory.keys())
        if missing_inventory:
            raise RuntimeError("persistent-state startup inventory is incomplete: " + ", ".join(missing_inventory))
        plan = startup_provider.build_priming_plan(
            runtime_config=runtime_config,
            inventory=inventory,
            model_config=engine_client.model_config,
        )
        service.configure_bootstrap_intervals(plan.served_intervals_ms)

        async def run_plan() -> list[Any]:
            observations: list[Any] = []
            for round_spec in plan.rounds:

                async def execute(spec: Any, leases: tuple[Any, ...]) -> Any:
                    return await startup_provider.execute_priming_round(
                        engine_client=engine_client,
                        round_spec=spec,
                        leases=leases,
                    )

                observations.append(
                    await run_service_priming_round(
                        service=service,
                        round_spec=round_spec,
                        execute=execute,
                        rollback_timeout_s=float(runtime_config.release_convergence_timeout_s),
                    )
                )
            return observations

        observations = await asyncio.wait_for(
            run_plan(),
            timeout=float(runtime_config.startup_priming_timeout_s),
        )
        compile_kwargs = dict(plan.compile_kwargs)
        startup_receipt = dict(compile_kwargs.get("startup_priming_receipt", {}))
        if startup_receipt:
            startup_receipt.update(
                {
                    "configured_population_ceiling": (runtime_config.priming_configured_population_ceiling),
                    "runtime_tombstone_allowance": (runtime_config.runtime_tombstone_allowance),
                    "resolved_max_tombstones": runtime_config.max_tombstones,
                    "startup_priming_timeout_s": (runtime_config.startup_priming_timeout_s),
                }
            )
            compile_kwargs["startup_priming_receipt"] = startup_receipt
        if "derating_factor" in compile_kwargs:
            compile_kwargs["derating_factor"] = Fraction(compile_kwargs["derating_factor"])
        compiled = compile_provisional_service_profile(
            _service_executions(observations),
            **compile_kwargs,
        )
        admission_config = derive_admission_controller_config(
            runtime_config,
            supported_intervals_ms=compiled.compiled_demand.intervals_ms,
        )
        service.seal_startup_profile(
            admission_config=admission_config,
            compiled_service_profile=compiled,
        )
        return service
    except BaseException:
        service.shutdown()
        raise
