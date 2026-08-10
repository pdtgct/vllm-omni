# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Model-owned startup priming plan for Nemotron streaming ASR."""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from fractions import Fraction
from types import SimpleNamespace
from typing import Any

from vllm_omni.engine.persistent_state_capacity import (
    ServiceExecutionTier,
    ServiceProfileContext,
)
from vllm_omni.engine.persistent_state_priming import (
    ServicePrimingBudgetCell,
    ServicePrimingBudgetDescriptor,
    ServicePrimingRound,
    validate_service_priming_plan,
)
from vllm_omni.model_executor.models.nemotron_asr.manifests import (
    CADENCES,
    RAW_SAMPLES_PER_CHUNK,
)

_PRIMING_SCENARIOS = (
    "ordinary",
    "forced_eou_then_chunk",
    "final_tail_then_flush",
)
_PRIMING_BUDGET_ANCHOR_COUNT = 11


def _max_visited_populations(population: int) -> int:
    if population == 1:
        return 1
    bisection_depth = (population - 2).bit_length()
    return min(
        population,
        _PRIMING_BUDGET_ANCHOR_COUNT + bisection_depth,
    )


@dataclass(frozen=True)
class NemotronServicePrimingPlan:
    """Finite eager-runner geometry plan and arithmetic compiler inputs."""

    rounds: tuple[ServicePrimingRound, ...]
    compile_kwargs: dict[str, Any]
    actual_plan_sha256: str
    actual_operation_count: int
    served_intervals_ms: tuple[int, ...]


class NemotronPersistentStateStartupProvider:
    """Construct ordinary synthetic requests below the public session API."""

    def build_priming_budget_descriptor(
        self,
        *,
        configured_population_ceiling: int,
        trailing_rounds: int,
    ) -> ServicePrimingBudgetDescriptor:
        """Declare the maximum plan before either control plane exists."""

        if configured_population_ceiling <= 0 or trailing_rounds <= 0:
            raise ValueError("priming budget inputs must be positive")
        tiers = [("single", 1)]
        if configured_population_ceiling > 1:
            tiers.append(("eager-bulk", configured_population_ceiling))
        repetitions = trailing_rounds + 1
        max_visited = _max_visited_populations(configured_population_ceiling)
        canary_populations = 1 if configured_population_ceiling == 1 else 2
        declared_operations = (
            2 * len(CADENCES) * repetitions * configured_population_ceiling * (max_visited + 2 * canary_populations)
        )
        return ServicePrimingBudgetDescriptor(
            policy_version="nemotron-service-priming-v2",
            configured_population_ceiling=configured_population_ceiling,
            cells=tuple(
                ServicePrimingBudgetCell(
                    geometry_id=geometry_id,
                    tier_id=tier_id,
                    scenario_id=scenario_id,
                    max_active_population=population,
                    repetitions=repetitions,
                )
                for geometry_id in range(len(CADENCES))
                for tier_id, population in tiers
                for scenario_id in _PRIMING_SCENARIOS
            ),
            declared_bootstrap_operation_budget=declared_operations,
        )

    def build_priming_plan(
        self,
        *,
        runtime_config: Any,
        inventory: dict[str, Any],
        model_config: Any,
    ) -> NemotronServicePrimingPlan:
        """Build the served configuration's covered executable plan."""

        maximum_population = min(
            int(inventory["effective_capacity"]),
            int(inventory["execution_claim_ceiling"]),
        )
        if maximum_population <= 0:
            raise RuntimeError("persistent-state priming requires positive effective capacity")
        trailing_rounds = int(runtime_config.service_profile_trailing_rounds)
        tiers = [
            ServiceExecutionTier(
                tier_id="single",
                max_active_population=1,
            )
        ]
        if maximum_population > 1:
            tiers.append(
                ServiceExecutionTier(
                    tier_id="eager-bulk",
                    max_active_population=maximum_population,
                )
            )
        hf_config = getattr(model_config, "hf_config", model_config)
        declared_lookaheads = getattr(
            hf_config,
            "supported_num_lookahead_tokens",
            None,
        )
        supported = None if declared_lookaheads is None else {int(value) for value in declared_lookaheads}
        admitted_geometries = tuple(
            (geometry_id, cadence)
            for geometry_id, (cadence, (_, lookahead)) in enumerate(CADENCES.items())
            if supported is None or lookahead in supported
        )
        if not admitted_geometries:
            raise ValueError("served configuration declares no supported manifest geometry")

        rounds: list[ServicePrimingRound] = []
        for geometry_id, cadence in admitted_geometries:
            service_interval_ms = int(cadence.removesuffix("ms"))
            for tier in tiers:
                for scenario_id in _PRIMING_SCENARIOS:
                    for repeat in range(trailing_rounds + 1):
                        rounds.append(
                            ServicePrimingRound(
                                round_id=(f"geometry-{geometry_id}-{tier.tier_id}-{scenario_id}-repeat-{repeat}"),
                                service_interval_ms=service_interval_ms,
                                geometry_id=geometry_id,
                                active_population=tier.max_active_population,
                                schema_id=str(inventory["schema_id"]),
                                profile_id=str(inventory["profile_id"]),
                                tier_id=tier.tier_id,
                                post_jit=repeat > 0,
                                continuously_loaded=True,
                                scenario_id=scenario_id,
                                expected_legal_parks_per_lease=(1 if scenario_id == "ordinary" else 2),
                            )
                        )
        derating = Fraction(str(runtime_config.service_profile_derating_factor))
        rounds_tuple = tuple(rounds)
        budget = runtime_config.priming_budget_descriptor
        plan_identity = validate_service_priming_plan(
            descriptor=budget,
            rounds=rounds_tuple,
        )
        context = ServiceProfileContext(
            pre_override_physical_bound=int(inventory["physical_capacity"]),
            allocated_pool=int(inventory["physical_capacity"]),
            count_cap=int(inventory["configured_limit"]),
            execution_claim_ceiling=int(inventory["execution_claim_ceiling"]),
            service_budget_source="measured_fallback",
            service_budget_coefficients=(),
            derating_factor=derating,
            slot_bytes=int(inventory["slot_bytes"]),
            execution_environment_key=str(inventory["execution_environment_key"]),
            precision_policy=str(inventory["precision_policy"]),
            state_profile=str(inventory["profile_id"]),
            compiler_version="persistent-state-service-v1",
            mixed_composition_policy="periodic_limited_preemption_edf",
        )
        reference_geometry_id, reference_cadence = max(
            admitted_geometries,
            key=lambda item: int(item[1].removesuffix("ms")),
        )
        return NemotronServicePrimingPlan(
            rounds=rounds_tuple,
            compile_kwargs={
                "execution_tiers": tuple(tiers),
                "max_population": maximum_population,
                "reference_interval_ms": int(reference_cadence.removesuffix("ms")),
                "reference_geometry_id": reference_geometry_id,
                "admitted_geometry_ids": tuple(geometry_id for geometry_id, _ in admitted_geometries),
                "trailing_rounds": trailing_rounds,
                "derating_factor": derating,
                "context": context,
                "evidence_class": "probe",
                "startup_priming_receipt": {
                    "budget_sha256": budget.sha256,
                    "bootstrap_operation_budget": (budget.bootstrap_operation_budget),
                    "actual_plan_sha256": plan_identity.sha256,
                    "actual_operation_count": plan_identity.operation_count,
                },
            },
            actual_plan_sha256=plan_identity.sha256,
            actual_operation_count=plan_identity.operation_count,
            served_intervals_ms=tuple(int(cadence.removesuffix("ms")) for _, cadence in admitted_geometries),
        )

    async def execute_priming_round(
        self,
        *,
        engine_client: Any,
        round_spec: ServicePrimingRound,
        leases: tuple[Any, ...],
    ) -> Any:
        """Drive one complete homogeneous round through legal park."""

        import numpy as np

        from vllm_omni.entrypoints.nemotron_session import (
            NemotronSessionLease,
        )
        from vllm_omni.model_executor.models.nemotron_asr.session import (
            NemotronRealtimeSession,
        )

        cadence = f"{round_spec.service_interval_ms}ms"
        if cadence not in CADENCES:
            raise ValueError(f"unknown priming cadence {cadence}")
        hf_config = getattr(
            engine_client.model_config,
            "hf_config",
            engine_client.model_config,
        )
        prompts = getattr(hf_config, "prompt_dictionary", None)
        if not isinstance(prompts, dict) or not prompts:
            raise RuntimeError("persistent-state priming requires the served prompt dictionary")
        locale = str(next(iter(prompts)))
        bound: list[NemotronSessionLease] = []
        for state_lease in leases:
            session = NemotronRealtimeSession.from_model_config(
                engine_client.model_config,
                cadence=cadence,
                locale=locale,
                with_ledger=True,
                request_id=str(state_lease.session_key),
                engine_epoch=str(state_lease.engine_epoch),
                lease_generation=int(state_lease.generation),
            )
            bound.append(
                NemotronSessionLease(
                    engine=engine_client,
                    session=session,
                    request_id=str(state_lease.session_key),
                    state_lease=state_lease,
                )
            )
        samples = np.zeros(
            RAW_SAMPLES_PER_CHUNK[cadence],
            dtype=np.float32,
        )
        started_ns = time.monotonic_ns()
        try:
            if round_spec.scenario_id == "ordinary":
                await asyncio.gather(*(lease.feed(samples) for lease in bound))
                completed_legal_parks = len(bound)
            elif round_spec.scenario_id == "forced_eou_then_chunk":
                await asyncio.gather(*(lease.force_segment() for lease in bound))
                await asyncio.gather(*(lease.feed(samples) for lease in bound))
                completed_legal_parks = 2 * len(bound)
            elif round_spec.scenario_id == "final_tail_then_flush":
                await asyncio.gather(*(lease.feed(samples[:-1]) for lease in bound))
                await asyncio.gather(*(lease.flush() for lease in bound))
                completed_legal_parks = 2 * len(bound)
            else:
                raise ValueError(f"unknown persistent-state priming scenario {round_spec.scenario_id}")
            elapsed_ns = time.monotonic_ns() - started_ns
        finally:
            await asyncio.gather(
                *(lease.abort() for lease in bound),
                return_exceptions=True,
            )
        return SimpleNamespace(
            completed_legal_parks=completed_legal_parks,
            elapsed_ns=elapsed_ns,
            completed_model_rows=None,
            dummy_run=False,
            is_profile=False,
        )


NEMOTRON_PERSISTENT_STATE_STARTUP = NemotronPersistentStateStartupProvider()


__all__ = [
    "NEMOTRON_PERSISTENT_STATE_STARTUP",
    "NemotronPersistentStateStartupProvider",
    "NemotronServicePrimingPlan",
]
