# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Shared compiled persistent-state fixtures for engine and entrypoint tests."""

import importlib
from fractions import Fraction
from typing import Any


def compile_with_canaries(
    *,
    ordinary_ns: int = 100,
    forced_ns: int = 200,
    terminal_ns: int = 200,
    omit_scenario: str | None = None,
    completed_legal_parks: int = 2,
    continuously_loaded: bool = True,
    canary_interval_ms: int = 80,
    derating_factor: Fraction = Fraction(1, 2),
    admission_policy: str = "profile",
    control_upper_ns_by_window_and_population: dict[int, tuple[int, ...]] | None = None,
) -> Any:
    """Compile ordinary and exceptional rounds through the production compiler."""
    module = importlib.import_module("vllm_omni.engine.persistent_state_capacity")
    execution_type = module.ServiceRoundExecution
    executions = [
        execution_type(
            scenario_id="ordinary",
            tier_id="single",
            active_population=1,
            elapsed_ns=ordinary_ns,
            service_interval_ms=80,
            geometry_id=0,
            completed_legal_parks=1,
            completed_model_rows=None,
            post_jit=True,
            continuously_loaded=True,
            dummy_run=False,
            is_profile=False,
        )
    ]
    for scenario_id, elapsed_ns in (
        ("forced_eou_then_chunk", forced_ns),
        ("final_tail_then_flush", terminal_ns),
    ):
        if scenario_id == omit_scenario:
            continue
        executions.extend(
            (
                execution_type(
                    scenario_id=scenario_id,
                    tier_id="single",
                    active_population=1,
                    elapsed_ns=10_000,
                    service_interval_ms=canary_interval_ms,
                    geometry_id=0,
                    completed_legal_parks=completed_legal_parks,
                    completed_model_rows=None,
                    post_jit=False,
                    continuously_loaded=continuously_loaded,
                    dummy_run=False,
                    is_profile=False,
                ),
                execution_type(
                    scenario_id=scenario_id,
                    tier_id="single",
                    active_population=1,
                    elapsed_ns=elapsed_ns,
                    service_interval_ms=canary_interval_ms,
                    geometry_id=0,
                    completed_legal_parks=completed_legal_parks,
                    completed_model_rows=None,
                    post_jit=True,
                    continuously_loaded=continuously_loaded,
                    dummy_run=False,
                    is_profile=False,
                ),
            )
        )
    context = module.ServiceProfileContext(
        pre_override_physical_bound=2,
        allocated_pool=1,
        count_cap=1,
        execution_claim_ceiling=1,
        service_budget_source="measured_fallback",
        service_budget_coefficients=(),
        derating_factor=derating_factor,
        slot_bytes=6_314_936,
        execution_environment_key="control-canary-test",
        precision_policy="fp32",
        state_profile="nemotron-asr-fp32-v1",
        compiler_version="persistent-state-capacity-v3",
        mixed_composition_policy="periodic_limited_preemption_edf",
    )
    return module.compile_provisional_service_profile(
        tuple(executions),
        execution_tiers=(
            module.ServiceExecutionTier(
                tier_id="single",
                max_active_population=1,
            ),
        ),
        max_population=1,
        reference_interval_ms=80,
        reference_geometry_id=0,
        admitted_geometry_ids=(0,),
        trailing_rounds=1,
        derating_factor=derating_factor,
        context=context,
        control_upper_ns_by_window_and_population=control_upper_ns_by_window_and_population,
        control_dominance_sha256=None,
        evidence_class="probe",
        admission_policy=admission_policy,
    )
