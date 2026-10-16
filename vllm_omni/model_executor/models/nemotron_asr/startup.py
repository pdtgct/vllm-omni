# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Model-owned startup priming plan for Nemotron streaming ASR."""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from fractions import Fraction
from types import SimpleNamespace
from typing import Any

from vllm.logger import init_logger

from vllm_omni.engine.persistent_state_capacity import (
    CanaryChunkTiming,
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

logger = init_logger(__name__)

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
class NemotronPrimingRound(ServicePrimingRound):
    """Detached model selection; no configuration object survives plan build."""

    outer_constants: tuple[tuple[str, int], ...] | None = None


@dataclass(frozen=True)
class OuterPrimingContext:
    worker_pid: int
    identity: tuple[Any, ...]
    pre_fused: int
    post_fused: int


@dataclass(frozen=True)
class NemotronServicePrimingPlan:
    """Finite eager-runner geometry plan and arithmetic compiler inputs."""

    rounds: tuple[ServicePrimingRound, ...]
    compile_kwargs: dict[str, Any]
    actual_plan_sha256: str
    actual_operation_count: int
    served_intervals_ms: tuple[int, ...]
    outer_constants: tuple[tuple[str, int], ...] | None = None


class NemotronPersistentStateStartupProvider:
    """Construct ordinary synthetic requests below the public session API."""

    def served_intervals_ms(self, *, model_config: Any) -> tuple[int, ...]:
        """Resolve model-declared cadence support without synthetic inputs."""

        hf_config = getattr(model_config, "hf_config", model_config)
        declared = getattr(
            hf_config,
            "supported_num_lookahead_tokens",
            None,
        )
        supported = None if declared is None else {int(value) for value in declared}
        intervals = tuple(
            int(cadence.removesuffix("ms"))
            for cadence, (_, lookahead) in CADENCES.items()
            if supported is None or lookahead in supported
        )
        if not intervals:
            raise ValueError("served configuration declares no supported manifest geometry")
        return intervals

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
        served_intervals = self.served_intervals_ms(model_config=model_config)
        admitted_geometries = tuple(
            (geometry_id, cadence)
            for geometry_id, (cadence, (_, lookahead)) in enumerate(CADENCES.items())
            if int(cadence.removesuffix("ms")) in served_intervals
        )
        if not admitted_geometries:
            raise ValueError("served configuration declares no supported manifest geometry")

        hf_config = getattr(model_config, "hf_config", model_config)
        selected = getattr(hf_config, "experimental_outer_tensor_fusion", False)
        if type(selected) is not bool:
            raise ValueError("experimental_outer_tensor_fusion must be a boolean")
        outer_constants = None
        if selected:
            from vllm_omni.model_executor.models.nemotron_asr.outer_tensor_fusion import model_constants

            outer_constants = tuple(model_constants(hf_config).items())
        rounds: list[ServicePrimingRound] = []
        for geometry_id, cadence in admitted_geometries:
            service_interval_ms = int(cadence.removesuffix("ms"))
            for tier in tiers:
                for scenario_id in _PRIMING_SCENARIOS:
                    for repeat in range(trailing_rounds + 1):
                        rounds.append(
                            NemotronPrimingRound(
                                outer_constants=outer_constants,
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
            compiler_version="persistent-state-service-v2",
            mixed_composition_policy="periodic_limited_preemption_edf",
        )
        reference_geometry_id, reference_cadence = max(
            admitted_geometries,
            key=lambda item: int(item[1].removesuffix("ms")),
        )
        return NemotronServicePrimingPlan(
            outer_constants=outer_constants,
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

    # @spec PORT-PERF-025, PORT-PERF-022
    async def validate_priming_boundary(
        self,
        *,
        engine_client: Any,
        plan: NemotronServicePrimingPlan,
        phase: str,
        boundary_context: OuterPrimingContext | None = None,
    ) -> OuterPrimingContext | None:
        if plan.outer_constants is None:
            return None
        snapshot = await _outer_snapshot(engine_client, plan.outer_constants, boundary_context)
        if phase == "before":
            return snapshot
        if phase != "after" or boundary_context is None:
            raise ValueError("invalid outer tensor fusion startup boundary")
        _positive_outer_delta(boundary_context, snapshot)
        return boundary_context

    # @spec PORT-STATE-026, PORT-PERF-005, PORT-PERF-006, PORT-PERF-007
    async def execute_priming_round(
        self,
        *,
        engine_client: Any,
        round_spec: ServicePrimingRound,
        leases: tuple[Any, ...],
        boundary_context: OuterPrimingContext | None = None,
    ) -> Any:
        """Drive one complete homogeneous round through legal park."""

        import numpy as np

        from vllm_omni.entrypoints.nemotron_session import (
            NemotronSessionLease,
        )
        from vllm_omni.model_executor.models.nemotron_asr.profiling import (
            end_startup_round,
            start_startup_round,
        )
        from vllm_omni.model_executor.models.nemotron_asr.session import (
            NemotronRealtimeSession,
        )

        outer_constants = getattr(round_spec, "outer_constants", None)
        before_outer = None
        if outer_constants is not None:
            if boundary_context is None:
                raise RuntimeError("outer tensor fusion startup lacks its pre-plan context")
            before_outer = await _outer_snapshot(engine_client, outer_constants, boundary_context)
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
        range_id: int | None = None
        try:
            range_id = start_startup_round(round_spec.round_id)
            started_ns = time.monotonic_ns()
            canary_wait_ns = 0
            canary_wait_start_ns = None
            canary_timer_start_ns = None
            following_chunk_timings: tuple[CanaryChunkTiming, ...] = ()
            if round_spec.scenario_id == "ordinary":
                await asyncio.gather(*(lease.feed(samples) for lease in bound))
                completed_legal_parks = len(bound)
            elif round_spec.scenario_id == "forced_eou_then_chunk":
                # An EOU token is a continuing control row. Register the fresh
                # request through the ordinary CHUNK path first; that common
                # setup work is deliberately outside the two-park canary.
                await asyncio.gather(*(lease.feed(samples) for lease in bound))
                await asyncio.gather(*(lease.wait_for_legal_park() for lease in bound))
                release_timings = tuple(lease.stage_forced_eou_then_chunk(samples) for lease in bound)
                eligible_ns = max(timing.eligible_ns for timing in release_timings)
                canary_wait_start_ns = time.monotonic_ns()
                now_ns = canary_wait_start_ns
                while now_ns < eligible_ns:
                    await asyncio.sleep((eligible_ns - now_ns) / 1_000_000_000)
                    now_ns = time.monotonic_ns()
                started_ns = time.monotonic_ns()
                canary_wait_ns = started_ns - canary_wait_start_ns
                canary_timer_start_ns = started_ns
                await asyncio.gather(*(lease.dispatch_staged_priming_chain() for lease in bound))
                completed_legal_parks = 2 * len(bound)
            elif round_spec.scenario_id == "final_tail_then_flush":
                await asyncio.gather(*(lease.feed(samples[:-1]) for lease in bound))
                await asyncio.gather(*(lease.flush() for lease in bound))
                completed_legal_parks = 2 * len(bound)
            else:
                raise ValueError(f"unknown persistent-state priming scenario {round_spec.scenario_id}")
            elapsed_ns = time.monotonic_ns() - started_ns
            ending_range_id = range_id
            range_id = None
            end_startup_round(ending_range_id)
            if round_spec.scenario_id == "forced_eou_then_chunk":
                timings = []
                for lease, timing in zip(bound, release_timings):
                    submitted_ns = lease.session.accepted_audio.prior_actual_submission_ns
                    if submitted_ns is None:
                        raise RuntimeError("canary following CHUNK lacks a submission timestamp")
                    timings.append(
                        CanaryChunkTiming(
                            true_ready_ns=timing.true_ready_ns,
                            prior_actual_submission_ns=timing.prior_actual_submission_ns,
                            cadence_ns=timing.cadence_ns,
                            eligible_ns=timing.eligible_ns,
                            submitted_ns=submitted_ns,
                        )
                    )
                following_chunk_timings = tuple(timings)
        finally:
            try:
                if range_id is not None:
                    ending_range_id = range_id
                    range_id = None
                    try:
                        end_startup_round(ending_range_id)
                    except BaseException:
                        # A body error is already propagating; retain its cause.
                        logger.warning("NVTX startup range cleanup failed", exc_info=True)
            finally:
                await asyncio.gather(
                    *(lease.abort() for lease in bound),
                    return_exceptions=True,
                )
        if before_outer is not None:
            after_outer = await _outer_snapshot(engine_client, outer_constants, boundary_context)
            _check_outer_monotone(before_outer, after_outer)
            # The forced chain requires exactly EOU then CHUNK in the existing
            # session authority; successful completion exercises zero-CHUNK too.
            # Final-tail/flush may legitimately perform no model work.
            if round_spec.scenario_id in {"ordinary", "forced_eou_then_chunk"}:
                _positive_outer_delta(before_outer, after_outer)
        return SimpleNamespace(
            completed_legal_parks=completed_legal_parks,
            elapsed_ns=elapsed_ns,
            canary_wait_ns=canary_wait_ns,
            canary_wait_start_ns=canary_wait_start_ns,
            canary_timer_start_ns=canary_timer_start_ns,
            following_chunk_timings=following_chunk_timings,
            completed_model_rows=None,
            dummy_run=False,
            is_profile=False,
        )


def _check_outer_monotone(before: OuterPrimingContext, after: OuterPrimingContext) -> None:
    if (
        before.worker_pid != after.worker_pid
        or before.identity != after.identity
        or after.pre_fused < before.pre_fused
        or after.post_fused < before.post_fused
    ):
        raise RuntimeError("outer tensor fusion worker identity/counters changed")


def _positive_outer_delta(before: OuterPrimingContext, after: OuterPrimingContext) -> None:
    _check_outer_monotone(before, after)
    if after.pre_fused <= before.pre_fused or after.post_fused <= before.post_fused:
        raise RuntimeError("outer tensor fusion startup has no positive executing-region coverage")


def _nonnegative_count(value: Any) -> int:
    if type(value) is not int or value < 0:
        raise ValueError("outer tensor fusion receipt counter is not a nonnegative integer")
    return value


async def _outer_snapshot(
    engine: Any,
    expected_constants: tuple[tuple[str, int], ...],
    baseline: OuterPrimingContext | None,
) -> OuterPrimingContext:
    from vllm_omni.model_executor.models.nemotron_asr.outer_tensor_fusion import FAMILIES, PROFILE, _freeze

    responses = await engine.collective_rpc("nemotron_execution_profile_receipt")
    if not isinstance(responses, (list, tuple)) or len(responses) != 1:
        raise RuntimeError("outer tensor fusion requires exactly one stage/replica receipt")
    worker_responses = responses[0]
    if not isinstance(worker_responses, (list, tuple)) or len(worker_responses) != 1:
        raise RuntimeError("outer tensor fusion requires exactly one expected worker receipt")
    envelope = worker_responses[0]
    if not isinstance(envelope, dict) or "persistent_state_warmup_complete" not in envelope:
        raise RuntimeError("outer tensor fusion worker receipt envelope is malformed")
    if envelope.get("persistent_state_warmup_complete") is not True:
        raise RuntimeError("outer tensor fusion worker warmup is incomplete")
    pid = envelope.get("worker_pid")
    if type(pid) is not int or pid <= 0:
        raise ValueError("outer tensor fusion worker PID is missing")
    receipt = envelope.get("execution_profile", {})
    block = receipt.get("outer_tensor_fusion")
    if not isinstance(block, dict):
        raise RuntimeError("outer tensor fusion receipt is missing")
    if (
        receipt.get("schema") != "nemotron-execution-profile/1"
        or receipt.get("emission") != {"lane": "native_burst", "profile": PROFILE}
        or receipt.get("encoder", {}).get("arm") != "eager-graphed"
        or receipt.get("encoder", {}).get("ready") is not True
        or receipt.get("decode", {}).get("arm") != "dense-graphed"
        or receipt.get("decode", {}).get("ready") is not True
        or receipt.get("decode", {}).get("captured_keys") != [[g, 1] for g in range(5)]
        or receipt.get("batch_invariant_mode", "off") != "off"
        or block.get("schema") != "nemotron-outer-tensor-fusion/1"
        or block.get("arm") != "compiled-static"
        or block.get("profile") != PROFILE
        or block.get("constants") != dict(expected_constants)
        or any(block.get(field) is not True for field in ("warmup_complete", "sealed", "ready"))
        or block.get("prepared") != list(FAMILIES)
        or block.get("real_profile_covered") != list(FAMILIES)
    ):
        raise RuntimeError("outer tensor fusion startup receipt does not match frozen launch")
    compiler = block.get("compiler", {})
    if (
        compiler.get("backend") != "inductor"
        or compiler.get("fullgraph") is not True
        or compiler.get("dynamic") is not False
        or compiler.get("options") != {"triton.cudagraphs": False}
        or not isinstance(compiler.get("torch"), str)
        or not isinstance(compiler.get("triton"), str)
    ):
        raise ValueError("outer tensor fusion compiler identity is invalid")
    context = block.get("context", {})
    context_booleans = (
        "deterministic_algorithms",
        "deterministic_warn_only",
        "grad_enabled",
        "inference_mode",
        "autocast_cpu_enabled",
        "autocast_cuda_enabled",
    )
    context_dtypes = ("default_dtype", "autocast_cpu_dtype", "autocast_cuda_dtype")
    if (
        set(context) != set(context_booleans + context_dtypes + ("dispatch_modes",))
        or any(type(context.get(key)) is not bool for key in context_booleans)
        or any(context.get(key) not in {"float16", "bfloat16", "float32", "float64"} for key in context_dtypes)
        or context.get("inference_mode") is not True
        or context.get("grad_enabled") is not False
        or context.get("dispatch_modes") != []
    ):
        raise ValueError("outer tensor fusion compiler context is invalid")
    declared = block.get("declared", [])
    from vllm_omni.model_executor.models.nemotron_asr.outer_tensor_fusion import expected_declarations

    if not declared or not declared[0].get("inputs"):
        raise ValueError("outer tensor fusion declarations are missing")
    device = declared[0]["inputs"][0].get("device")
    if not isinstance(device, str) or not device.startswith("cuda:"):
        raise ValueError("outer tensor fusion declaration is not CUDA")
    if (
        declared != expected_declarations(dict(expected_constants), device)
        or any(type(value) is not int for value in block["constants"].values())
        or any(
            type(item["requires_grad"]) is not bool
            or any(type(item[key]) is not int for key in ("storage_offset_bytes", "object_group", "storage_group"))
            or any(type(value) is not int for key in ("shape", "stride") for value in item[key])
            for declaration in declared
            for item in declaration["inputs"]
        )
    ):
        raise ValueError("outer tensor fusion declared tensor inventory is invalid")
    frames = block.get("frames", {})
    if (
        frames.get("accepted") != dict.fromkeys(FAMILIES, 1)
        or frames.get("at_seal") != dict.fromkeys(FAMILIES, 1)
        or any(type(v) is not int for key in ("accepted", "at_seal") for v in frames.get(key, {}).values())
        or _nonnegative_count(frames.get("backend_attempts")) != 7
        or _nonnegative_count(frames.get("attempts_at_seal")) != 7
        or _nonnegative_count(frames.get("post_seal_attempts")) != 0
        or _nonnegative_count(frames.get("undeclared_attempts")) != 0
    ):
        raise RuntimeError("outer tensor fusion backend/frame counters changed after seal")
    dispatch = block.get("dispatch", {})
    for region in ("pre", "post"):
        counts = dispatch.get(region, {})
        for field in ("fused_calls", "eager_calls", "fused_at_seal", "eager_at_seal"):
            _nonnegative_count(counts.get(field))
        if counts["eager_calls"] != counts["eager_at_seal"] or counts["fused_calls"] < counts["fused_at_seal"]:
            raise RuntimeError("outer tensor fusion startup selected eager or reset counters")
    identity = _freeze({key: value for key, value in block.items() if key not in ("frames", "dispatch")})
    identity += (
        _freeze(frames),
        tuple((r, dispatch[r]["fused_at_seal"], dispatch[r]["eager_at_seal"]) for r in ("pre", "post")),
    )
    snapshot = OuterPrimingContext(pid, identity, dispatch["pre"]["fused_calls"], dispatch["post"]["fused_calls"])
    if baseline is not None:
        _check_outer_monotone(baseline, snapshot)
    return snapshot


NEMOTRON_PERSISTENT_STATE_STARTUP = NemotronPersistentStateStartupProvider()


__all__ = [
    "NEMOTRON_PERSISTENT_STATE_STARTUP",
    "NemotronPersistentStateStartupProvider",
    "NemotronServicePrimingPlan",
]
