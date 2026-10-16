# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Model-owned startup priming plan for Nemotron streaming ASR."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import sys
import time
from dataclasses import dataclass
from dataclasses import field as dataclass_field
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
class ChunkObservationPlan:
    expected_json: str
    roster_json: str


@dataclass
class ChunkPrimingContext:
    plan: ChunkObservationPlan
    samples: list[dict[str, Any]] = dataclass_field(default_factory=list)
    failed: bool = False
    emitted: bool = False


@dataclass(frozen=True)
class NemotronServicePrimingPlan:
    """Finite eager-runner geometry plan and arithmetic compiler inputs."""

    rounds: tuple[ServicePrimingRound, ...]
    compile_kwargs: dict[str, Any]
    actual_plan_sha256: str
    actual_operation_count: int
    served_intervals_ms: tuple[int, ...]
    outer_constants: tuple[tuple[str, int], ...] | None = None
    chunk_observation: ChunkObservationPlan | None = None


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
        # @spec PORT-PERF-033: observation lives only on this provider plan.
        observe = getattr(hf_config, "experimental_chunk_graph_startup_attestation", False)
        if type(observe) is not bool:
            raise ValueError("experimental_chunk_graph_startup_attestation must be a boolean")
        chunk_observation = None
        if observe:
            if inventory["configured_limit"] != 1 or inventory["execution_claim_ceiling"] != 1:
                raise ValueError("CHUNK attestation requires configured cap1 execution")
            chunk_observation = _chunk_observation_plan(
                hf_config, model_config, runtime_config, rounds_tuple, maximum_population, derating
            )
        return NemotronServicePrimingPlan(
            chunk_observation=chunk_observation,
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

    # @spec PORT-PERF-025, PORT-PERF-022, PORT-PERF-035
    async def validate_priming_boundary(
        self,
        *,
        engine_client: Any,
        plan: NemotronServicePrimingPlan,
        phase: str,
        boundary_context: OuterPrimingContext | ChunkPrimingContext | None = None,
    ) -> OuterPrimingContext | ChunkPrimingContext | None:
        if plan.chunk_observation is not None:
            if phase == "before":
                return ChunkPrimingContext(plan.chunk_observation)
            if phase != "after" or not isinstance(boundary_context, ChunkPrimingContext):
                raise ValueError("invalid CHUNK attestation startup boundary")
            _emit_chunk_observation(plan, boundary_context)
            return boundary_context
        if plan.outer_constants is None:
            return None
        snapshot = await _outer_snapshot(engine_client, plan.outer_constants, boundary_context)
        if phase == "before":
            return snapshot
        if phase != "after" or boundary_context is None:
            raise ValueError("invalid outer tensor fusion startup boundary")
        _positive_outer_delta(boundary_context, snapshot)
        return boundary_context

    # @spec PORT-STATE-026, PORT-PERF-005, PORT-PERF-006, PORT-PERF-007, PORT-PERF-034
    async def execute_priming_round(
        self,
        *,
        engine_client: Any,
        round_spec: ServicePrimingRound,
        leases: tuple[Any, ...],
        boundary_context: OuterPrimingContext | ChunkPrimingContext | None = None,
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
        chunk_context = boundary_context if isinstance(boundary_context, ChunkPrimingContext) else None
        sample_identity = None
        if chunk_context is not None:
            sample_identity = next(
                (
                    item
                    for item in json.loads(chunk_context.plan.roster_json)
                    if item["round_id"] == round_spec.round_id
                ),
                None,
            )
        try:
            if sample_identity is not None:
                roster = json.loads(chunk_context.plan.roster_json)
                if (
                    chunk_context.failed
                    or len(chunk_context.samples) >= 3
                    or (sample_identity != roster[len(chunk_context.samples)])
                ):
                    raise RuntimeError("CHUNK attestation round roster is out of order or already failed")
                before_request_ns, before_response_ns, before_chunk = await _chunk_snapshot(
                    engine_client, chunk_context
                )
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
            ended_ns = time.monotonic_ns()
            elapsed_ns = ended_ns - started_ns
            ending_range_id = range_id
            range_id = None
            end_startup_round(ending_range_id)
            if sample_identity is not None:
                after_request_ns, after_response_ns, after_chunk = await _chunk_snapshot(engine_client, chunk_context)
                _check_chunk_interval(before_chunk, after_chunk, json.loads(chunk_context.plan.expected_json))
                if completed_legal_parks != 1:
                    raise RuntimeError("CHUNK attestation round did not complete one legal park")
                if not (
                    before_request_ns
                    <= before_response_ns
                    <= started_ns
                    <= ended_ns
                    <= after_request_ns
                    <= after_response_ns
                ):
                    raise ValueError("CHUNK attestation interval clocks are out of order")
                chunk_context.samples.append(
                    sample_identity
                    | {
                        "completed_legal_parks": completed_legal_parks,
                        "elapsed_ns": elapsed_ns,
                        "timer_start_ns": started_ns,
                        "timer_end_ns": ended_ns,
                        "before_request_ns": before_request_ns,
                        "before_response_ns": before_response_ns,
                        "after_request_ns": after_request_ns,
                        "after_response_ns": after_response_ns,
                        "before": before_chunk,
                        "after": after_chunk,
                    }
                )
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
        except BaseException:
            if chunk_context is not None:
                chunk_context.failed = True
            raise
        finally:
            primary_error = sys.exc_info()[1]
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
                cleanup = await asyncio.gather(
                    *(lease.abort() for lease in bound),
                    return_exceptions=True,
                )
                if chunk_context is not None:
                    errors = [error for error in cleanup if isinstance(error, BaseException)]
                    if errors:
                        chunk_context.failed = True
                        if primary_error is None:
                            raise errors[0]
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


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)


def _chunk_observation_plan(
    hf: Any, model: Any, runtime: Any, rounds: tuple[ServicePrimingRound, ...], maximum: int, derating: Fraction
) -> ChunkObservationPlan:
    from vllm_omni.model_executor.models.nemotron_asr.batch_invariance import installed_batch_invariant_mode
    from vllm_omni.model_executor.models.nemotron_asr.chunk_bucket_graph import chunk_graph_selection
    from vllm_omni.model_executor.models.nemotron_asr.native_burst import NATIVE_BURST_FIVE_CADENCE_PROFILE
    from vllm_omni.model_executor.models.nemotron_asr.precision import FP32_BRINGUP, resolve_precision_policy

    legacy, cells = chunk_graph_selection(hf)
    domain = sorted({(r.geometry_id, r.active_population) for r in rounds})
    if (
        legacy
        or cells not in (frozenset(), frozenset({(0, 1)}))
        or maximum != 1
        or domain != [(g, 1) for g in range(5)]
        or getattr(hf, "experimental_native_burst", False) is not True
        or getattr(hf, "experimental_native_burst_profile", None) != NATIVE_BURST_FIVE_CADENCE_PROFILE
        or tuple(getattr(hf, "supported_num_lookahead_tokens", ())) != (0, 1, 3, 6, 13)
        or getattr(hf, "encoder_execution_arm", None) != "eager-graphed"
        or getattr(hf, "decode_dispatch_arm", None) != "dense-graphed"
        or getattr(hf, "experimental_outer_tensor_fusion", False) is not False
        or str(getattr(model, "dtype", None)) != "torch.float32"
        or resolve_precision_policy(hf) is not FP32_BRINGUP
        or installed_batch_invariant_mode().enabled
    ):
        raise ValueError("CHUNK attestation requires native-five FP32 BI-off cap1 graph geometry domain")
    roster = [
        dict(
            plan_ordinal=ordinal,
            round_id=r.round_id,
            geometry_id=r.geometry_id,
            tier_id=r.tier_id,
            active_population=r.active_population,
            scenario_id=r.scenario_id,
            post_jit=r.post_jit,
            repeat=int(r.round_id.rsplit("-", 1)[1]),
        )
        for ordinal, r in enumerate(rounds)
        if r.geometry_id == 0
        and r.tier_id == "single"
        and r.active_population == 1
        and r.scenario_id == "ordinary"
        and r.post_jit
    ]
    if runtime.service_profile_trailing_rounds != 3 or [r["repeat"] for r in roster] != [1, 2, 3]:
        raise ValueError("CHUNK attestation requires exactly three trailing ordinary rounds")
    expected = dict(
        geometry_cells=[list(cell) for cell in domain],
        chunk_cells=[list(cell) for cell in sorted(cells)],
        encoder_arm=hf.encoder_execution_arm,
        decode_arm=hf.decode_dispatch_arm,
        dtype=str(model.dtype).removeprefix("torch."),
        batch_invariant_mode="off",
        emission_lane="native_burst",
        emission_profile=hf.experimental_native_burst_profile,
        max_num_seqs=maximum,
        admission_policy=runtime.admission_policy,
        derating_numerator=derating.numerator,
        derating_denominator=derating.denominator,
        trailing_rounds=runtime.service_profile_trailing_rounds,
    )
    return ChunkObservationPlan(_canonical_json(expected), _canonical_json(roster))


def _chunk_profile(raw: Any, expected: dict[str, Any]) -> tuple[int, Any, int, dict[tuple[int, int], int]]:
    if not isinstance(raw, list) or len(raw) != 1 or not isinstance(raw[0], list) or len(raw[0]) != 1:
        raise ValueError("CHUNK attestation requires exactly one stage/worker")
    envelope = raw[0][0]
    if not isinstance(envelope, dict) or envelope.get("persistent_state_warmup_complete") is not True:
        raise ValueError("CHUNK attestation worker is not ready")
    pid = envelope.get("worker_pid")
    if type(pid) is not int or pid <= 0:
        raise ValueError("CHUNK attestation worker PID is invalid")
    profile = envelope.get("execution_profile", {})
    if not isinstance(profile, dict):
        raise ValueError("CHUNK attestation execution profile is malformed")
    encoder, decode = profile.get("encoder", {}), profile.get("decode", {})
    if not isinstance(encoder, dict) or not isinstance(decode, dict):
        raise ValueError("CHUNK attestation execution inventories are malformed")
    domain, selected = expected["geometry_cells"], expected["chunk_cells"]
    complement = [cell for cell in domain if cell not in selected]

    def inventory(value: Any, wanted: Any) -> bool:
        return (
            isinstance(value, list)
            and all(isinstance(cell, list) and len(cell) == 2 and all(type(n) is int for n in cell) for cell in value)
            and value == wanted
        )

    if (
        profile.get("schema") != "nemotron-execution-profile/1"
        or profile.get("emission") != {"lane": expected["emission_lane"], "profile": expected["emission_profile"]}
        or profile.get("batch_invariant_mode", "off") != expected["batch_invariant_mode"]
        or "outer_tensor_fusion" in profile
        or encoder.get("arm") != expected["encoder_arm"]
        or encoder.get("ready") is not True
        or decode.get("arm") != expected["decode_arm"]
        or decode.get("ready") is not True
        or encoder.get("warmup_geometries") != list(range(5))
        or encoder.get("warmup_populations") != [1]
        or any(type(n) is not int for n in encoder["warmup_geometries"] + encoder["warmup_populations"])
        or not inventory(encoder.get("warmup_cells"), domain)
        or not inventory(encoder.get("captured_keys"), complement)
        or not inventory(decode.get("captured_keys"), domain)
    ):
        raise ValueError("CHUNK attestation execution inventory differs from frozen launch")
    if not selected:
        if "chunk" in profile or "chunk_graph_keys" in encoder or "reserved_chunk_graph_keys" in encoder:
            raise ValueError("CHUNK attestation empty selection has CHUNK authority")
        return pid, None, 0, {}
    block = profile.get("chunk", {})
    if not isinstance(block, dict):
        raise ValueError("CHUNK attestation graph receipt is malformed")
    sources = block.get("source_sha256", {})
    source_names = {
        "advance.py",
        "encoder.py",
        "encoder_execution.py",
        "decode_graph.py",
        "rnnt.py",
        "chunk_bucket_graph.py",
        "nemotron_asr.py",
        "profile_execution.py",
    }
    if (
        block.get("ready") is not True
        or type(block.get("worker_pid")) is not int
        or block["worker_pid"] != pid
        or not isinstance(block.get("instance_id"), str)
        or not block["instance_id"]
        or not isinstance(sources, dict)
        or set(sources) != source_names
        or any(
            not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest)
            for digest in sources.values()
        )
        or not inventory(encoder.get("chunk_graph_keys"), selected)
        or not inventory(encoder.get("reserved_chunk_graph_keys"), selected)
    ):
        raise ValueError("CHUNK attestation graph identity is invalid")
    rows = block.get("cells")
    if not isinstance(rows, list) or len(rows) != 1 or not isinstance(rows[0], dict):
        raise ValueError("CHUNK attestation graph inventory is invalid")
    row = rows[0]
    if any(
        type(row.get(k)) is not int or row[k] != v
        for k, v in (("geometry", 0), ("encoder_population", 1), ("decoder_tier", 1))
    ):
        raise ValueError("CHUNK attestation selected cell is invalid")
    replays = _nonnegative_count(row.get("successful_replays"))
    fallbacks: dict[tuple[int, int], int] = {}
    fallback_rows = block.get("fallbacks")
    if not isinstance(fallback_rows, list):
        raise ValueError("CHUNK attestation fallback inventory is invalid")
    for row in fallback_rows:
        if not isinstance(row, dict):
            raise ValueError("CHUNK attestation fallback cell is malformed")
        g, n = row.get("geometry"), row.get("encoder_population")
        if type(g) is not int or type(n) is not int or [g, n] not in domain or (g, n) in fallbacks:
            raise ValueError("CHUNK attestation fallback cell is invalid or duplicated")
        fallbacks[g, n] = _nonnegative_count(row.get("successful_calls"))
    return pid, (block["instance_id"], sources), replays, fallbacks


def _check_chunk_interval(before: Any, after: Any, expected: dict[str, Any], *, positive: bool = True) -> None:
    pre_pid, pre_identity, pre_count, pre_fallback = _chunk_profile(before, expected)
    pid, identity, count, fallback = _chunk_profile(after, expected)
    if (
        pid != pre_pid
        or identity != pre_identity
        or count < pre_count
        or any(fallback.get(cell, 0) < value for cell, value in pre_fallback.items())
        or fallback.get((0, 1), 0) != pre_fallback.get((0, 1), 0)
        or positive
        and expected["chunk_cells"]
        and count <= pre_count
    ):
        raise ValueError("CHUNK attestation worker identity, replay or fallback interval is invalid")


async def _chunk_snapshot(engine: Any, context: ChunkPrimingContext) -> tuple[int, int, Any]:
    requested = time.monotonic_ns()
    raw = await engine.collective_rpc("nemotron_execution_profile_receipt")
    responded = time.monotonic_ns()
    encoded = _canonical_json(raw)
    if len(encoded.encode("utf-8")) > 16384:
        raise ValueError("CHUNK attestation snapshot exceeds size limit")
    raw = json.loads(encoded)
    expected = json.loads(context.plan.expected_json)
    _chunk_profile(raw, expected)
    if context.samples:
        _check_chunk_interval(context.samples[-1]["after"], raw, expected, positive=False)
    return requested, responded, raw


def _emit_chunk_observation(plan: NemotronServicePrimingPlan, context: ChunkPrimingContext) -> None:
    roster = json.loads(context.plan.roster_json)
    if (
        context.plan != plan.chunk_observation
        or context.failed
        or context.emitted
        or len(context.samples) != 3
        or [{key: sample[key] for key in roster[0]} for sample in context.samples] != roster
    ):
        raise RuntimeError("CHUNK attestation has an incomplete or failed round roster")
    data = dict(
        schema="nemotron-chunk-startup-attestation/1",
        api_pid=os.getpid(),
        worker_pid=context.samples[0]["before"][0][0]["worker_pid"],
        plan_sha256=plan.actual_plan_sha256,
        roster_sha256=hashlib.sha256(context.plan.roster_json.encode("utf-8")).hexdigest(),
        expected=json.loads(context.plan.expected_json),
        samples=context.samples,
        verdict="PASS",
    )
    metadata = data | {"samples": [sample | {"before": None, "after": None} for sample in context.samples]}
    encoded = _canonical_json(data).encode("utf-8")
    if len(_canonical_json(metadata).encode("utf-8")) > 32768 or len(encoded) > 131072:
        raise ValueError("CHUNK attestation metadata or total size limit exceeded")
    context.emitted = True
    logger.info(
        "Nemotron CHUNK startup attestation %s",
        _canonical_json({"data": data, "data_sha256": hashlib.sha256(encoded).hexdigest()}),
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
        raise ValueError("execution receipt counter is not a nonnegative integer")
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
