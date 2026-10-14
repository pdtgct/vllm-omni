# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Diagnostic-only, bounded startup leg attribution; no device reads."""

from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Any

logger = logging.getLogger(__name__)

# Diagnostic counters exist only between explicit startup RPCs. No growing history.
worker_counts: dict[str, dict[str, int]] | None = None


def worker_trace_control(request_ids: tuple[str, ...] | None) -> Any:
    global worker_counts
    if request_ids is not None:
        worker_counts = {
            key: dict.fromkeys(("CHUNK", "REPLAY", "EOU", "FLUSH", "echo", "invalid"), 0) for key in request_ids
        }
        return None
    result = worker_counts
    worker_counts = None
    return result


def count_worker_row(request_id: str, starts_chunk: bool, phase: int | None, status: int) -> None:
    """Count committed row invocations from the already-consumed host status copy.

    Phase 0 is park echo, 1/2 queue emission/drain, 3 FLUSH, 4/5 forced
    EOU. CHUNK's host binding distinguishes it from REPLAY in phases 1/2.
    Counts include setup and any echo drained during observation cleanup.
    """
    counts = None if worker_counts is None else worker_counts.get(request_id)
    if counts is None:
        return
    if status or phase is None:
        role = "invalid"
    elif starts_chunk:
        role = "CHUNK"
    elif phase == 0:
        role = "echo"
    elif phase == 3:
        role = "FLUSH"
    elif phase in (4, 5):
        role = "EOU"
    elif phase in (1, 2):
        role = "REPLAY"
    else:
        role = "invalid"
    counts[role] += 1


class PrimingLegTrace:
    """One session's fixed-size host measurements, allocated only for the probe."""

    def __init__(self, scenario: str, request_id: str) -> None:
        self.scenario = scenario
        self.request_id = request_id
        self.timestamps: dict[str, int | None] = dict.fromkeys(
            ("tS", "t0", "tE", "tC", "tP", "tail_submission", "tail_park", "flush_submission", "flush_park")
        )
        self.sleeps = {"count": 0, "requested_ns": 0, "actual_ns": 0, "max_actual_ns": 0}
        self.sleep_events: list[tuple[int, int]] = []
        self.emitted_tokens = 0
        self.output_batches = 0
        self.preexpire_requested_ns = 0
        self.preexpire_actual_ns = 0

    def submitted(self, kind: str, stamp_ns: int) -> None:
        if kind == "regular":
            key = "tS" if self.scenario == "forced_eou_then_chunk" and self.timestamps["tS"] is None else "tC"
        elif kind == "final_tail":
            key = "tail_submission"
            self.timestamps["tC"] = stamp_ns
        elif kind == "flush":
            key = "flush_submission"
        else:
            return
        self.timestamps[key] = stamp_ns

    def output(self, kind: str | None, token_count: int, parked: bool) -> None:
        self.emitted_tokens += token_count
        self.output_batches += 1
        if not parked:
            return
        if kind == "forced_eou":
            key = "tE"
        elif kind == "final_tail":
            key = "tail_park"
        elif kind is None:
            key = "flush_park"
        elif self.timestamps["tC"] is not None:
            key = "tP"
        else:
            return  # Setup park is deliberately untimed.
        stamp_ns = time.monotonic_ns()
        self.timestamps[key] = stamp_ns
        if kind == "final_tail":
            self.timestamps["tP"] = stamp_ns

    def slept(self, requested_ns: int, actual_ns: int) -> None:
        self.sleeps["count"] += 1
        self.sleeps["requested_ns"] += requested_ns
        self.sleeps["actual_ns"] += actual_ns
        self.sleeps["max_actual_ns"] = max(self.sleeps["max_actual_ns"], actual_ns)
        if len(self.sleep_events) < 8:
            self.sleep_events.append((requested_ns, actual_ns))

    def payload(self, t0: int) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "timestamps_ns": {key: None if value is None else value - t0 for key, value in self.timestamps.items()},
            "sleeps": self.sleeps,
            "sleep_events_requested_actual_ns": self.sleep_events,
            "sleep_events_omitted": self.sleeps["count"] - len(self.sleep_events),
            "emitted_tokens": self.emitted_tokens,
            "output_batches": self.output_batches,
            "preexpire_requested_ns": self.preexpire_requested_ns,
            "preexpire_actual_ns": self.preexpire_actual_ns,
        }


async def preexpire_setup(legs: list[PrimingLegTrace], cadence_ns: int) -> None:
    # Diagnostic attribution probe: all setup deadlines expire before timer reset.
    submissions = [leg.timestamps["tS"] for leg in legs]
    if not submissions or any(stamp is None for stamp in submissions):
        raise RuntimeError("priming preexpire requires each setup submission timestamp")
    deadline = max(stamp for stamp in submissions if stamp is not None) + cadence_ns
    started = time.monotonic_ns()
    remaining = max(0, deadline - started)
    for leg in legs:
        leg.preexpire_requested_ns = remaining
    while remaining > 0:
        await asyncio.sleep(remaining / 1e9)
        remaining = max(0, deadline - time.monotonic_ns())
    elapsed = time.monotonic_ns() - started
    for leg in legs:
        leg.preexpire_actual_ns = elapsed


async def worker_trace_rpc(engine: Any, request_ids: tuple[str, ...] | None) -> Any:
    try:
        return await engine.collective_rpc(
            method="priming_leg_trace",
            args=(request_ids,),
            stage_ids=[0],
            timeout=5.0,
        )
    except Exception as error:
        # Diagnostics must leave timing/profile policy intact if RPC is unavailable.
        return {"unavailable": type(error).__name__}


def log_trace(
    spec: Any, legs: list[PrimingLegTrace], t0: int, elapsed_ns: int, preexpire: bool, role_counts: Any
) -> None:
    suffix = spec.round_id.rsplit("-repeat-", 1)[-1]
    payload = {
        "scenario": spec.scenario_id,
        "geometry": spec.geometry_id,
        "interval_ms": spec.service_interval_ms,
        "population": spec.active_population,
        "round_id": spec.round_id,
        "round_index": int(suffix) if suffix.isdecimal() else None,
        "post_jit": spec.post_jit,
        "elapsed_ns": elapsed_ns,
        "preexpire_applied": preexpire,
        "legs": [leg.payload(t0) for leg in legs],
        "role_counts": role_counts,
        "count_scope": "observation_including_setup_and_cleanup;per_worker_committed_rows",
    }
    logger.info("PRIMING_LEG_TRACE %s", json.dumps(payload, separators=(",", ":")))
