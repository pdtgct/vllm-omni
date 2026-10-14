# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Experimental startup attribution: real stream/ledger, fake engine and clock."""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import replace
from types import SimpleNamespace

import pytest

from tests.entrypoints.test_nemotron_session import EOU_ID, PARK_ID, FakeAsyncOmni, _out
from vllm_omni.engine.persistent_state_priming import ServicePrimingRound
from vllm_omni.model_executor.models.nemotron_asr import startup

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.fixture
def harness(monkeypatch):
    from vllm_omni.entrypoints import nemotron_session

    clock = SimpleNamespace(ns=1_000_000_000, sleeps=[], rpc=[])
    monkeypatch.delenv("VLLM_OMNI_PRIMING_LEG_TRACE", raising=False)
    monkeypatch.delenv("VLLM_OMNI_PRIMING_CANARY_PREEXPIRE", raising=False)
    monkeypatch.setattr(startup.time, "monotonic_ns", lambda: clock.ns)

    async def sleep(delay):
        clock.sleeps.append(delay)
        clock.ns += round(delay * 1e9)

    monkeypatch.setattr(asyncio, "sleep", sleep)

    async def render(prompt):
        return SimpleNamespace(prompt=prompt)

    monkeypatch.setattr(nemotron_session, "_render_factory", lambda engine: render)

    def script(item, index):
        # Advance fake device/transport time; EOU is queued before it parks.
        clock.ns += 10_000_000
        if item["prompt_token_ids"] == [EOU_ID]:
            return [_out([PARK_ID])]
        return [_out([7, PARK_ID])]

    engine = FakeAsyncOmni(script=script)
    original_generate = engine.generate

    async def generate(**kwargs):
        prompts = kwargs["prompt"]

        async def unwrap():
            async for prompt in prompts:
                yield prompt.prompt

        kwargs["prompt"] = unwrap()
        async for output in original_generate(**kwargs):
            yield output

    engine.generate = generate

    async def rpc(**kwargs):
        clock.rpc.append((clock.ns, kwargs))
        return [[{"session-0": {"CHUNK": 2, "REPLAY": 0, "EOU": 1, "FLUSH": 0, "echo": 0}}]]

    engine.collective_rpc = rpc
    lease = SimpleNamespace(
        session_key="session-0",
        engine_epoch="epoch-a",
        generation=1,
        schema_id="schema",
        profile_id="profile",
        binding_token="binding",
    )
    return clock, engine, (lease,)


async def run(harness, scenario):
    _, engine, leases = harness
    spec = ServicePrimingRound(
        round_id=f"geometry-1-single-{scenario}-repeat-2",
        service_interval_ms=160,
        geometry_id=1,
        active_population=1,
        schema_id="schema",
        profile_id="profile",
        post_jit=True,
        scenario_id=scenario,
        expected_legal_parks_per_lease=1 if scenario == "ordinary" else 2,
    )
    return await startup.NEMOTRON_PERSISTENT_STATE_STARTUP.execute_priming_round(
        engine_client=engine,
        round_spec=spec,
        leases=leases,
    )


def lines(caplog, prefix="PRIMING_LEG_TRACE "):
    return [
        json.loads(record.message.removeprefix(prefix))
        for record in caplog.records
        if record.message.startswith(prefix)
    ]


@pytest.mark.asyncio
async def test_disabled_allocates_no_collector_or_rpc_or_diagnostic_sleep(harness, caplog, monkeypatch):
    from vllm_omni.model_executor.models.nemotron_asr import priming_trace

    monkeypatch.setattr(priming_trace, "PrimingLegTrace", lambda *a, **k: pytest.fail("disabled allocation"))
    caplog.set_level(logging.INFO)
    result = await run(harness, "ordinary")
    assert result.elapsed_ns == 10_000_000
    assert harness[0].sleeps == harness[0].rpc == []
    assert lines(caplog) == []


@pytest.mark.asyncio
@pytest.mark.parametrize("scenario", ["ordinary", "forced_eou_then_chunk", "final_tail_then_flush"])
async def test_trace_real_submission_park_and_sleep_seams(harness, caplog, monkeypatch, scenario):
    monkeypatch.setenv("VLLM_OMNI_PRIMING_LEG_TRACE", "1")
    caplog.set_level(logging.INFO)
    result = await run(harness, scenario)
    [row] = lines(caplog)
    assert (row["scenario"], row["geometry"], row["interval_ms"], row["population"]) == (scenario, 1, 160, 1)
    assert row["round_index"] == 2 and row["post_jit"] is True
    assert row["elapsed_ns"] == result.elapsed_ns
    assert row["preexpire_applied"] is False
    assert row["role_counts"][0][0]["session-0"]["EOU"] == 1
    assert len(harness[0].rpc) == 2
    [leg] = row["legs"]
    assert leg["timestamps_ns"]["t0"] == 0
    if scenario == "forced_eou_then_chunk":
        assert leg["timestamps_ns"] == {
            "tS": -10_000_000,
            "t0": 0,
            "tE": 10_000_000,
            "tC": 150_000_000,
            "tP": 160_000_000,
            "tail_submission": None,
            "tail_park": None,
            "flush_submission": None,
            "flush_park": None,
        }
        assert leg["sleeps"]["requested_ns"] == leg["sleeps"]["actual_ns"] == 140_000_000
        assert leg["sleep_events_requested_actual_ns"] == [[140_000_000, 140_000_000]]
        assert leg["sleep_events_omitted"] == 0
        assert leg["emitted_tokens"] == 5  # Includes the untimed setup CHUNK.
    elif scenario == "ordinary":
        assert leg["timestamps_ns"]["tS"] is None
        assert leg["timestamps_ns"]["tC"] == 0
        assert leg["timestamps_ns"]["tP"] == 10_000_000
    else:
        assert leg["timestamps_ns"]["tC"] == 0
        assert leg["timestamps_ns"]["tP"] == 10_000_000
        assert leg["timestamps_ns"]["tail_submission"] == 0
        assert leg["timestamps_ns"]["tail_park"] == 10_000_000
        assert leg["timestamps_ns"]["flush_submission"] == 10_000_000
        assert leg["timestamps_ns"]["flush_park"] == 20_000_000
        assert leg["timestamps_ns"]["tE"] is None


@pytest.mark.asyncio
@pytest.mark.parametrize("trace", [False, True])
@pytest.mark.parametrize("scenario", ["ordinary", "forced_eou_then_chunk", "final_tail_then_flush"])
async def test_preexpire_is_untimed_forced_only(harness, caplog, monkeypatch, trace, scenario):
    monkeypatch.setenv("VLLM_OMNI_PRIMING_CANARY_PREEXPIRE", "1")
    if trace:
        monkeypatch.setenv("VLLM_OMNI_PRIMING_LEG_TRACE", "1")
    caplog.set_level(logging.INFO)
    result = await run(harness, scenario)
    if scenario == "forced_eou_then_chunk":
        assert harness[0].sleeps == [0.15]
        assert result.elapsed_ns == 20_000_000
    else:
        assert harness[0].sleeps == []
    if trace:
        [row] = lines(caplog)
        assert row["preexpire_applied"] == (scenario == "forced_eou_then_chunk")
        if scenario == "forced_eou_then_chunk":
            [leg] = row["legs"]
            assert leg["timestamps_ns"]["tS"] == -160_000_000
            assert leg["timestamps_ns"]["tE"] == leg["timestamps_ns"]["tC"] == 10_000_000
            assert leg["sleeps"]["count"] == 0
    else:
        assert lines(caplog) == []
        assert harness[0].rpc == []


def test_worker_counts_use_existing_host_progress_and_bound_request_set():
    from vllm_omni.model_executor.models.nemotron_asr import priming_trace

    priming_trace.worker_trace_control(("request-a",))
    try:
        for starts_chunk, phase in ((True, 1), (False, 1), (False, 2), (False, 4), (False, 5), (False, 3), (False, 0)):
            priming_trace.count_worker_row("request-a", starts_chunk, phase, 0)
        priming_trace.count_worker_row("not-registered", True, 1, 0)
        assert priming_trace.worker_trace_control(None) == {
            "request-a": {"CHUNK": 1, "REPLAY": 2, "EOU": 2, "FLUSH": 1, "echo": 1, "invalid": 0}
        }
        assert priming_trace.worker_counts is None
    finally:
        priming_trace.worker_trace_control(None)


@pytest.mark.asyncio
@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("scenario", ["ordinary", "forced_eou_then_chunk", "final_tail_then_flush"])
async def test_raw_observation_logged_before_compiler_can_fail(monkeypatch, caplog, enabled, scenario):
    from tests.engine.test_persistent_state_service_priming import _PrimingService, _round
    from vllm_omni.engine.persistent_state_priming import run_service_priming_round

    monkeypatch.setenv("VLLM_OMNI_PRIMING_LEG_TRACE", "1" if enabled else "0")
    caplog.set_level(logging.INFO)
    spec = replace(_round(), scenario_id=scenario, expected_legal_parks_per_lease=1 if scenario == "ordinary" else 2)

    async def execute(spec, leases):
        return SimpleNamespace(
            elapsed_ns=42,
            completed_legal_parks=2 * spec.expected_legal_parks_per_lease,
            completed_model_rows=None,
            dummy_run=False,
            is_profile=False,
        )

    observation = await run_service_priming_round(service=_PrimingService([]), round_spec=spec, execute=execute)
    # This is the observation handed to the compiler; its log must already exist.
    rows = lines(caplog, "PRIMING_OBSERVATION ")
    assert len(rows) == int(enabled)
    if enabled:
        assert rows[0]["elapsed_ns"] == observation.elapsed_ns == 42
        assert rows[0]["scenario_id"] == scenario


@pytest.mark.parametrize("enabled", [False, True])
def test_commit_sink_counts_host_rows_including_suppressed_echo(monkeypatch, enabled):
    import torch

    from vllm_omni.model_executor.models.nemotron_asr import priming_trace
    from vllm_omni.model_executor.models.nemotron_asr.commit_sink import BoundedCommitSink

    # Exercise the actual consumption seam with CPU copies, without a CUDA event.
    sink = object.__new__(BoundedCommitSink)
    binding = SimpleNamespace(request_id="a", admission_generation=1, is_chunk=False)
    sink._staged = SimpleNamespace(bindings=(binding,), progress=object())
    sink._event = None
    sink._status_host = torch.tensor([0])
    sink._progress_host = torch.tensor([[1, 0, 0]])  # Echo: absent from published progress.
    sink._staged_lease_ok = True
    sink._staged_records = ()
    if enabled:
        priming_trace.worker_trace_control(("a",))
    else:
        monkeypatch.setattr(priming_trace, "count_worker_row", lambda *args: pytest.fail("disabled count"))
    try:
        reports, _, _ = sink.collect()
        assert reports[0].progress_key is None
        result = priming_trace.worker_trace_control(None)
        if enabled:
            assert result["a"]["echo"] == 1
        else:
            assert result is None
    finally:
        priming_trace.worker_trace_control(None)


@pytest.mark.asyncio
async def test_preexpire_uses_latest_setup_and_rechecks_early_wake(harness, monkeypatch):
    from vllm_omni.model_executor.models.nemotron_asr import priming_trace

    clock = harness[0]
    legs = [priming_trace.PrimingLegTrace("forced_eou_then_chunk", str(i)) for i in range(2)]
    legs[0].timestamps["tS"] = clock.ns - 100_000_000
    legs[1].timestamps["tS"] = clock.ns - 50_000_000

    async def early_sleep(delay):
        clock.sleeps.append(delay)
        clock.ns += 50_000_000 if len(clock.sleeps) == 1 else round(delay * 1e9)

    monkeypatch.setattr(asyncio, "sleep", early_sleep)
    await priming_trace.preexpire_setup(legs, 160_000_000)
    assert clock.sleeps == [0.11, 0.06]
    assert clock.ns == 1_110_000_000
    clock.sleeps.clear()
    await priming_trace.preexpire_setup(legs, 160_000_000)
    assert clock.sleeps == []  # Already expired; no gratuitous sleep(0).


@pytest.mark.asyncio
async def test_unavailable_worker_rpc_is_explicit_and_preserves_observation(harness, monkeypatch, caplog):
    monkeypatch.setenv("VLLM_OMNI_PRIMING_LEG_TRACE", "1")
    caplog.set_level(logging.INFO)

    async def unavailable(**kwargs):
        raise RuntimeError("unavailable diagnostic RPC")

    harness[1].collective_rpc = unavailable
    result = await run(harness, "ordinary")
    [row] = lines(caplog)
    assert result.elapsed_ns == 10_000_000
    assert row["role_counts"] == {"unavailable": "RuntimeError"}
