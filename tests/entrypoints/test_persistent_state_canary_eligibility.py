# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Production canary timing over real accepted audio, streaming and lease code."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from tests.engine.test_persistent_state_service_priming import _PrimingService
from tests.entrypoints.test_nemotron_session import FakeAsyncOmni, _default_script
from tests.entrypoints.test_persistent_state_startup_composition import _Provider, _runtime, _startup_inventory

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _provider_fixture(monkeypatch, *, clock=None, sleep=None):
    from vllm_omni.engine.persistent_state_priming import ServicePrimingRound
    from vllm_omni.entrypoints import nemotron_session as binding
    from vllm_omni.model_executor.models.nemotron_asr import accepted_audio, startup, streaming

    original = binding.NemotronSessionLease
    bound = []

    def make_lease(**kwargs):
        kwargs.pop("state_lease")

        async def render(prompt):
            return prompt

        lease = original(**kwargs, render=render)
        bound.append(lease)
        return lease

    monkeypatch.setattr(binding, "NemotronSessionLease", make_lease)
    engine = FakeAsyncOmni()
    if clock is not None:

        def script(item, index):
            clock[0] += 10
            return _default_script(item, index)

        engine.script = script
        fake_time = SimpleNamespace(monotonic_ns=lambda: clock[0], time=lambda: 1.0)
        monkeypatch.setattr(startup, "time", fake_time)
        monkeypatch.setattr(accepted_audio, "time", fake_time)
        monkeypatch.setattr(streaming, "time", fake_time)
    if sleep is not None:
        monkeypatch.setattr(startup, "asyncio", SimpleNamespace(gather=asyncio.gather, sleep=sleep))
    spec = ServicePrimingRound(
        round_id="canary",
        service_interval_ms=80,
        geometry_id=0,
        active_population=2,
        schema_id="schema-a",
        profile_id="profile-a",
        scenario_id="forced_eou_then_chunk",
        expected_legal_parks_per_lease=2,
    )
    leases = tuple(SimpleNamespace(session_key=f"canary-{i}", engine_epoch="epoch", generation=i + 1) for i in range(2))
    return startup.NEMOTRON_PERSISTENT_STATE_STARTUP, engine, spec, leases, bound


@pytest.mark.asyncio
async def test_population_canary_stages_before_wait_and_waits_for_latest(monkeypatch):
    """@spec PORT-STATE-026, PORT-PERF-005, PORT-PERF-008: staging preserves canonical ordering and excludes the latest eligibility wait."""
    clock = [1_000]
    waits = []

    async def sleep(delay):
        assert len(engine.prompts) == 2  # registration only, no timed operation dispatched
        assert all(
            [u.kind for u in lease.session.accepted_audio.ready_units] == ["forced_eou", "regular"] for lease in bound
        )
        waits.append(round(delay * 1e9))
        clock[0] += round(delay * 1e9)

    provider, engine, spec, leases, bound = _provider_fixture(monkeypatch, clock=clock, sleep=sleep)
    result = await asyncio.wait_for(
        provider.execute_priming_round(engine_client=engine, round_spec=spec, leases=leases), timeout=1.0
    )
    assert result.elapsed_ns == 40  # four timed operations, ten ns each
    assert result.completed_legal_parks == 4
    assert len(engine.prompts) == 6
    assert len(result.following_chunk_timings) == 2
    latest = max(w.eligible_ns for w in result.following_chunk_timings)
    assert result.canary_timer_start_ns == latest
    assert result.canary_timer_start_ns - result.canary_wait_start_ns == result.canary_wait_ns
    assert result.canary_wait_ns == sum(waits) > 79_000_000
    assert {w.prior_actual_submission_ns for w in result.following_chunk_timings} == {1000, 1010}
    for w in result.following_chunk_timings:
        assert w.true_ready_ns == 1020
        assert w.cadence_ns == 80_000_000
        assert w.eligible_ns == max(w.true_ready_ns, w.prior_actual_submission_ns + w.cadence_ns)
        assert result.canary_timer_start_ns <= w.submitted_ns <= result.canary_timer_start_ns + result.elapsed_ns
    assert all(lease.session.accepted_audio.snapshot().parked_samples == 2560 for lease in bound)


@pytest.mark.asyncio
async def test_eligibility_wait_is_inside_round_timeout_and_rolls_back(monkeypatch):
    """@spec PORT-PERF-005, PORT-PERF-008: untimed waiting still consumes the bounded round and releases every lease."""
    reached_wait = asyncio.Event()

    async def sleep(delay):
        reached_wait.set()
        await asyncio.Event().wait()

    provider, engine, spec, leases, bound = _provider_fixture(monkeypatch, sleep=sleep)
    from vllm_omni.engine.persistent_state_priming import run_service_priming_round

    service = _PrimingService([])

    async def execute(spec, ignored):
        return await provider.execute_priming_round(engine_client=engine, round_spec=spec, leases=leases)

    with pytest.raises(asyncio.TimeoutError):
        await asyncio.wait_for(
            run_service_priming_round(service=service, round_spec=spec, execute=execute), timeout=0.1
        )
    assert reached_wait.is_set()
    assert service.resident == 0
    assert len(engine.prompts) == 2
    assert len(engine.aborted) == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("timeout", (0.25, 1.12))
async def test_round_timeout_must_exceed_largest_served_cadence_before_reserve(monkeypatch, timeout):
    """@spec PORT-PERF-005: reject an impossible per-round budget before any transient reservation."""
    from vllm_omni.engine import persistent_state_startup as startup

    events = []

    class Service:
        def __init__(self, *args, **kwargs):
            pass

        async def bootstrap_handshake(self):
            return _startup_inventory()

        def configure_bootstrap_intervals(self, intervals):
            pass

        def shutdown(self):
            events.append("shutdown")

    async def reserve(**kwargs):
        pytest.fail("invalid timeout reached first reserve")

    monkeypatch.setattr(startup, "PersistentStateService", Service)
    monkeypatch.setattr(startup, "run_service_priming_round", reserve)
    runtime = _runtime()
    runtime.startup_priming_round_timeout_s = timeout
    with pytest.raises(ValueError, match="round_timeout.*1120|1120.*round_timeout"):
        await startup.prepare_persistent_state_service(
            engine_client=SimpleNamespace(model_config=object()),
            stage_client=object(),
            runtime_config=runtime,
            startup_provider=_Provider(events),
            host_fatal_callback=lambda *_: None,
        )
    assert events[-1] == "shutdown"


@pytest.mark.asyncio
async def test_feed_rejects_empty_piece_without_starting_consumer():
    """@spec PORT-STATE-026: empty control wakes are never caller audio pieces."""
    import numpy as np

    from tests.entrypoints.test_nemotron_session import _make_lease

    lease, _ = _make_lease(FakeAsyncOmni(), cadence="80ms")
    try:
        with pytest.raises(ValueError, match="empty"):
            await asyncio.wait_for(lease.feed(np.empty(0, dtype=np.float32)), 0.1)
        assert lease._task is None
        assert lease._audio.empty()
    finally:
        await lease.abort()


@pytest.mark.asyncio
async def test_staged_dispatch_collects_receipt_without_feed(monkeypatch):
    """@spec PORT-STATE-026: dispatch consumes the staged receipt and single ticket directly."""
    import numpy as np

    from tests.entrypoints.test_nemotron_session import _make_lease

    engine = FakeAsyncOmni()
    lease, _ = _make_lease(engine, cadence="80ms")
    samples = np.zeros(1280, dtype=np.float32)
    try:
        await lease.feed(samples)
        await lease.wait_for_legal_park()
        lease.stage_forced_eou_then_chunk(samples)

        async def forbidden(*args, **kwargs):
            pytest.fail("staged dispatch must not call feed")

        monkeypatch.setattr(lease, "feed", forbidden)
        await asyncio.wait_for(lease.dispatch_staged_priming_chain(), 1)
        assert len(engine.prompts) == 3
        assert lease.session.accepted_audio.snapshot().parked_samples == 2560
    finally:
        await lease.abort()


@pytest.mark.asyncio
@pytest.mark.parametrize("reject", (False, True))
async def test_staging_obeys_audio_acceptance_lifecycle_callbacks(monkeypatch, reject):
    """@spec PORT-STATE-026: staged audio uses the canonical lifecycle fence and idle renewal."""
    import numpy as np

    from tests.entrypoints.test_nemotron_session import _make_lease

    lease, _ = _make_lease(FakeAsyncOmni(), cadence="80ms")
    samples = np.zeros(1280, dtype=np.float32)
    calls = []

    def before():
        calls.append("before")
        if reject:
            raise RuntimeError("lifecycle fence")

    try:
        await lease.feed(samples)
        await lease.wait_for_legal_park()
        monkeypatch.setattr(lease, "_require_unexpired_lifecycle", before)
        monkeypatch.setattr(lease, "_arm_session_lifecycle_timeout", lambda kind: calls.append(kind))
        if reject:
            with pytest.raises(RuntimeError, match="lifecycle fence"):
                lease.stage_forced_eou_then_chunk(samples)
            assert lease.session.accepted_audio.snapshot().accepted_samples == 1280
            assert calls == ["before"]
        else:
            lease.stage_forced_eou_then_chunk(samples)
            assert calls == ["before", "idle"]
            assert lease.session.accepted_audio.snapshot().accepted_samples == 2560
    finally:
        await lease.abort()


@pytest.mark.asyncio
async def test_legal_park_surfaces_consumer_failure_immediately():
    """@spec PORT-PERF-005: consumer failure interrupts legal-park waiting with the original cause."""
    import numpy as np

    from tests.entrypoints.test_nemotron_session import _make_lease

    lease, _ = _make_lease(FakeAsyncOmni(), cadence="80ms")
    rendering = asyncio.Event()
    fail = asyncio.Event()
    error = RuntimeError("consumer render failed")

    async def render(prompt):
        rendering.set()
        await fail.wait()
        raise error

    lease._render = render
    feeding = asyncio.create_task(lease.feed(np.zeros(1280, dtype=np.float32)))
    try:
        await asyncio.wait_for(rendering.wait(), 1)
        parking = asyncio.create_task(lease.wait_for_legal_park())
        await asyncio.sleep(0)
        fail.set()
        with pytest.raises(RuntimeError) as caught:
            await asyncio.wait_for(parking, 0.1)
        assert caught.value is error
    finally:
        await lease.abort()
        await asyncio.gather(feeding, return_exceptions=True)


@pytest.mark.asyncio
@pytest.mark.parametrize("defect", ("samples", "tickets"))
async def test_staged_dispatch_checks_receipt_identity(monkeypatch, defect):
    """@spec PORT-STATE-026: a staged receipt must cover its exact samples and one CHUNK ticket."""
    from dataclasses import replace

    import numpy as np

    from tests.entrypoints.test_nemotron_session import _make_lease

    lease, _ = _make_lease(FakeAsyncOmni(), cadence="80ms")
    samples = np.zeros(1280, dtype=np.float32)
    try:
        await lease.feed(samples)
        await lease.wait_for_legal_park()
        lease.stage_forced_eou_then_chunk(samples)
        next_piece = lease._ledger.next_piece

        async def malformed():
            receipt = await next_piece()
            return replace(
                receipt,
                **({"samples_consumed": receipt.samples_consumed + 1} if defect == "samples" else {"tickets": ()}),
            )

        monkeypatch.setattr(lease._ledger, "next_piece", malformed)
        with pytest.raises(AssertionError, match="receipt sample|one ticket"):
            await lease.dispatch_staged_priming_chain()
    finally:
        await lease.abort()
