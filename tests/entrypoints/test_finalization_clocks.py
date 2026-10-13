# SPDX-License-Identifier: Apache-2.0
"""@spec PORT-SESS-005, PORT-SESS-016, PORT-SESS-017, PORT-SESS-018."""

import asyncio
from types import SimpleNamespace

import pytest

from vllm_omni.entrypoints.session_lifecycle import SessionLifecycleDeadline

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.mark.asyncio
async def test_progress_rearms_stall_but_never_absolute(monkeypatch):
    """@spec PORT-SESS-016, PORT-SESS-017: silent progress can drain past 42s."""
    loop = asyncio.get_running_loop()
    now = [loop.time()]
    monkeypatch.setattr(loop, "time", lambda: now[0])
    observed = []

    async def expire(kind):
        observed.append(kind)

    deadline = SessionLifecycleDeadline(
        idle_timeout_s=60, finalization_timeout_s=100, finalization_stall_timeout_s=10, on_expire=expire
    )
    deadline.arm("finalization")
    absolute = deadline.absolute_deadline
    for sequence in range(1, 13):
        now[0] += 8
        assert deadline.observe_progress((sequence, 0, 1))
        assert deadline.absolute_deadline == absolute
        handle = deadline._handle
        assert not deadline.observe_progress((sequence - 1, 999, 9))
        assert not deadline.observe_progress((sequence, 0, 1))
        assert deadline._handle is handle
    assert now[0] > absolute - 58
    assert not deadline.expired
    now[0] = absolute
    deadline._deadline_reached(deadline.generation, "finalization")
    await asyncio.sleep(0)
    assert deadline.expired_clock == "absolute"
    assert observed == ["finalization"]
    await deadline.close()


@pytest.mark.asyncio
async def test_stall_expiry_and_stale_callback(monkeypatch):
    """@spec PORT-SESS-005, PORT-SESS-016."""
    loop = asyncio.get_running_loop()
    now = [loop.time()]
    monkeypatch.setattr(loop, "time", lambda: now[0])

    async def expire(kind):
        pass

    deadline = SessionLifecycleDeadline(
        idle_timeout_s=60, finalization_timeout_s=100, finalization_stall_timeout_s=10, on_expire=expire
    )
    deadline.arm("finalization")
    stale = deadline.generation
    now[0] += 5
    deadline.observe_progress((1, 0, 1))
    deadline._deadline_reached(stale, "finalization")
    assert not deadline.expired
    now[0] += 10
    deadline._deadline_reached(deadline.generation, "finalization")
    assert deadline.expired_clock == "stall"
    await deadline.close()


def test_bound_derivation_and_operator_max():
    """@spec PORT-SESS-016, PORT-SESS-017: tail counted once, zero audio controls."""
    from vllm_omni.entrypoints.session_lifecycle import derive_finalization_bounds

    args = dict(
        regular_samples=321,
        chunk_samples=160,
        cadence_s=0.01,
        largest_eligibility_gap_s=1.12,
        response_bound_s=2,
        initial_eligibility_delay_s=0.5,
        final_tail_then_flush_s=3,
        adapter_margin_s=0.1,
        operator_stall_s=1,
        operator_absolute_s=1,
    )
    bounds = derive_finalization_bounds(**args)
    assert bounds.stall_s == pytest.approx(3.22)
    assert bounds.absolute_s == pytest.approx(9.6)
    assert bounds.stall_winner == bounds.absolute_winner == "derived"
    bounds = derive_finalization_bounds(
        **{**args, "regular_samples": 0, "operator_stall_s": 10, "operator_absolute_s": 20}
    )
    assert (bounds.stall_s, bounds.absolute_s) == (10, 20)
    assert bounds.stall_winner == bounds.absolute_winner == "operator"
    with pytest.raises((ValueError, OverflowError)):
        derive_finalization_bounds(**{**args, "response_bound_s": float("inf")})


@pytest.mark.parametrize(
    "name", ["streaming_session_finalization_stall_timeout", "streaming_session_finalization_response_bound_s"]
)
@pytest.mark.parametrize("value", [0, -1, float("inf"), float("nan")])
def test_config_rejects_invalid_clock_values(name, value):
    """@spec PORT-SESS-017, PORT-INT-005."""
    from vllm_omni.engine.persistent_state_config import PersistentStateRuntimeConfig

    with pytest.raises(ValueError, match=name):
        PersistentStateRuntimeConfig.from_vllm_config(
            SimpleNamespace(additional_config={"persistent_state_admission_policy": "hard_cap", name: value})
        )


def test_terminal_claim_fences_generation_and_first_wins():
    """@spec PORT-SESS-018."""
    from vllm_omni.model_executor.models.nemotron_asr.accepted_audio import AcceptedAudioAuthority

    authority = AcceptedAudioAuthority(
        request_id="a",
        engine_epoch="e",
        lease_generation=3,
        chunk_samples=160,
        capacity_samples=1600,
        carrier_sequence_modulus=100,
    )
    assert not authority.claim_terminal("completed", lease_generation=2)
    assert authority.claim_terminal("finalization_timeout", lease_generation=3)
    assert not authority.claim_terminal("completed", lease_generation=3)
    assert authority.terminal_cause == "finalization_timeout"


@pytest.mark.parametrize(
    "policy,covered,qualified,source",
    [
        ("profile", True, False, "profile"),
        ("profile", True, True, "profile"),
        ("hard_cap", True, False, "advisory"),
        ("hard_cap", False, False, "advisory"),
    ],
)
def test_response_sources(policy, covered, qualified, source):
    """@spec PORT-SESS-016: competing geometry and derating are charged."""
    from fractions import Fraction

    from vllm_omni.entrypoints.session_lifecycle import finalization_response_bounds

    profile = SimpleNamespace(
        receipt=SimpleNamespace(
            allocated_pool=2,
            execution_claim_ceiling=2,
            count_cap=2,
            maximum_charged_population=2 if covered else 1,
            admission_policy=policy,
        ),
        service_interval_ms_by_geometry={0: 160, 1: 320},
        homogeneous_capacity_by_interval={160: 2, 320: 2},
        fragmentation_duration_ns_by_geometry_and_population={0: (100000000, 200000000), 1: (300000000, 400000000)},
        control_upper_ns_by_window_and_population=None,
        derating_factor=Fraction(1, 2),
        qualified=qualified,
        observations=(
            SimpleNamespace(
                scenario_id="final_tail_then_flush",
                active_population=2 if covered else 1,
                elapsed_ns=500000000,
                censored=False,
                post_jit=True,
            ),
        ),
    )
    result = finalization_response_bounds(profile, service_interval_ms=160, configured_s=7)
    assert result.source == source
    assert result.unqualified is (source != "profile" or not qualified)
    assert result.response_s == pytest.approx(
        7 if source == "configured" else 0.16 if qualified else 1.2 if covered else 0.8
    )
    assert result.tail_s == pytest.approx(14 if source == "configured" else 1)


def test_no_profile_uses_explicit_assumption():
    """@spec PORT-SESS-016."""
    from vllm_omni.entrypoints.session_lifecycle import finalization_response_bounds

    result = finalization_response_bounds(None, service_interval_ms=160, configured_s=7)
    assert (result.response_s, result.tail_s, result.source, result.unqualified) == (7, 14, "configured", True)


def test_authoritative_progress_rejects_wrong_identity_sequence_and_echo():
    """@spec PORT-SESS-016: neither tokens nor observer hooks authorize progress."""
    import numpy as np

    from vllm_omni.model_executor.models.nemotron_asr.accepted_audio import AcceptedAudioAuthority

    authority = AcceptedAudioAuthority(
        request_id="a",
        engine_epoch="e",
        lease_generation=3,
        chunk_samples=160,
        capacity_samples=1600,
        carrier_sequence_modulus=100,
    )
    authority.accept(np.zeros(160, dtype=np.float32))
    unit = authority.dispatch_next()
    assert unit is not None
    assert authority.observe_model_progress("a", 3, (0, 0, 1)) == (0, 0, 1)
    assert authority.observe_model_progress("a", 2, (0, 1, 1)) is None
    assert authority.observe_model_progress("b", 3, (0, 1, 1)) is None
    assert authority.observe_model_progress("a", 3, (2, 1, 2)) is None
    assert authority.observe_model_progress("a", 3, (0, 0, 1)) is None
    assert authority.observe_model_progress("a", 3, (0, 2, 1)) == (0, 2, 1)
    assert authority.observe_model_progress("a", 3, (0, 1, 1)) is None
    assert authority.observe_model_progress("a", 3, (0, 2, 2)) == (0, 2, 2)
    assert authority.observe_model_progress("a", 3, (0, 2, 2)) is None


def test_regular_work_excludes_final_tail_residual():
    """@spec PORT-SESS-017: final tail is counted once, including zero samples."""
    import numpy as np

    from vllm_omni.model_executor.models.nemotron_asr.accepted_audio import AcceptedAudioAuthority

    authority = AcceptedAudioAuthority(
        request_id="a",
        engine_epoch="e",
        lease_generation=3,
        chunk_samples=160,
        capacity_samples=1600,
        carrier_sequence_modulus=100,
    )
    authority.accept(np.zeros(345, dtype=np.float32))
    authority.begin_finalize()
    assert authority.finalization_work() == (320, 0)
    assert authority.snapshot().outstanding_samples == 345


@pytest.mark.asyncio
async def test_transcript_ticks_and_unfenced_outputs_cannot_rearm():
    """@spec PORT-SESS-016: adapter output without a model receipt is inert."""
    from vllm_omni.entrypoints.session_lifecycle import consume_session_progress
    from vllm_omni.model_executor.models.nemotron_asr.accepted_audio import AcceptedAudioAuthority

    authority = AcceptedAudioAuthority(
        request_id="a",
        engine_epoch="e",
        lease_generation=3,
        chunk_samples=160,
        capacity_samples=1600,
        carrier_sequence_modulus=100,
    )

    async def expire(kind):
        pass

    deadline = SessionLifecycleDeadline(
        idle_timeout_s=60, finalization_timeout_s=100, finalization_stall_timeout_s=10, on_expire=expire
    )
    deadline.arm("finalization")
    handle = deadline._handle
    for output in [
        SimpleNamespace(outputs=[]),
        SimpleNamespace(outputs=[SimpleNamespace(text="delta", token_ids=[1])]),
    ]:
        consume_session_progress(deadline, SimpleNamespace(accepted_audio=authority), output)
        assert deadline._handle is handle
    await deadline.close()


@pytest.mark.asyncio
async def test_silent_progress_completes_after_old_absolute_floor(monkeypatch):
    """@spec PORT-SESS-016, PORT-SESS-017: 64s drain completes without expiry."""
    loop = asyncio.get_running_loop()
    now = [loop.time()]
    monkeypatch.setattr(loop, "time", lambda: now[0])
    expired = []

    async def expire(kind):
        expired.append(kind)

    deadline = SessionLifecycleDeadline(
        idle_timeout_s=60, finalization_timeout_s=100, finalization_stall_timeout_s=10, on_expire=expire
    )
    deadline.arm("finalization")
    for sequence in range(8):
        now[0] += 8
        deadline.observe_progress((sequence, 0, 2))
    await deadline.close()
    deadline._deadline_reached(deadline.generation, "finalization")
    assert expired == []
    assert not deadline.expired


@pytest.mark.asyncio
async def test_checked_arithmetic_failure_uses_finalization_timeout():
    """@spec PORT-SESS-017."""
    from vllm_omni.entrypoints.session_lifecycle import arm_session_finalization

    observed = []

    async def expire(kind):
        observed.append(kind)

    deadline = SessionLifecycleDeadline(idle_timeout_s=60, finalization_timeout_s=40, on_expire=expire)
    authority = SimpleNamespace(chunk_samples=160, finalization_work=lambda: (10**400, 0))
    arm_session_finalization(deadline, SimpleNamespace(accepted_audio=authority), None)
    await deadline.close()
    assert observed == ["finalization"]
    assert deadline.receipt["derivation_error"]


def test_terminal_claim_closes_acceptance_before_cleanup_await():
    """@spec PORT-SESS-014, PORT-SESS-018."""
    import numpy as np

    from vllm_omni.model_executor.models.nemotron_asr.accepted_audio import AcceptedAudioAuthority

    authority = AcceptedAudioAuthority(
        request_id="a",
        engine_epoch="e",
        lease_generation=3,
        chunk_samples=160,
        capacity_samples=1600,
        carrier_sequence_modulus=100,
    )
    assert authority.claim_terminal("disconnect", lease_generation=3)
    with pytest.raises(ValueError):
        authority.accept(np.zeros(160, dtype=np.float32))
    assert authority.snapshot().accepted_samples == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("operator_absolute,expected_warning", [(40, True), (10000, False)])
@pytest.mark.parametrize("adapter", ["native", "leased"])
async def test_adapter_receipt_warns_on_winning_configured_assumption(
    caplog, operator_absolute, expected_warning, adapter
):
    """@spec PORT-SESS-017: fallback stays 10s and its winning bound is visible."""
    import json

    from vllm_omni.entrypoints.nemotron_session import NemotronSessionLease
    from vllm_omni.entrypoints.openai.realtime_connection import RealtimeConnection

    async def expire(kind):
        pass

    deadline = SessionLifecycleDeadline(idle_timeout_s=60, finalization_timeout_s=operator_absolute, on_expire=expire)
    authority = SimpleNamespace(chunk_samples=1280, finalization_work=lambda: (640000, 0))
    session = SimpleNamespace(accepted_audio=authority)
    service = SimpleNamespace(
        runtime_config=SimpleNamespace(session_finalization_stall_timeout=10 if expected_warning else 10000)
    )
    owner = SimpleNamespace(
        _session_lifecycle=deadline, _persistent_state_service=service, _session=session, _nemotron_session=session
    )
    adapter_type = RealtimeConnection if adapter == "native" else NemotronSessionLease
    with caplog.at_level("INFO"):
        adapter_type._arm_session_lifecycle_timeout(owner, "finalization")
    await deadline.close()
    receipt = json.loads(
        next(
            r.message.removeprefix("Finalization receipt ")
            for r in caplog.records
            if r.message.startswith("Finalization receipt ")
        )
    )
    assert receipt["response_s"] == 10
    assert receipt["stall_winner"] == ("derived" if expected_warning else "operator")
    assert receipt["absolute_winner"] == ("derived" if expected_warning else "operator")
    assert receipt["absolute_from_configured_fallback"] is expected_warning
    assert receipt["unqualified"] is True
    assert "unqualified operational assumption" in receipt["assumption_note"]
    warnings = [r for r in caplog.records if r.levelname == "WARNING"]
    assert bool(warnings) is expected_warning
    if expected_warning:
        assert "configured" in warnings[0].message
        assert receipt["absolute_s"] == pytest.approx(5020.1)


@pytest.mark.asyncio
async def test_adapter_derives_clock_operands_from_installed_modes(monkeypatch):
    """@spec PORT-SESS-016, PORT-SESS-017: installed sample rate/cadences own timing."""
    from vllm_omni.entrypoints import session_lifecycle as lifecycle
    from vllm_omni.model_executor.models.nemotron_asr import manifests

    monkeypatch.setitem(manifests.FRONTEND_CONSTANTS, "sample_rate", 8000)
    monkeypatch.setattr(manifests, "RAW_SAMPLES_PER_CHUNK", {"test": 24000})

    async def expire(kind):
        pass

    deadline = SessionLifecycleDeadline(idle_timeout_s=60, finalization_timeout_s=1, on_expire=expire)
    authority = SimpleNamespace(chunk_samples=16000, finalization_work=lambda: (32000, 0))
    service = SimpleNamespace(
        runtime_config=SimpleNamespace(session_finalization_response_bound_s=0.5, session_finalization_stall_timeout=1)
    )
    lifecycle.arm_session_finalization(deadline, SimpleNamespace(accepted_audio=authority), service)
    await deadline.close()
    assert deadline.receipt["absolute_s"] == pytest.approx(5.1)
    assert deadline.receipt["stall_s"] == pytest.approx(3.6)
    assert deadline.receipt["adapter_margin_s"] == lifecycle.FINALIZATION_ADAPTER_MARGIN_S
