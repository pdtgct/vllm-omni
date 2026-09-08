# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Phase-5 tests-first: the production streaming-metrics families.

``vllm_omni.metrics.streaming`` family declarations are REAL (they
register on import exactly as production will), so the family-shape tests
below (existence, names, types, label sets, bucket ladders) are GREEN
today. Every ``OmniStreamingMetrics`` observe method unconditionally
raises ``NotImplementedError`` (no ``log_stats`` gating logic yet), so the
behavioral tests (gating, verbatim model_name label) are RED today and
turn green without being rewritten once Phase 6 implements them (the LID
tests-first rule, matching ``test_manifests.py``).
"""

from __future__ import annotations

import re
from typing import Any

import pytest
from prometheus_client import REGISTRY, generate_latest

from vllm_omni.metrics import definitions as defs
from vllm_omni.metrics.streaming import OmniStreamingMetrics

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

_MODEL = "test-streaming-model"


def _streaming_lines(out: str) -> list[str]:
    """Exposition lines for this module's families only.

    The default registry also carries live process collectors
    (``process_cpu_seconds_total``, ``process_resident_memory_bytes``,
    ``python_gc_*``) whose values move on their own between two scrapes,
    so a whole-registry diff is not a stable assertion — it fails
    whenever CPU time ticks or a GC runs mid-test, which depends on what
    ran before. Scope the comparison to the families under test.
    """
    return [line for line in out.splitlines() if line.startswith(defs.METRIC_PREFIX + "streaming_")]


_EXPECTED_FAMILIES: dict[str, tuple[str, tuple[str, ...]]] = {
    # exposition name -> (type, labelnames)
    defs.STREAMING_SESSIONS_ACTIVE: ("gauge", defs.STREAMING_CADENCE_LABELS),
    defs.STREAMING_SESSIONS_FINISHED + "_total": ("counter", defs.STREAMING_FINISHED_LABELS),
    defs.STREAMING_CHUNK_LATENCY_S: ("histogram", defs.STREAMING_CHUNK_LATENCY_LABELS),
    defs.STREAMING_CHUNKS + "_total": ("counter", defs.STREAMING_CHUNKS_LABELS),
    defs.STREAMING_DEADLINE_MISSES + "_total": ("counter", defs.STREAMING_DEADLINE_MISS_LABELS),
    defs.STREAMING_BACKLOG_CHUNKS: ("gauge", defs.STREAMING_BACKLOG_LABELS),
    defs.STREAMING_BACKLOG_OVERFLOWS + "_total": ("counter", defs.STREAMING_OVERFLOW_LABELS),
    defs.STREAMING_SESSION_OPEN_REJECTIONS + "_total": ("counter", defs.STREAMING_OPEN_REJECTION_LABELS),
    defs.STREAMING_ADMISSION_REJECTIONS + "_total": ("counter", defs.STREAMING_ADMISSION_REJECTION_LABELS),
    defs.PERSISTENT_STATE_SLOTS: ("gauge", defs.PERSISTENT_STATE_SLOT_LABELS),
    defs.STREAMING_INPUT_AUDIO_SECONDS + "_total": ("counter", defs.STREAMING_INPUT_AUDIO_LABELS),
    defs.STREAMING_CHUNK_BATCH_SIZE: ("histogram", defs.STREAMING_BATCH_SIZE_LABELS),
}


@pytest.fixture(scope="module")
def scrape() -> str:
    return generate_latest(REGISTRY).decode()


# ---------------------------------------------------------------------------
# PORT-OBS-001: family existence, names, types, label sets, bucket ladders.
# ---------------------------------------------------------------------------


class TestFamilyRegistration:
    # @spec PORT-OBS-001
    def test_all_production_families_present(self, scrape: str) -> None:
        for name in _EXPECTED_FAMILIES:
            assert f"# HELP {name}" in scrape, f"missing streaming family: {name}"

    # @spec PORT-OBS-001
    def test_family_types_match_the_design_table(self, scrape: str) -> None:
        for name, (expected_type, _labels) in _EXPECTED_FAMILIES.items():
            assert f"# TYPE {name} {expected_type}" in scrape, f"{name} should be a {expected_type}"

    # @spec PORT-OBS-001
    def test_families_carry_model_name_plus_their_declared_labels(self) -> None:
        # model_name is implicit on every family (PORT-OBS-001); the
        # remaining labelnames are exactly the design table's per-family
        # set, no more, no less.
        from vllm_omni.metrics import streaming as mod

        by_name = {
            defs.STREAMING_SESSIONS_ACTIVE: mod._sessions_active_family,
            defs.STREAMING_SESSIONS_FINISHED: mod._sessions_finished_family,
            defs.STREAMING_CHUNK_LATENCY_S: mod._chunk_latency_family,
            defs.STREAMING_CHUNKS: mod._chunks_family,
            defs.STREAMING_DEADLINE_MISSES: mod._deadline_misses_family,
            defs.STREAMING_BACKLOG_CHUNKS: mod._backlog_family,
            defs.STREAMING_BACKLOG_OVERFLOWS: mod._backlog_overflow_family,
            defs.STREAMING_SESSION_OPEN_REJECTIONS: mod._open_rejections_family,
            defs.STREAMING_ADMISSION_REJECTIONS: mod._admission_rejections_family,
            defs.PERSISTENT_STATE_SLOTS: mod._persistent_state_slots_family,
            defs.STREAMING_INPUT_AUDIO_SECONDS: mod._input_audio_seconds_family,
            defs.STREAMING_CHUNK_BATCH_SIZE: mod._chunk_batch_size_family,
        }
        expected_by_name = {
            defs.STREAMING_SESSIONS_ACTIVE: defs.STREAMING_CADENCE_LABELS,
            defs.STREAMING_SESSIONS_FINISHED: defs.STREAMING_FINISHED_LABELS,
            defs.STREAMING_CHUNK_LATENCY_S: defs.STREAMING_CHUNK_LATENCY_LABELS,
            defs.STREAMING_CHUNKS: defs.STREAMING_CHUNKS_LABELS,
            defs.STREAMING_DEADLINE_MISSES: defs.STREAMING_DEADLINE_MISS_LABELS,
            defs.STREAMING_BACKLOG_CHUNKS: defs.STREAMING_BACKLOG_LABELS,
            defs.STREAMING_BACKLOG_OVERFLOWS: defs.STREAMING_OVERFLOW_LABELS,
            defs.STREAMING_SESSION_OPEN_REJECTIONS: defs.STREAMING_OPEN_REJECTION_LABELS,
            defs.STREAMING_ADMISSION_REJECTIONS: defs.STREAMING_ADMISSION_REJECTION_LABELS,
            defs.PERSISTENT_STATE_SLOTS: defs.PERSISTENT_STATE_SLOT_LABELS,
            defs.STREAMING_INPUT_AUDIO_SECONDS: defs.STREAMING_INPUT_AUDIO_LABELS,
            defs.STREAMING_CHUNK_BATCH_SIZE: defs.STREAMING_BATCH_SIZE_LABELS,
        }
        for name, family in by_name.items():
            assert tuple(sorted(family._labelnames)) == tuple(sorted(expected_by_name[name])), name

    # @spec PORT-OBS-001, PORT-OBS-007
    def test_open_rejections_carries_no_cadence_label(self) -> None:
        # PORT-OBS-007: the rejected value is untrusted client input and
        # deliberately does not become a label.
        assert "cadence_ms" not in defs.STREAMING_OPEN_REJECTION_LABELS

    # @spec PORT-OBS-001
    def test_chunk_latency_bucket_ladder_carries_every_admitted_cadence(self) -> None:
        # Histogram bucket samples only appear after a child is created via
        # .labels(...) — observe directly on the module-level family (not
        # through OmniStreamingMetrics, which is unimplemented) so this
        # test pins the declaration, not the stub wrapper.
        from vllm_omni.metrics import streaming as mod

        mod._chunk_latency_family.labels(model_name=_MODEL, cadence_ms="560", chunk_type="regular").observe(0.05)
        out = generate_latest(REGISTRY).decode()
        for cadence_s in ("0.08", "0.16", "0.32", "0.56", "1.12"):
            pattern = rf'{re.escape(defs.STREAMING_CHUNK_LATENCY_S)}_bucket\{{[^}}]*le="{re.escape(cadence_s)}"'
            assert re.search(pattern, out), f"latency ladder missing cadence-edge bucket le={cadence_s}"

    # @spec PORT-OBS-001
    def test_batch_size_bucket_ladder_is_the_fixed_import_time_series(self) -> None:
        from vllm_omni.metrics import streaming as mod

        mod._chunk_batch_size_family.labels(model_name=_MODEL, stage="0", replica="0", cadence_ms="560").observe(4)
        out = generate_latest(REGISTRY).decode()
        for edge in ("1.0", "2.0", "5.0", "10.0", "20.0", "50.0", "100.0", "200.0", "500.0", "1000.0", "2000.0"):
            pattern = rf'{re.escape(defs.STREAMING_CHUNK_BATCH_SIZE)}_bucket\{{[^}}]*le="{re.escape(edge)}"'
            assert re.search(pattern, out), f"batch-size ladder missing le={edge}"


# ---------------------------------------------------------------------------
# Enum completeness (PORT-OBS-001's bounded label enumerations).
# ---------------------------------------------------------------------------


class TestEnumCompleteness:
    # @spec PORT-OBS-001
    def test_cadence_ms_values(self) -> None:
        assert defs.STREAMING_CADENCE_MS_VALUES == ("80", "160", "320", "560", "1120")

    # @spec PORT-OBS-001
    def test_chunk_types(self) -> None:
        assert defs.STREAMING_CHUNK_TYPES == ("regular", "final_tail")

    # @spec PORT-OBS-001
    def test_chunk_outcomes(self) -> None:
        assert defs.STREAMING_CHUNK_OUTCOMES == ("parked", "aborted", "error")

    # @spec PORT-OBS-001, PORT-OBS-006
    def test_finished_reasons(self) -> None:
        assert defs.STREAMING_FINISHED_REASONS == ("completed", "aborted", "error")

    # @spec PORT-OBS-001, PORT-OBS-005
    def test_overflow_kinds(self) -> None:
        assert defs.STREAMING_OVERFLOW_KINDS == ("input_queue", "carrier", "receipt")

    # @spec PORT-OBS-001, PORT-OBS-007
    def test_open_rejection_reasons(self) -> None:
        assert defs.STREAMING_OPEN_REJECTION_REASONS == ("model", "cadence", "locale", "config")

    # @spec PORT-OBS-001, PORT-OBS-007
    def test_admission_rejection_reasons(self) -> None:
        assert defs.STREAMING_ADMISSION_REJECTION_REASONS == (
            "capacity",
            "unavailable",
            "unsupported",
        )

    # @spec PORT-OBS-001, PORT-OBS-010
    def test_persistent_state_slot_kinds(self) -> None:
        assert defs.PERSISTENT_STATE_SLOT_KINDS == (
            "resident",
            "safety_reserve",
            "physical_capacity",
            "configured_limit",
            "effective_capacity",
        )

    # @spec PORT-OBS-001
    def test_latency_bucket_ladder_is_exact(self) -> None:
        assert defs.STREAMING_LATENCY_BUCKETS == (
            0.01,
            0.02,
            0.04,
            0.08,
            0.16,
            0.32,
            0.56,
            0.84,
            1.12,
            1.68,
            2.24,
            4.48,
            10.0,
        )

    # @spec PORT-OBS-001
    def test_batch_size_bucket_ladder_is_exact(self) -> None:
        assert defs.STREAMING_BATCH_SIZE_BUCKETS == (1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000)


# ---------------------------------------------------------------------------
# No-unregister discipline (module-level singletons on the default
# registry, matching prometheus.py/modality.py; no unregister helper).
# ---------------------------------------------------------------------------


class TestNoUnregisterDiscipline:
    # @spec PORT-OBS-001
    def test_streaming_module_defines_no_unregister_helper(self) -> None:
        import inspect

        import vllm_omni.metrics.streaming as mod

        source = inspect.getsource(mod)
        assert re.search(r"\.unregister\(|\bdef unregister\b", source) is None

    # @spec PORT-OBS-001
    def test_streaming_module_uses_the_process_default_registry(self) -> None:
        # Families are constructed with no explicit registry= kwarg, so
        # they land on prometheus_client's implicit default REGISTRY —
        # the same convention as prometheus.py/modality.py.
        import inspect

        import vllm_omni.metrics.streaming as mod

        source = inspect.getsource(mod)
        assert "registry=" not in source


# ---------------------------------------------------------------------------
# PORT-OBS-002: log_stats gating. RED — the stub raises unconditionally.
# ---------------------------------------------------------------------------


class TestLogStatsGating:
    # @spec PORT-OBS-002
    def test_disabled_collection_returns_without_recording_or_raising(self) -> None:
        metrics = OmniStreamingMetrics(model_name=_MODEL, log_stats=False)
        before = _streaming_lines(generate_latest(REGISTRY).decode())

        # Disabled collection must return silently and leave every
        # streaming sample untouched — stronger than checking one
        # family, and unaffected by collectors this module does not own.
        metrics.inc_sessions_active("560")

        after = _streaming_lines(generate_latest(REGISTRY).decode())
        assert after == before

    # @spec PORT-OBS-002
    def test_disabled_collection_covers_every_observe_method(self) -> None:
        metrics = OmniStreamingMetrics(model_name=_MODEL, log_stats=False)
        calls = [
            lambda: metrics.inc_sessions_active("560"),
            lambda: metrics.observe_session_finished("560", "completed"),
            lambda: metrics.inc_open_rejection("model"),
            lambda: metrics.inc_admission_rejection("capacity"),
            lambda: metrics.observe_persistent_state_slots(
                "0",
                "0",
                {
                    "resident": 1,
                    "safety_reserve": 2,
                    "physical_capacity": 8,
                    "configured_limit": 7,
                    "effective_capacity": 6,
                },
            ),
            lambda: metrics.inc_input_audio_seconds("560", 0.56),
            lambda: metrics.observe_chunk_latency("560", "regular", 0.1),
            lambda: metrics.observe_chunk_outcome("560", "regular", "parked"),
            lambda: metrics.inc_deadline_miss("560", "regular"),
            lambda: metrics.inc_backlog("560"),
            lambda: metrics.dec_backlog("560"),
            lambda: metrics.inc_backlog_overflow("input_queue"),
            lambda: metrics.observe_chunk_batch_size("0", "0", "560", 4),
        ]
        for call in calls:
            call()  # must not raise once log_stats gating is implemented

        out = generate_latest(REGISTRY).decode()
        assert f'{defs.PERSISTENT_STATE_SLOTS}{{kind="resident",model_name="{_MODEL}"' not in out


# ---------------------------------------------------------------------------
# Served-alias model_name: caller-supplied string used verbatim as the
# `model_name` label (PORT-OBS-001).
# ---------------------------------------------------------------------------


class TestServedAliasVerbatim:
    # @spec PORT-OBS-001
    def test_model_name_alias_appears_verbatim_in_the_scraped_label(self) -> None:
        alias = "my-served-alias:v1"
        metrics = OmniStreamingMetrics(model_name=alias, log_stats=True)
        metrics.inc_sessions_active("320")

        out = generate_latest(REGISTRY).decode()
        pattern = (
            rf"{re.escape(defs.STREAMING_SESSIONS_ACTIVE)}\{{"
            rf'cadence_ms="320",model_name="{re.escape(alias)}"'
        )
        assert re.search(pattern, out), "alias must be used verbatim, never transformed"


# ---------------------------------------------------------------------------
# Enabled-path behavior via scrape deltas (correction 6). Every call below
# hits an unconditional NotImplementedError today (OmniStreamingMetrics'
# Phase-5 stub), so each test is RED with that failure mode; each pins the
# exact real behavior Phase 6 must produce.
# ---------------------------------------------------------------------------


def _count_value(out: str, prefix: str) -> float | None:
    for line in out.splitlines():
        if line.startswith(prefix):
            return float(line.split()[-1])
    return None


class TestEnabledPathScrapeDeltas:
    # @spec PORT-OBS-006
    def test_inc_sessions_active_produces_exact_gauge_delta(self) -> None:
        metrics = OmniStreamingMetrics(model_name="delta-model-1", log_stats=True)
        before = (
            _count_value(
                generate_latest(REGISTRY).decode(),
                f'{defs.STREAMING_SESSIONS_ACTIVE}{{cadence_ms="560",model_name="delta-model-1"}}',
            )
            or 0.0
        )

        metrics.inc_sessions_active("560")

        after = _count_value(
            generate_latest(REGISTRY).decode(),
            f'{defs.STREAMING_SESSIONS_ACTIVE}{{cadence_ms="560",model_name="delta-model-1"}}',
        )
        assert after == before + 1.0

    # @spec PORT-OBS-006
    def test_observe_session_finished_produces_exact_counter_delta(self) -> None:
        metrics = OmniStreamingMetrics(model_name="delta-model-2", log_stats=True)
        prefix = (
            f'{defs.STREAMING_SESSIONS_FINISHED}_total{{cadence_ms="560",'
            'model_name="delta-model-2",reason="completed"}'
        )
        before = _count_value(generate_latest(REGISTRY).decode(), prefix) or 0.0

        metrics.observe_session_finished("560", "completed")

        after = _count_value(generate_latest(REGISTRY).decode(), prefix)
        assert after == before + 1.0

    # @spec PORT-OBS-007
    def test_inc_open_rejection_produces_exact_counter_delta(self) -> None:
        metrics = OmniStreamingMetrics(model_name="delta-model-3", log_stats=True)
        prefix = f'{defs.STREAMING_SESSION_OPEN_REJECTIONS}_total{{model_name="delta-model-3",reason="cadence"}}'
        before = _count_value(generate_latest(REGISTRY).decode(), prefix) or 0.0

        metrics.inc_open_rejection("cadence")

        after = _count_value(generate_latest(REGISTRY).decode(), prefix)
        assert after == before + 1.0

    # @spec PORT-OBS-007
    def test_inc_admission_rejection_produces_exact_counter_delta(self) -> None:
        metrics = OmniStreamingMetrics(model_name="delta-model-admission", log_stats=True)
        prefix = f'{defs.STREAMING_ADMISSION_REJECTIONS}_total{{model_name="delta-model-admission",reason="capacity"}}'
        before = _count_value(generate_latest(REGISTRY).decode(), prefix) or 0.0

        metrics.inc_admission_rejection("capacity")

        after = _count_value(generate_latest(REGISTRY).decode(), prefix)
        assert after == before + 1.0

    # @spec PORT-OBS-010
    def test_persistent_state_slots_replace_the_manager_projection(self) -> None:
        metrics = OmniStreamingMetrics(model_name="delta-model-state-slots", log_stats=True)
        first = {
            "resident": 1,
            "safety_reserve": 2,
            "physical_capacity": 8,
            "configured_limit": 7,
            "effective_capacity": 6,
        }
        metrics.observe_persistent_state_slots("0", "0", first)

        for kind, expected in first.items():
            prefix = (
                f'{defs.PERSISTENT_STATE_SLOTS}{{kind="{kind}",'
                'model_name="delta-model-state-slots",replica="0",stage="0"}'
            )
            assert _count_value(generate_latest(REGISTRY).decode(), prefix) == expected

        metrics.observe_persistent_state_slots("0", "0", {**first, "resident": 3})
        resident_prefix = (
            f'{defs.PERSISTENT_STATE_SLOTS}{{kind="resident",'
            'model_name="delta-model-state-slots",replica="0",stage="0"}'
        )
        assert _count_value(generate_latest(REGISTRY).decode(), resident_prefix) == 3

    # @spec PORT-OBS-006
    def test_inc_input_audio_seconds_produces_exact_counter_delta(self) -> None:
        metrics = OmniStreamingMetrics(model_name="delta-model-4", log_stats=True)
        prefix = f'{defs.STREAMING_INPUT_AUDIO_SECONDS}_total{{cadence_ms="560",model_name="delta-model-4"}}'
        before = _count_value(generate_latest(REGISTRY).decode(), prefix) or 0.0

        metrics.inc_input_audio_seconds("560", 0.56)

        after = _count_value(generate_latest(REGISTRY).decode(), prefix)
        assert after == pytest.approx((before or 0.0) + 0.56)

    # @spec PORT-OBS-004
    def test_observe_chunk_latency_produces_exact_histogram_count_delta(self) -> None:
        metrics = OmniStreamingMetrics(model_name="delta-model-5", log_stats=True)
        prefix = (
            f'{defs.STREAMING_CHUNK_LATENCY_S}_count{{cadence_ms="560",'
            'chunk_type="regular",model_name="delta-model-5"}'
        )
        before = _count_value(generate_latest(REGISTRY).decode(), prefix) or 0.0

        metrics.observe_chunk_latency("560", "regular", 0.1)

        after = _count_value(generate_latest(REGISTRY).decode(), prefix)
        assert after == before + 1.0

    # @spec PORT-OBS-004, PORT-OBS-005
    def test_observe_chunk_outcome_produces_exact_counter_delta(self) -> None:
        metrics = OmniStreamingMetrics(model_name="delta-model-6", log_stats=True)
        prefix = (
            f'{defs.STREAMING_CHUNKS}_total{{cadence_ms="560",chunk_type="regular",'
            'model_name="delta-model-6",outcome="parked"}'
        )
        before = _count_value(generate_latest(REGISTRY).decode(), prefix) or 0.0

        metrics.observe_chunk_outcome("560", "regular", "parked")

        after = _count_value(generate_latest(REGISTRY).decode(), prefix)
        assert after == before + 1.0

    # @spec PORT-OBS-004
    def test_inc_deadline_miss_produces_exact_counter_delta(self) -> None:
        metrics = OmniStreamingMetrics(model_name="delta-model-7", log_stats=True)
        prefix = (
            f'{defs.STREAMING_DEADLINE_MISSES}_total{{cadence_ms="560",'
            'chunk_type="regular",model_name="delta-model-7"}'
        )
        before = _count_value(generate_latest(REGISTRY).decode(), prefix) or 0.0

        metrics.inc_deadline_miss("560", "regular")

        after = _count_value(generate_latest(REGISTRY).decode(), prefix)
        assert after == before + 1.0

    # @spec PORT-OBS-005
    def test_inc_and_dec_backlog_produce_exact_gauge_deltas(self) -> None:
        metrics = OmniStreamingMetrics(model_name="delta-model-8", log_stats=True)
        prefix = f'{defs.STREAMING_BACKLOG_CHUNKS}{{cadence_ms="560",model_name="delta-model-8"}}'
        base = _count_value(generate_latest(REGISTRY).decode(), prefix) or 0.0

        metrics.inc_backlog("560")
        after_inc = _count_value(generate_latest(REGISTRY).decode(), prefix)
        assert after_inc == base + 1.0

        metrics.dec_backlog("560")
        after_dec = _count_value(generate_latest(REGISTRY).decode(), prefix)
        assert after_dec == base

    # @spec PORT-OBS-005
    def test_inc_backlog_overflow_produces_exact_counter_delta(self) -> None:
        metrics = OmniStreamingMetrics(model_name="delta-model-9", log_stats=True)
        prefix = f'{defs.STREAMING_BACKLOG_OVERFLOWS}_total{{kind="input_queue",model_name="delta-model-9"}}'
        before = _count_value(generate_latest(REGISTRY).decode(), prefix) or 0.0

        metrics.inc_backlog_overflow("input_queue")

        after = _count_value(generate_latest(REGISTRY).decode(), prefix)
        assert after == before + 1.0

    # @spec PORT-OBS-008, PORT-OBS-009
    def test_observe_chunk_batch_size_produces_exact_histogram_count_delta(self) -> None:
        metrics = OmniStreamingMetrics(model_name="delta-model-10", log_stats=True)
        prefix = (
            f'{defs.STREAMING_CHUNK_BATCH_SIZE}_count{{cadence_ms="560",'
            'model_name="delta-model-10",replica="0",stage="0"}'
        )
        before = _count_value(generate_latest(REGISTRY).decode(), prefix) or 0.0

        metrics.observe_chunk_batch_size("0", "0", "560", 4)

        after = _count_value(generate_latest(REGISTRY).decode(), prefix)
        assert after == before + 1.0


class TestBacklogIdempotency:
    # @spec PORT-OBS-005
    def test_double_terminal_disposition_does_not_double_decrement_backlog(self) -> None:
        """A double-fire terminal disposition on the SAME ready handle
        (e.g. an abort racing a natural park) must decrement backlog
        exactly once, never twice, through the observer adapter."""
        from vllm_omni.metrics.streaming import PrometheusStreamingObserver

        metrics = OmniStreamingMetrics(model_name="idempotent-backlog-model", log_stats=True)
        observer = PrometheusStreamingObserver(metrics)
        prefix = f'{defs.STREAMING_BACKLOG_CHUNKS}{{cadence_ms="560",model_name="idempotent-backlog-model"}}'
        base = _count_value(generate_latest(REGISTRY).decode(), prefix) or 0.0

        observer.session_opened(session_key="idem-key", cadence_ms="560")
        handle = observer.unit_ready(session_key="idem-key", cadence_ms="560", chunk_type="regular", ready_stamp_s=0.0)
        observer.unit_parked(handle, park_stamp_s=0.1)
        after_first_terminal = _count_value(generate_latest(REGISTRY).decode(), prefix)
        assert after_first_terminal == base  # +1 at ready, -1 at park -> net 0

        # A racing second terminal call on the SAME handle must be a no-op.
        observer.unit_cleared(handle, outcome="aborted")
        after_second_terminal = _count_value(generate_latest(REGISTRY).decode(), prefix)
        assert after_second_terminal == base


class TestStrictLatencyMissBoundary:
    # @spec PORT-OBS-004
    def test_latency_exactly_equal_to_cadence_is_not_a_miss(self) -> None:
        """Equality lands in the on-budget bucket; only STRICTLY greater
        latency counts as a miss (port-design.md §Production metric
        export)."""
        from vllm_omni.metrics.streaming import PrometheusStreamingObserver

        metrics = OmniStreamingMetrics(model_name="miss-boundary-model", log_stats=True)
        observer = PrometheusStreamingObserver(metrics)
        miss_prefix = (
            f'{defs.STREAMING_DEADLINE_MISSES}_total{{cadence_ms="560",'
            'chunk_type="regular",model_name="miss-boundary-model"}'
        )
        before = _count_value(generate_latest(REGISTRY).decode(), miss_prefix) or 0.0

        observer.session_opened(session_key="fixture-key", cadence_ms="560")
        handle = observer.unit_ready(
            session_key="fixture-key", cadence_ms="560", chunk_type="regular", ready_stamp_s=0.0
        )
        observer.unit_parked(handle, park_stamp_s=0.56)  # exactly the 560ms cadence period

        # Lead-authorized fix (Phase-6 round 2, Q1a): coerce both sides
        # identically — an absent series IS the correct no-observation
        # outcome, not a value distinct from a coerced 0.0.
        after = _count_value(generate_latest(REGISTRY).decode(), miss_prefix) or 0.0
        assert after == before  # equality must NOT count as a miss

    # @spec PORT-OBS-004
    def test_latency_strictly_over_cadence_is_a_miss(self) -> None:
        from vllm_omni.metrics.streaming import PrometheusStreamingObserver

        metrics = OmniStreamingMetrics(model_name="miss-boundary-model-2", log_stats=True)
        observer = PrometheusStreamingObserver(metrics)
        miss_prefix = (
            f'{defs.STREAMING_DEADLINE_MISSES}_total{{cadence_ms="560",'
            'chunk_type="regular",model_name="miss-boundary-model-2"}'
        )
        before = _count_value(generate_latest(REGISTRY).decode(), miss_prefix) or 0.0

        observer.session_opened(session_key="fixture-key", cadence_ms="560")
        handle = observer.unit_ready(
            session_key="fixture-key", cadence_ms="560", chunk_type="regular", ready_stamp_s=0.0
        )
        observer.unit_parked(handle, park_stamp_s=0.560001)  # strictly over

        after = _count_value(generate_latest(REGISTRY).decode(), miss_prefix)
        assert after == before + 1.0


class TestParkedVsCleared:
    # @spec PORT-OBS-004, PORT-OBS-005
    def test_cleared_unit_observes_no_latency_and_no_miss(self) -> None:
        """A cleared (aborted/error) unit must never observe chunk
        latency or a deadline miss — only its outcome-labeled counter."""
        from vllm_omni.metrics.streaming import PrometheusStreamingObserver

        metrics = OmniStreamingMetrics(model_name="parked-vs-cleared-model", log_stats=True)
        observer = PrometheusStreamingObserver(metrics)
        latency_count_prefix = (
            f'{defs.STREAMING_CHUNK_LATENCY_S}_count{{cadence_ms="560",'
            'chunk_type="regular",model_name="parked-vs-cleared-model"}'
        )
        miss_prefix = (
            f'{defs.STREAMING_DEADLINE_MISSES}_total{{cadence_ms="560",'
            'chunk_type="regular",model_name="parked-vs-cleared-model"}'
        )
        outcome_prefix = (
            f'{defs.STREAMING_CHUNKS}_total{{cadence_ms="560",chunk_type="regular",'
            'model_name="parked-vs-cleared-model",outcome="aborted"}'
        )
        latency_before = _count_value(generate_latest(REGISTRY).decode(), latency_count_prefix) or 0.0
        miss_before = _count_value(generate_latest(REGISTRY).decode(), miss_prefix) or 0.0
        outcome_before = _count_value(generate_latest(REGISTRY).decode(), outcome_prefix) or 0.0

        observer.session_opened(session_key="fixture-key", cadence_ms="560")
        handle = observer.unit_ready(
            session_key="fixture-key", cadence_ms="560", chunk_type="regular", ready_stamp_s=0.0
        )
        observer.unit_cleared(handle, outcome="aborted")

        # Lead-authorized fix (Phase-6 round 2, Q1a): coerce identically
        # on both sides — an absent series IS the correct no-observation
        # outcome for latency/miss (never touched for a cleared unit).
        out = generate_latest(REGISTRY).decode()
        assert (_count_value(out, latency_count_prefix) or 0.0) == latency_before
        assert (_count_value(out, miss_prefix) or 0.0) == miss_before
        assert (_count_value(out, outcome_prefix) or 0.0) == outcome_before + 1.0


class TestBoundedEnumRejection:
    # @spec PORT-OBS-001, PORT-OBS-007
    def test_unknown_open_rejection_reason_is_rejected_or_skipped_never_a_new_label(self) -> None:
        metrics = OmniStreamingMetrics(model_name="enum-reject-model", log_stats=True)
        before = generate_latest(REGISTRY).decode()

        try:
            metrics.inc_open_rejection("not-a-bounded-reason")
        except (ValueError, NotImplementedError):
            pass

        after = generate_latest(REGISTRY).decode()
        assert 'reason="not-a-bounded-reason"' not in after
        # No new label VALUE may appear on the family regardless of path taken.
        assert after == before or 'reason="not-a-bounded-reason"' not in after

    # @spec PORT-OBS-001, PORT-OBS-005
    def test_unknown_overflow_kind_is_rejected_or_skipped_never_a_new_label(self) -> None:
        metrics = OmniStreamingMetrics(model_name="enum-reject-model-2", log_stats=True)

        try:
            metrics.inc_backlog_overflow("not-a-bounded-kind")
        except (ValueError, NotImplementedError):
            pass

        after = generate_latest(REGISTRY).decode()
        assert 'kind="not-a-bounded-kind"' not in after

    # @spec PORT-OBS-001, PORT-OBS-004
    def test_unknown_chunk_outcome_is_rejected_or_skipped_never_a_new_label(self) -> None:
        metrics = OmniStreamingMetrics(model_name="enum-reject-model-3", log_stats=True)

        try:
            metrics.observe_chunk_outcome("560", "regular", "not-a-bounded-outcome")
        except (ValueError, NotImplementedError):
            pass

        after = generate_latest(REGISTRY).decode()
        assert 'outcome="not-a-bounded-outcome"' not in after


# ---------------------------------------------------------------------------
# Keyed session lifecycle + bounded per-session observer state
# (PORT-OBS-003/006 as amended by the A27 topology cascade, notes PR #106).
# ---------------------------------------------------------------------------


def _keyed_observer(model_name: str) -> Any:
    from vllm_omni.metrics.streaming import PrometheusStreamingObserver

    metrics = OmniStreamingMetrics(model_name=model_name, log_stats=True)
    return PrometheusStreamingObserver(metrics)


def _gauge(prefix: str) -> float:
    return _count_value(generate_latest(REGISTRY).decode(), prefix) or 0.0


class TestKeyedSessionLifecycle:
    # @spec PORT-OBS-003, PORT-OBS-006
    def test_first_open_increments_active_and_finish_uses_opens_cadence(self) -> None:
        """Open records the cadence; finish takes no cadence argument and
        resolves the label from the record captured at open."""
        observer = _keyed_observer("keyed-lc-1")
        active = f'{defs.STREAMING_SESSIONS_ACTIVE}{{cadence_ms="560",model_name="keyed-lc-1"}}'
        finished = (
            f'{defs.STREAMING_SESSIONS_FINISHED}_total{{cadence_ms="560",model_name="keyed-lc-1",reason="completed"}}'
        )
        base_active, base_fin = _gauge(active), _gauge(finished)

        observer.session_opened(session_key="req-a", cadence_ms="560")
        assert _gauge(active) == base_active + 1

        observer.session_finished(session_key="req-a", reason="completed")
        assert _gauge(active) == base_active
        assert _gauge(finished) == base_fin + 1

    # @spec PORT-OBS-003
    def test_duplicate_open_of_an_active_session_is_a_no_op(self) -> None:
        observer = _keyed_observer("keyed-lc-2")
        active = f'{defs.STREAMING_SESSIONS_ACTIVE}{{cadence_ms="560",model_name="keyed-lc-2"}}'
        base = _gauge(active)
        observer.session_opened(session_key="req-b", cadence_ms="560")
        observer.session_opened(session_key="req-b", cadence_ms="560")
        assert _gauge(active) == base + 1
        observer.session_finished(session_key="req-b", reason="completed")
        assert _gauge(active) == base

    # @spec PORT-OBS-003
    def test_conflicting_cadence_on_duplicate_open_is_logged_and_ignored(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        observer = _keyed_observer("keyed-lc-3")
        active_560 = f'{defs.STREAMING_SESSIONS_ACTIVE}{{cadence_ms="560",model_name="keyed-lc-3"}}'
        base = _gauge(active_560)
        observer.session_opened(session_key="req-c", cadence_ms="560")
        with caplog.at_level("WARNING", logger="vllm_omni.metrics.streaming"):
            observer.session_opened(session_key="req-c", cadence_ms="1120")
        assert any("cadence" in rec.message for rec in caplog.records)
        # The record keeps the FIRST cadence; finish resolves to it.
        observer.session_finished(session_key="req-c", reason="completed")
        assert _gauge(active_560) == base

    # @spec PORT-OBS-003, PORT-OBS-006
    def test_unknown_and_duplicate_finish_are_no_ops(self) -> None:
        observer = _keyed_observer("keyed-lc-4")
        active = f'{defs.STREAMING_SESSIONS_ACTIVE}{{cadence_ms="560",model_name="keyed-lc-4"}}'
        base = _gauge(active)
        observer.session_finished(session_key="never-opened", reason="completed")
        assert _gauge(active) == base
        observer.session_opened(session_key="req-d", cadence_ms="560")
        observer.session_finished(session_key="req-d", reason="completed")
        observer.session_finished(session_key="req-d", reason="error")
        assert _gauge(active) == base

    # @spec PORT-OBS-003
    def test_session_key_is_never_a_label(self) -> None:
        observer = _keyed_observer("keyed-lc-5")
        observer.session_opened(session_key="SECRET-CORRELATION-KEY", cadence_ms="560")
        observer.session_finished(session_key="SECRET-CORRELATION-KEY", reason="completed")
        assert "SECRET-CORRELATION-KEY" not in generate_latest(REGISTRY).decode()

    # @spec PORT-OBS-003, PORT-OBS-005
    def test_finish_clears_remaining_handles_as_error_and_releases_the_record(self) -> None:
        """Divergence defense-in-depth: units still waiting/in-flight at
        terminal finish are cleared as error (backlog returns to zero,
        outcome=error counted) and the record is released — a late
        disposition afterwards is a no-op."""
        observer = _keyed_observer("keyed-lc-6")
        backlog = f'{defs.STREAMING_BACKLOG_CHUNKS}{{cadence_ms="560",model_name="keyed-lc-6"}}'
        err = (
            f"{defs.STREAMING_CHUNKS}_total"
            f'{{cadence_ms="560",chunk_type="regular",model_name="keyed-lc-6",outcome="error"}}'
        )
        base_b, base_e = _gauge(backlog), _gauge(err)

        observer.session_opened(session_key="req-f", cadence_ms="560")
        h1 = observer.unit_ready(session_key="req-f", cadence_ms="560", chunk_type="regular", ready_stamp_s=0.0)
        observer.unit_ready(session_key="req-f", cadence_ms="560", chunk_type="regular", ready_stamp_s=0.1)
        assert _gauge(backlog) == base_b + 2

        observer.session_finished(session_key="req-f", reason="error")
        assert _gauge(backlog) == base_b
        assert _gauge(err) == base_e + 2

        # Late disposition after release: full no-op.
        observer.unit_parked(h1, park_stamp_s=1.0)
        assert _gauge(backlog) == base_b

    # @spec PORT-OBS-003
    def test_clear_all_outstanding_returns_the_number_cleared(self) -> None:
        observer = _keyed_observer("keyed-lc-7")
        observer.session_opened(session_key="req-g", cadence_ms="560")
        h = observer.unit_ready(session_key="req-g", cadence_ms="560", chunk_type="regular", ready_stamp_s=0.0)
        observer.unit_minted(h)
        observer.unit_ready(session_key="req-g", cadence_ms="560", chunk_type="regular", ready_stamp_s=0.1)
        assert observer.clear_all_outstanding("req-g", outcome="aborted") == 2
        assert observer.clear_all_outstanding("req-g", outcome="aborted") == 0

    # @spec PORT-OBS-003
    def test_no_state_outlives_the_session_record(self) -> None:
        """The bounded-state contract: after open -> ready -> park ->
        finish, the observer holds nothing for the session — no
        process-lifetime disposed-set, no orphan waiting/in-flight maps."""
        observer = _keyed_observer("keyed-lc-8")
        observer.session_opened(session_key="req-h", cadence_ms="560")
        h = observer.unit_ready(session_key="req-h", cadence_ms="560", chunk_type="regular", ready_stamp_s=0.0)
        observer.unit_minted(h)
        observer.complete_inflight("req-h")
        observer.unit_parked(h, park_stamp_s=0.2)
        observer.session_finished(session_key="req-h", reason="completed")
        assert observer.session_record_count() == 0


class TestStrictLifecycleTransitions:
    """Review round (2026-07-28): adversarial transitions must not
    corrupt backlog or publish a false terminal reason."""

    # @spec PORT-OBS-003, PORT-OBS-005, PORT-SESS-001
    def test_double_ready_double_mint_cannot_corrupt_backlog(self) -> None:
        """Reviewer repro (F1): two readies + two mints used to orphan
        the first in-flight handle — terminal cleanup released the
        record with backlog stuck at 1. A mint while another unit is in
        flight is an illegal transition and must no-op, leaving the
        second unit waiting (PORT-SESS-001's one-in-flight)."""
        observer = _keyed_observer("strict-lc-1")
        backlog = f'{defs.STREAMING_BACKLOG_CHUNKS}{{cadence_ms="560",model_name="strict-lc-1"}}'
        base = _gauge(backlog)

        observer.session_opened(session_key="req-m1", cadence_ms="560")
        h1 = observer.unit_ready(session_key="req-m1", cadence_ms="560", chunk_type="regular", ready_stamp_s=0.0)
        h2 = observer.unit_ready(session_key="req-m1", cadence_ms="560", chunk_type="regular", ready_stamp_s=0.1)
        observer.unit_minted(h1)
        observer.unit_minted(h2)  # illegal while h1 in flight: no-op
        assert observer.complete_inflight("req-m1") is h1
        observer.unit_parked(h1, park_stamp_s=0.2)
        observer.unit_minted(h2)  # now legal
        assert observer.complete_inflight("req-m1") is h2
        observer.unit_parked(h2, park_stamp_s=0.3)
        observer.session_finished(session_key="req-m1", reason="completed")

        assert _gauge(backlog) == base
        assert observer.session_record_count() == 0

    # @spec PORT-OBS-003, PORT-SESS-001
    def test_duplicate_and_foreign_mints_are_no_ops(self) -> None:
        observer = _keyed_observer("strict-lc-2")
        observer.session_opened(session_key="req-m2", cadence_ms="560")
        h = observer.unit_ready(session_key="req-m2", cadence_ms="560", chunk_type="regular", ready_stamp_s=0.0)
        observer.unit_minted(h)
        observer.unit_minted(h)  # duplicate: idempotent
        assert observer.complete_inflight("req-m2") is h

        # A foreign handle (never made ready for this session) must not
        # displace the in-flight unit.
        from vllm_omni.metrics.streaming_transport import ChunkReadyHandle

        foreign = ChunkReadyHandle(session_key="req-m2", cadence_ms="560", chunk_type="regular", ready_stamp_s=9.9)
        observer.unit_minted(foreign)
        assert observer.complete_inflight("req-m2") is h
        observer.unit_parked(h, park_stamp_s=0.2)
        observer.session_finished(session_key="req-m2", reason="completed")
        assert observer.session_record_count() == 0

    # @spec PORT-OBS-003, PORT-OBS-006
    def test_completed_finish_with_outstanding_normalizes_to_error(self) -> None:
        """Reviewer F4: when the divergence defense fires, the effective
        reason is error — never the caller's completed alongside error
        chunk outcomes."""
        observer = _keyed_observer("strict-lc-3")
        completed = (
            f'{defs.STREAMING_SESSIONS_FINISHED}_total{{cadence_ms="560",model_name="strict-lc-3",reason="completed"}}'
        )
        errored = (
            f'{defs.STREAMING_SESSIONS_FINISHED}_total{{cadence_ms="560",model_name="strict-lc-3",reason="error"}}'
        )
        base_c, base_e = _gauge(completed), _gauge(errored)
        observer.session_opened(session_key="req-m3", cadence_ms="560")
        observer.unit_ready(session_key="req-m3", cadence_ms="560", chunk_type="regular", ready_stamp_s=0.0)
        observer.session_finished(session_key="req-m3", reason="completed")
        assert _gauge(completed) == base_c
        assert _gauge(errored) == base_e + 1

    # @spec PORT-OBS-003
    def test_ready_before_open_is_ignored_loudly(self, caplog: pytest.LogCaptureFixture) -> None:
        """Reviewer F6 / PR #106 wording: open-at-construction is
        authoritative; a ready without an open creates no record, moves
        no backlog, and warns — it never manufactures implicit session
        state that could shadow the correlation invariant."""
        observer = _keyed_observer("strict-lc-4")
        backlog = f'{defs.STREAMING_BACKLOG_CHUNKS}{{cadence_ms="560",model_name="strict-lc-4"}}'
        base = _gauge(backlog)
        with caplog.at_level("WARNING", logger="vllm_omni.metrics.streaming"):
            h = observer.unit_ready(
                session_key="never-opened", cadence_ms="560", chunk_type="regular", ready_stamp_s=0.0
            )
        assert any("un-opened" in rec.message for rec in caplog.records)
        assert _gauge(backlog) == base
        assert observer.session_record_count() == 0
        observer.unit_parked(h, park_stamp_s=0.1)  # inert handle: no-op
        assert _gauge(backlog) == base

    # @spec PORT-OBS-003
    def test_unit_ready_requires_an_explicit_session_key(self) -> None:
        """Reviewer F6: the correlation key has no default — keyless
        compatibility exists only on the unobserved path."""
        import inspect

        from vllm_omni.metrics.streaming import PrometheusStreamingObserver

        param = inspect.signature(PrometheusStreamingObserver.unit_ready).parameters["session_key"]
        assert param.default is inspect.Parameter.empty


# @spec PORT-OBS-002, PORT-OBS-003, PORT-OBS-004
class TestServiceTiming:
    def observer(self, monkeypatch: Any, *, enabled: bool = True, stats: bool = True) -> Any:
        from vllm_omni.metrics.streaming import PrometheusStreamingObserver

        monkeypatch.setenv("VLLM_OMNI_SERVICE_TIMING", "1" if enabled else "0")
        return PrometheusStreamingObserver(OmniStreamingMetrics(model_name=_MODEL, log_stats=stats))

    def ready(self, observer: Any, *, kind: str = "regular", stamp: float = 1.0) -> Any:
        return observer.unit_ready(session_key="timing", cadence_ms="160", chunk_type=kind, ready_stamp_s=stamp)

    def test_disabled_and_stats_disabled_have_no_trace(self, monkeypatch: Any) -> None:
        for enabled, stats in ((False, True), (True, False)):
            observer = self.observer(monkeypatch, enabled=enabled, stats=stats)
            observer.session_opened(session_key="timing", cadence_ms="160")
            assert observer.service_timing("timing") is None

    def test_equal_handles_keep_identity_and_one_disposition(self, monkeypatch: Any) -> None:
        observer = self.observer(monkeypatch)
        observer.session_opened(session_key="timing", cadence_ms="160")
        a, b = self.ready(observer), self.ready(observer)
        assert a == b and a is not b
        observer.unit_minted(b)
        observer.unit_parked(b, park_stamp_s=1.2)
        assert observer.complete_inflight("timing") is None
        assert len(observer._sessions["timing"].waiting) == 1
        assert observer._sessions["timing"].waiting[0] is a
        observer.unit_minted(a)
        assert observer.complete_inflight("timing") is a
        observer.unit_parked(a, park_stamp_s=1.3)
        assert observer._sessions["timing"].waiting == []

    def test_complete_raw_trace_export_once_and_decomposition(self, monkeypatch: Any) -> None:
        import json

        from vllm_omni.metrics import streaming as mod

        records = []
        monkeypatch.setattr(mod.logger, "info", lambda fmt, payload: records.append(json.loads(payload)))
        observer = self.observer(monkeypatch)
        observer.session_opened(session_key="timing", cadence_ms="160")
        trace = observer.service_timing("timing")
        for seq, kind, r, e, s, p in (
            (0, "regular", 1.0, 1_000_000_000, 1_020_000_000, 1.3),
            (1, "final_tail", 1.0, 1_180_000_000, 1_310_000_000, 1.4),
        ):
            handle = self.ready(observer, kind=kind, stamp=r)
            observer.unit_minted(handle)
            trace.submitted(handle, seq, seq, kind, e, s)
            observer.unit_parked(handle, park_stamp_s=p)
        assert records == []
        observer.session_finished(session_key="timing", reason="completed")
        observer.session_finished(session_key="timing", reason="completed")
        assert len(records) == 1
        record = records[0]
        assert record["valid"] and record["complete"]
        assert record["capacity"] == 256 and record["schema"] == 1
        assert record["count"] == 2 and record["overflow"] == 0
        a, b = record["units"]
        assert a["r"] == b["r"] == 1.0
        assert a["p"] == 1.3 and b["p"] == 1.4
        assert b["e"] == 1.18 and b["s"] == 1.31
        assert b["p"] - b["r"] == pytest.approx((b["e"] - b["r"]) + (b["s"] - b["e"]) + (b["p"] - b["s"]))
        assert max(b["e"], a["p"]) - b["e"] == pytest.approx(0.12)
        assert observer.session_record_count() == 0

    @pytest.mark.parametrize("fault", ["gap", "duplicate", "ordering", "missing", "cleared"])
    def test_invalid_trace_never_fails_serving(self, monkeypatch: Any, fault: str) -> None:
        def run(mutation: str | None) -> dict[str, Any]:
            observer = self.observer(monkeypatch)
            observer.session_opened(session_key="timing", cadence_ms="160")
            trace = observer.service_timing("timing")
            for seq, kind, r, e, s, p in (
                (0, "regular", 1.0, 1_000_000_000, 1_100_000_000, 1.2),
                (1, "final_tail", 2.0, 2_000_000_000, 2_100_000_000, 2.2),
            ):
                handle = self.ready(observer, kind=kind, stamp=r)
                observer.unit_minted(handle)
                if not (mutation == "missing" and seq == 0):
                    # Changing both authoritative coordinates creates exactly
                    # one logical gap, without a second carrier-modulo fault.
                    identity = 2 if mutation == "gap" and seq == 1 else seq
                    trace.submitted(handle, identity, identity, kind, e, s)
                if mutation == "cleared" and seq == 0:
                    observer.unit_cleared(handle, outcome="error")
                else:
                    park = 1.05 if mutation == "ordering" and seq == 0 else p
                    observer.unit_parked(handle, park_stamp_s=park)
                if mutation == "duplicate" and seq == 0:
                    assert trace.valid, "ordinary first disposition must be valid"
                    observer.unit_parked(handle, park_stamp_s=p)
                    assert not trace.valid, "duplicate disposition must invalidate at the event"
            if mutation == "gap":
                assert not trace.valid, "logical gap must invalidate at submission"
            elif mutation not in ("duplicate", "missing"):
                assert trace.valid, "this mutation must be detected by terminal validation"
            result: dict[str, Any] = trace.finish("completed")
            observer.session_finished(session_key="timing", reason="completed")
            assert observer.session_record_count() == 0
            return result

        baseline = run(None)
        assert baseline["valid"] and baseline["complete"]
        assert baseline["count"] == baseline["audio_count"] == 2
        assert [row["kind"] for row in baseline["units"]] == ["regular", "final_tail"]
        result = run(fault)
        assert not result["valid"], f"single {fault} mutation must invalidate a complete baseline"
        assert result["count"] == result["audio_count"] == 2
        assert result["reason"] == baseline["reason"] == "completed"
        # Gap/duplicate/order fail validity while preserving audio completeness;
        # missing/cleared fail specifically because required timing is absent.
        assert result["complete"] is (fault not in ("missing", "cleared"))
        if fault == "gap":
            assert [row["logical_sequence"] for row in result["units"]] == [0, 2]
            assert all(row["carrier_sequence"] == row["logical_sequence"] for row in result["units"])
        elif fault == "duplicate":
            assert result["units"] == baseline["units"], "duplicate must not overwrite any raw sample"
        elif fault == "ordering":
            assert result["units"][0]["p"] == 1.05  # no raw clamping
            assert result["units"][0]["s"] == 1.1
            assert result["units"][1] == baseline["units"][1]
        elif fault == "missing":
            assert result["units"][0]["s"] is None
            assert result["units"][0]["disposition"] == "parked"
        else:
            assert result["units"][0]["p"] is None
            assert result["units"][0]["disposition"] == "error"

    def test_fixed_capacity_even_after_overflow(self, monkeypatch: Any) -> None:
        observer = self.observer(monkeypatch)
        observer.session_opened(session_key="timing", cadence_ms="160")
        trace = observer.service_timing("timing")
        slots = trace.slots
        for seq in range(300):
            h = self.ready(observer)
            observer.unit_minted(h)
            trace.submitted(h, seq, seq, "regular", 1_000_000_000, 1_100_000_000)
            observer.unit_parked(h, park_stamp_s=1.2)
        result = trace.finish("completed")
        assert trace.slots is slots and len(slots) == 256
        assert len(result["units"]) == 256
        assert result["overflow"] == 44 and not result["valid"]

    def test_controls_are_typed_partial_and_consume_capacity(self, monkeypatch: Any) -> None:
        observer = self.observer(monkeypatch)
        observer.session_opened(session_key="timing", cadence_ms="160")
        trace = observer.service_timing("timing")
        trace.submitted(None, 0, None, "forced_eou", None, 1_000_000_000)
        h = self.ready(observer, kind="final_tail")
        observer.unit_minted(h)
        trace.submitted(h, 1, 1, "final_tail", 1_000_000_000, 1_100_000_000)
        observer.unit_parked(h, park_stamp_s=1.2)
        result = trace.finish("completed")
        control = result["units"][0]
        assert control["kind"] == "forced_eou"
        assert control["r"] is control["e"] is control["p"] is None
        assert result["complete"] and result["valid"]
        assert result["completion_scope"] == "ordinary_audio"

    def test_export_failure_cannot_prevent_cleanup(self, monkeypatch: Any) -> None:
        from vllm_omni.metrics import streaming as mod

        def broken(*args: Any, **kwargs: Any) -> None:
            raise RuntimeError("logger failed")

        observer = self.observer(monkeypatch)
        observer.session_opened(session_key="timing", cadence_ms="160")
        self.ready(observer)
        monkeypatch.setattr(mod.logger, "info", broken)
        monkeypatch.setattr(mod.logger, "exception", broken)
        observer.session_finished(session_key="timing", reason="error")
        assert observer.session_record_count() == 0


# @spec PORT-OBS-002, PORT-OBS-003
@pytest.mark.parametrize("stage", ["ready", "disposed", "finish"])
def test_service_timing_observer_failure_preserves_cleanup(monkeypatch: Any, stage: str) -> None:
    from vllm_omni.metrics.streaming import PrometheusStreamingObserver

    monkeypatch.setenv("VLLM_OMNI_SERVICE_TIMING", "1")
    observer = PrometheusStreamingObserver(OmniStreamingMetrics(model_name=_MODEL))
    observer.session_opened(session_key="failure", cadence_ms="160")
    trace = observer.service_timing("failure")
    assert trace is not None

    def broken(*args: Any, **kwargs: Any) -> None:
        raise RuntimeError("trace failed")

    monkeypatch.setattr(trace, stage, broken)
    handle = observer.unit_ready(session_key="failure", cadence_ms="160", chunk_type="final_tail", ready_stamp_s=1.0)
    observer.unit_minted(handle)
    observer.unit_parked(handle, park_stamp_s=1.2)
    assert observer.complete_inflight("failure") is None
    observer.session_finished(session_key="failure", reason="completed")
    assert observer.session_record_count() == 0


# @spec PORT-OBS-003, PORT-OBS-004, PORT-STATE-026
@pytest.mark.parametrize("fault", ["overlap", "carrier"])
def test_service_timing_rejects_cross_unit_identity_and_order(monkeypatch: Any, fault: str) -> None:
    helper = TestServiceTiming()
    observer = helper.observer(monkeypatch)
    observer.session_opened(session_key="timing", cadence_ms="160")
    trace = observer.service_timing("timing")
    for seq, kind, s, p in ((0, "regular", 1_000_000_000, 1.5), (1, "final_tail", 1_400_000_000, 1.6)):
        h = helper.ready(observer, kind=kind)
        observer.unit_minted(h)
        trace.submitted(h, seq, 9 if fault == "carrier" else seq, kind, 1_000_000_000, s)
        observer.unit_parked(h, park_stamp_s=p if fault == "overlap" else s / 1e9 + 0.01)
    result = trace.finish("completed")
    assert result["complete"]
    assert not result["valid"]


# @spec PORT-OBS-002, PORT-OBS-003
@pytest.mark.parametrize("count", [256, 257])
def test_service_timing_exact_capacity_boundary(monkeypatch: Any, count: int) -> None:
    helper = TestServiceTiming()
    observer = helper.observer(monkeypatch)
    observer.session_opened(session_key="timing", cadence_ms="160")
    trace = observer.service_timing("timing")
    slots = trace.slots
    for seq in range(count):
        kind = "final_tail" if seq == count - 1 else "regular"
        handle = helper.ready(observer, kind=kind, stamp=float(seq + 1))
        observer.unit_minted(handle)
        trace.submitted(handle, seq, seq, kind, (seq + 1) * 10**9, (seq + 1) * 10**9)
        observer.unit_parked(handle, park_stamp_s=seq + 1.1)
    record = trace.finish("completed")
    assert trace.slots is slots and len(slots) == 256
    assert record["count"] == 256 and record["overflow"] == count - 256
    assert record["complete"] is (count == 256)
    assert record["valid"] is (count == 256)


# @spec PORT-OBS-002, PORT-OBS-003
def test_service_timing_all_log_methods_and_metric_failure_allow_cleanup(monkeypatch: Any) -> None:
    from vllm_omni.metrics import streaming as mod

    helper = TestServiceTiming()
    observer = helper.observer(monkeypatch)
    observer.session_opened(session_key="timing", cadence_ms="160")
    helper.ready(observer)

    def broken(*args: Any, **kwargs: Any) -> None:
        raise RuntimeError("sink failed")

    for method in ("info", "warning", "exception"):
        monkeypatch.setattr(mod.logger, method, broken)
    monkeypatch.setattr(observer.metrics, "observe_session_finished", broken)
    observer.session_finished(session_key="timing", reason="completed")
    assert observer.session_record_count() == 0


# @spec PORT-OBS-003, PORT-OBS-004
def test_service_timing_total_and_audio_counts_exclude_controls(monkeypatch: Any) -> None:
    helper = TestServiceTiming()
    observer = helper.observer(monkeypatch)
    observer.session_opened(session_key="timing", cadence_ms="160")
    trace = observer.service_timing("timing")
    for seq, kind in enumerate(("regular", "forced_eou", "final_tail")):
        if kind == "forced_eou":
            trace.submitted(None, seq, None, kind, None, (seq + 1) * 10**9)
        else:
            handle = helper.ready(observer, kind=kind, stamp=float(seq + 1))
            observer.unit_minted(handle)
            trace.submitted(handle, seq, seq, kind, (seq + 1) * 10**9, (seq + 1) * 10**9)
            observer.unit_parked(handle, park_stamp_s=seq + 1.1)
    record = trace.finish("completed")
    assert record["valid"] and record["complete"]
    assert record["count"] == len(record["units"]) == 3
    assert record["audio_count"] == 2
    assert record["audio_count"] == sum(row["kind"] in ("regular", "final_tail") for row in record["units"])
    assert record["units"][2]["predecessor_attribution"] == "control_unobserved"


# @spec PORT-OBS-002, PORT-OBS-003
@pytest.mark.parametrize(
    "field,value",
    [
        ("logical_sequence", None),
        ("logical_sequence", -1),
        ("carrier_sequence", 0),
        ("s_ns", None),
        ("r", 1.0),
        ("e_ns", 1_000_000_000),
        ("p", 1.1),
        ("disposition", "parked"),
        ("handle", object()),
        ("kind", "unknown"),
    ],
)
def test_service_timing_rejects_malformed_control_shape(monkeypatch: Any, field: str, value: Any) -> None:
    helper = TestServiceTiming()
    observer = helper.observer(monkeypatch)
    observer.session_opened(session_key="timing", cadence_ms="160")
    trace = observer.service_timing("timing")
    trace.submitted(None, 0, None, "forced_eou", None, 1_000_000_000)
    handle = helper.ready(observer, kind="final_tail", stamp=2.0)
    observer.unit_minted(handle)
    trace.submitted(handle, 1, 1, "final_tail", 2_000_000_000, 2_000_000_000)
    observer.unit_parked(handle, park_stamp_s=2.1)
    # A broken collaborator may corrupt a partially filled diagnostic record;
    # terminal validation must reject it without touching serving lifecycle.
    setattr(trace.slots[0], field, value)
    record = trace.finish("completed")
    assert not record["valid"]
    assert record["count"] == 2
    assert record["audio_count"] == 1
    observer.session_finished(session_key="timing", reason="completed")
    assert observer.session_record_count() == 0
