# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""CPU tests for the bounded, Prometheus-free engine timing companion.

Load the neutral module directly, as the import-graph tests do, so these tests
also run on hosts without the optional vLLM/Torch package dependencies.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from collections import deque
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.fixture
def transport(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    path = Path(__file__).resolve().parents[2] / "vllm_omni/metrics/streaming_transport.py"
    spec = importlib.util.spec_from_file_location("_engine_timing_test_transport", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    return module


def _identity(transport, sequence=0, kind="regular", generation=7):
    trace = transport.ServiceTimingTrace("session-1")
    unit = SimpleNamespace(
        logical_sequence=sequence,
        carrier_sequence=None if kind == "forced_eou" else sequence % (2**24),
        kind=kind,
    )
    raw = trace.input_identity(None if kind == "flush" else unit, engine_epoch="engine-A", lease_generation=generation)
    return trace, transport.engine_service_timing_identity({"meta": {"service_timing": raw}})


def test_identity_keeps_carrier_wrap_and_generations_distinct(transport):
    trace, identity = _identity(transport, 2**24 + 3)
    assert identity == {
        "session": "session-1",
        "engine_epoch": "engine-A",
        "lease_generation": 7,
        "logical_sequence": 2**24 + 3,
        "carrier_sequence": 3,
        "kind": "regular",
    }
    assert trace.finish("aborted")["lease_generation"] == 7
    assert trace.finish("aborted")["engine_epoch"] == "engine-A"


@pytest.mark.parametrize("kind", ["regular", "final_tail", "forced_eou", "flush"])
def test_all_explicit_unit_kinds_round_trip(transport, kind):
    _, identity = _identity(transport, kind=kind)
    assert identity["kind"] == kind


def test_generation_change_invalidates_trace_without_relabelling_previous_units(transport):
    trace, identity = _identity(transport)
    unit = SimpleNamespace(logical_sequence=1, carrier_sequence=1, kind="regular")
    with pytest.raises(ValueError, match="generation changed"):
        trace.input_identity(unit, engine_epoch="engine-A", lease_generation=8)
    assert trace.valid is False and trace.lease_generation == 7
    engine_trace = transport.EngineServiceTimingTrace("internal-1", identity)
    _, changed = _identity(transport, generation=8)
    engine_trace.record(SimpleNamespace(), "update_arrival", identity=changed)
    assert engine_trace.finish("aborted")["valid"] is False


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("logical_sequence", True),
        ("lease_generation", -1),
        ("engine_epoch", 1),
        ("carrier_sequence", 42),
        ("kind", "unknown"),
        ("extra", "not allowed"),
    ],
)
def test_bad_identity_is_not_inferred(transport, key, value):
    _, identity = _identity(transport)
    identity[key] = value
    assert transport.engine_service_timing_identity({"meta": {"service_timing": json.dumps(identity)}}) is None


@pytest.mark.parametrize(
    "info",
    [{}, {"meta": {}}, {"meta": {"service_timing": "{"}}, {"meta": {"service_timing": '{"session":"s"}'}}],
    ids=["missing-payload", "missing-identity", "malformed-present", "schema-invalid-present"],
)
def test_existing_trace_invalidates_missing_or_malformed_update_identity(transport, info):
    _, identity = _identity(transport)
    trace = transport.EngineServiceTimingTrace("internal-1", identity)
    assert trace.read_input_identity(info) is None
    assert trace.finish("completed")["valid"] is False
    assert trace.identity == identity
    assert trace.events == [] and trace.overflow == 0


def test_prompt_metadata_json_round_trip_preserves_explicit_identity(transport):
    trace = transport.ServiceTimingTrace("session-1")
    unit = SimpleNamespace(logical_sequence=2, carrier_sequence=2, kind="final_tail")
    prompt = {
        "prompt_token_ids": [123],
        "additional_information": {
            "meta": {"service_timing": trace.input_identity(unit, engine_epoch="engine-A", lease_generation=7)}
        },
    }
    decoded = json.loads(json.dumps(prompt))
    identity = transport.engine_service_timing_identity(decoded["additional_information"])
    engine_trace = transport.EngineServiceTimingTrace("internal-1", identity)
    assert engine_trace.read_input_identity(decoded["additional_information"]) == identity
    assert engine_trace.valid is True
    assert identity["logical_sequence"] == 2 and identity["lease_generation"] == 7
    assert decoded["prompt_token_ids"] == [123]


def test_flush_and_input_end_preserve_validity_without_audio_sequence(transport):
    _, identity = _identity(transport)
    trace = transport.EngineServiceTimingTrace("internal-1", identity)
    _, flush_identity = _identity(transport, kind="flush")
    info = {"meta": {"service_timing": json.dumps(flush_identity)}}
    trace.identity = trace.read_input_identity(info)
    request = SimpleNamespace(_omni_segment_generation=2)
    trace.record(request, "legal_park", timestamp=1.0)
    trace.record_input_end(request, timestamp=2.0)
    assert trace.valid is True
    assert [event["identity"]["kind"] for event in trace.events] == ["flush", "input_end"]
    assert all(event["identity"]["logical_sequence"] is None for event in trace.events)
    assert all(event["identity"]["carrier_sequence"] is None for event in trace.events)
    assert trace.identity == flush_identity


def test_park_snapshot_survives_update_and_reused_session_generation(transport):
    _, predecessor = _identity(transport, 0)
    _, successor = _identity(transport, 1)
    request = SimpleNamespace(_omni_segment_generation=11, streaming_queue=deque([object()]))
    trace = transport.EngineServiceTimingTrace("internal-1", predecessor)
    trace.record(request, "update_arrival", timestamp=2.0, identity=successor)
    trace.record(request, "legal_park", timestamp=3.0)
    trace.identity = successor
    request._omni_segment_generation = 12
    trace.record(request, "scheduled", timestamp=4.0)
    assert trace.events[0]["identity"] == successor
    assert trace.events[1]["identity"] == predecessor
    assert trace.events[1]["segment_generation"] == 11
    assert trace.events[2]["identity"] == successor
    assert trace.events[2]["segment_generation"] == 12
    assert trace.events[2]["timestamp"] == 4.0
    _, reused = _identity(transport, 0, generation=8)
    assert reused != predecessor


def test_bounded_trace_reports_loss_and_batch_context(transport):
    _, identity = _identity(transport)
    trace = transport.EngineServiceTimingTrace("internal-1", identity)
    trace.capacity = 2
    request = SimpleNamespace()
    context = {"num_scheduled_tokens": {"internal-1": 1, "other": 3}}
    trace.record(request, "schedule_batch", timestamp=1.0, schedule_context=context)
    trace.record(request, "legal_park", timestamp=2.0)
    trace.record(request, "scheduled", timestamp=3.0)
    result = trace.finish("FINISHED_ABORTED")
    assert result["count"] == 2 and result["overflow"] == 1
    assert result["valid"] is False and result["end"] is True
    assert result["events"][0]["schedule_context"] == context
    assert result["process_id"] > 0 and result["clock_host"]
    assert result["schedule_semantics"].endswith("not_gpu_start")


def test_disabled_and_broken_observations_cannot_break_cleanup(transport, monkeypatch):
    request = SimpleNamespace(status="FINISHED_ABORTED")
    transport.record_engine_service_timing(request, "legal_park")
    assert vars(request) == {"status": "FINISHED_ABORTED"}
    _, identity = _identity(transport)
    trace = transport.EngineServiceTimingTrace("internal-1", identity)
    request._omni_service_timing = trace

    def fail(*args, **kwargs):
        raise RuntimeError("diagnostic failure")

    monkeypatch.setattr(trace, "record", fail)
    transport.record_engine_service_timing(request, "legal_park")
    assert trace.valid is False
    monkeypatch.setattr(transport._logger, "info", fail)
    transport.finish_engine_service_timing(request)
    assert request._omni_service_timing is None
    transport.finish_engine_service_timing(request)


def test_trace_is_exported_once_on_cleanup(transport, monkeypatch):
    _, identity = _identity(transport)
    trace = transport.EngineServiceTimingTrace("internal-1", identity)
    request = SimpleNamespace(status="FINISHED_STOPPED", _omni_service_timing=trace)
    records = []
    monkeypatch.setattr(transport._logger, "info", lambda fmt, payload: records.append(json.loads(payload)))
    transport.finish_engine_service_timing(request)
    transport.finish_engine_service_timing(request)
    assert len(records) == 1
    assert records[0]["end"] is True
