# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The omni output envelope and the scheduler-side batch-stat forward.

A27 topology cascade (lead-approved amendment 6): the omni schedulers
must construct ``OmniEngineCoreOutputs`` — the owning envelope type —
not vanilla ``EngineCoreOutputs``. The vanilla type is a slotted msgspec
struct that cannot carry ``streaming_chunk_batch_stats`` at all: on the
A27 GPU round the field could never cross the proc boundary, and
enabling the scheduler gate alone would have raised ``AttributeError``.

These tests drive the scheduler's REAL forwarding seam (the method
``update_from_output`` calls) on scheduler-constructed envelopes, then
round-trip them through the same msgspec encoder/decoder pair the
engine-core boundary uses — not only hand-constructed objects.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from vllm_omni.engine import OmniEngineCoreOutputs

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _round_trip(outputs: Any) -> Any:
    """Encode/decode with the engine-core boundary's own machinery."""
    from vllm.v1.serial_utils import MsgpackDecoder, MsgpackEncoder

    encoder = MsgpackEncoder()
    decoder = MsgpackDecoder(OmniEngineCoreOutputs)
    return decoder.decode(encoder.encode(outputs))


def _ar_scheduler(*, log_stats: bool) -> Any:
    from vllm_omni.core.sched.omni_ar_scheduler import OmniARScheduler

    sched: Any = OmniARScheduler.__new__(OmniARScheduler)
    sched.log_stats = log_stats
    return sched


# @spec PORT-OBS-008, PORT-OBS-009
def test_ar_scheduler_module_binds_the_omni_envelope() -> None:
    """Every construction site in the AR scheduler builds the omni
    envelope: the module-level name is bound to the omni type, so no
    site can silently construct a vanilla (field-less) envelope."""
    from vllm_omni.core.sched import omni_ar_scheduler as m

    assert m.EngineCoreOutputs is OmniEngineCoreOutputs


# @spec PORT-OBS-008, PORT-OBS-009
def test_generation_scheduler_module_binds_the_omni_envelope() -> None:
    """Amendment 6: the generation scheduler already emits
    ``OmniEngineCoreOutput`` elements; its enclosing envelope must be
    the omni type too, not a perpetuated invariant split."""
    from vllm_omni.core.sched import omni_generation_scheduler as m

    assert m.EngineCoreOutputs is OmniEngineCoreOutputs


# @spec PORT-OBS-008, PORT-OBS-009
def test_enabled_nonempty_stats_forward_and_round_trip() -> None:
    sched = _ar_scheduler(log_stats=True)
    engine_core_outputs: dict[int, Any] = {}
    runner_output = SimpleNamespace(streaming_chunk_batch_stats=[("560", 3), ("1120", 1)])

    sched._forward_streaming_batch_stats(engine_core_outputs, runner_output)

    assert list(engine_core_outputs) == [0]
    eco = engine_core_outputs[0]
    assert isinstance(eco, OmniEngineCoreOutputs)

    decoded = _round_trip(eco)
    got = [tuple(pair) for pair in decoded.streaming_chunk_batch_stats]
    assert got == [("560", 3), ("1120", 1)]


# @spec PORT-OBS-008, PORT-OBS-009
def test_enabled_empty_stats_forward_and_round_trip() -> None:
    """A transaction that executed no nonempty CHUNK bucket drains
    ``[]`` — forwarded as-is (the sink's defensive skip handles it),
    and it survives the boundary."""
    sched = _ar_scheduler(log_stats=True)
    engine_core_outputs: dict[int, Any] = {0: OmniEngineCoreOutputs()}
    runner_output = SimpleNamespace(streaming_chunk_batch_stats=[])

    sched._forward_streaming_batch_stats(engine_core_outputs, runner_output)

    decoded = _round_trip(engine_core_outputs[0])
    assert decoded.streaming_chunk_batch_stats == []


# @spec PORT-OBS-008, PORT-OBS-009
def test_vanilla_runner_output_on_an_idle_step_is_forwarded_as_none() -> None:
    """GPU-round regression pin (2026-07-28, second round): idle
    scheduler steps carry vLLM's vanilla ``ModelRunnerOutput`` (the
    shared empty singleton), which has no
    ``streaming_chunk_batch_stats`` attribute at all. Reading it
    unconditionally raised ``AttributeError`` inside the engine-core
    busy loop and KILLED THE ENGINE on the first idle step after a
    generation. The forward must treat an attribute-less runner output
    exactly like "nothing drained"."""
    from vllm.v1.outputs import ModelRunnerOutput

    sched = _ar_scheduler(log_stats=True)
    engine_core_outputs: dict[int, Any] = {0: OmniEngineCoreOutputs()}
    vanilla = ModelRunnerOutput(
        req_ids=[],
        req_id_to_index={},
        sampled_token_ids=[],
        logprobs=None,
        prompt_logprobs_dict={},
        pooler_output=[],
    )
    assert not hasattr(vanilla, "streaming_chunk_batch_stats")

    sched._forward_streaming_batch_stats(engine_core_outputs, vanilla)

    assert engine_core_outputs[0].streaming_chunk_batch_stats is None


# @spec PORT-OBS-002, PORT-OBS-009
def test_disabled_stats_forward_nothing_and_synthesize_nothing() -> None:
    """Statistics disabled: no ``streaming_chunk_batch_stats`` payload
    ever leaves the scheduler, and no otherwise-unneeded envelope is
    synthesized for a step that would send nothing else."""
    sched = _ar_scheduler(log_stats=False)
    engine_core_outputs: dict[int, Any] = {}
    runner_output = SimpleNamespace(streaming_chunk_batch_stats=[("560", 3)])

    sched._forward_streaming_batch_stats(engine_core_outputs, runner_output)

    assert engine_core_outputs == {}

    # With an envelope already present, the field stays None.
    present: dict[int, Any] = {0: OmniEngineCoreOutputs()}
    sched._forward_streaming_batch_stats(present, runner_output)
    assert present[0].streaming_chunk_batch_stats is None
