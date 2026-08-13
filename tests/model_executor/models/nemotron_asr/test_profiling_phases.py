# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The NVTX phase contract: closed names, free when off, safe when on."""

from __future__ import annotations

from contextlib import AbstractContextManager
from types import SimpleNamespace

import pytest
import torch

from vllm_omni.model_executor.models.nemotron_asr import nemotron_asr, profiling

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


# @spec PORT-OBS-001
def test_phase_names_are_a_closed_declared_set() -> None:
    """Digests compare across runs only if span names are stable, so the
    set lives in one place and every name is `port.`-prefixed."""
    assert profiling.PHASES == (
        "port.ingest",
        "port.carrier_pack",
        "port.carrier_h2d",
        "port.multimodal_merge",
        "port.featurize",
        "port.encode",
        "port.decode",
        "port.scatter",
        "port.park",
    )
    assert all(name.startswith("port.") for name in profiling.PHASES)


# @spec PORT-OBS-001
def test_ranges_are_preallocated_and_shared() -> None:
    """The hot path runs ~12,500 transactions/s at target; a phase must
    not allocate per entry. The same object comes back every call."""
    for name in profiling.PHASES:
        assert profiling.phase(name) is profiling.phase(name)


# @spec PORT-OBS-001
def test_an_undeclared_name_is_inert_rather_than_fatal() -> None:
    """A typo must not take down serving; the missing span shows up in
    the digest, which is where a profiling mistake belongs."""
    with profiling.phase("port.not_a_phase"):
        pass


# @spec PORT-OBS-001
def test_a_phase_body_that_raises_still_leaves_the_range() -> None:
    """Unbalanced push/pop corrupts every later range in the capture."""
    with pytest.raises(ValueError):
        with profiling.phase("port.encode"):
            raise ValueError("boom")
    with profiling.phase("port.encode"):
        pass


# @spec PORT-OBS-001
def test_disabled_by_default_so_serving_pays_nothing() -> None:
    """Profiling is a capture-harness decision, never a serving default."""
    assert profiling.enabled() is False
    assert type(profiling.phase("port.encode")).__name__ == "_NullPhase"


# @spec PORT-PERF-007
def test_carrier_ranges_cover_the_production_pack_transfer_merge_join(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The actual carrier produced by packing must reach the merge seam."""
    events: list[tuple[str, str]] = []

    class RecordingPhase(AbstractContextManager[None]):
        def __init__(self, name: str) -> None:
            self.name = name

        def __enter__(self) -> None:
            events.append(("enter", self.name))

        def __exit__(self, *exc_info: object) -> None:
            del exc_info
            events.append(("exit", self.name))

    monkeypatch.setattr(
        nemotron_asr,
        "_CARRIER_PACK_PHASE",
        RecordingPhase("port.carrier_pack"),
    )
    monkeypatch.setattr(
        nemotron_asr,
        "_CARRIER_H2D_PHASE",
        RecordingPhase("port.carrier_h2d"),
    )
    monkeypatch.setattr(
        nemotron_asr,
        "_MULTIMODAL_MERGE_PHASE",
        RecordingPhase("port.multimodal_merge"),
    )
    parameter = torch.nn.Parameter(torch.empty(0))
    model = SimpleNamespace(
        config=SimpleNamespace(hidden_size=16),
        parameters=lambda: iter((parameter,)),
    )
    envelope = torch.arange(9, dtype=torch.float32)

    rows = nemotron_asr.NemotronASRForRNNT.embed_multimodal(
        model,
        audio=[envelope],
    )
    merged = nemotron_asr.NemotronASRForRNNT.embed_input_ids(
        model,
        torch.tensor([1, 2]),
        rows,
        is_multimodal=torch.tensor([True, False]),
    )

    assert events == [
        ("enter", "port.carrier_pack"),
        ("exit", "port.carrier_pack"),
        ("enter", "port.carrier_h2d"),
        ("exit", "port.carrier_h2d"),
        ("enter", "port.multimodal_merge"),
        ("exit", "port.multimodal_merge"),
    ]
    torch.testing.assert_close(merged[0, :9], envelope)
    assert torch.count_nonzero(merged[1]) == 0
