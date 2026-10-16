# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""The NVTX phase contract: closed names, free when off, safe when on."""

from __future__ import annotations

import pytest

from vllm_omni.model_executor.models.nemotron_asr import profiling

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


# @spec PORT-OBS-001
def test_phase_names_are_a_closed_declared_set() -> None:
    """Digests compare across runs only if span names are stable, so the
    set lives in one place and every name is `port.`-prefixed."""
    assert profiling.PHASES == (
        "port.ingest",
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
def test_startup_round_labels_are_exact_and_separate() -> None:
    expected = {
        f"geometry-{geometry}-single-ordinary-repeat-{repeat}": (
            f"port.startup.measurement.geometry-{geometry}-single-ordinary-repeat-{repeat}"
        )
        for geometry in (0, 1)
        for repeat in range(4)
    }
    assert profiling.STARTUP_ROUND_NAMES == expected
    assert not set(expected.values()).intersection(profiling._RANGES)


# @spec PORT-PERF-007
def test_startup_round_ids_support_zero_and_out_of_order_close(monkeypatch: pytest.MonkeyPatch) -> None:
    events: list[tuple[str, object]] = []
    handles = iter((0, 17))

    def start(name: str) -> int:
        events.append(("start", name))
        return next(handles)

    monkeypatch.setattr(profiling, "_ENABLED", True)
    monkeypatch.setattr(profiling.torch.cuda.nvtx, "range_start", start)
    monkeypatch.setattr(profiling.torch.cuda.nvtx, "range_end", lambda handle: events.append(("end", handle)))
    first = profiling.start_startup_round("geometry-0-single-ordinary-repeat-0")
    second = profiling.start_startup_round("geometry-1-single-ordinary-repeat-3")
    assert (first, second) == (0, 17)
    profiling.end_startup_round(first)
    profiling.end_startup_round(second)
    profiling.end_startup_round(None)
    assert events == [
        ("start", "port.startup.measurement.geometry-0-single-ordinary-repeat-0"),
        ("start", "port.startup.measurement.geometry-1-single-ordinary-repeat-3"),
        ("end", 0),
        ("end", 17),
    ]


# @spec PORT-PERF-007
@pytest.mark.parametrize("enabled", (False, True))
def test_startup_round_disabled_or_unregistered_is_inert(monkeypatch: pytest.MonkeyPatch, enabled: bool) -> None:
    monkeypatch.setattr(profiling, "_ENABLED", enabled)

    def forbidden(*args: object) -> None:
        pytest.fail("inert startup correlation called NVTX")

    monkeypatch.setattr(profiling.torch.cuda.nvtx, "range_start", forbidden)
    monkeypatch.setattr(profiling.torch.cuda.nvtx, "range_end", forbidden)
    names = [
        "geometry-0-single-forced_eou_then_chunk-repeat-1",
        "geometry-0-single-final_tail_then_flush-repeat-1",
        "geometry-0-bulk-ordinary-repeat-1",
        "geometry-2-single-ordinary-repeat-1",
        "geometry-0-single-ordinary-repeat-4",
    ]
    if not enabled:
        names.append("geometry-0-single-ordinary-repeat-1")
    for name in names:
        assert profiling.start_startup_round(name) is None
    profiling.end_startup_round(None)


# @spec PORT-PERF-007
def test_startup_round_disabled_does_not_format_or_lookup(monkeypatch: pytest.MonkeyPatch) -> None:
    class UnusableIdentity:
        def __str__(self) -> str:
            raise AssertionError("disabled correlation formatted identity")

        def __hash__(self) -> int:
            raise AssertionError("disabled correlation looked up identity")

    monkeypatch.setattr(profiling, "_ENABLED", False)
    assert profiling.start_startup_round(UnusableIdentity()) is None
