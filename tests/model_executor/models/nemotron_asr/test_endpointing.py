# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests-first contract for bounded cache-aware RNN-T endpointing."""

from __future__ import annotations

import copy
import importlib
import inspect
from typing import Any, NoReturn

import pytest

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _fail(message: str) -> NoReturn:
    pytest.fail(message, pytrace=False)
    raise AssertionError(message)


def _module() -> Any:
    try:
        return importlib.import_module(
            "vllm_omni.model_executor.models.nemotron_asr.endpointing"
        )
    except ModuleNotFoundError:
        _fail("PORT-SEG-001 missing bounded cache-aware endpointing")


def _policy(
    *,
    stop_history_ms: int = 240,
    residue_frames: int = 2,
    capacity: int = 16,
) -> Any:
    return _module().EndpointPolicy.resolve(
        mode="greedy_blank",
        stop_history_ms=stop_history_ms,
        residue_frames=residue_frames,
        frame_stride_ms=80,
        history_capacity_frames=capacity,
    )


def _book(capacity: int = 16) -> Any:
    return _module().EndpointBook.initial(history_capacity_frames=capacity)


def _word_start(label: int) -> bool:
    return label in {10, 20, 30}


def _observe(
    book: Any,
    frames: list[list[int]],
    *,
    policy: Any | None = None,
    final_tail: bool = False,
) -> Any:
    return _module().observe_chunk(
        book,
        frames,
        is_word_start=_word_start,
        policy=policy or _policy(),
        final_tail=final_tail,
    )


def test_reference_policy_derives_threshold_and_active_window() -> None:
    # @spec PORT-SEG-001
    policy = _module().EndpointPolicy.resolve(
        mode="greedy_blank",
        stop_history_ms=800,
        residue_frames=2,
        frame_stride_ms=80,
        history_capacity_frames=12,
    )

    assert policy.mode == "greedy_blank"
    assert policy.stop_history_ms == 800
    assert policy.threshold_frames == 10
    assert policy.residue_frames == 2
    assert policy.active_window_frames == 12
    assert policy.history_capacity_frames == 12

    rounded = _module().EndpointPolicy.resolve(
        mode="greedy_blank",
        stop_history_ms=100,
        residue_frames=2,
        frame_stride_ms=80,
        history_capacity_frames=8,
    )
    assert rounded.threshold_frames == 2
    assert rounded.active_window_frames == 4


@pytest.mark.parametrize(
    ("kwargs", "pattern"),
    [
        ({"mode": "vad"}, "mode"),
        ({"stop_history_ms": 0}, "stop_history"),
        ({"stop_history_ms": -1}, "stop_history"),
        ({"residue_frames": -1}, "residue"),
        ({"frame_stride_ms": 0}, "stride"),
        ({"history_capacity_frames": 4}, "capacity"),
    ],
)
def test_invalid_or_uninstalled_policy_fails_closed(
    kwargs: dict[str, Any], pattern: str
) -> None:
    # @spec PORT-SEG-001 / PORT-RTC-001
    values: dict[str, Any] = {
        "mode": "greedy_blank",
        "stop_history_ms": 240,
        "residue_frames": 2,
        "frame_stride_ms": 80,
        "history_capacity_frames": 16,
    }
    values.update(kwargs)
    with pytest.raises(ValueError, match=pattern):
        _module().EndpointPolicy.resolve(**values)


def test_disabled_policy_is_explicit_and_has_no_active_window() -> None:
    # @spec PORT-SEG-001
    policy = _module().EndpointPolicy.resolve(
        mode="disabled",
        stop_history_ms=None,
        residue_frames=0,
        frame_stride_ms=80,
        history_capacity_frames=16,
    )
    assert policy.mode == "disabled"
    assert policy.threshold_frames == 0
    assert policy.active_window_frames == 0


@pytest.mark.parametrize(
    ("labels", "expected_name"),
    [
        ([], "BLANK"),
        ([10], "WORD_START"),
        ([11], "NON_WORD_START"),
        ([10, 11], "NON_WORD_START"),
        ([11, 20], "WORD_START"),
    ],
)
def test_one_dense_endpoint_symbol_uses_the_final_label_of_each_frame(
    labels: list[int], expected_name: str
) -> None:
    # @spec PORT-DEC-001 / PORT-SEG-002
    symbol = _module().frame_symbol(labels, is_word_start=_word_start)
    assert symbol.name == expected_name


def test_multiple_labels_remain_in_transcript_order_but_use_one_frame_slot() -> None:
    # @spec PORT-DEC-001 / PORT-SEG-002
    transition = _observe(_book(), [[10, 11, 20], [], [21]])

    assert transition.emitted_labels == (10, 11, 20, 21)
    assert tuple(symbol.name for symbol in transition.observed_symbols) == (
        "WORD_START",
        "BLANK",
        "NON_WORD_START",
    )
    assert transition.next_book.history_length == 3


def test_endpoint_ring_is_fixed_width_and_keeps_newest_frame_symbols() -> None:
    # @spec PORT-SEG-002 / PORT-STATE-021
    book = _book(capacity=5)
    policy = _policy(capacity=5)
    expected: list[str] = []

    assert len(book.history) == 5
    assert all(symbol.name == "BLANK" for symbol in book.history)
    assert book.history_length == 0

    for index in range(40):
        labels = [10] if index % 3 == 0 else []
        transition = _observe(book, [labels], policy=policy)
        book = transition.next_book
        expected.append("WORD_START" if labels else "BLANK")
        expected = expected[-5:]

        assert len(book.history) == 5
        assert book.history_length <= 5
        assert [symbol.name for symbol in book.newest(book.history_length)] == expected


@pytest.mark.parametrize(
    ("silence_frames", "detected"),
    [(3, False), (4, False), (5, True)],
)
def test_greedy_blank_threshold_is_strict(
    silence_frames: int, detected: bool
) -> None:
    # @spec PORT-SEG-002
    policy = _policy(stop_history_ms=240, residue_frames=2, capacity=8)
    book = _book(capacity=8)
    book = _observe(book, [[10]], policy=policy).next_book
    transition = _observe(book, [[] for _ in range(silence_frames)], policy=policy)

    assert transition.detector_trace.silent_frames == silence_frames - 1
    assert transition.is_eou is detected


def test_residue_shortens_effective_end_without_moving_newest_pivot() -> None:
    # @spec PORT-SEG-001 / PORT-SEG-002
    policy = _policy(stop_history_ms=240, residue_frames=2, capacity=8)
    transition = _observe(
        _observe(_book(8), [[10]], policy=policy).next_book,
        [[], [], [], [], []],
        policy=policy,
    )

    assert transition.detector_trace.active_window_length == 5
    assert transition.detector_trace.pivot == 4
    assert transition.detector_trace.effective_end == 3
    assert transition.detector_trace.silent_frames == 4
    assert transition.is_eou is True


def test_disabled_and_final_tail_never_make_model_eou_decisions() -> None:
    # @spec PORT-SESS-003 / PORT-SEG-002 / PORT-SEG-003
    enabled = _policy()
    disabled = _module().EndpointPolicy.resolve(
        mode="disabled",
        stop_history_ms=None,
        residue_frames=0,
        frame_stride_ms=80,
        history_capacity_frames=16,
    )
    book = _observe(_book(), [[10]], policy=enabled).next_book
    silence: list[list[int]] = [[] for _ in range(8)]

    assert _observe(book, silence, policy=disabled).is_eou is False
    assert _observe(book, silence, policy=enabled, final_tail=True).is_eou is False


def test_regular_unit_eligibility_is_independent_of_session_phase() -> None:
    # @spec PORT-SESS-003 / PORT-SEG-003
    signature = inspect.signature(_module().observe_chunk)
    assert "final_tail" in signature.parameters
    assert "session_phase" not in signature.parameters

    policy = _policy()
    book = _observe(_book(), [[10]], policy=policy).next_book
    assert _observe(
        book,
        [[], [], [], [], []],
        policy=policy,
        final_tail=False,
    ).is_eou is True


def test_eou_disarms_and_continuing_silence_cannot_emit_empty_segments() -> None:
    # @spec PORT-SEG-002 / PORT-SEG-007
    policy = _policy()
    book = _observe(_book(), [[10]], policy=policy).next_book
    detected = _observe(book, [[], [], [], [], []], policy=policy)
    assert detected.is_eou is True
    assert detected.next_book.endpoint_armed is False
    assert detected.next_book.segment_has_output is False
    assert detected.next_book.segment_generation == 1

    repeated = _observe(
        detected.next_book,
        [[], [], [], [], []],
        policy=policy,
    )
    assert repeated.is_eou is False
    assert repeated.next_book.segment_generation == 1

    rearmed = _observe(repeated.next_book, [[20]], policy=policy)
    assert rearmed.next_book.endpoint_armed is True
    assert rearmed.next_book.segment_has_output is True

    detected_again = _observe(
        rearmed.next_book,
        [[], [], [], [], []],
        policy=policy,
    )
    assert detected_again.is_eou is True
    assert detected_again.next_book.segment_generation == 2


def test_one_chunk_produces_at_most_one_boolean_eou_decision() -> None:
    # @spec PORT-SEG-002
    transition = _observe(
        _observe(_book(), [[10]], policy=_policy()).next_book,
        [[], [], [], [], [], [], [], []],
    )

    assert isinstance(transition.is_eou, bool)
    assert not hasattr(transition, "eou_detected_at")
    assert not hasattr(transition, "tail_output")
    assert not hasattr(transition, "clipped_output")


def test_detected_chunk_orders_all_labels_then_eou_then_park() -> None:
    # @spec PORT-SEG-003 / PORT-INT-002
    compose = _module().compose_chunk_output

    assert compose(
        labels=[7, 8, 9],
        is_eou=True,
        eou_token_id=101,
        park_token_id=103,
    ) == [7, 8, 9, 101, 103]
    assert compose(
        labels=[],
        is_eou=False,
        eou_token_id=101,
        park_token_id=103,
    ) == [103]


def test_output_budget_adds_eou_without_reducing_label_capacity() -> None:
    # @spec PORT-DEC-005 / PORT-INT-002
    emission_bound = _module().max_emission_tokens

    assert emission_bound(max_valid_frames=14, max_symbols_per_step=10) == 142
    with pytest.raises(ValueError, match="142|emission"):
        _module().validate_emission_budget(
            caller_max_tokens=141,
            max_valid_frames=14,
            max_symbols_per_step=10,
        )
    assert (
        _module().validate_emission_budget(
            caller_max_tokens=1_000,
            max_valid_frames=14,
            max_symbols_per_step=10,
        )
        == 142
    )


def test_failed_candidate_does_not_mutate_prior_endpoint_book() -> None:
    # @spec PORT-SEG-003 / PORT-SEG-007
    policy = _policy()
    book = _observe(_book(), [[10]], policy=policy).next_book
    before = copy.deepcopy(book)

    candidate = _observe(book, [[], [], [], [], []], policy=policy)
    assert candidate.is_eou is True
    assert book == before
    # The transition becomes authoritative only when its next_book is
    # committed by the surrounding state transaction.
    assert candidate.next_book != before


def test_forced_eou_with_output_emits_control_and_increments_generation() -> None:
    # @spec PORT-SEG-004 / PORT-SEG-007
    module = _module()
    book = _observe(_book(), [[10]], policy=_policy()).next_book

    forced = module.apply_forced_eou(
        book,
        request_generation=7,
        expected_request_generation=7,
        eou_token_id=101,
        park_token_id=103,
    )

    assert forced.output_ids == (101, 103)
    assert forced.reason == "forced"
    assert forced.empty_segment is False
    assert forced.next_book.segment_generation == 1
    assert forced.next_book.endpoint_armed is False
    assert forced.next_book.segment_has_output is False


def test_forced_eou_without_output_is_ordered_park_only_noop() -> None:
    # @spec PORT-SEG-004
    forced = _module().apply_forced_eou(
        _book(),
        request_generation=7,
        expected_request_generation=7,
        eou_token_id=101,
        park_token_id=103,
    )

    assert forced.output_ids == (103,)
    assert forced.empty_segment is True
    assert forced.completion is None
    assert forced.next_book.segment_generation == 0


def test_model_eou_winning_same_generation_coalesces_queued_force() -> None:
    # @spec PORT-SEG-004 / PORT-SEG-007
    module = _module()
    policy = _policy()
    armed = _observe(_book(), [[10]], policy=policy).next_book
    queued = module.queue_forced_eou(armed, request_generation=7)
    model = _observe(
        queued,
        [[], [], [], [], []],
        policy=policy,
    )
    assert model.is_eou is True
    assert model.next_book.segment_generation == 1

    forced = module.apply_forced_eou(
        model.next_book,
        request_generation=7,
        expected_request_generation=7,
        eou_token_id=101,
        park_token_id=103,
    )
    assert forced.coalesced is True
    assert forced.output_ids == (103,)
    assert forced.completion is None
    assert forced.next_book.segment_generation == 1


def test_stale_forced_generation_is_ignored_and_preserves_book() -> None:
    # @spec PORT-SEG-007
    book = _observe(_book(), [[10]], policy=_policy()).next_book
    before = copy.deepcopy(book)

    ignored = _module().apply_forced_eou(
        book,
        request_generation=6,
        expected_request_generation=7,
        eou_token_id=101,
        park_token_id=103,
    )

    assert ignored is None
    assert book == before
