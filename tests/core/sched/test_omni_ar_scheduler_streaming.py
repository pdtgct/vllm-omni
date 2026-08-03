"""Unit tests for Omni AR streaming-session async placeholder handling."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

# Imports must run in this order: vllm_omni applies patches to vllm.v1.request before
# Request / StreamingUpdate are bound in this module. Ruff isort would reorder them.
# isort: off
import vllm_omni  # noqa: F401 - import for side effects (patch vLLM)
import vllm_omni.core.sched.omni_ar_scheduler as scheduler_mod
from vllm.sampling_params import SamplingParams
from vllm.v1.request import Request, RequestStatus, StreamingUpdate
from vllm_omni.core.sched.output import OmniNewRequestData
from vllm_omni.core.sched.omni_ar_scheduler import OmniARScheduler

# isort: on

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _make_scheduler(*, stage_id: int = 0) -> OmniARScheduler:
    sched = OmniARScheduler.__new__(OmniARScheduler)
    sched._new_prompt_len_snapshot = {}
    sched.vllm_config = SimpleNamespace(model_config=SimpleNamespace(stage_id=stage_id))
    sched.num_waiting_for_streaming_input = 0
    sched.log_stats = False
    sched.chunk_transfer_adapter = None
    return sched


def _make_request() -> Request:
    return Request(
        request_id="req-ar-streaming-test",
        prompt_token_ids=[1, 2, 3],
        sampling_params=SamplingParams(max_tokens=8),
        pooling_params=None,
        arrival_time=100.0,
        block_hasher=None,
    )


def _make_update(prompt_token_ids: list[int] | None = None) -> StreamingUpdate:
    return StreamingUpdate(
        mm_features=None,
        prompt_token_ids=[10, 20] if prompt_token_ids is None else prompt_token_ids,
        max_tokens=32,
        arrival_time=200.0,
        sampling_params=SamplingParams(max_tokens=16),
    )


def test_stage0_streaming_update_discards_outstanding_async_placeholder_token() -> None:
    sched = _make_scheduler(stage_id=0)
    session = _make_request()
    session.status = RequestStatus.WAITING_FOR_STREAMING_REQ
    session.append_output_token_ids([7, 8, 9])
    session.num_computed_tokens = 6
    session.num_output_placeholders = 1
    session.spec_token_ids = [-1]

    sched._update_request_as_session(session, _make_update([10, 20]))

    assert session.async_tokens_to_discard == 1
    assert session.num_output_placeholders == 0
    assert session.spec_token_ids == []
    # The async placeholder makes token 9 unconfirmed, so only 7 and 8 are
    # carried into the next streaming prompt before the new chunk tokens.
    assert session.prompt_token_ids == [1, 2, 3, 7, 8, 10, 20]
    assert list(session._all_token_ids) == [1, 2, 3, 7, 8, 10, 20]
    assert session._output_token_ids == []
    assert session.num_prompt_tokens == 7
    assert sched._new_prompt_len_snapshot[session.request_id] == 2


def test_stage0_streaming_update_keeps_all_computed_tokens_without_placeholder() -> None:
    sched = _make_scheduler(stage_id=0)
    session = _make_request()
    session.status = RequestStatus.WAITING_FOR_STREAMING_REQ
    session.append_output_token_ids([7, 8, 9])
    session.num_computed_tokens = 6
    session.num_output_placeholders = 0

    sched._update_request_as_session(session, _make_update([10, 20]))

    assert getattr(session, "async_tokens_to_discard", 0) == 0
    assert session.num_output_placeholders == 0
    assert session.prompt_token_ids == [1, 2, 3, 7, 8, 9, 10, 20]
    assert list(session._all_token_ids) == [1, 2, 3, 7, 8, 9, 10, 20]
    assert session._output_token_ids == []
    assert session.num_prompt_tokens == 8
    assert sched._new_prompt_len_snapshot[session.request_id] == 2


def test_scheduler_rewrap_preserves_v2_prefill_tokens(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # @spec PORT-MIG-005 / PORT-MIG-006
    scheduler = _make_scheduler()
    scheduler.waiting = []
    scheduler.running = []
    scheduler.input_coordinator = None
    scheduler._consume_pending_connector_output = lambda model_mode: None
    scheduler._process_pending_input_timeouts = lambda: None
    scheduler._should_defer_waiting_admission = lambda: False
    scheduler.get_finished_requests_needing_kv_transfer = lambda: {}
    scheduler._wrap_omni_scheduler_output = (
        lambda output, **kwargs: output
    )
    request = SimpleNamespace(
        external_req_id="external-v2",
        prompt_embeds=None,
        additional_information=None,
    )
    scheduler.requests = {"req-v2": request}
    prefill_token_ids = [401, 402, 403]
    base_new_request = SimpleNamespace(
        req_id="req-v2",
        prompt_token_ids=[401],
        mm_features=[],
        sampling_params=None,
        pooling_params=None,
        block_ids=([9],),
        num_computed_tokens=0,
        lora_request=None,
        prompt_embeds=None,
        prompt_is_token_ids=[True],
        prefill_token_ids=prefill_token_ids,
    )
    scheduler_output = SimpleNamespace(
        scheduled_new_reqs=[base_new_request]
    )
    monkeypatch.setattr(
        scheduler_mod.VLLMScheduler,
        "schedule",
        lambda self, throttle_prefills=False: scheduler_output,
    )

    output = scheduler.schedule()

    wrapped = output.scheduled_new_reqs[0]
    assert isinstance(wrapped, OmniNewRequestData)
    assert wrapped.prefill_token_ids is prefill_token_ids
