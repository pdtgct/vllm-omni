# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Model-transaction status delivery at the scheduler boundary."""

from types import SimpleNamespace

import pytest
from vllm.v1.request import RequestStatus

from vllm_omni.core.sched.omni_scheduler_mixin import OmniSchedulerMixin
from vllm_omni.outputs import OmniConnectorOutput

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_model_failures_are_finished_without_an_input_coordinator() -> None:
    scheduler = OmniSchedulerMixin.__new__(OmniSchedulerMixin)
    scheduler.requests = {"bad": object(), "healthy": object()}
    scheduler.input_coordinator = None
    scheduler._latest_omni_connector_output = OmniConnectorOutput(
        model_status={"bad": 512, "healthy": 0},
        model_failed_req_ids={"bad"},
    )
    calls: list[tuple[set[str], RequestStatus]] = []
    scheduler.finish_requests = lambda ids, status: calls.append((set(ids), status))

    scheduler._consume_pending_connector_output(model_mode="ar")

    assert calls == [({"bad"}, RequestStatus.FINISHED_ERROR)]
    assert scheduler._latest_omni_connector_output is None


def test_stale_model_failure_identity_is_ignored() -> None:
    scheduler = OmniSchedulerMixin.__new__(OmniSchedulerMixin)
    scheduler.requests = {"healthy": object()}
    scheduler.input_coordinator = SimpleNamespace(
        update_request_metadata=lambda *args, **kwargs: None,
        process_pending_full_payload_inputs=lambda *args, **kwargs: None,
    )
    scheduler.waiting = []
    scheduler.running = []
    scheduler._latest_omni_connector_output = OmniConnectorOutput(
        model_status={"gone": 512},
        model_failed_req_ids={"gone"},
    )
    calls: list[tuple[set[str], RequestStatus]] = []
    scheduler.finish_requests = lambda ids, status: calls.append((set(ids), status))

    scheduler._consume_pending_connector_output(model_mode="ar")

    assert calls == []
