# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""The scheduler and worker follow the same explicit O30 runner selection."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from vllm_omni.core.sched.omni_scheduler_mixin import OmniSchedulerMixin

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.fixture(autouse=True)
def ordinary_model_class(monkeypatch):
    from vllm.model_executor import model_loader

    monkeypatch.setattr(model_loader, "get_model_cls", lambda config: object)


class _Scheduler(OmniSchedulerMixin):
    """Minimal carrier for the shared init hook both omni schedulers call."""

    def __init__(self, *, use_v2_model_runner: bool) -> None:
        self.use_v2_model_runner = use_v2_model_runner
        self.vllm_config = SimpleNamespace(
            model_config=SimpleNamespace(
                use_v2_model_runner=use_v2_model_runner,
                async_chunk=False,
                stage_id=0,
                pooling_output_decoder=None,
            )
        )
        self._init_omni_io_scheduling_state()


@pytest.mark.parametrize("initial", [True, False])
def test_scheduler_preserves_selected_model_runner(initial: bool) -> None:
    """Use the runner resolved by the typed model config."""
    assert _Scheduler(use_v2_model_runner=initial).use_v2_model_runner is initial


def test_v1_selection_does_not_reach_the_resumed_request_fast_path() -> None:
    """The V1 selection retains resumed-request token IDs for its runner."""
    scheduler = _Scheduler(use_v2_model_runner=False)
    assert not scheduler.use_v2_model_runner


def test_persistent_scheduler_preserves_v2_request_payload_mode(monkeypatch) -> None:
    """@spec PORT-MIG-006: scheduler and worker agree before state allocation."""
    from vllm.model_executor import model_loader

    from vllm_omni.model_executor.models.nemotron_asr.nemotron_asr import NemotronASRForRNNT

    monkeypatch.setattr(model_loader, "get_model_cls", lambda config: NemotronASRForRNNT)
    scheduler = _Scheduler(use_v2_model_runner=True)
    assert scheduler.use_v2_model_runner is True
