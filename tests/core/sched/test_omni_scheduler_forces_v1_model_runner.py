# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""The omni scheduler must stay on the v1 model runner, like every omni worker.

vLLM 0.28 resolved this config to v1; 0.29 ends its fallback chain with v2. When
only the workers forced v1, the scheduler took its v2 fast path -- which omits
resumed-request token ids -- while the v1 runner still read them, so resuming a
request raised ``KeyError: <req_id>`` in ``_update_states`` and killed the engine
core (Multi-Replica Startup Test).

The default is upstream's to change, so this pins Omni's side of the contract
rather than the default itself.
"""

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
                async_chunk=False,
                stage_id=0,
                pooling_output_decoder=None,
            )
        )
        self._init_omni_io_scheduling_state()


@pytest.mark.parametrize("initial", [True, False])
def test_scheduler_ends_on_v1_model_runner(initial: bool) -> None:
    """Whatever upstream resolves, the omni scheduler runs v1."""
    assert _Scheduler(use_v2_model_runner=initial).use_v2_model_runner is False


def test_v2_default_does_not_reach_the_resumed_request_fast_path() -> None:
    """The v2 fast path is what dropped resumed-request token ids.

    ``OmniGenerationScheduler`` branches on ``use_v2_model_runner`` when it
    assembles SchedulerOutput; forcing it false keeps the v1 assembly that
    carries ``all_token_ids`` for resumed requests, which the v1 runner reads.
    """
    scheduler = _Scheduler(use_v2_model_runner=True)

    assert not scheduler.use_v2_model_runner, (
        "scheduler would take the v2 fast path while omni workers run v1; "
        "resumed requests then KeyError in _update_states"
    )


def test_persistent_scheduler_preserves_v2_request_payload_mode(monkeypatch) -> None:
    """@spec PORT-MIG-006: scheduler and worker agree before state allocation."""
    from vllm.model_executor import model_loader

    from vllm_omni.model_executor.models.nemotron_asr.nemotron_asr import NemotronASRForRNNT

    monkeypatch.setattr(model_loader, "get_model_cls", lambda config: NemotronASRForRNNT)
    scheduler = _Scheduler(use_v2_model_runner=True)
    assert scheduler.use_v2_model_runner is True
