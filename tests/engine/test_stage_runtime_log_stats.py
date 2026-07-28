# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Host-statistics threading into the stage procs (A27 amendment 5).

The A27 GPU round proved the API path's ``log_stats=True`` was dropped
on the way to the stage-proc scheduler: ``create_stage_runtime`` had no
``log_stats`` parameter, the replica launch hardcoded ``log_stats=False``,
and the logical-stage output processor was built without the flag — so
the scheduler-side batch-stat forward never ran in any real serve.

Seam-existence pins in the style this suite already uses for the
api-server-count invariant: signatures and call-site shapes, cheap on
CPU, loud on regression. The composed truth is proven on the pod.
"""

from __future__ import annotations

import inspect

import pytest

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


# @spec PORT-OBS-008
def test_create_stage_runtime_accepts_log_stats() -> None:
    from vllm_omni.engine.stage_runtime import create_stage_runtime

    assert "log_stats" in inspect.signature(create_stage_runtime).parameters


# @spec PORT-OBS-008
def test_replica_launch_threads_the_runtime_flag_not_a_hardcoded_false() -> None:
    from vllm_omni.engine import stage_runtime as m

    src = inspect.getsource(m.StageRuntime)
    assert "log_stats=False" not in src
    assert "log_stats=self._log_stats" in src


# @spec PORT-OBS-008
def test_output_processor_receives_the_flag() -> None:
    from vllm_omni.engine import stage_runtime as m

    src = inspect.getsource(m.StageRuntime)
    assert "build_llm_stage_output_processor(" in src
    # The call site passes the runtime's flag rather than defaulting.
    start = src.index("build_llm_stage_output_processor(")
    call = src[start : start + 220]
    assert "log_stats" in call


# @spec PORT-OBS-008
def test_engine_passes_its_flag_into_the_runtime_factory() -> None:
    from vllm_omni.engine import async_omni_engine as m

    src = inspect.getsource(m.AsyncOmniEngine)
    start = src.index("create_stage_runtime(")
    call = src[start : src.index(")", src.index("request_queue", start))]
    assert "log_stats=self._log_stats" in call


# @spec PORT-OBS-008
def test_headless_derives_the_flag_from_its_own_cli() -> None:
    """Headless stage procs are configured by the headless process's own
    CLI — already correctly wired; pinned here so it stays that way."""
    from vllm_omni.entrypoints.cli import serve as m

    src = inspect.getsource(m.run_headless)
    assert "log_stats" in src
    assert "disable_log_stats" in src
