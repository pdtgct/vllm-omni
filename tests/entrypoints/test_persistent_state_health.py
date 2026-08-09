# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Native readiness semantics for persistent-state authority and load."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from vllm_omni.engine.persistent_state_service import (
    PersistentStateServiceUnavailable,
)
from vllm_omni.entrypoints.openai.api_server import health

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class _Engine:
    def __init__(self, error: BaseException | None = None) -> None:
        self.error = error
        self.health_calls = 0

    async def check_health(self) -> None:
        self.health_calls += 1
        if self.error is not None:
            raise self.error


def _request(engine: _Engine) -> SimpleNamespace:
    return SimpleNamespace(
        app=SimpleNamespace(state=SimpleNamespace(engine_client=engine))
    )


async def _health_response(engine: _Engine) -> Any:
    try:
        return await health(_request(engine))  # type: ignore[arg-type]
    except PersistentStateServiceUnavailable as error:
        pytest.fail(
            "PORT-STATE-030 native health leaked state-service failure "
            f"instead of returning 503: {error}",
            pytrace=False,
        )


@pytest.mark.asyncio  # type: ignore[untyped-decorator]
async def test_load_pressure_remains_ready() -> None:
    """@spec PORT-STATE-029 / PORT-STATE-030: shed is not fault."""

    engine = _Engine()

    response = await _health_response(engine)

    assert response.status_code == 200
    assert engine.health_calls == 1


@pytest.mark.asyncio  # type: ignore[untyped-decorator]
async def test_state_authority_loss_maps_to_503_instead_of_generic_500() -> None:
    """@spec PORT-STATE-022 / PORT-STATE-030."""

    engine = _Engine(
        PersistentStateServiceUnavailable(
            "persistent-state authority is unhandshaked"
        )
    )

    response = await _health_response(engine)

    assert response.status_code == 503
    assert b"unhandshaked" in response.body


@pytest.mark.asyncio  # type: ignore[untyped-decorator]
@pytest.mark.parametrize("state", ["restart_escalation", "stopping"])
async def test_restart_escalation_and_stopping_are_unready(
    state: str,
) -> None:
    """@spec PORT-STATE-014 / PORT-STATE-030."""

    engine = _Engine(PersistentStateServiceUnavailable(state))

    response = await _health_response(engine)

    assert response.status_code == 503
    assert state.encode() in response.body
