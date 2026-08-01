# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Mechanical contracts for the initial resident-only state connector."""

from __future__ import annotations

import pytest

from tests.model_executor.persistent_state._helpers import (
    connector_capabilities,
    invoke_connector_operation,
    mutation_snapshot,
    new_connector,
    require_persistent_state_module,
)


def test_connector_capability_is_exactly_resident() -> None:
    """@spec PORT-STATE-011: the initial connector advertises only resident."""

    module = require_persistent_state_module()
    connector = new_connector(module)

    assert connector_capabilities(connector) == {"resident"}


def test_unsupported_connector_operations_fail_before_mutation() -> None:
    """@spec PORT-STATE-011: save/load/transfer/drop fail before mutation."""

    module = require_persistent_state_module()
    connector = new_connector(module)

    for operation in ("save", "load", "transfer", "durable_drop"):
        before = mutation_snapshot(connector)
        with pytest.raises(Exception) as exc_info:
            invoke_connector_operation(connector, operation)

        message = str(exc_info.value).lower()
        assert "unsupported_capability" in message or "unsupported" in message
        assert mutation_snapshot(connector) == before
