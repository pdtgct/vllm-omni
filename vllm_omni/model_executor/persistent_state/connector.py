# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Explicit capability boundary for persistent-state location changes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, NoReturn


class UnsupportedPersistentStateCapability(RuntimeError):  # noqa: N818
    """A requested location operation is outside the resident-only profile."""


@dataclass(frozen=True)
class PersistentStateConnector:
    """Initial connector advertising only cohosted resident state."""

    capabilities: frozenset[str] = frozenset({"resident"})

    def __post_init__(self) -> None:
        # @spec PORT-STATE-011
        if self.capabilities != frozenset({"resident"}):
            raise ValueError("initial persistent-state connector is resident-only")

    @staticmethod
    def _unsupported(operation: str) -> NoReturn:
        raise UnsupportedPersistentStateCapability(f"unsupported_capability: persistent-state {operation}")

    def save(self, *args: Any, **kwargs: Any) -> NoReturn:
        del args, kwargs
        self._unsupported("save")

    def load(self, *args: Any, **kwargs: Any) -> NoReturn:
        del args, kwargs
        self._unsupported("load")

    def transfer(self, *args: Any, **kwargs: Any) -> NoReturn:
        del args, kwargs
        self._unsupported("transfer")

    def durable_drop(self, *args: Any, **kwargs: Any) -> NoReturn:
        del args, kwargs
        self._unsupported("durable_drop")
