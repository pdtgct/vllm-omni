# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Derivation of the engine's advertised task set from stage metadata.

Kept free of engine and vLLM imports so the derivation is unit-testable
without a GPU or a built vLLM: it reads only structural attributes of the
per-stage metadata records (PORT-CAP-001).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

# Derived tasks that predate declared-capability advertisement; a stage
# declaring the same string adds nothing.
_COMPREHENSION_TASK = "generate"
_AUDIO_OUTPUT_TASK = "speech"
_DEFAULT_TASKS: tuple[str, ...] = ("generate",)


@runtime_checkable
class StageTaskMetadata(Protocol):
    """The stage-metadata surface the derivation reads.

    Structural on purpose: ``StageRuntimeInfo`` satisfies it without this
    module importing the engine package.
    """

    @property
    def final_output_type(self) -> object: ...

    @property
    def declared_tasks(self) -> tuple[str, ...]: ...


def derive_supported_tasks(
    stage_metadata: Sequence[StageTaskMetadata],
    *,
    has_comprehension_stage: bool,
) -> tuple[str, ...]:
    """Derive the engine's advertised task set.

    The declared-capability union runs after the historical rules and is a
    plain capability list: no task string is special-cased, so a new serving
    surface becomes advertisable by stage declaration alone.

    Args:
        stage_metadata: Per-stage runtime metadata, in stage order.
        has_comprehension_stage: Whether any stage client is a comprehension
            stage.

    Returns:
        The advertised task names; ``("generate",)`` when nothing is derived.
    """
    supported_tasks: set[str] = set()
    if has_comprehension_stage:
        supported_tasks.add(_COMPREHENSION_TASK)
    if any(meta.final_output_type == "audio" for meta in stage_metadata):
        supported_tasks.add(_AUDIO_OUTPUT_TASK)
    for meta in stage_metadata:
        supported_tasks.update(meta.declared_tasks)
    return tuple(supported_tasks) if supported_tasks else _DEFAULT_TASKS
