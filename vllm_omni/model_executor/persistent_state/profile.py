# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fail-closed validation for the first persistent-state execution profile."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from .spec import PersistentStateSpec


# @spec PORT-STATE-018
def validate_persistent_state_profile(profile: Mapping[str, Any]) -> None:
    singleton_axes = (
        "stage_count",
        "replica_count",
        "gpu_count",
        "tp",
        "pp",
        "dcp",
        "pcp",
        "dp",
    )
    for field in singleton_axes:
        if profile.get(field) != 1:
            raise ValueError(f"persistent-state profile requires {field}=1")
    for field in ("persistent_schema_ids", "model_ids"):
        values = tuple(profile.get(field, ()))
        if len(values) != 1 or not values[0]:
            raise ValueError(f"persistent-state profile requires exactly one {field}")


# @spec PORT-STATE-002
def validate_persistent_state_declarations(
    declarations: Sequence[Mapping[str, Any]],
) -> None:
    if len(declarations) != 1:
        raise ValueError("persistent-state profile requires exactly one declaration")
    declaration = declarations[0]
    spec = declaration.get("spec")
    if type(spec) is not PersistentStateSpec:
        raise TypeError("persistent-state declaration must use direct PersistentStateSpec")
    if declaration.get("schema_id") != spec.schema_id:
        raise ValueError("persistent-state declaration schema mismatch")
    layer_name = declaration.get("layer_name")
    profile_id = declaration.get("profile_id")
    if not isinstance(layer_name, str) or not layer_name:
        raise ValueError("persistent-state declaration layer name must be nonempty")
    if not isinstance(profile_id, str) or not profile_id:
        raise ValueError("persistent-state declaration profile id must be nonempty")
