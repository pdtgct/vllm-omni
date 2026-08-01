# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shared adapters for the Phase-5 persistent-state contract tests.

The approved PORT intent names the persistent-state types and behaviors, but it
does not yet choose their public Python module or every constructor/method
spelling.  Keep those provisional choices here so the contract tests fail as
ordinary test failures until the production seam exists, and so Sol can update
one file after the public surface is reviewed.
"""

from __future__ import annotations

import asyncio
import copy
import importlib
import inspect
from collections.abc import Awaitable, Callable, Mapping, Sequence
from types import ModuleType
from typing import Any, NoReturn, cast

import pytest
import torch
from vllm.sampling_params import SamplingParams
from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.single_type_kv_cache_manager import SingleTypeKVCacheManager
from vllm.v1.request import Request

# Provisional public surface.  The pinned intent deliberately leaves this
# module boundary to the implementation review; do not spread this assumption
# into the individual tests.
PERSISTENT_STATE_MODULE = "vllm_omni.model_executor.persistent_state"


def _fail(message: str) -> NoReturn:
    pytest.fail(message, pytrace=False)
    raise AssertionError(message)


def require_persistent_state_module() -> ModuleType:
    """Load the future substrate module from inside a test call."""

    try:
        return importlib.import_module(PERSISTENT_STATE_MODULE)
    except Exception as exc:
        _fail(
            "Missing or unusable future persistent-state seam "
            f"{PERSISTENT_STATE_MODULE!r}: {exc}"
        )


def require_symbol(module: ModuleType, name: str) -> Any:
    """Return a required future symbol, reporting a test failure if absent."""

    try:
        return getattr(module, name)
    except AttributeError:
        _fail(f"{PERSISTENT_STATE_MODULE!r} must expose {name!r}")


def require_callable(module: ModuleType, name: str) -> Callable[..., Any]:
    """Return a required callable future seam."""

    value = require_symbol(module, name)
    if not callable(value):
        _fail(f"{PERSISTENT_STATE_MODULE!r}.{name} must be callable")
    return cast(Callable[..., Any], value)


def _select_parameter(
    parameters: Mapping[str, inspect.Parameter], aliases: Sequence[str]
) -> str | None:
    """Select the first constructor spelling accepted by a future type."""

    for name in aliases:
        if name in parameters:
            return name
    if any(parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in parameters.values()):
        return aliases[0]
    return None


def _construct(
    cls: type[Any], values: Mapping[str, tuple[Sequence[str], Any]], label: str
) -> Any:
    """Construct a future value using the canonical contract fields."""

    try:
        parameters = inspect.signature(cls).parameters
    except (TypeError, ValueError) as exc:
        _fail(f"Cannot inspect future {label} constructor: {exc}")

    kwargs: dict[str, Any] = {}
    for canonical, (aliases, value) in values.items():
        selected = _select_parameter(parameters, aliases)
        if selected is not None:
            kwargs[selected] = value

    required = {
        name
        for name, parameter in parameters.items()
        if parameter.default is inspect.Parameter.empty
        and parameter.kind
        in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY)
    }
    missing = sorted(required.difference(kwargs))
    if missing:
        _fail(
            f"Future {label} constructor has undecided required fields "
            f"{missing!r}; update the centralized test helper after review"
        )

    try:
        return cls(**kwargs)
    except Exception as exc:
        _fail(f"Future {label} could not be constructed from the contract fields: {exc}")


def descriptor_field(descriptor: Any, *names: str) -> Any:
    """Read a descriptor field while keeping spelling adaptation centralized."""

    for name in names:
        if hasattr(descriptor, name):
            return getattr(descriptor, name)
    _fail(f"Persistent-state descriptor must expose one of {names!r}")


def spec_field(spec: Any, *names: str) -> Any:
    """Read a spec field while keeping spelling adaptation centralized."""

    for name in names:
        if hasattr(spec, name):
            return getattr(spec, name)
    _fail(f"PersistentStateSpec must expose one of {names!r}")


def make_descriptor(
    module: ModuleType,
    *,
    name: str,
    shape: tuple[int, ...],
    dtype: torch.dtype,
    offset_bytes: int,
    alignment_bytes: int,
) -> Any:
    """Create one generic, non-Nemotron descriptor for layout tests."""

    descriptor_cls = require_symbol(module, "PersistentStateDescriptor")
    return _construct(
        descriptor_cls,
        {
            "name": (("name", "semantic_name", "state_name"), name),
            "shape": (("shape",), shape),
            "dtype": (("dtype",), dtype),
            "offset": (("offset_bytes", "byte_offset", "offset"), offset_bytes),
            "alignment": (("alignment_bytes", "alignment"), alignment_bytes),
            "initializer": (("initializer", "fresh_initializer", "fresh_init"), torch.zeros),
        },
        "PersistentStateDescriptor",
    )


def make_generic_spec(module: ModuleType, variant: str = "base") -> Any:
    """Create a small aggregate spec without pinning model tensor shapes."""

    descriptor_one = make_descriptor(
        module,
        name="generic.float_state",
        shape=(2, 3),
        dtype=torch.float32,
        offset_bytes=0,
        alignment_bytes=16,
    )
    second_offset = 32 if variant != "shifted" else 40
    second_shape = (3,) if variant != "reshaped" else (4,)
    descriptor_two = make_descriptor(
        module,
        name="generic.integer_state",
        shape=second_shape,
        dtype=torch.int64,
        offset_bytes=second_offset,
        alignment_bytes=8,
    )
    spec_cls = require_symbol(module, "PersistentStateSpec")
    return _construct(
        spec_cls,
        {
            "descriptors": (("descriptors", "tensor_descriptors", "layout"),
                            (descriptor_one, descriptor_two)),
            "page_size": (("page_size_bytes", "page_size"), 64),
            "block_size": (("block_size",), 1),
            "state_name": (("state_name", "name", "semantic_name"), "generic-state"),
            "persistence_class": (("persistence_class", "kind"), "resident"),
        },
        "PersistentStateSpec",
    )


def make_manager(
    module: ModuleType, *, num_gpu_blocks: int = 8
) -> tuple[SingleTypeKVCacheManager, BlockPool, Any]:
    """Construct the future manager over a real vLLM block pool."""

    spec = make_generic_spec(module)
    manager_cls = require_symbol(module, "PersistentStateManager")
    block_pool = BlockPool(
        num_gpu_blocks=num_gpu_blocks,
        enable_caching=False,
        hash_block_size=1,
    )
    manager = _construct(
        manager_cls,
        {
            "spec": (("kv_cache_spec", "spec"), spec),
            "block_pool": (("block_pool",), block_pool),
            "enable_caching": (("enable_caching",), False),
            "kv_cache_group_id": (("kv_cache_group_id", "group_id"), 0),
            "scheduler_block_size": (("scheduler_block_size",), 1),
        },
        "PersistentStateManager",
    )
    if not isinstance(manager, SingleTypeKVCacheManager):
        _fail("PersistentStateManager must subclass core SingleTypeKVCacheManager")
    return manager, block_pool, spec


def state_binding_pair(manager: Any, request_id: str) -> tuple[Any, Any]:
    """Read the manager's authoritative physical-slot/generation binding."""

    method = None
    for name in ("get_state_binding", "get_binding"):
        candidate = getattr(manager, name, None)
        if callable(candidate):
            method = candidate
            break
    if method is None:
        _fail(
            "PersistentStateManager must expose a reviewed binding accessor "
            "(expected get_state_binding(request_id))"
        )
    binding = method(request_id)
    if binding is None:
        _fail(f"No persistent-state binding returned for {request_id!r}")
    slot_id = descriptor_field(binding, "slot_id", "physical_slot_id", "block_id")
    generation = descriptor_field(binding, "generation", "slot_generation")
    return slot_id, generation


def real_request(request_id: str = "request-1") -> Request:
    """Build the real vLLM Request needed by manager cache APIs."""

    return Request(
        request_id=request_id,
        prompt_token_ids=[1],
        sampling_params=SamplingParams(max_tokens=1),
        pooling_params=None,
    )


def _call_payload_validator(
    fn: Callable[..., Any], payload: Any, label: str
) -> Any:
    """Call a future validator with either one payload or named fields."""

    try:
        parameters = inspect.signature(fn).parameters
    except (TypeError, ValueError) as exc:
        _fail(f"Cannot inspect {label}: {exc}")

    required = [
        parameter
        for parameter in parameters.values()
        if parameter.default is inspect.Parameter.empty
        and parameter.kind
        in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY)
    ]
    if len(required) == 1 and required[0].name in {
        "profile",
        "execution_profile",
        "declarations",
        "persistent_declarations",
    }:
        if required[0].kind == inspect.Parameter.KEYWORD_ONLY:
            return fn(**{required[0].name: payload})
        return fn(payload)
    if isinstance(payload, Mapping):
        return fn(**payload)
    return fn(payload)


def validate_execution_profile(module: ModuleType, profile: Mapping[str, Any]) -> Any:
    """Invoke the reviewed execution-profile validator."""

    fn = require_callable(module, "validate_persistent_state_profile")
    return _call_payload_validator(fn, profile, "validate_persistent_state_profile")


def validate_declarations(module: ModuleType, declarations: Sequence[Mapping[str, Any]]) -> Any:
    """Invoke the reviewed persistent-layer declaration validator."""

    fn = require_callable(module, "validate_persistent_state_declarations")
    return _call_payload_validator(fn, declarations, "validate_persistent_state_declarations")


def declaration(spec: Any, layer_name: str, schema_id: Any) -> dict[str, Any]:
    """Build one generic declaration payload for the profile validator."""

    return {
        "layer_name": layer_name,
        "spec": spec,
        "schema_id": schema_id,
        "profile_id": "generic-profile",
    }


def new_connector(module: ModuleType) -> Any:
    """Construct the future resident-only connector."""

    connector_cls = require_symbol(module, "PersistentStateConnector")
    return _construct(
        connector_cls,
        {"capabilities": (("capabilities", "supported_capabilities"), frozenset({"resident"}))},
        "PersistentStateConnector",
    )


def connector_capabilities(connector: Any) -> set[str]:
    """Return the connector capability names as a normalized set."""

    capabilities = getattr(connector, "capabilities", None)
    if callable(capabilities):
        capabilities = capabilities()
    if capabilities is None:
        capabilities = getattr(connector, "supported_capabilities", None)
        if callable(capabilities):
            capabilities = capabilities()
    if capabilities is None:
        _fail("PersistentStateConnector must expose its capability set")
    if isinstance(capabilities, Mapping):
        return {str(name) for name, enabled in capabilities.items() if enabled}
    return {str(name) for name in capabilities}


def _operation_argument(parameter_name: str) -> Any:
    """Supply inert, named test values to an unsupported connector operation."""

    lowered = parameter_name.lower()
    if "request" in lowered or lowered.endswith("_id"):
        return "request-1"
    if "capabil" in lowered or lowered in {"operation", "kind", "action"}:
        return "resident"
    if "lease" in lowered:
        return object()
    if "state" in lowered or "snapshot" in lowered or "payload" in lowered:
        return {"sentinel": True}
    return object()


def invoke_connector_operation(connector: Any, operation: str) -> Any:
    """Invoke one unsupported operation through the provisional public surface."""

    method_names = {
        "save": ("save", "save_async"),
        "load": ("load", "load_async"),
        "transfer": ("transfer", "transfer_async"),
        "durable_drop": ("durable_drop", "drop", "durable_drop_async", "drop_async"),
    }[operation]
    method = next((getattr(connector, name, None) for name in method_names), None)
    if method is None:
        method = getattr(connector, "request", None) or getattr(connector, "execute", None)
    if not callable(method):
        _fail(
            f"PersistentStateConnector must expose a reviewed {operation!r} "
            "operation (or request/execute dispatcher)"
        )

    try:
        parameters = inspect.signature(method).parameters
    except (TypeError, ValueError) as exc:
        _fail(f"Cannot inspect connector {operation} operation: {exc}")
    args: list[Any] = []
    kwargs: dict[str, Any] = {}
    for parameter in parameters.values():
        if parameter.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
            continue
        if parameter.default is not inspect.Parameter.empty:
            continue
        value = operation if parameter.name in {"operation", "kind", "action"} else _operation_argument(parameter.name)
        if parameter.kind == inspect.Parameter.KEYWORD_ONLY:
            kwargs[parameter.name] = value
        else:
            args.append(value)
    if method.__name__ in {"request", "execute"} and not any(
        parameter.name in {"operation", "kind", "action"} for parameter in parameters.values()
    ):
        kwargs["operation"] = operation
    result = method(*args, **kwargs)
    if inspect.isawaitable(result):
        return asyncio.run(_resolve_awaitable(result))
    return result


async def _resolve_awaitable(value: Awaitable[Any]) -> Any:
    return await value


def mutation_snapshot(value: Any) -> Any:
    """Take a comparison-friendly snapshot of connector-owned mutable state."""

    state = getattr(value, "__dict__", None)
    if state is None:
        return repr(value)
    try:
        return copy.deepcopy(state)
    except Exception:
        return repr(state)
