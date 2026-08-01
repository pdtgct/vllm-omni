# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Mechanical P3 exit contracts for the generic persistent-state substrate.

The intent fixes the behavior, but the production module and several public
method spellings remain an implementation-review decision.  All such names
are kept in ``PROVISIONAL_SURFACE`` below so the tests have one explicit seam
for that review.  Missing future symbols are reported as ordinary test
failures by the shared helper; no import-time stubs or module poisoning are
used.
"""

from __future__ import annotations

import inspect
import logging
import math
from collections.abc import Callable, Mapping, Sequence
from types import ModuleType
from typing import Any, NoReturn, cast

import pytest
import torch
from vllm.v1.core.block_pool import BlockPool

from tests.model_executor.persistent_state._helpers import (
    descriptor_field,
    make_generic_spec,
    make_manager,
    require_persistent_state_module,
    spec_field,
)

# The intent leaves the Python boundary and exact call spellings to the
# implementation review.  Keep every new assumption here rather than in the
# individual EARS tests.
PROVISIONAL_SURFACE: dict[str, tuple[str, ...]] = {
    "capacity_manager_class": ("PersistentStateManager",),
    "storage_allocate": (
        "allocate_persistent_state_storage",
        "allocate_raw_state_storage",
        "allocate_persistent_state_views",
    ),
    "storage_initialize": (
        "initialize_persistent_state_slot",
        "initialize_state_slot",
        "initialize_fresh_state_slot",
    ),
    "batch_class": ("PersistentStateBatch",),
    "batch_validate": (
        "validate_persistent_state_batch",
        "preflight_persistent_state_batch",
    ),
    "batch_gather": (
        "gather_persistent_state_batch",
        "gather_state_batch",
    ),
    "batch_scatter": (
        "scatter_persistent_state_batch",
        "scatter_state_batch",
    ),
    "binding_class": ("StateBinding", "PersistentStateBinding"),
}

_MISSING = object()
_NO_DEFAULT = object()


def _fail(message: str) -> NoReturn:
    pytest.fail(message, pytrace=False)
    raise AssertionError(message)


def _surface_callable(module: ModuleType, key: str) -> Callable[..., Any]:
    names = PROVISIONAL_SURFACE[key]
    for name in names:
        candidate = getattr(module, name, None)
        if callable(candidate):
            return cast(Callable[..., Any], candidate)
    _fail(f"{module.__name__!r} must expose one reviewed callable for {key!r}: {names!r}")


def _surface_class(module: ModuleType, key: str) -> type[Any]:
    names = PROVISIONAL_SURFACE[key]
    for name in names:
        candidate = getattr(module, name, None)
        if isinstance(candidate, type):
            return candidate
    _fail(f"{module.__name__!r} must expose one reviewed type for {key!r}: {names!r}")


def _parameter_names(fn: Callable[..., Any]) -> set[str]:
    try:
        return set(inspect.signature(fn).parameters)
    except (TypeError, ValueError) as exc:
        _fail(f"Cannot inspect future contract callable {fn!r}: {exc}")


def _call_contract(
    fn: Callable[..., Any],
    values: Mapping[str, tuple[Sequence[str], Any]],
    label: str,
    *,
    preserve_exceptions: bool = False,
) -> Any:
    """Call a reviewed seam while centralizing spelling adapters.

    A required parameter that is not represented in ``values`` is a deliberate
    red test for an API decision, not a reason to guess an architectural
    object.  Optional parameters are supplied only when their reviewed alias
    is present.
    """

    try:
        parameters = inspect.signature(fn).parameters
    except (TypeError, ValueError) as exc:
        _fail(f"Cannot inspect future {label}: {exc}")

    selected: dict[str, Any] = {}
    used_values: set[str] = set()
    for canonical, (aliases, value) in values.items():
        for parameter_name in aliases:
            if parameter_name in parameters:
                selected[parameter_name] = value
                used_values.add(canonical)
                break

    has_var_kwargs = any(parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in parameters.values())
    if has_var_kwargs:
        for canonical, (aliases, value) in values.items():
            if canonical not in used_values:
                selected.setdefault(aliases[0], value)

    required = {
        name
        for name, parameter in parameters.items()
        if parameter.default is inspect.Parameter.empty
        and parameter.kind
        in (
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        )
    }
    missing = sorted(required.difference(selected))
    if missing:
        _fail(
            f"Future {label} has undecided required parameters {missing!r}; "
            "centralize the reviewed API spelling in this test module"
        )

    positional: list[Any] = []
    keyword: dict[str, Any] = {}
    for name, parameter in parameters.items():
        if name not in selected:
            continue
        if parameter.kind == inspect.Parameter.POSITIONAL_ONLY:
            positional.append(selected[name])
        elif parameter.kind not in (
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        ):
            keyword[name] = selected[name]
    try:
        return fn(*positional, **keyword)
    except Exception as exc:
        if preserve_exceptions:
            raise
        _fail(f"Future {label} rejected the contract call: {exc}")


def _read_value(value: Any, names: Sequence[str], label: str, *, default: Any = _NO_DEFAULT) -> Any:
    result = _try_read_value(value, names)
    if result is not _MISSING:
        return result
    if default is not _NO_DEFAULT:
        return default
    _fail(f"{label} must expose one of {tuple(names)!r}")


def _try_read_value(value: Any, names: Sequence[str]) -> Any:
    if isinstance(value, Mapping):
        for name in names:
            if name in value:
                return value[name]
    for name in names:
        if hasattr(value, name):
            candidate = getattr(value, name)
            if callable(candidate):
                try:
                    candidate = candidate()
                except TypeError:
                    continue
            return candidate
    return _MISSING


def _read_nested_value(value: Any, names: Sequence[str], label: str, *, default: Any = _NO_DEFAULT) -> Any:
    candidates = [value]
    for container_name in (
        "capacity",
        "capacity_fingerprint",
        "inventory",
        "capacity_summary",
    ):
        nested = _try_read_value(value, (container_name,))
        if nested is not _MISSING:
            candidates.append(nested)
    for candidate in candidates:
        result = _try_read_value(candidate, names)
        if result is not _MISSING:
            return result
    return _read_value(value, names, label, default=default)


def _construct_capacity_manager(
    module: ModuleType,
    *,
    num_gpu_blocks: int,
    safety_reserve_slots: int,
    max_resident_sessions: int,
    preserve_exceptions: bool = False,
) -> tuple[Any, BlockPool, Any]:
    spec = make_generic_spec(module)
    manager_cls = _surface_class(module, "capacity_manager_class")
    block_pool = BlockPool(
        num_gpu_blocks=num_gpu_blocks,
        enable_caching=False,
        hash_block_size=1,
    )
    values = {
        "spec": (("kv_cache_spec", "spec", "persistent_state_spec"), spec),
        "block_pool": (("block_pool",), block_pool),
        "enable_caching": (("enable_caching",), False),
        "kv_cache_group_id": (("kv_cache_group_id", "group_id"), 0),
        "scheduler_block_size": (("scheduler_block_size",), 1),
        "safety_reserve_slots": (
            (
                "safety_reserve_slots",
                "persistent_state_safety_reserve_slots",
                "safety_reserve",
                "reserve_slots",
            ),
            safety_reserve_slots,
        ),
        "max_resident_sessions": (
            (
                "max_resident_sessions",
                "configured_max_resident_sessions",
                "max_sessions",
            ),
            max_resident_sessions,
        ),
    }
    parameters = _parameter_names(manager_cls)
    for key in ("safety_reserve_slots", "max_resident_sessions"):
        aliases = values[key][0]
        if not any(alias in parameters for alias in aliases):
            _fail(f"PersistentStateManager must accept {key!r}; reviewed aliases are {aliases!r}")
    manager = _call_contract(
        manager_cls,
        values,
        "PersistentStateManager",
        preserve_exceptions=preserve_exceptions,
    )
    return manager, block_pool, spec


def _capacity_values(manager: Any) -> tuple[int, int, int, int]:
    physical = _read_nested_value(
        manager,
        (
            "physical_capacity",
            "physical_real_slots",
            "physical_slots",
            "real_slots",
        ),
        "physical capacity",
    )
    safety = _read_nested_value(
        manager,
        (
            "safety_reserve_slots",
            "persistent_state_safety_reserve_slots",
            "safety_reserve",
            "reserve_slots",
        ),
        "safety reserve",
    )
    configured = _read_nested_value(
        manager,
        (
            "max_resident_sessions",
            "configured_max_resident_sessions",
            "configured_limit",
            "max_sessions",
        ),
        "configured resident-session limit",
    )
    effective = _read_nested_value(
        manager,
        (
            "effective_capacity",
            "effective_slots",
            "resident_capacity",
            "capacity",
        ),
        "effective capacity",
    )
    return int(physical), int(safety), int(configured), int(effective)


def test_capacity_reserves_null_block_and_clamps_effective_sessions(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """@spec PORT-STATE-004, PORT-STATE-009: capacity is physical, safe, and clamped."""

    module = require_persistent_state_module()
    manager, block_pool, _ = _construct_capacity_manager(
        module,
        num_gpu_blocks=6,
        safety_reserve_slots=2,
        max_resident_sessions=10,
    )
    physical, safety, configured, effective = _capacity_values(manager)

    assert block_pool.null_block.is_null
    assert block_pool.null_block.block_id == 0
    assert physical == 5
    assert safety == 2
    assert configured == 10
    assert effective == 3
    assert safety >= 0
    assert effective == min(physical - safety, configured)

    capped_manager, _, _ = _construct_capacity_manager(
        module,
        num_gpu_blocks=6,
        safety_reserve_slots=2,
        max_resident_sessions=2,
    )
    assert _capacity_values(capped_manager)[-1] == 2

    with caplog.at_level(logging.WARNING):
        over_cap_manager, _, _ = _construct_capacity_manager(
            module,
            num_gpu_blocks=6,
            safety_reserve_slots=2,
            max_resident_sessions=99,
        )
    over_cap_message = " ".join(record.getMessage() for record in caplog.records).lower()
    assert "clamp" in over_cap_message or "capacity" in over_cap_message
    assert _capacity_values(over_cap_manager)[-1] == 3

    with pytest.raises((ValueError, RuntimeError), match="(?i)zero|capacity|reserve"):
        _construct_capacity_manager(
            module,
            num_gpu_blocks=3,
            safety_reserve_slots=2,
            max_resident_sessions=10,
            preserve_exceptions=True,
        )
    with pytest.raises((ValueError, RuntimeError), match="(?i)negative|reserve"):
        _construct_capacity_manager(
            module,
            num_gpu_blocks=6,
            safety_reserve_slots=-1,
            max_resident_sessions=10,
            preserve_exceptions=True,
        )


def _storage_parts(storage: Any) -> tuple[torch.Tensor, Any]:
    if isinstance(storage, (tuple, list)) and len(storage) == 2:
        raw, views = storage
    else:
        raw = _read_value(
            storage,
            ("raw", "raw_storage", "buffer", "storage", "aggregate"),
            "raw persistent-state allocation",
        )
        views = _read_value(
            storage,
            ("views", "typed_views", "component_views", "descriptor_views"),
            "typed persistent-state views",
        )
    if not isinstance(raw, torch.Tensor):
        _fail("raw persistent-state allocation must be a torch.Tensor")
    return raw, views


def _descriptor_view(views: Any, descriptor: Any, index: int, slot: int) -> torch.Tensor:
    name = descriptor_field(descriptor, "name", "semantic_name", "state_name")
    candidates: list[Any] = []
    if isinstance(views, Mapping):
        for key in (name, index, str(index)):
            if key in views:
                candidates.append(views[key])
    elif isinstance(views, Sequence) and not isinstance(views, (str, bytes)):
        candidates.extend(views)
    if not candidates:
        _fail(f"typed views must contain descriptor {name!r}")
    view = candidates[0] if isinstance(views, Mapping) else candidates[index]
    shape = tuple(descriptor_field(descriptor, "shape"))
    if tuple(view.shape) == shape:
        return cast(torch.Tensor, view)
    if len(view.shape) == len(shape) + 1 and tuple(view.shape[1:]) == shape:
        return cast(torch.Tensor, view[slot])
    _fail(f"typed view for {name!r} has shape {tuple(view.shape)!r}; expected {shape!r} or a leading slot dimension")


def _storage_initializer(module: ModuleType, storage: Any) -> Callable[..., Any]:
    names = PROVISIONAL_SURFACE["storage_initialize"]
    for name in names:
        candidate = getattr(storage, name, None)
        if callable(candidate):
            return cast(Callable[..., Any], candidate)
    return _surface_callable(module, "storage_initialize")


def _initialize_storage_slot(
    module: ModuleType,
    storage: Any,
    raw: torch.Tensor,
    spec: Any,
    *,
    slot: int,
    generation: int,
) -> Any:
    initializer = _storage_initializer(module, storage)
    views = _read_value(
        storage,
        ("views", "typed_views", "component_views", "descriptor_views"),
        "typed persistent-state views",
        default=None,
    )
    return _call_contract(
        initializer,
        {
            "storage": (("storage", "allocation", "state_storage"), storage),
            "raw": (("raw", "raw_storage", "buffer", "aggregate"), raw),
            "views": (("views", "typed_views", "component_views"), views),
            "spec": (("spec", "persistent_state_spec", "state_spec"), spec),
            "slot": (("slot", "slot_index", "block_id", "physical_slot_id"), slot),
            "generation": (("generation", "slot_generation"), generation),
            "fresh": (("fresh", "fresh_init", "initialize_fresh"), True),
        },
        "persistent-state fresh-slot initializer",
    )


def test_raw_aggregate_views_cover_page_and_fresh_init_erases_padding() -> None:
    """@spec PORT-STATE-003, PORT-STATE-009, PORT-STATE-011: typed views and fresh generations cover every byte."""

    module = require_persistent_state_module()
    spec = make_generic_spec(module)
    allocator = _surface_callable(module, "storage_allocate")
    storage = _call_contract(
        allocator,
        {
            "spec": (("spec", "persistent_state_spec", "state_spec"), spec),
            "num_slots": (("num_slots", "slot_count", "num_blocks"), 2),
            "device": (("device",), torch.device("cpu")),
        },
        "raw persistent-state allocator",
    )
    raw, views = _storage_parts(storage)
    assert raw.dtype == torch.uint8
    assert raw.is_contiguous()
    raw_bytes = raw.reshape(-1)
    page_size = int(spec_field(spec, "page_size_bytes", "page_size"))
    assert raw_bytes.numel() == 2 * page_size

    descriptors = tuple(spec_field(spec, "descriptors", "tensor_descriptors", "layout"))
    intervals: list[tuple[int, int]] = []
    for index, descriptor in enumerate(descriptors):
        view = _descriptor_view(views, descriptor, index, slot=0)
        shape = tuple(descriptor_field(descriptor, "shape"))
        dtype = descriptor_field(descriptor, "dtype")
        offset = int(descriptor_field(descriptor, "offset_bytes", "byte_offset", "offset"))
        size = math.prod(shape) * torch.empty((), dtype=dtype).element_size()
        assert view.dtype == dtype
        assert tuple(view.shape) == shape
        assert view.numel() * view.element_size() == size
        assert view.data_ptr() == raw_bytes.data_ptr() + offset
        intervals.append((offset, offset + size))
    assert all(0 <= start < end <= page_size for start, end in intervals)
    assert all(right <= next_left for (_, right), (next_left, _) in zip(sorted(intervals), sorted(intervals)[1:]))

    sorted_intervals = sorted(intervals)
    padding_intervals: list[tuple[int, int]] = []
    cursor = 0
    for start, end in sorted_intervals:
        if cursor < start:
            padding_intervals.append((cursor, start))
        cursor = end
    if cursor < page_size:
        padding_intervals.append((cursor, page_size))

    raw_bytes[:page_size].fill_(0xA5)
    _initialize_storage_slot(module, storage, raw, spec, slot=0, generation=1)
    for index, descriptor in enumerate(descriptors):
        view = _descriptor_view(views, descriptor, index, slot=0)
        assert torch.equal(view, torch.zeros_like(view))
    for start, end in padding_intervals:
        assert torch.count_nonzero(raw_bytes[start:end]) == 0

    raw_bytes[:page_size].fill_(0x7C)
    _initialize_storage_slot(module, storage, raw, spec, slot=0, generation=2)
    assert torch.count_nonzero(raw_bytes[:page_size] == 0x7C) == 0
    for index, descriptor in enumerate(descriptors):
        view = _descriptor_view(views, descriptor, index, slot=0)
        assert torch.equal(view, torch.zeros_like(view))


def _binding_for(manager: Any, request_id: str) -> Any:
    for name in (
        "get_state_binding",
        "get_persistent_state_binding",
        "get_binding",
        "binding_for",
    ):
        method = getattr(manager, name, None)
        if callable(method):
            binding = method(request_id)
            if binding is None:
                _fail(f"manager binding accessor returned no binding for {request_id!r}")
            return binding
    _fail("PersistentStateManager must expose a reviewed state-binding accessor")


def _binding_field(binding: Any, names: Sequence[str], label: str) -> Any:
    return _read_value(binding, names, f"manager-issued {label}")


def _manager_fallback(manager: Any, names: Sequence[str]) -> Any:
    return _try_read_value(manager, names)


def _live_row(manager: Any, request_id: str, order: int, spec: Any) -> dict[str, Any]:
    binding = _binding_for(manager, request_id)
    binding_cls = _surface_class(require_persistent_state_module(), "binding_class")
    if not isinstance(binding, binding_cls):
        _fail("manager-issued binding must use the typed StateBinding class")
    schema = _try_read_value(binding, ("schema_id", "state_schema_id", "schema"))
    if schema is _MISSING:
        schema = spec_field(spec, "schema_id", "state_schema_id", "schema")
    profile = _try_read_value(
        binding,
        ("profile_id", "execution_profile_id", "profile", "execution_profile"),
    )
    if profile is _MISSING:
        profile = _manager_fallback(
            manager,
            ("profile_id", "execution_profile_id", "profile", "execution_profile"),
        )
    if profile is _MISSING:
        _fail("manager-issued binding must expose a profile identity")
    stage = _try_read_value(binding, ("stage", "stage_id", "pipeline_stage"))
    if stage is _MISSING:
        stage = _manager_fallback(manager, ("stage", "stage_id", "pipeline_stage"))
    if stage is _MISSING:
        _fail("manager-issued binding must expose a stage identity")
    replica = _try_read_value(binding, ("replica", "replica_id", "replica_rank"))
    if replica is _MISSING:
        replica = _manager_fallback(manager, ("replica", "replica_id", "replica_rank"))
    if replica is _MISSING:
        _fail("manager-issued binding must expose a replica identity")
    return {
        "binding": binding,
        "request_id": _binding_field(binding, ("request_id", "session_id"), "request"),
        "generation": _binding_field(binding, ("generation", "slot_generation"), "generation"),
        "schema_id": schema,
        "profile_id": profile,
        "stage": stage,
        "replica": replica,
        "slot_id": _binding_field(binding, ("slot_id", "physical_slot_id", "block_id"), "physical slot"),
        "fresh": _binding_field(binding, ("fresh", "fresh_init"), "fresh flag"),
        "order": order,
        "role": "resident",
        "no_state": False,
        "request_key": request_id,
    }


def _dummy_row(order: int) -> dict[str, Any]:
    return {
        "binding": None,
        "request_id": None,
        "generation": None,
        "schema_id": None,
        "profile_id": None,
        "stage": None,
        "replica": None,
        "slot_id": None,
        "fresh": False,
        "order": order,
        "role": "no_state",
        "no_state": True,
        "request_key": None,
    }


def _make_batch(
    module: ModuleType,
    rows: Sequence[Mapping[str, Any]],
    *,
    preserve_exceptions: bool = False,
) -> Any:
    batch_cls = _surface_class(module, "batch_class")
    materialized = tuple(dict(row) for row in rows)
    bindings = tuple(row["binding"] for row in materialized)
    values = {
        "rows": (("rows", "state_rows", "batch_rows"), materialized),
        "bindings": (("bindings", "state_bindings", "live_bindings"), bindings),
        "request_ids": (
            ("request_ids", "requests", "row_request_ids"),
            tuple(row["request_id"] for row in materialized),
        ),
        "generations": (
            ("generations", "row_generations", "slot_generations"),
            tuple(row["generation"] for row in materialized),
        ),
        "schema_ids": (
            ("schema_ids", "row_schema_ids", "schemas"),
            tuple(row["schema_id"] for row in materialized),
        ),
        "profile_ids": (
            ("profile_ids", "row_profile_ids", "profiles"),
            tuple(row["profile_id"] for row in materialized),
        ),
        "stages": (
            ("stages", "stage_ids", "row_stages"),
            tuple(row["stage"] for row in materialized),
        ),
        "replicas": (
            ("replicas", "replica_ids", "row_replicas"),
            tuple(row["replica"] for row in materialized),
        ),
        "slot_ids": (
            ("slot_ids", "block_ids", "physical_slot_ids", "row_slot_ids"),
            tuple(row["slot_id"] for row in materialized),
        ),
        "fresh_flags": (
            ("fresh_flags", "fresh", "row_fresh_flags"),
            tuple(row["fresh"] for row in materialized),
        ),
        "roles": (
            ("roles", "row_roles", "state_roles"),
            tuple(row["role"] for row in materialized),
        ),
        "no_state": (
            ("no_state", "no_state_rows", "row_is_no_state"),
            tuple(row["no_state"] for row in materialized),
        ),
        "row_order": (("row_order", "order", "indices"), tuple(range(len(rows)))),
    }
    return _call_contract(
        batch_cls,
        values,
        "PersistentStateBatch",
        preserve_exceptions=preserve_exceptions,
    )


def _batch_operation(
    module: ModuleType,
    batch: Any,
    manager: Any,
    key: str,
    extra: Mapping[str, tuple[Sequence[str], Any]] | None = None,
    *,
    preserve_exceptions: bool = False,
) -> Any:
    method_names = {
        "batch_validate": ("validate", "preflight_validate", "validate_against"),
        "batch_gather": ("gather", "gather_state", "gather_persistent_state"),
        "batch_scatter": ("scatter", "scatter_state", "scatter_persistent_state"),
    }
    method = next(
        (getattr(batch, name, None) for name in method_names[key]),
        None,
    )
    if not callable(method):
        method = _surface_callable(module, key)
    values: dict[str, tuple[Sequence[str], Any]] = {
        "batch": (("batch", "state_batch", "persistent_state_batch"), batch),
        "manager": (("manager", "state_manager", "persistent_state_manager"), manager),
    }
    if extra:
        values.update(extra)
    return _call_contract(
        method,
        values,
        f"PersistentStateBatch {key}",
        preserve_exceptions=preserve_exceptions,
    )


def _clone_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [dict(row) for row in rows]


def test_typed_batch_validates_mixed_rows_before_gather_and_scatter() -> None:
    """@spec PORT-STATE-007, PORT-STATE-008, PORT-STATE-011: typed row validation gates ordered gather/scatter."""

    module = require_persistent_state_module()

    manager, block_pool, spec = make_manager(module, num_gpu_blocks=5)
    for request_id in ("request-1", "request-2", "request-3"):
        manager.allocate_new_blocks(request_id, 1, 1)
    rows = [
        _live_row(manager, "request-1", 0, spec),
        _dummy_row(1),
        _live_row(manager, "request-2", 2, spec),
        _live_row(manager, "request-3", 3, spec),
    ]
    batch = _make_batch(module, rows)
    _batch_operation(module, batch, manager, "batch_validate")

    gather_calls: list[Any] = []

    def gatherer(*args: Any, **kwargs: Any) -> Any:
        values = list(args) + list(kwargs.values())
        request_id = _MISSING
        for value in values:
            if isinstance(value, Mapping):
                request_id = _try_read_value(value, ("request_id", "request"))
            else:
                request_id = _try_read_value(value, ("request_id", "session_id"))
            if request_id is not _MISSING:
                break
        if request_id is _MISSING:
            _fail("batch gatherer must receive the typed live-row request identity")
        gather_calls.append(request_id)
        return {"request_id": request_id}

    _batch_operation(
        module,
        batch,
        manager,
        "batch_gather",
        {
            "gatherer": (("gatherer", "read_row", "row_reader", "reader"), gatherer),
        },
    )
    assert gather_calls == ["request-1", "request-2", "request-3"]

    bad_rows: list[tuple[str, list[dict[str, Any]]]] = []
    for field, bad_value in (
        ("request_id", "wrong-request"),
        ("generation", 10**9),
        ("schema_id", "wrong-schema"),
        ("profile_id", "wrong-profile"),
        ("stage", "wrong-stage"),
        ("replica", "wrong-replica"),
        ("slot_id", block_pool.num_gpu_blocks + 1),
        ("fresh", False),
    ):
        mutated = _clone_rows(rows)
        mutated[0][field] = bad_value
        bad_rows.append((field, mutated))

    reordered = _clone_rows(rows)
    reordered[0], reordered[2] = reordered[2], reordered[0]
    bad_rows.append(("row_order", reordered))
    duplicate_slot = _clone_rows(rows)
    duplicate_slot[3]["slot_id"] = duplicate_slot[0]["slot_id"]
    bad_rows.append(("duplicate_live_slot", duplicate_slot))
    malformed_dummy = _clone_rows(rows)
    malformed_dummy[1]["no_state"] = False
    malformed_dummy[1]["role"] = "resident"
    bad_rows.append(("dummy_role", malformed_dummy))

    for label, mutated in bad_rows:
        try:
            invalid_batch = _make_batch(module, mutated, preserve_exceptions=True)
        except (AssertionError, TypeError, ValueError, RuntimeError):
            continue
        with pytest.raises((AssertionError, TypeError, ValueError, RuntimeError)):
            _batch_operation(
                module,
                invalid_batch,
                manager,
                "batch_validate",
                preserve_exceptions=True,
            )
        invalid_gather_calls: list[Any] = []

        def invalid_gatherer(*args: Any, **kwargs: Any) -> Any:
            invalid_gather_calls.append((args, kwargs))
            return None

        with pytest.raises((AssertionError, TypeError, ValueError, RuntimeError)):
            _batch_operation(
                module,
                invalid_batch,
                manager,
                "batch_gather",
                {
                    "gatherer": (
                        ("gatherer", "read_row", "row_reader", "reader"),
                        invalid_gatherer,
                    ),
                },
                preserve_exceptions=True,
            )
        assert not invalid_gather_calls, label

    scatter_calls: list[Any] = []

    def scatterer(*args: Any, **kwargs: Any) -> None:
        state = kwargs.get("state") or kwargs.get("row_state") or kwargs.get("output")
        if state is None:
            state = next((arg for arg in args if isinstance(arg, Mapping)), None)
        if isinstance(state, Mapping) and "row" in state:
            scatter_calls.append(state["row"])
        else:
            scatter_calls.append(state)

    _batch_operation(
        module,
        batch,
        manager,
        "batch_scatter",
        {
            "row_states": (
                ("row_states", "states", "outputs", "emissions", "working_states"),
                ({"row": "first"}, None, {"row": "failed"}, {"row": "last"}),
            ),
            "row_statuses": (
                ("row_statuses", "statuses", "successes", "results"),
                ("clean", "no_state", "failed", "clean"),
            ),
            "scatterer": (
                ("scatterer", "scatter_row", "commit_row", "writer"),
                scatterer,
            ),
        },
    )
    assert scatter_calls == ["first", "last"]
