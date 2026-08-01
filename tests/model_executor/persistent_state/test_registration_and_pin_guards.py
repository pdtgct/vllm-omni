# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Registration, profile, and vLLM-pin guards for the P3 substrate."""

from __future__ import annotations

import ast
import hashlib
import inspect
import textwrap
from pathlib import Path
from typing import Any

import pytest
import torch
from vllm.model_executor.models.config import MODELS_CONFIG_MAP
from vllm.v1.core.single_type_kv_cache_manager import (
    FullAttentionManager,
    register_all_kvcache_specs,
)
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    UniformTypeKVCacheSpecs,
)
from vllm.v1.kv_cache_spec_registry import KVCacheSpecRegistry
from vllm.v1.worker.gpu.attn_utils import init_attn_backend
from vllm.v1.worker.gpu.model_runner import GPUModelRunner as GPUModelRunnerV2
from vllm.v1.worker.gpu_model_runner import GPUModelRunner as GPUModelRunnerV1
from vllm.v1.worker.utils import prepare_kernel_block_sizes

from tests.model_executor.persistent_state._helpers import (
    declaration,
    make_generic_spec,
    require_persistent_state_module,
    require_symbol,
    spec_field,
    validate_declarations,
    validate_execution_profile,
)


def test_platform_registration_runs_after_core_builtins_and_is_idempotent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """@spec PORT-STATE-002: platform registration is ordered and repeat-safe."""

    module = require_persistent_state_module()
    spec_cls = require_symbol(module, "PersistentStateSpec")
    manager_cls = require_symbol(module, "PersistentStateManager")
    registry_module = inspect.getmodule(KVCacheSpecRegistry)
    assert registry_module is not None

    monkeypatch.setattr(registry_module, "_REGISTRY_KVCACHESPEC_LIST", {})
    events: list[type[Any]] = []
    original_register = KVCacheSpecRegistry.register

    def record_registration(
        kvcache_spec_cls: type[Any],
        manager_class: Any = None,
        uniform_type_base_spec: Any = None,
    ) -> None:
        events.append(kvcache_spec_cls)
        original_register(kvcache_spec_cls, manager_class, uniform_type_base_spec)

    monkeypatch.setattr(KVCacheSpecRegistry, "register", record_registration)
    from vllm import platforms as vllm_platforms

    from vllm_omni.platforms import current_omni_platform

    monkeypatch.setattr(vllm_platforms, "current_platform", current_omni_platform)
    before_models = dict(MODELS_CONFIG_MAP)

    register_all_kvcache_specs(None)

    registry = registry_module._REGISTRY_KVCACHESPEC_LIST
    assert spec_cls in registry
    assert registry[spec_cls].manager_class is manager_cls
    assert registry[spec_cls].uniform_type_base_spec is spec_cls

    builtin_positions = [
        index
        for index, registered_cls in enumerate(events)
        if registered_cls is FullAttentionSpec
    ]
    custom_positions = [
        index for index, registered_cls in enumerate(events) if registered_cls is spec_cls
    ]
    assert builtin_positions
    assert custom_positions
    assert min(custom_positions) > max(builtin_positions)

    metadata_before_repeat = registry[spec_cls]
    current_omni_platform.register_custom_kv_cache_specs(None)
    current_omni_platform.register_custom_kv_cache_specs(None)
    assert registry[spec_cls] == metadata_before_repeat
    assert dict(MODELS_CONFIG_MAP) == before_models

    with pytest.raises(AssertionError, match="Conflicting registration"):
        KVCacheSpecRegistry.register(
            spec_cls,
            FullAttentionManager,
            uniform_type_base_spec=spec_cls,
        )


def test_persistent_profile_rejects_non_singleton_topology_and_pools() -> None:
    """@spec PORT-STATE-018: only the initial single-GPU profile is admitted."""

    module = require_persistent_state_module()
    profile = {
        "stage_count": 1,
        "replica_count": 1,
        "gpu_count": 1,
        "tp": 1,
        "pp": 1,
        "dcp": 1,
        "pcp": 1,
        "dp": 1,
        "persistent_schema_ids": ("schema-a",),
        "model_ids": ("model-a",),
    }
    validate_execution_profile(module, profile)

    for field in ("stage_count", "replica_count", "gpu_count", "tp", "pp", "dcp", "pcp", "dp"):
        invalid = dict(profile)
        invalid[field] = 2
        with pytest.raises(Exception):
            validate_execution_profile(module, invalid)

    mixed_schema = dict(profile)
    mixed_schema["persistent_schema_ids"] = ("schema-a", "schema-b")
    with pytest.raises(Exception):
        validate_execution_profile(module, mixed_schema)

    multi_model = dict(profile)
    multi_model["model_ids"] = ("model-a", "model-b")
    with pytest.raises(Exception):
        validate_execution_profile(module, multi_model)


def test_profile_rejects_unknown_wrapped_mixed_and_multiple_state_declarations() -> None:
    """@spec PORT-STATE-002: declaration validation fails before allocation."""

    module = require_persistent_state_module()
    spec = make_generic_spec(module)
    shifted = make_generic_spec(module, variant="shifted")
    schema_id = spec_field(spec, "schema_id", "state_schema_id", "schema")
    shifted_schema_id = spec_field(shifted, "schema_id", "state_schema_id", "schema")

    validate_declarations(module, [declaration(spec, "persistent_state", schema_id)])

    wrapped = UniformTypeKVCacheSpecs(block_size=1, kv_cache_specs={"inner": spec})
    with pytest.raises(Exception):
        validate_declarations(
            module,
            [declaration(wrapped, "persistent_state", schema_id)],
        )

    unknown = FullAttentionSpec(
        block_size=1,
        num_kv_heads=1,
        head_size=1,
        dtype=torch.float32,
    )
    with pytest.raises(Exception):
        validate_declarations(module, [declaration(unknown, "attention", schema_id)])

    with pytest.raises(Exception):
        validate_declarations(
            module,
            [
                declaration(spec, "state-a", schema_id),
                declaration(shifted, "state-b", shifted_schema_id),
            ],
        )

    with pytest.raises(Exception):
        validate_declarations(
            module,
            [
                declaration(spec, "state-a", schema_id),
                declaration(spec, "state-b", schema_id),
            ],
        )


def _function_node(function: Any) -> ast.FunctionDef | ast.AsyncFunctionDef:
    source = textwrap.dedent(inspect.getsource(function))
    tree = ast.parse(source)
    node = tree.body[0]
    assert isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    return node


def _ast_digest(function: Any) -> str:
    node = _function_node(function)
    return hashlib.sha256(ast.dump(node, include_attributes=False).encode()).hexdigest()


def _called_names(function: Any) -> set[str]:
    names: set[str] = set()
    for node in ast.walk(_function_node(function)):
        if not isinstance(node, ast.Call):
            continue
        if isinstance(node.func, ast.Name):
            names.add(node.func.id)
        elif isinstance(node.func, ast.Attribute):
            names.add(node.func.attr)
    return names


def test_vllm_v025_pin_guard_covers_state_sensitive_core_seams() -> None:
    """@spec PORT-MIG-006: core source drift blocks re-qualification."""

    expected_digests = {
        "v1.get_kv_cache_spec": "d3c876929690fd2da697a26459cf571e5ceab46ca730b66258b3f3d6b8dd622a",
        "v1.initialize_kv_cache": "b72595b866e97bccb6abbe8be35c9ca11881b9907f10615ac4c1d1f66dd37e22",
        "v1._reshape_kv_cache_tensors": "b1b1de1461b370909694d2651f46ab5e67f5c0ae12a7df55daa844d9cf1f9825",
        "v2.get_kv_cache_spec": "bcd0fd78fc1d097fedcc6186d31e20eb5aa668f16fa0f1e84e8ada4d75be82c6",
        "v2.initialize_kv_cache": "122f31eea583d0739475e80b2baf8a83471fb9562b3da55f8c1be9801900b40e",
        "prepare_kernel_block_sizes": "a4c432e936764f26da4ad6265583424e4e2ce61dc3b816783a5c27a0d62b402d",
        "v1.initialize_attn_backend": "74264253a816755eedd814f51c6bc191f4aff5abc50f989ad941140be9cc2b39",
        "v2.init_attn_backend": "0cbad5ff9617912114bc6e83b3f5e9ad6c8fcb5814daf514d9f54a52a8ca5199",
    }
    functions = {
        "v1.get_kv_cache_spec": GPUModelRunnerV1.get_kv_cache_spec,
        "v1.initialize_kv_cache": GPUModelRunnerV1.initialize_kv_cache,
        "v1._reshape_kv_cache_tensors": GPUModelRunnerV1._reshape_kv_cache_tensors,
        "v2.get_kv_cache_spec": GPUModelRunnerV2.get_kv_cache_spec,
        "v2.initialize_kv_cache": GPUModelRunnerV2.initialize_kv_cache,
        "prepare_kernel_block_sizes": prepare_kernel_block_sizes,
        "v1.initialize_attn_backend": GPUModelRunnerV1.initialize_attn_backend,
        "v2.init_attn_backend": init_attn_backend,
    }
    assert set(functions) == set(expected_digests)
    for name, function in functions.items():
        assert _ast_digest(function) == expected_digests[name], name

    assert {"get_kv_cache_spec", "replace"} <= _called_names(GPUModelRunnerV1.get_kv_cache_spec)
    assert {
        "initialize_attn_backend",
        "prepare_kernel_block_sizes",
        "initialize_kv_cache_tensors",
    } <= _called_names(GPUModelRunnerV1.initialize_kv_cache)
    assert {"_reshape_attention_kv_cache", "isinstance"} <= _called_names(
        GPUModelRunnerV1._reshape_kv_cache_tensors
    )
    assert {"init_attn_backend", "init_kv_cache"} <= _called_names(
        GPUModelRunnerV2.initialize_kv_cache
    )
    assert {"select_common_block_size", "NotImplementedError"} <= _called_names(
        prepare_kernel_block_sizes
    )
    assert {"prepare_kernel_block_sizes", "AttentionGroup"} <= _called_names(
        init_attn_backend
    )


def test_omni_sources_do_not_mutate_core_models_config_map() -> None:
    """@spec PORT-MIG-006: the migration has no MODELS_CONFIG_MAP mutation."""

    repo_root = Path(__file__).resolve().parents[3]
    mutations: list[str] = []
    for path in sorted((repo_root / "vllm_omni").rglob("*.py")):
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            target_nodes: list[ast.AST] = []
            if isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign, ast.Delete)):
                if isinstance(node, ast.Assign):
                    target_nodes.extend(node.targets)
                elif isinstance(node, ast.Delete):
                    target_nodes.extend(node.targets)
                else:
                    target_nodes.append(node.target)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                if node.func.attr in {"update", "clear", "pop", "setdefault", "__setitem__"}:
                    target_nodes.append(node.func.value)
            for target in target_nodes:
                if any(
                    isinstance(name, ast.Name) and name.id == "MODELS_CONFIG_MAP"
                    for name in ast.walk(target)
                ):
                    mutations.append(f"{path}:{getattr(node, 'lineno', '?')}")
    assert not mutations, mutations


def test_persistent_spec_has_no_mamba_identity() -> None:
    """@spec PORT-MIG-006: the native spec never inherits MambaSpec."""

    module = require_persistent_state_module()
    spec_cls = require_symbol(module, "PersistentStateSpec")
    from vllm.v1.kv_cache_interface import MambaSpec

    assert not issubclass(spec_cls, MambaSpec)
    assert all(base is not MambaSpec for base in spec_cls.__mro__)
