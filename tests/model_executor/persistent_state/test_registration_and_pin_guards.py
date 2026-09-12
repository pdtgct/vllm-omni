# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Registration, profile, and vLLM-pin guards for the P3 substrate."""

from __future__ import annotations

import ast
import hashlib
import inspect
import textwrap
from pathlib import Path
from types import SimpleNamespace
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
from vllm.v1.worker.utils import allocate_kv_cache, prepare_kernel_block_sizes

from tests.model_executor.persistent_state._helpers import (
    declaration,
    make_generic_spec,
    require_persistent_state_module,
    require_symbol,
    spec_field,
    validate_declarations,
    validate_execution_profile,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


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

    import vllm_omni.platforms as omni_platforms
    from vllm_omni.engine.stage_engine_core_proc import (
        _install_omni_platform_for_stage_core,
    )

    resolved_omni_platform = SimpleNamespace(
        is_unspecified=lambda: False,
        register_custom_kv_cache_specs=(omni_platforms.OmniPlatform.register_custom_kv_cache_specs),
    )
    monkeypatch.setattr(
        omni_platforms,
        "_current_omni_platform",
        resolved_omni_platform,
    )
    monkeypatch.setattr(vllm_platforms, "current_platform", object())
    _install_omni_platform_for_stage_core()
    assert vllm_platforms.current_platform is resolved_omni_platform
    before_models = dict(MODELS_CONFIG_MAP)

    register_all_kvcache_specs(None)

    registry = registry_module._REGISTRY_KVCACHESPEC_LIST
    assert spec_cls in registry
    assert registry[spec_cls].manager_class is manager_cls
    assert registry[spec_cls].uniform_type_base_spec is spec_cls

    builtin_positions = [index for index, registered_cls in enumerate(events) if registered_cls is FullAttentionSpec]
    custom_positions = [index for index, registered_cls in enumerate(events) if registered_cls is spec_cls]
    assert builtin_positions
    assert custom_positions
    assert min(custom_positions) > max(builtin_positions)

    metadata_before_repeat = registry[spec_cls]
    resolved_omni_platform.register_custom_kv_cache_specs(None)
    resolved_omni_platform.register_custom_kv_cache_specs(None)
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


def _source_digest(function: Any) -> str:
    source = textwrap.dedent(inspect.getsource(function)).strip() + "\n"
    return hashlib.sha256(source.encode()).hexdigest()


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


def test_vllm_v029_source_guard_covers_state_sensitive_core_seams() -> None:
    """@spec PORT-MIG-006: core source drift blocks re-qualification.

    Audited against v0.29.0 source 98dff2a81d747d1dba01a47f939f48c3526d4206.
    This is a source compatibility guard, not installed-wheel/GPU qualification.
    V1's removed reshape hook is now the shared byte-layout allocator, reached
    through initialize_kv_cache_tensors. Pin both ends of that replacement.
    """

    expected_digests = {
        "v1.get_kv_cache_spec": "9942da0efdf23c20af79d0707fddf0712e71dfe8884ce1d2b3019a235fb40bd9",
        "v1.initialize_kv_cache": "abf417d0b3bcef6ef61aa373bb6469b61c5ec6854086d27dd6ffcfa98e6a2b66",
        "v1.initialize_kv_cache_tensors": "97b72e9f6a9d1ee2b67d53bc24308360bb4019a97b098f26897e15aa5dd6ebde",
        "v2.get_kv_cache_spec": "9307d484402f25986a64c645b9da713939e698e71595b46c4338b40c10c0620b",
        "v2.initialize_kv_cache": "fcb8388b0a24e9b81a49d27e25f96a090eddd27f29da914c7d477cb58a8896f5",
        "prepare_kernel_block_sizes": "2fe8d5c03098041cfe7e4c5b6320d290a41c8d110acbc9f4f97084f2e7be891d",
        "v1.initialize_attn_backend": "e7446cc68d31de806a7edc9adc99746ed3edc0908604e7a0d41c7754b4e5e465",
        "v2.init_attn_backend": "07074ed6cc04d6c9aeb542dba5991099bde0c211e298c3860fad4d0c297de80f",
    }
    expected_digests["allocate_kv_cache"] = "567365d6ba1f8cfbf8e87d070e116d81984bc4a03ab7bd50517345e4e16363a2"
    functions = {
        "v1.get_kv_cache_spec": GPUModelRunnerV1.get_kv_cache_spec,
        "v1.initialize_kv_cache": GPUModelRunnerV1.initialize_kv_cache,
        "v1.initialize_kv_cache_tensors": GPUModelRunnerV1.initialize_kv_cache_tensors,
        "v2.get_kv_cache_spec": GPUModelRunnerV2.get_kv_cache_spec,
        "v2.initialize_kv_cache": GPUModelRunnerV2.initialize_kv_cache,
        "prepare_kernel_block_sizes": prepare_kernel_block_sizes,
        "v1.initialize_attn_backend": GPUModelRunnerV1.initialize_attn_backend,
        "v2.init_attn_backend": init_attn_backend,
        "allocate_kv_cache": allocate_kv_cache,
    }
    assert set(functions) == set(expected_digests)
    for name, function in functions.items():
        assert _source_digest(function) == expected_digests[name], name

    # Discovery still visits attention layers; v0.29 asks the backend to
    # customize attention specs instead of replacing their block size inline.
    assert {"get_kv_cache_spec", "customize_spec"} <= _called_names(GPUModelRunnerV1.get_kv_cache_spec)
    assert {
        "initialize_attn_backend",
        "prepare_kernel_block_sizes",
        "initialize_kv_cache_tensors",
    } <= _called_names(GPUModelRunnerV1.initialize_kv_cache)
    assert {"allocate_kv_cache", "bind_kv_cache"} <= _called_names(GPUModelRunnerV1.initialize_kv_cache_tensors)
    assert {"create_kv_cache_views", "zeros"} <= _called_names(allocate_kv_cache)
    # V2 now accepts profiling/allocation context and sizes tables through
    # the spec contract; persistent partitioning must precede this path.
    assert {"init_attn_backend", "init_kv_cache", "max_num_blocks_per_req"} <= _called_names(
        GPUModelRunnerV2.initialize_kv_cache
    )
    for runner in (GPUModelRunnerV1, GPUModelRunnerV2):
        assert {"is_profiling", "kv_cache_allocation_context"} <= set(
            inspect.signature(runner.initialize_kv_cache).parameters
        )
    assert {"select_common_block_size", "NotImplementedError"} <= _called_names(prepare_kernel_block_sizes)
    assert {"prepare_kernel_block_sizes", "AttentionGroup"} <= _called_names(init_attn_backend)


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
                if any(isinstance(name, ast.Name) and name.id == "MODELS_CONFIG_MAP" for name in ast.walk(target)):
                    mutations.append(f"{path}:{getattr(node, 'lineno', '?')}")
    assert not mutations, mutations


def test_persistent_spec_has_no_mamba_identity() -> None:
    """@spec PORT-MIG-006: the native spec never inherits MambaSpec."""

    module = require_persistent_state_module()
    spec_cls = require_symbol(module, "PersistentStateSpec")
    from vllm.v1.kv_cache_interface import MambaSpec

    assert not issubclass(spec_cls, MambaSpec)
    assert all(base is not MambaSpec for base in spec_cls.__mro__)
