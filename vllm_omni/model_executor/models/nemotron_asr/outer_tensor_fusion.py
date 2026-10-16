# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Finite, model-owned execution of the two outer transaction predicates."""

from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
from importlib.metadata import PackageNotFoundError, version
from types import FunctionType, MappingProxyType
from typing import Any

import torch

from vllm_omni.model_executor.models.nemotron_asr.manifests import CADENCES, FRONTEND_CONSTANTS, SESSION_LIMITS

FAMILIES = ("pre", "post-control", "post-g0", "post-g1", "post-g2", "post-g3", "post-g4")
PROFILE = "fp32-sync-single-device-five-cadence-v1"
_PRE_NAMES = (
    "input_ids",
    "is_chunk",
    "plan_geometry",
    "prior_prompt",
    "queue",
    "book",
    "frontend_counters",
    "cadence_frames",
)
_POST_NAMES = (
    "token_ids",
    "token_lengths",
    "roles",
    "chunk_rows",
    "baseline_queue",
    "baseline_book",
    "prompt_index",
    "incoming_status",
    "proposed_rows",
    "proposed_queue",
    "proposed_book",
    "proposed_status",
    "plan_geometry",
)
_PRE_CONSTANTS = (
    "queue_capacity",
    "blank_id",
    "placeholder_id",
    "park_id",
    "eou_token_id",
    "raw_tail_capacity",
    "mel_tail_capacity",
    "hop_length",
    "n_fft",
)
_POST_CONSTANTS = ("park_id", "blank_id", "eou_token_id")


def model_constants(config: Any) -> dict[str, int]:
    from vllm_omni.model_executor.models.nemotron_asr.rnnt import MAX_SYMBOLS_PER_STEP

    return dict(
        blank_id=int(config.num_asr_labels),
        placeholder_id=int(config.audio_chunk_token_id),
        park_id=int(config.eos_token_id),
        eou_token_id=int(config.eou_token_id),
        queue_capacity=SESSION_LIMITS["queue_capacity"],
        raw_tail_capacity=FRONTEND_CONSTANTS["raw_tail_capacity"],
        mel_tail_capacity=FRONTEND_CONSTANTS["pre_encode_cache_frames"],
        hop_length=FRONTEND_CONSTANTS["hop_length"],
        n_fft=FRONTEND_CONSTANTS["n_fft"],
        carrier_width=int(config.hidden_size),
        max_symbols_per_step=MAX_SYMBOLS_PER_STEP,
    )


# @spec PORT-PERF-012, PORT-PERF-013
def validate_selection(runtime: Any, config: Any, policy: Any) -> bool:
    enabled = getattr(config, "experimental_outer_tensor_fusion", False)
    if type(enabled) is not bool:
        raise ValueError("experimental_outer_tensor_fusion must be a boolean")
    if not enabled:
        return False
    from vllm_omni.model_executor.models.nemotron_asr.batch_invariance import installed_batch_invariant_mode
    from vllm_omni.model_executor.models.nemotron_asr.native_burst import validate_native_burst_config
    from vllm_omni.model_executor.models.nemotron_asr.precision import FP32_BRINGUP

    if installed_batch_invariant_mode().enabled:
        raise ValueError("outer tensor fusion requires batch invariance off")
    if (
        policy is not FP32_BRINGUP
        or runtime.model_config.dtype != torch.float32
        or runtime.scheduler_config.max_num_seqs != 1
        or getattr(config, "encoder_execution_arm", None) != "eager-graphed"
        or getattr(config, "decode_dispatch_arm", None) != "dense-graphed"
        or getattr(config, "experimental_native_burst", False) is not True
        or getattr(config, "experimental_native_burst_profile", None) != PROFILE
        or bool(getattr(config, "chunk_bucket_graph_populations", ()))
    ):
        raise ValueError("outer tensor fusion requires the FP32 singleton native five-cadence graph profile")
    validate_native_burst_config(runtime, hf_config=config, profile=PROFILE)
    return True


def _tensor_descriptors(names: tuple[str, ...], tensors: tuple[torch.Tensor, ...]) -> list[dict[str, Any]]:
    objects: dict[int, int] = {}
    storages: dict[int, int] = {}
    result = []
    for index, (name, tensor) in enumerate(zip(names, tensors, strict=True)):
        # StorageImpl identity preserves empty-storage and view equivalence; it
        # is canonicalized locally and never exported as an address or key.
        object_group = objects.setdefault(id(tensor), index)
        storage_group = storages.setdefault(tensor.untyped_storage()._cdata, index)
        result.append(
            dict(
                name=name,
                shape=list(tensor.shape),
                dtype=str(tensor.dtype).removeprefix("torch."),
                device=str(tensor.device),
                stride=list(tensor.stride()),
                storage_offset_bytes=tensor.storage_offset() * tensor.element_size(),
                requires_grad=tensor.requires_grad,
                object_group=object_group,
                storage_group=storage_group,
            )
        )
    return result


def _freeze(value: Any) -> Any:
    if isinstance(value, dict):
        return tuple((key, _freeze(item)) for key, item in value.items())
    if isinstance(value, list):
        return tuple(_freeze(item) for item in value)
    return value


# @spec PORT-PERF-015
def tensor_metadata_key(tensors: tuple[torch.Tensor, ...]) -> tuple[Any, ...]:
    return _freeze(_tensor_descriptors(tuple(str(i) for i in range(len(tensors))), tensors))


def _context() -> dict[str, Any]:
    from torch.overrides import _get_current_function_mode_stack
    from torch.utils._python_dispatch import _get_current_dispatch_mode_stack

    return dict(
        default_dtype=str(torch.get_default_dtype()).removeprefix("torch."),
        deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
        deterministic_warn_only=torch.is_deterministic_algorithms_warn_only_enabled(),
        grad_enabled=torch.is_grad_enabled(),
        inference_mode=torch.is_inference_mode_enabled(),
        autocast_cpu_enabled=torch.is_autocast_enabled("cpu"),
        autocast_cpu_dtype=str(torch.get_autocast_dtype("cpu")).removeprefix("torch."),
        autocast_cuda_enabled=torch.is_autocast_enabled("cuda"),
        autocast_cuda_dtype=str(torch.get_autocast_dtype("cuda")).removeprefix("torch."),
        dispatch_modes=[
            f"{type(mode).__module__}.{type(mode).__qualname__}"
            for mode in (*_get_current_dispatch_mode_stack(), *_get_current_function_mode_stack())
        ],
    )


# @spec PORT-PERF-015
def compiler_context_key() -> tuple[Any, ...]:
    return _freeze(_context())


@contextmanager
def _compiler_scope(*, sealed: bool):
    from vllm_omni.model_executor.models.nemotron_asr.encoder_execution import _COMPILER_BUDGET_LOCK

    with _COMPILER_BUDGET_LOCK:
        with torch._dynamo.config.patch(
            cache_size_limit=7,
            accumulated_cache_size_limit=256,
            suppress_errors=False,
            error_on_recompile=sealed,
            fail_on_recompile_limit_hit=True,
        ):
            yield


# @spec PORT-PERF-015, PORT-PERF-016, PORT-PERF-017, PORT-PERF-019, PORT-PERF-020, PORT-PERF-021, PORT-PERF-022
class OuterTensorExecution:
    """One pre and six post fullgraph keys, sealed before model profiling."""

    def __init__(self, *, constants: dict[str, int]):
        required = set(_PRE_CONSTANTS) | {"carrier_width", "max_symbols_per_step"}
        if set(constants) != required or any(type(v) is not int for v in constants.values()):
            raise ValueError("outer tensor fusion constants must be complete strict integers")
        self.constants = MappingProxyType(dict(constants))
        self.sealed = False
        self._failed = False
        self._context: dict[str, Any] = {}
        self._declared: list[dict[str, Any]] = []
        self._compiled: dict[str, Any] = {}
        self._functions: dict[str, Any] = {}
        self._keys: dict[tuple[Any, ...], str] = {}
        self._accepted = dict.fromkeys(FAMILIES, 0)
        self._at_seal = dict(self._accepted)
        self._attempts = self._attempts_at_seal = self._post_seal = self._undeclared = 0
        self._pre_fused = self._pre_eager = self._post_fused = self._post_eager = 0
        self._dispatch_at_seal = (0, 0, 0, 0)
        self._covered: set[str] = set()
        self._preparing: str | None = None
        try:
            triton_version = version("triton")
        except PackageNotFoundError:
            triton_version = "unavailable"
        self._compiler = dict(
            backend="inductor",
            torch=torch.__version__,
            triton=triton_version,
            fullgraph=True,
            dynamic=False,
            options={"triton.cudagraphs": False},
        )

    @property
    def ready(self) -> bool:
        return self.sealed and not self._failed and self._covered == set(FAMILIES)

    def _examples(self, device: torch.device):
        from vllm_omni.model_executor.models.nemotron_asr.advance import ROLE_CHUNK, ROLE_FLUSH
        from vllm_omni.model_executor.models.nemotron_asr.rnnt import BOOK_GEOMETRY, QUEUE_LAST_LABEL

        c = self.constants

        def zeros(*shape, dtype=torch.int64):
            return torch.zeros(*shape, dtype=dtype, device=device)

        book = zeros(1, 7, dtype=torch.int32)
        book[:, QUEUE_LAST_LABEL] = c["blank_id"]
        pre = dict(
            input_ids=torch.full((1,), c["placeholder_id"], device=device, dtype=torch.int64),
            is_chunk=torch.ones(1, device=device, dtype=torch.bool),
            plan_geometry=zeros(1),
            prior_prompt=zeros(1),
            queue=zeros(1, c["queue_capacity"], dtype=torch.int32),
            book=book,
            frontend_counters=zeros(1, 8),
            cadence_frames=torch.full((1,), 8, device=device, dtype=torch.int64),
        )
        yield "pre", None, pre
        for geometry in (None, *range(len(CADENCES))):
            count = int(geometry is not None)
            width = (
                0 if geometry is None else (tuple(CADENCES.values())[geometry][1] + 1) * c["max_symbols_per_step"] + 1
            )
            baseline_book = book.clone()
            baseline_book[:, BOOK_GEOMETRY] = geometry or 0
            rows = zeros(1, c["carrier_width"], dtype=torch.float32)
            rows[:, 0] = c["park_id"]
            post = dict(
                token_ids=zeros(count, width, dtype=torch.int32),
                token_lengths=zeros(count, dtype=torch.int32),
                roles=torch.full(
                    (1,), ROLE_FLUSH if geometry is None else ROLE_CHUNK, device=device, dtype=torch.int64
                ),
                chunk_rows=torch.arange(count, device=device, dtype=torch.int64),
                baseline_queue=zeros(1, c["queue_capacity"], dtype=torch.int32),
                baseline_book=baseline_book,
                prompt_index=zeros(1),
                incoming_status=zeros(1, dtype=torch.int32),
                proposed_rows=rows,
                proposed_queue=zeros(1, c["queue_capacity"], dtype=torch.int32),
                proposed_book=baseline_book.clone(),
                proposed_status=zeros(1, dtype=torch.int32),
                plan_geometry=torch.full((1,), geometry or 0, device=device, dtype=torch.int64),
            )
            yield "post-control" if geometry is None else f"post-g{geometry}", geometry, post

    def _function(self, region: str):
        from vllm_omni.model_executor.models.nemotron_asr.advance import (
            _outer_post_tensor_predicates,
            _outer_pre_tensor_predicates,
        )

        return _outer_pre_tensor_predicates if region == "pre" else _outer_post_tensor_predicates

    def _isolated_function(self, region: str):
        if region not in self._functions:
            fn = self._function(region)
            # Dynamo caches on Python code identity. Two private code objects
            # isolate this model's budgets without resetting process caches or
            # duplicating the predicate implementation.
            isolated = FunctionType(fn.__code__.replace(), fn.__globals__, fn.__name__, fn.__defaults__, fn.__closure__)
            isolated.__kwdefaults__ = fn.__kwdefaults__
            self._functions[region] = isolated
        return self._functions[region]

    def _constant_args(self, region: str) -> dict[str, int]:
        return {name: self.constants[name] for name in (_PRE_CONSTANTS if region == "pre" else _POST_CONSTANTS)}

    def _backend(self, family: str):
        def compile_frame(graph, inputs, **kwargs):
            self._attempts += 1
            if self.sealed:
                self._post_seal += 1
                raise RuntimeError("outer tensor fusion backend entered after seal")
            if self._preparing != family:
                self._undeclared += 1
                raise RuntimeError("undeclared outer tensor fusion compilation")
            if self._accepted[family]:
                raise RuntimeError("outer tensor fusion second frame exceeds family budget")
            from torch._inductor.compile_fx import compile_fx

            options = kwargs.pop("options", None)
            result = compile_fx(graph, inputs, config_patches=options, **kwargs)
            self._accepted[family] += 1
            return result

        return compile_frame

    def warmup(self, device: torch.device) -> None:
        if self._failed:
            raise RuntimeError("outer tensor fusion preparation failed; fresh worker required")
        if self.sealed:
            if (
                self._accepted != self._at_seal
                or self._attempts != self._attempts_at_seal
                or tuple(self._compiled) != FAMILIES
                or len(self._keys) != len(FAMILIES)
                or self._accepted != dict.fromkeys(FAMILIES, 1)
            ):
                raise RuntimeError("outer tensor fusion sealed inventory changed")
            return
        try:
            if device.type != "cuda" or not torch.is_inference_mode_enabled():
                raise RuntimeError("outer tensor fusion preparation requires CUDA inference mode")
            self._context = _context()
            if self._context["dispatch_modes"]:
                raise RuntimeError("outer tensor fusion cannot prepare under a dispatch mode")
            for family, geometry, inputs in self._examples(device):
                region = "pre" if family == "pre" else "post"
                names = _PRE_NAMES if region == "pre" else _POST_NAMES
                tensors = tuple(inputs[name] for name in names)
                self._declared.append(
                    dict(id=family, region=region, geometry=geometry, inputs=_tensor_descriptors(names, tensors))
                )
                self._keys[(region, tensor_metadata_key(tensors))] = family
                fn = self._function(region)
                constants = self._constant_args(region)
                self._preparing = family
                with _compiler_scope(sealed=False):
                    compiled = torch.compile(
                        self._isolated_function(region),
                        backend=self._backend(family),
                        fullgraph=True,
                        dynamic=False,
                        options={"triton.cudagraphs": False},
                    )
                    expected = fn(**inputs, **constants)
                    for _ in range(2):
                        if region == "pre":
                            self._pre_fused += 1
                        else:
                            self._post_fused += 1
                        actual = compiled(**inputs, **constants)
                        expected_tuple = expected if isinstance(expected, tuple) else (expected,)
                        actual_tuple = actual if isinstance(actual, tuple) else (actual,)
                        for observed, reference in zip(actual_tuple, expected_tuple, strict=True):
                            if not torch.equal(observed, reference):
                                raise RuntimeError("outer tensor fusion preparation changed predicate output")
                    if self._accepted[family] != 1:
                        raise RuntimeError("outer tensor fusion preparation requires one accepted frame")
                self._compiled[family] = compiled
            if self._declared != expected_declarations(dict(self.constants), str(tensors[0].device)):
                raise RuntimeError("outer tensor fusion canonical metadata differs from declared inventory")
            self._preparing = None
            self._at_seal = dict(self._accepted)
            self._attempts_at_seal = self._attempts
            self._dispatch_at_seal = (self._pre_fused, self._pre_eager, self._post_fused, self._post_eager)
            self.sealed = True
        except BaseException:
            self._failed = True
            raise

    def _run(self, region: str, values: dict[str, Any], *, capture: bool, endpoint_enabled: bool):
        names = _PRE_NAMES if region == "pre" else _POST_NAMES
        constants = self._constant_args(region)
        for name in tuple(values):
            if name in constants:
                if values.pop(name) != constants[name]:
                    # Endpoint-absent valid probes use the original eager rule.
                    if name != "eou_token_id" or endpoint_enabled:
                        raise ValueError(f"outer tensor fusion frozen constant mismatch: {name}")
        if not endpoint_enabled:
            constants["eou_token_id"] = None
        tensors = tuple(values[name] for name in names)
        family = None
        if (
            self.sealed
            and not capture
            and endpoint_enabled
            and all(t.device.type == "cuda" and t.is_contiguous() for t in tensors)
            and compiler_context_key() == _freeze(self._context)
        ):
            family = self._keys.get((region, tensor_metadata_key(tensors)))
        if family is None:
            if region == "pre":
                self._pre_eager += 1
            else:
                self._post_eager += 1
            return self._function(region)(**values, **constants)
        if region == "pre":
            self._pre_fused += 1
        else:
            self._post_fused += 1
        with _compiler_scope(sealed=True):
            result = self._compiled[family](**values, **constants)
        self._covered.add(family)
        return result

    def run_pre(self, *, capture: bool = False, endpoint_enabled: bool = True, **values):
        return self._run("pre", values, capture=capture, endpoint_enabled=endpoint_enabled)

    def run_post(self, *, capture: bool = False, endpoint_enabled: bool = True, **values):
        return self._run("post", values, capture=capture, endpoint_enabled=endpoint_enabled)

    def receipt(self) -> dict[str, Any]:
        pf, pe, qf, qe = self._dispatch_at_seal
        return deepcopy(
            dict(
                schema="nemotron-outer-tensor-fusion/1",
                arm="compiled-static",
                profile=PROFILE,
                constants=dict(self.constants),
                compiler=self._compiler,
                context=self._context,
                declared=self._declared,
                prepared=[f for f in FAMILIES if f in self._compiled],
                real_profile_covered=[f for f in FAMILIES if f in self._covered],
                warmup_complete=self.sealed,
                sealed=self.sealed,
                ready=self.ready,
                frames=dict(
                    accepted=self._accepted,
                    at_seal=self._at_seal,
                    backend_attempts=self._attempts,
                    attempts_at_seal=self._attempts_at_seal,
                    post_seal_attempts=self._post_seal,
                    undeclared_attempts=self._undeclared,
                ),
                dispatch=dict(
                    pre=dict(
                        fused_calls=self._pre_fused, eager_calls=self._pre_eager, fused_at_seal=pf, eager_at_seal=pe
                    ),
                    post=dict(
                        fused_calls=self._post_fused, eager_calls=self._post_eager, fused_at_seal=qf, eager_at_seal=qe
                    ),
                ),
            )
        )


def expected_declarations(constants: dict[str, int], device: str) -> list[dict[str, Any]]:
    """Describe the finite canonical scratch ABI without tensor allocation."""
    queue = constants["queue_capacity"]
    result = []
    for family in FAMILIES:
        geometry = int(family[-1]) if family.startswith("post-g") else None
        if family == "pre":
            names = _PRE_NAMES
            shapes = ((1,), (1,), (1,), (1,), (1, queue), (1, 7), (1, 8), (1,))
            dtypes = ("int64", "bool", "int64", "int64", "int32", "int32", "int64", "int64")
        else:
            names = _POST_NAMES
            count = int(geometry is not None)
            width = (
                0
                if geometry is None
                else (tuple(CADENCES.values())[geometry][1] + 1) * constants["max_symbols_per_step"] + 1
            )
            shapes = (
                (count, width),
                (count,),
                (1,),
                (count,),
                (1, queue),
                (1, 7),
                (1,),
                (1,),
                (1, constants["carrier_width"]),
                (1, queue),
                (1, 7),
                (1,),
                (1,),
            )
            dtypes = (
                "int32",
                "int32",
                "int64",
                "int64",
                "int32",
                "int32",
                "int64",
                "int32",
                "float32",
                "int32",
                "int32",
                "int32",
                "int64",
            )
        inputs = [
            dict(
                name=name,
                shape=list(shape),
                dtype=dtype,
                device=device,
                stride=[max(shape[1], 1), 1] if len(shape) == 2 else [1],
                storage_offset_bytes=0,
                requires_grad=False,
                object_group=index,
                storage_group=index,
            )
            for index, (name, shape, dtype) in enumerate(zip(names, shapes, dtypes, strict=True))
        ]
        result.append(dict(id=family, region="pre" if family == "pre" else "post", geometry=geometry, inputs=inputs))
    return result
