# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Feature-gated execution for the static streaming encoder transition.

All arms execute the same encoder, language-conditioning, and padded-row
zeroing function. Inductor specializes the compiled arms for exact tensor
shapes with compiler-owned CUDA graphs disabled. The dense-graphed arm then
captures that compiled transition with explicit, stable graph-owned scratch.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable, Generator
from contextlib import contextmanager
from dataclasses import dataclass, field
from threading import RLock
from typing import TYPE_CHECKING, Any, NamedTuple, Protocol, TypeVar, cast

import torch

from vllm_omni.model_executor.models.nemotron_asr.decode_graph import (
    GraphRuntime,
    platform_graph_runtime,
)
from vllm_omni.model_executor.models.nemotron_asr.encoder import (
    StreamingCaches,
    stream_step,
)
from vllm_omni.model_executor.models.nemotron_asr.profiling import phase

logger = logging.getLogger(__name__)

_Result = TypeVar("_Result")

if TYPE_CHECKING:
    from vllm_omni.model_executor.models.nemotron_asr.nemotron_asr import (
        NemotronASRCore,
    )


class EncoderCaches(Protocol):
    """Structural cache surface shared by resident and profile execution."""

    channel: Any
    time: Any
    valid: torch.Tensor
    left_context: int

    def graph_storage(self) -> EncoderCacheStorage:
        """Return every mutable cache tensor, including all valid slots."""
        ...

    def empty_like(self) -> EncoderCaches:
        """Build the same adapter class/layout over fresh tensor storage."""
        ...


EncoderTransition = Callable[
    [
        torch.Tensor,
        EncoderCaches,
        torch.Tensor,
        torch.Tensor,
        int,
        torch.Tensor,
    ],
    tuple[torch.Tensor, torch.Tensor],
]


class TensorSignature(NamedTuple):
    """Static tensor metadata admitted after warmup."""

    shape: tuple[int, ...]
    stride: tuple[int, ...]
    dtype: str
    device: str
    layout: str
    requires_grad: bool
    storage_offset: int
    tensor_type: str
    dispatch_keys: str


class EncoderCacheStorage(NamedTuple):
    """All mutable cache tensors behind one encoder cache adapter."""

    channel: tuple[torch.Tensor, ...]
    time: tuple[torch.Tensor, ...]
    valid: tuple[torch.Tensor, ...]


class EncoderGeometryShape(NamedTuple):
    """One manifest-backed encoder shape for a declared geometry."""

    cadence_frames: int
    mel_width: int
    out_width: int


class CompilerTensorState(NamedTuple):
    """Compiler-visible state for one model-owned tensor."""

    identity: int
    shape: tuple[int, ...]
    stride: tuple[int, ...]
    dtype: str
    device: str
    layout: str
    dispatch_keys: str
    requires_grad: bool


class CompilerModelState(NamedTuple):
    """Complete compiler-visible parameter, buffer, and module state."""

    parameters: tuple[tuple[str, CompilerTensorState], ...]
    buffers: tuple[tuple[str, CompilerTensorState], ...]
    modules: tuple[tuple[str, bool], ...]


class EncoderSignature(NamedTuple):
    """Complete static input contract for one encoder specialization."""

    mel: TensorSignature
    channel: tuple[TensorSignature, ...]
    time: tuple[TensorSignature, ...]
    valid: TensorSignature
    left_context: int
    out_offsets: TensorSignature
    out_lengths: TensorSignature
    out_width: int
    prompt_index: TensorSignature
    cache_type: str


def _tensor_signature(tensor: torch.Tensor) -> TensorSignature:
    return TensorSignature(
        tuple(tensor.shape),
        tuple(tensor.stride()),
        str(tensor.dtype),
        str(tensor.device),
        str(tensor.layout),
        bool(tensor.requires_grad),
        int(tensor.storage_offset()),
        f"{type(tensor).__module__}.{type(tensor).__qualname__}",
        str(torch._C._dispatch_keys(tensor)),
    )


def _compiler_tensor_state(tensor: torch.Tensor) -> CompilerTensorState:
    stride = tuple(tensor.stride()) if tensor.layout == torch.strided else ()
    return CompilerTensorState(
        identity=id(tensor),
        shape=tuple(tensor.shape),
        stride=stride,
        dtype=str(tensor.dtype),
        device=str(tensor.device),
        layout=str(tensor.layout),
        dispatch_keys=str(torch._C._dispatch_keys(tensor)),
        requires_grad=bool(tensor.requires_grad),
    )


def _compiled_transition_model_state(core: Any) -> CompilerModelState:
    roots = (
        ("encoder", core.encoder),
        ("lid", core.lid),
    )

    def qualified(root: str, name: str) -> str:
        return f"{root}.{name}" if name else root

    return CompilerModelState(
        parameters=tuple(
            (qualified(root, name), _compiler_tensor_state(parameter))
            for root, module in roots
            for name, parameter in module.named_parameters()
        ),
        buffers=tuple(
            (qualified(root, name), _compiler_tensor_state(buffer))
            for root, module in roots
            for name, buffer in module.named_buffers()
        ),
        modules=tuple(
            (qualified(root, name), bool(child.training))
            for root, module in roots
            for name, child in module.named_modules()
        ),
    )


def _compiled_transition_runner_device(
    core: Any,
    *,
    activation_dtype: torch.dtype,
) -> torch.device:
    """Validate finalized compute metadata and return the runner device."""
    parameters: tuple[torch.nn.Parameter, ...] = tuple(core.encoder.parameters()) + tuple(core.lid.parameters())
    if not parameters:
        raise ValueError("compiled-static encoder transition has no device authority")
    devices = {parameter.device for parameter in parameters}
    if len(devices) != 1:
        raise ValueError("compiled-static encoder transition spans multiple model devices")
    dtypes = {parameter.dtype for parameter in parameters}
    if dtypes != {activation_dtype}:
        raise ValueError(
            "compiled-static encoder transition parameter dtype differs from "
            f"activation policy: parameters={sorted(map(str, dtypes))}, "
            f"activation={activation_dtype}"
        )
    return next(iter(devices))


def _changed_model_state_name(
    expected: CompilerModelState,
    actual: CompilerModelState,
) -> str | None:
    for expected_family, actual_family in (
        (expected.parameters, actual.parameters),
        (expected.buffers, actual.buffers),
    ):
        expected_values = dict(expected_family)
        actual_values = dict(actual_family)
        added = sorted(actual_values.keys() - expected_values.keys())
        if added:
            return added[0]
        removed = sorted(expected_values.keys() - actual_values.keys())
        if removed:
            return removed[0]
        for name, state in expected_values.items():
            if actual_values[name] != state:
                return name
    expected_modules = dict(expected.modules)
    actual_modules = dict(actual.modules)
    added_modules = sorted(actual_modules.keys() - expected_modules.keys())
    if added_modules:
        return added_modules[0]
    removed_modules = sorted(expected_modules.keys() - actual_modules.keys())
    if removed_modules:
        return removed_modules[0]
    for name, training in expected_modules.items():
        if actual_modules[name] != training:
            prefix = f"{name}." if name else ""
            return f"{prefix}training"
    return None


# @spec PORT-PERF-010
def encoder_geometry_shape(
    core: NemotronASRCore,
    geometry: int,
) -> EncoderGeometryShape:
    """Derive one encoder shape from the canonical geometry manifest."""
    from vllm_omni.model_executor.models.nemotron_asr.manifests import (
        CADENCES,
        FRONTEND_CONSTANTS,
    )

    labels = tuple(CADENCES)
    if isinstance(geometry, bool) or not isinstance(geometry, int):
        raise ValueError("encoder geometry id must be an integer")
    if not 0 <= geometry < len(labels):
        raise ValueError(f"unknown encoder geometry id {geometry}")
    _, right_context = CADENCES[labels[geometry]]
    cadence_frames = int(FRONTEND_CONSTANTS["subsampling_factor"]) * (int(right_context) + 1)
    mel_width = int(FRONTEND_CONSTANTS["pre_encode_cache_frames"]) + cadence_frames
    length = torch.tensor(
        [mel_width],
        dtype=torch.int64,
        device="cpu",
    )
    out_width = int(core.encoder.pre_encode.output_lengths(length)[0])
    return EncoderGeometryShape(
        cadence_frames=cadence_frames,
        mel_width=mel_width,
        out_width=out_width,
    )


def _tensor_group_signature(value: Any) -> tuple[TensorSignature, ...]:
    if isinstance(value, torch.Tensor):
        return (_tensor_signature(value),)
    return tuple(_tensor_signature(tensor) for tensor in value)


def _transition_tensor_items(
    mel: torch.Tensor,
    caches: EncoderCaches,
    out_offsets: torch.Tensor,
    out_lengths: torch.Tensor,
    prompt_index: torch.Tensor,
) -> tuple[tuple[str, torch.Tensor], ...]:
    items: list[tuple[str, torch.Tensor]] = [("mel", mel)]
    for family_name in ("channel", "time"):
        value = getattr(caches, family_name)
        tensors = (value,) if isinstance(value, torch.Tensor) else tuple(value)
        items.extend((f"caches.{family_name}[{index}]", tensor) for index, tensor in enumerate(tensors))
    items.extend(
        (
            ("caches.valid", caches.valid),
            ("out_offsets", out_offsets),
            ("out_lengths", out_lengths),
            ("prompt_index", prompt_index),
        )
    )
    return tuple(items)


def _validate_transition_tensor_contract(
    mel: torch.Tensor,
    caches: EncoderCaches,
    out_offsets: torch.Tensor,
    out_lengths: torch.Tensor,
    prompt_index: torch.Tensor,
    *,
    runner_device: torch.device,
) -> None:
    expected_dtypes = {
        "mel": torch.float32,
        "caches.valid": torch.int64,
        "out_offsets": torch.int64,
        "out_lengths": torch.int64,
        "prompt_index": torch.int64,
    }
    for name, tensor in _transition_tensor_items(
        mel,
        caches,
        out_offsets,
        out_lengths,
        prompt_index,
    ):
        if tensor.device != runner_device:
            raise ValueError(f"encoder transition tensor {name} differs from runner device")
        if tensor.layout != torch.strided:
            raise ValueError(f"encoder transition tensor {name} must use strided layout")
        expected_dtype = expected_dtypes.get(name, torch.float32)
        if tensor.dtype != expected_dtype:
            raise ValueError(f"encoder transition tensor {name} must use {expected_dtype}")
        dispatch_keys = str(torch._C._dispatch_keys(tensor))
        if "Autograd" in dispatch_keys or "ADInplaceOrView" in dispatch_keys:
            raise ValueError(f"encoder transition tensor {name} is outside serving inference mode")


def _encoder_signature(
    mel: torch.Tensor,
    caches: EncoderCaches,
    out_offsets: torch.Tensor,
    out_lengths: torch.Tensor,
    out_width: int,
    prompt_index: torch.Tensor,
) -> EncoderSignature:
    """Return the complete static compiler input contract."""
    return EncoderSignature(
        _tensor_signature(mel),
        _tensor_group_signature(caches.channel),
        _tensor_group_signature(caches.time),
        _tensor_signature(caches.valid),
        int(caches.left_context),
        _tensor_signature(out_offsets),
        _tensor_signature(out_lengths),
        int(out_width),
        _tensor_signature(prompt_index),
        f"{type(caches).__module__}.{type(caches).__qualname__}",
    )


def _cache_storage(caches: EncoderCaches) -> EncoderCacheStorage:
    storage_factory = getattr(caches, "graph_storage", None)
    if not callable(storage_factory):
        raise TypeError("dense-graphed encoder caches require graph_storage()")
    storage = storage_factory()
    if not isinstance(storage, EncoderCacheStorage):
        raise TypeError("encoder cache graph_storage() returned malformed storage")
    if not storage.channel or not storage.time or not storage.valid:
        raise ValueError("dense-graphed encoder cache storage is incomplete")
    if any(not isinstance(tensor, torch.Tensor) for family in storage for tensor in family):
        raise TypeError("dense-graphed encoder cache storage must contain tensors")
    return storage


def _cache_storage_signature(
    storage: EncoderCacheStorage,
) -> tuple[tuple[TensorSignature, ...], ...]:
    return tuple(tuple(_tensor_signature(tensor) for tensor in family) for family in storage)


def _copy_cache_storage_(
    destination: EncoderCacheStorage,
    source: EncoderCacheStorage,
) -> None:
    # Layouts are validated once, before any staging mutation. Graph-owned
    # storage remains fixed for the lifetime of the published domain.
    for destination_family, source_family in zip(destination, source, strict=True):
        for destination_tensor, source_tensor in zip(
            destination_family,
            source_family,
            strict=True,
        ):
            destination_tensor.copy_(source_tensor)


def _cuda_memory_snapshot(
    device: torch.device,
    *,
    stage: str,
    key: tuple[int, int] | None = None,
) -> dict[str, Any] | None:
    """Record diagnostic graph memory without becoming capacity authority."""
    if device.type != "cuda":
        return None
    free_bytes, total_bytes = torch.cuda.mem_get_info(device)
    snapshot: dict[str, Any] = {
        "stage": stage,
        "free_bytes": int(free_bytes),
        "total_bytes": int(total_bytes),
        "allocated_bytes": int(torch.cuda.memory_allocated(device)),
        "reserved_bytes": int(torch.cuda.memory_reserved(device)),
    }
    if key is not None:
        snapshot["key"] = [key[0], key[1]]
    return snapshot


@dataclass
class _EncoderGraphEntry:
    """Stable storage and one no-argument platform graph wrapper."""

    key: tuple[int, int]
    signature: EncoderSignature
    storage_signature: tuple[tuple[TensorSignature, ...], ...]
    mel: torch.Tensor
    caches: EncoderCaches
    out_offsets: torch.Tensor
    out_lengths: torch.Tensor
    out_width: int
    prompt_index: torch.Tensor
    raw: torch.Tensor
    conditioned: torch.Tensor
    descriptor: Any
    wrapper: Any

    def cache_storage(self) -> EncoderCacheStorage:
        return _cache_storage(self.caches)

    def output_tuple(self) -> tuple[torch.Tensor, ...]:
        storage = self.cache_storage()
        return (
            self.raw,
            self.conditioned,
            *storage.channel,
            *storage.time,
            *storage.valid,
        )


# Dynamo's cache limits are process-global. Nemotron enters this lock only
# during startup-owned, pre-admission compilation so nested or overlapping
# Nemotron scopes cannot restore stale values.
_COMPILER_BUDGET_LOCK = RLock()
_COMPILED_ARMS = frozenset({"compiled-static", "dense-graphed"})


# @spec PORT-PERF-009
@contextmanager
def _compiler_specialization_budget(
    specializations: int,
) -> Generator[None, None, None]:
    """Scope Dynamo's process-global limits to one declared-domain call."""
    if specializations <= 0:
        raise ValueError("compiler specialization budget must be positive")
    with _COMPILER_BUDGET_LOCK:
        config = torch._dynamo.config
        original_cache_size = int(config.cache_size_limit)
        original_accumulated_cache_size = int(config.accumulated_cache_size_limit)
        try:
            config.cache_size_limit = max(
                original_cache_size,
                specializations,
            )
            config.accumulated_cache_size_limit = max(
                original_accumulated_cache_size,
                specializations,
            )
            yield
        finally:
            config.cache_size_limit = original_cache_size
            config.accumulated_cache_size_limit = original_accumulated_cache_size


@dataclass
class ResolvedEncoderExecution:
    """One startup selection plus its sealed static-shape authority."""

    arm: str
    transition: EncoderTransition
    warmup_geometries: tuple[int, ...] = ()
    warmup_populations: tuple[int, ...] = ()
    t_cap: int = 0
    _history_frames: int = 0
    _geometry_shapes: dict[int, EncoderGeometryShape] = field(
        default_factory=dict,
        repr=False,
    )
    _core: Any | None = field(default=None, repr=False)
    _runner_device: torch.device | None = field(default=None, repr=False)
    _invocations: int = 0
    _last_signature: EncoderSignature | None = None
    _warmup_signatures: dict[tuple[int, int], EncoderSignature] = field(default_factory=dict)
    _profile_signature: tuple[tuple[int, int], EncoderSignature] | None = None
    _allowed_signatures: frozenset[EncoderSignature] = frozenset()
    _active_cell: tuple[int, int] | None = None
    _sealed: bool = False
    _failed: bool = False
    _model_state: CompilerModelState | None = None
    _compiled_transition: EncoderTransition | None = field(default=None, repr=False)
    _vllm_config: Any | None = field(default=None, repr=False)
    _graph_runtime: GraphRuntime | None = field(default=None, repr=False)
    _graph_entries: dict[tuple[int, int], _EncoderGraphEntry] = field(
        default_factory=dict,
        repr=False,
    )
    _staging_cell: tuple[int, int] | None = field(default=None, repr=False)
    _pending_graph_entries: dict[tuple[int, int], _EncoderGraphEntry] = field(
        default_factory=dict,
        repr=False,
    )
    _memory_diagnostics: list[dict[str, Any]] = field(
        default_factory=list,
        repr=False,
    )

    @property
    def ready(self) -> bool:
        """Whether the selected arm is safe to admit served work."""
        if self.arm == "eager":
            return True
        if self.arm == "dense-graphed":
            expected = {
                (geometry, population) for geometry in self.warmup_geometries for population in self.warmup_populations
            }
            return self._sealed and set(self._graph_entries) == expected and not self._failed
        return self._sealed and not self._failed

    @property
    def cell_active(self) -> bool:
        """Whether product-owned startup code authorized the current call."""
        return self._active_cell is not None

    def warmup_cell(
        self,
        *,
        geometry: int,
        population: int,
        invoke: Callable[[], Any],
    ) -> None:
        """Execute and attest exactly one compiler invocation for a cell."""
        if self.arm not in _COMPILED_ARMS:
            raise ValueError("encoder warmup cells require a compiled encoder arm")
        self._raise_if_failed()
        if self._sealed:
            raise ValueError(f"{self.arm} encoder execution is already sealed")
        cell = (int(geometry), int(population))
        if cell in self._warmup_signatures:
            raise ValueError(f"encoder warmup cell {cell} was repeated")
        if cell[0] not in self.warmup_geometries:
            raise ValueError(f"encoder warmup geometry {cell[0]} was not declared")
        if cell[1] not in self.warmup_populations:
            raise ValueError(f"encoder warmup population {cell[1]} was not declared")
        try:
            before = self._invocations
            self._invoke_declared_cell(cell=cell, invoke=invoke)
            if self._invocations != before + 1 or self._last_signature is None:
                raise ValueError("encoder warmup cell must execute exactly one transition")
            observed_population = self._last_signature.mel.shape[0]
            if observed_population != cell[1]:
                raise ValueError("encoder warmup population differs from executed batch")
            if self._profile_signature is not None:
                profile_cell, profile_signature = self._profile_signature
                if cell == profile_cell and self._last_signature != profile_signature:
                    raise ValueError("encoder warmup signature differs from memory profile")
            self._assert_model_state()
            self._warmup_signatures[cell] = self._last_signature
        except Exception:
            self._discard()
            raise

    def profile_cell(
        self,
        *,
        geometry: int,
        population: int,
        invoke: Callable[[], _Result],
    ) -> _Result:
        """Authorize and attest the sole activation-memory profile cell."""
        if self.arm not in _COMPILED_ARMS:
            return invoke()
        self._raise_if_failed()
        if self._sealed:
            raise ValueError(f"{self.arm} encoder execution is already sealed")
        if self._profile_signature is not None:
            raise ValueError(f"{self.arm} encoder memory profile was repeated")
        cell = (int(geometry), int(population))
        try:
            before = self._invocations
            result = self._invoke_declared_cell(cell=cell, invoke=invoke)
            if self._invocations != before + 1 or self._last_signature is None:
                raise ValueError("encoder memory profile must execute exactly one transition")
            self._assert_model_state()
            self._profile_signature = (cell, self._last_signature)
            return result
        except Exception:
            self._discard()
            raise

    def _invoke_declared_cell(
        self,
        *,
        cell: tuple[int, int],
        invoke: Callable[[], _Result],
    ) -> _Result:
        if cell[0] not in self.warmup_geometries:
            raise ValueError(f"encoder geometry {cell[0]} was not declared")
        if cell[1] not in self.warmup_populations:
            raise ValueError(f"encoder population {cell[1]} was not declared")
        if self._active_cell is not None:
            raise ValueError(f"{self.arm} encoder cell invocation overlapped")
        self._active_cell = cell
        try:
            with torch.inference_mode():
                return invoke()
        finally:
            self._active_cell = None

    def _raise_if_failed(self) -> None:
        if self._failed:
            raise ValueError(f"{self.arm} encoder authority is failed and discarded")

    def _discard(self) -> None:
        self._failed = True
        self._sealed = False
        self._warmup_signatures.clear()
        self._profile_signature = None
        self._allowed_signatures = frozenset()
        self._graph_entries.clear()
        self._pending_graph_entries.clear()
        self._staging_cell = None

    def _materialize_preprofile_state(
        self,
        out_width: int,
    ) -> None:
        if self._core is None or self.t_cap <= 0:
            raise ValueError(f"{self.arm} encoder positional capacity is unavailable")
        required = int(out_width) + self._history_frames
        if required > self.t_cap:
            raise ValueError(
                "encoder.pos_enc.pe capacity is smaller than the transition window: "
                f"required={required}, declared={self.t_cap}"
            )
        if self._model_state is not None:
            return
        encoder = self._core.encoder
        activation_dtype = self._core.policy.dtype_for("activations")
        runner_device = _compiled_transition_runner_device(
            self._core,
            activation_dtype=activation_dtype,
        )
        reference = torch.empty(
            (),
            dtype=activation_dtype,
            device=runner_device,
        )
        encoder.pos_enc._extend(self.t_cap, reference)
        positional = encoder.pos_enc.pe
        expected_width = 2 * self.t_cap - 1
        if (
            tuple(positional.shape) != (1, expected_width, int(encoder.pos_enc.d_model))
            or positional.dtype != reference.dtype
            or positional.device != reference.device
            or positional.layout != torch.strided
        ):
            raise ValueError("encoder.pos_enc.pe materialization differs from declared capacity or activation metadata")
        dispatch_keys = str(torch._C._dispatch_keys(positional))
        if "Autograd" in dispatch_keys or "ADInplaceOrView" in dispatch_keys:
            raise ValueError("encoder.pos_enc.pe materialization is outside serving inference mode")
        self._runner_device = runner_device
        self._model_state = _compiled_transition_model_state(self._core)

    def _assert_model_state(self) -> None:
        if self._core is None or self._model_state is None:
            raise ValueError(f"{self.arm} encoder model state was not materialized")
        changed = _changed_model_state_name(
            self._model_state,
            _compiled_transition_model_state(self._core),
        )
        if changed is not None:
            raise ValueError(f"{self.arm} encoder model state changed at {changed}")

    # @spec PORT-PERF-009
    def warmup_domain(
        self,
        *,
        expected_cells: tuple[tuple[int, int], ...],
        invoke: Callable[[int, int], Any],
    ) -> None:
        """Compile and seal the complete finite specialization domain."""
        if self.arm not in _COMPILED_ARMS:
            raise ValueError("encoder warmup domain requires a compiled encoder arm")
        self._raise_if_failed()
        if self._sealed:
            raise ValueError(f"{self.arm} encoder execution is already sealed")
        if self._warmup_signatures:
            raise ValueError(f"{self.arm} encoder warmup already started")
        cells = tuple((int(geometry), int(population)) for geometry, population in expected_cells)
        if not cells:
            raise ValueError(f"{self.arm} encoder warmup domain is empty")
        if len(set(cells)) != len(cells):
            raise ValueError(f"{self.arm} encoder warmup cells repeat")
        expected_domain = {
            (geometry, population) for geometry in self.warmup_geometries for population in self.warmup_populations
        }
        if set(cells) != expected_domain:
            missing = sorted(expected_domain - set(cells))
            unexpected = sorted(set(cells) - expected_domain)
            raise ValueError(
                f"{self.arm} encoder warmup domain differs from authority: missing={missing}, unexpected={unexpected}"
            )
        try:
            if self._model_state is not None:
                self._assert_model_state()
            for geometry, population in cells:

                def invoke_cell(
                    geometry: int = geometry,
                    population: int = population,
                ) -> Any:
                    return invoke(geometry, population)

                self.warmup_cell(
                    geometry=geometry,
                    population=population,
                    invoke=invoke_cell,
                )
            self._seal()
            if self.arm == "dense-graphed":
                # Initial warmup intentionally specializes. After sealing,
                # staging and capture must reuse exactly that compiler domain.
                with _compiler_specialization_budget(len(cells)), torch._dynamo.config.patch(error_on_recompile=True):
                    self._capture_graph_domain(cells=cells, invoke=invoke)
        except Exception:
            self._discard()
            raise

    def _seal(self) -> None:
        """Seal the exact signatures proven by product-owned warmup."""
        if self.arm not in _COMPILED_ARMS:
            return
        self._assert_model_state()
        expected = {
            (geometry, population) for geometry in self.warmup_geometries for population in self.warmup_populations
        }
        observed = set(self._warmup_signatures)
        if observed != expected:
            missing = sorted(expected - observed)
            unexpected = sorted(observed - expected)
            raise ValueError(
                f"{self.arm} encoder warmup cells differ from authority: missing={missing}, unexpected={unexpected}"
            )
        self._allowed_signatures = frozenset(self._warmup_signatures.values())
        self._sealed = True

    def _seed_graph_entry(self, entry: _EncoderGraphEntry, *, variant: str = "full") -> None:
        """Restore row/position-varying inputs for full or mixed history parity."""
        if variant not in {"full", "mixed"}:
            raise ValueError(f"unknown encoder graph parity variant {variant!r}")
        population = int(entry.mel.shape[0])
        rows = torch.arange(population, dtype=torch.int64, device=entry.mel.device)
        storage = entry.cache_storage()
        for index, tensor in enumerate((entry.mel, *storage.channel, *storage.time)):
            # Bound the values while varying both row and within-row position.
            # This runs outside capture and never replaces compiler-visible storage.
            positions = torch.arange(tensor[0].numel(), device=tensor.device).reshape(1, *tensor.shape[1:])
            row_shape = (population,) + (1,) * (tensor.ndim - 1)
            values = 0.01 * (index + 1) + rows.reshape(row_shape) * 0.1 + positions.remainder(97) * 0.001
            tensor.copy_(values)
        valid = (
            torch.full_like(rows, self._history_frames)
            if variant == "full"
            else rows.remainder(self._history_frames + 1)
        )
        for tensor in storage.valid:
            tensor.copy_(valid.reshape(tensor.shape).to(tensor.dtype))
        if entry.out_width <= 0:
            raise ValueError("dense-graphed encoder output width must be positive")
        if variant == "full":
            entry.out_offsets.copy_(rows.remainder(2) * min(2, entry.out_width - 1))
            entry.out_lengths.copy_(entry.out_width - entry.out_offsets)
        else:
            zero_rows = rows.remainder(2) == 0
            entry.out_offsets.copy_(torch.where(zero_rows, min(2, entry.out_width), 0))
            entry.out_lengths.copy_(torch.where(zero_rows, 0, entry.out_width))
        # Compiler warmup established this authority before graph parity.
        core = self._core
        assert core is not None
        entry.prompt_index.copy_(rows.remainder(core.lid.num_prompts))
        entry.raw.zero_()
        entry.conditioned.zero_()

    def _new_graph_entry(
        self,
        *,
        cell: tuple[int, int],
        signature: EncoderSignature,
        mel: torch.Tensor,
        caches: EncoderCaches,
        out_offsets: torch.Tensor,
        out_lengths: torch.Tensor,
        out_width: int,
        prompt_index: torch.Tensor,
    ) -> _EncoderGraphEntry:
        compiled = self._compiled_transition
        runtime = self._graph_runtime
        if compiled is None or runtime is None or self._vllm_config is None:
            raise ValueError("dense-graphed encoder runtime authority is unavailable")
        empty_like = getattr(caches, "empty_like", None)
        if not callable(empty_like):
            raise TypeError("dense-graphed encoder caches require empty_like()")
        stable_caches = empty_like()
        if type(stable_caches) is not type(caches):
            raise TypeError("dense-graphed encoder cache factory changed adapter class")
        caller_storage = _cache_storage(caches)
        stable_storage = _cache_storage(stable_caches)
        caller_storage_signature = _cache_storage_signature(caller_storage)
        if _cache_storage_signature(stable_storage) != caller_storage_signature:
            raise ValueError("dense-graphed encoder cache factory changed tensor layout")

        stable_mel = torch.empty_like(mel)
        stable_offsets = torch.empty_like(out_offsets)
        stable_lengths = torch.empty_like(out_lengths)
        stable_prompt = torch.empty_like(prompt_index)
        descriptor = runtime.descriptor_factory(cell[1])

        # Discover result layouts before allocating strong graph outputs. This
        # executes the already-sealed compiled transition on graph-owned scratch.
        provisional = _EncoderGraphEntry(
            key=cell,
            signature=signature,
            storage_signature=caller_storage_signature,
            mel=stable_mel,
            caches=stable_caches,
            out_offsets=stable_offsets,
            out_lengths=stable_lengths,
            out_width=int(out_width),
            prompt_index=stable_prompt,
            raw=torch.empty(0, dtype=mel.dtype, device=mel.device),
            conditioned=torch.empty(0, dtype=mel.dtype, device=mel.device),
            descriptor=descriptor,
            wrapper=None,
        )
        self._seed_graph_entry(provisional)
        with runtime.forward_context(
            None,
            self._vllm_config,
            cudagraph_runtime_mode=runtime.eager_mode,
            batch_descriptor=descriptor,
        ):
            raw, conditioned = compiled(
                stable_mel,
                stable_caches,
                stable_offsets,
                stable_lengths,
                int(out_width),
                stable_prompt,
            )
        if not isinstance(raw, torch.Tensor) or not isinstance(conditioned, torch.Tensor):
            raise TypeError("dense-graphed encoder transition returned malformed outputs")
        entry = _EncoderGraphEntry(
            key=cell,
            signature=signature,
            storage_signature=caller_storage_signature,
            mel=stable_mel,
            caches=stable_caches,
            out_offsets=stable_offsets,
            out_lengths=stable_lengths,
            out_width=int(out_width),
            prompt_index=stable_prompt,
            raw=torch.empty_like(raw),
            conditioned=torch.empty_like(conditioned),
            descriptor=descriptor,
            wrapper=None,
        )

        def run() -> tuple[torch.Tensor, ...]:
            next_raw, next_conditioned = compiled(
                entry.mel,
                entry.caches,
                entry.out_offsets,
                entry.out_lengths,
                entry.out_width,
                entry.prompt_index,
            )
            entry.raw.copy_(next_raw)
            entry.conditioned.copy_(next_conditioned)
            return entry.output_tuple()

        entry.wrapper = runtime.wrapper_factory(
            run,
            self._vllm_config,
            runtime_mode=runtime.graph_mode,
        )
        return entry

    def _call_graph_entry(
        self,
        entry: _EncoderGraphEntry,
        *,
        mode: Any,
    ) -> tuple[torch.Tensor, ...]:
        runtime = self._graph_runtime
        if runtime is None or self._vllm_config is None:
            raise ValueError("dense-graphed encoder runtime authority is unavailable")
        with runtime.forward_context(
            None,
            self._vllm_config,
            cudagraph_runtime_mode=mode,
            batch_descriptor=entry.descriptor,
        ):
            output = entry.wrapper()
        expected_count = len(entry.output_tuple())
        if not isinstance(output, tuple) or len(output) != expected_count:
            raise TypeError("dense-graphed encoder wrapper returned malformed output")
        if any(not isinstance(tensor, torch.Tensor) for tensor in output):
            raise TypeError("dense-graphed encoder wrapper returned a non-tensor output")
        return output

    @staticmethod
    def _snapshot_graph_entry(
        output: tuple[torch.Tensor, ...],
    ) -> tuple[torch.Tensor, ...]:
        return tuple(tensor.clone() for tensor in output)

    @staticmethod
    def _assert_graph_equal(
        key: tuple[int, int],
        expected: tuple[torch.Tensor, ...],
        actual: tuple[torch.Tensor, ...],
    ) -> None:
        if len(expected) != len(actual) or any(
            not torch.equal(left, right) for left, right in zip(expected, actual, strict=True)
        ):
            raise RuntimeError(f"dense-graphed encoder capture/replay differs from compiled transition at key {key}")

    def _capture_graph_entry(self, entry: _EncoderGraphEntry) -> None:
        runtime = self._graph_runtime
        if runtime is None:
            raise ValueError("dense-graphed encoder runtime authority is unavailable")
        for variant in ("full", "mixed"):
            self._seed_graph_entry(entry, variant=variant)
            eager = self._snapshot_graph_entry(self._call_graph_entry(entry, mode=runtime.eager_mode))
            self._seed_graph_entry(entry, variant=variant)
            # Capture records work without executing it. Subsequent variants
            # reuse the same graph with changed contents and control values.
            if variant == "full":
                self._call_graph_entry(entry, mode=runtime.graph_mode)
                self._seed_graph_entry(entry, variant=variant)
            captured = self._snapshot_graph_entry(self._call_graph_entry(entry, mode=runtime.graph_mode))
            self._assert_graph_equal(entry.key, eager, captured)
            self._seed_graph_entry(entry, variant=variant)
            replayed = self._snapshot_graph_entry(self._call_graph_entry(entry, mode=runtime.graph_mode))
            self._assert_graph_equal(entry.key, eager, replayed)

    def _record_memory_diagnostic(
        self, device: torch.device, *, stage: str, key: tuple[int, int] | None = None
    ) -> None:
        """Keep diagnostics best-effort, including after an allocator failure."""
        try:
            snapshot = _cuda_memory_snapshot(device, stage=stage, key=key)
        except Exception as error:
            snapshot = {"stage": stage, "diagnostic_error": str(error)}
            if key is not None:
                snapshot["key"] = list(key)
        if snapshot is not None:
            self._memory_diagnostics.append(snapshot)

    def _capture_graph_domain(
        self,
        *,
        cells: tuple[tuple[int, int], ...],
        invoke: Callable[[int, int], Any],
    ) -> None:
        if self.arm != "dense-graphed" or not self._sealed:
            raise ValueError("encoder graph capture requires a sealed compiler domain")
        if self._graph_entries or self._pending_graph_entries:
            raise ValueError("dense-graphed encoder capture was repeated")
        runtime = self._graph_runtime
        device = self._runner_device
        if runtime is None or device is None:
            raise ValueError("dense-graphed encoder runtime device is unavailable")
        stage = "staging"
        cell: tuple[int, int] | None = None
        try:
            self._record_memory_diagnostic(device, stage="before-staging")
            for cell in cells:
                self._staging_cell = cell
                staging_cell = cell
                try:

                    def invoke_cell(cell: tuple[int, int] = staging_cell) -> Any:
                        return invoke(cell[0], cell[1])

                    self._invoke_declared_cell(
                        cell=cell,
                        invoke=invoke_cell,
                    )
                finally:
                    self._staging_cell = None
                if cell not in self._pending_graph_entries:
                    raise ValueError(f"encoder graph staging cell {cell} did not execute")
            self._record_memory_diagnostic(device, stage="after-staging")
            self._assert_model_state()

            stage = "capture"
            cell = None
            runtime.set_capture_enabled(True)
            try:
                with torch.inference_mode(), runtime.capture_context(device):
                    for cell in cells:
                        entry = self._pending_graph_entries[cell]
                        self._capture_graph_entry(entry)
                        self._record_memory_diagnostic(device, stage="after-capture", key=cell)
                    cell = None
                    runtime.synchronize(device)
            finally:
                runtime.set_capture_enabled(False)
            self._assert_model_state()
            self._graph_entries = dict(self._pending_graph_entries)
            self._pending_graph_entries.clear()
            self._record_memory_diagnostic(device, stage="after-all-captures")
        except Exception:
            # Neither the allocator probe nor a logging handler may replace the
            # causal startup exception. warmup_domain disposes partial authority.
            try:
                self._record_memory_diagnostic(device, stage=f"failure-{stage}", key=cell)
                logger.error(
                    "dense-graphed encoder startup failed: memory_diagnostics=%s", json.dumps(self._memory_diagnostics)
                )
            except Exception:
                pass
            raise

    def _stage_graph_entry(
        self,
        *,
        cell: tuple[int, int],
        signature: EncoderSignature,
        mel: torch.Tensor,
        caches: EncoderCaches,
        out_offsets: torch.Tensor,
        out_lengths: torch.Tensor,
        out_width: int,
        prompt_index: torch.Tensor,
    ) -> None:
        if cell in self._pending_graph_entries:
            raise ValueError(f"encoder graph staging cell {cell} was repeated")
        expected_signature = self._warmup_signatures.get(cell)
        if expected_signature is None or signature != expected_signature:
            raise ValueError("encoder graph staging signature differs from compiler warmup")
        self._pending_graph_entries[cell] = self._new_graph_entry(
            cell=cell,
            signature=signature,
            mel=mel,
            caches=caches,
            out_offsets=out_offsets,
            out_lengths=out_lengths,
            out_width=out_width,
            prompt_index=prompt_index,
        )

    def _replay_graph_entry(
        self,
        *,
        signature: EncoderSignature,
        mel: torch.Tensor,
        caches: EncoderCaches,
        out_offsets: torch.Tensor,
        out_lengths: torch.Tensor,
        prompt_index: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        matches = [entry for entry in self._graph_entries.values() if entry.signature == signature]
        if not matches:
            raise ValueError("dense-graphed encoder signature was not captured")
        if len(matches) != 1:
            raise ValueError("dense-graphed encoder signature maps to multiple graph keys")
        entry = matches[0]
        if type(caches) is not type(entry.caches):
            raise TypeError("dense-graphed encoder cache adapter class changed")
        caller_storage = _cache_storage(caches)
        if _cache_storage_signature(caller_storage) != entry.storage_signature:
            raise ValueError("dense-graphed encoder cache storage signature changed")

        # The finalized model is immutable through this execution domain's
        # lifetime, as in compiled-static. Reconfiguration requires a fresh
        # worker; serving does not traverse all model tensors on every replay.

        # All validation precedes staging, so an unknown key/signature cannot
        # mutate caller or graph scratch. The graph mutates only entry storage.
        runtime = self._graph_runtime
        if runtime is None:
            raise ValueError("dense-graphed encoder runtime authority is unavailable")
        with phase("port.encode.stage_in"):
            entry.mel.copy_(mel)
            _copy_cache_storage_(entry.cache_storage(), caller_storage)
            entry.out_offsets.copy_(out_offsets)
            entry.out_lengths.copy_(out_lengths)
            entry.prompt_index.copy_(prompt_index)
        with phase("port.encode.replay"):
            self._call_graph_entry(entry, mode=runtime.graph_mode)
        with phase("port.encode.stage_out"):
            _copy_cache_storage_(caller_storage, entry.cache_storage())
            return entry.raw.clone(), entry.conditioned.clone()

    def ready_receipt(self) -> dict[str, Any]:
        """Return the machine-readable resolved-arm readiness receipt."""
        receipt: dict[str, Any] = {
            "arm": self.arm,
            "ready": self.ready,
            "warmup_cells": [[geometry, population] for geometry, population in sorted(self._warmup_signatures)],
            "warmup_geometries": list(self.warmup_geometries),
            "warmup_populations": list(self.warmup_populations),
        }
        if self.arm == "dense-graphed":
            receipt["captured_keys"] = [[geometry, population] for geometry, population in sorted(self._graph_entries)]
        if self._memory_diagnostics:
            receipt["memory_diagnostics"] = list(self._memory_diagnostics)
        return receipt


def execute_encoder_transition(
    core: NemotronASRCore,
    mel: torch.Tensor,
    caches: EncoderCaches,
    out_offsets: torch.Tensor,
    out_lengths: torch.Tensor,
    out_width: int,
    prompt_index: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run the exact cache-aware encoder and conditioning transition."""
    encoded = stream_step(
        core.encoder,
        mel,
        cast(StreamingCaches, caches),
        out_offsets=out_offsets,
        out_lengths=out_lengths,
        out_width=out_width,
    )
    conditioned = core.lid(encoded, prompt_index=prompt_index)
    frame = torch.arange(out_width, device=mel.device).view(1, -1, 1)
    conditioned = torch.where(
        frame < out_lengths.view(-1, 1, 1),
        conditioned,
        conditioned.new_zeros(()),
    )
    return encoded, conditioned


# @spec PORT-PERF-009, PORT-PERF-010, PORT-PERF-011
def build_encoder_execution(
    core: NemotronASRCore,
    hf_config: Any,
    *,
    maximum_population: int,
    warmup_geometries: tuple[int, ...],
    vllm_config: Any | None = None,
    graph_runtime: GraphRuntime | None = None,
) -> ResolvedEncoderExecution:
    """Resolve the analysis-only encoder execution arm at startup.

    ``eager`` is the compatibility baseline when the field is absent.
    ``compiled-static`` uses a full, static graph and deliberately has no
    eager fallback: a graph break or compilation failure invalidates that
    experimental arm. ``dynamic=False`` may cache multiple exact-shape
    specializations; the profiling warmup must cover the measured shapes.
    ``dense-graphed`` retains that compiler domain and then explicitly captures
    every exact geometry/population transition through the platform graph seam.
    """
    if isinstance(maximum_population, bool) or not isinstance(maximum_population, int) or maximum_population <= 0:
        raise ValueError("maximum encoder population must be positive")
    geometries = tuple(int(geometry) for geometry in warmup_geometries)
    if not geometries:
        raise ValueError("encoder warmup geometries must not be empty")
    if any(geometry < 0 for geometry in geometries) or len(set(geometries)) != len(geometries):
        raise ValueError("encoder warmup geometries must be unique nonnegative ids")
    arm = getattr(hf_config, "encoder_execution_arm", None) or "eager"
    if arm not in {"eager", "compiled-static", "dense-graphed"}:
        raise ValueError(
            f"unknown encoder_execution_arm {arm!r} (known: ['compiled-static', 'dense-graphed', 'eager'])"
        )

    def transition(
        mel: torch.Tensor,
        caches: EncoderCaches,
        out_offsets: torch.Tensor,
        out_lengths: torch.Tensor,
        out_width: int,
        prompt_index: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return execute_encoder_transition(
            core,
            mel,
            caches,
            out_offsets,
            out_lengths,
            out_width,
            prompt_index,
        )

    if arm == "eager":
        return ResolvedEncoderExecution(arm=arm, transition=transition)
    if arm == "dense-graphed" and vllm_config is None:
        raise ValueError("dense-graphed encoder execution requires vllm_config")
    if arm == "dense-graphed":
        num_prompts = getattr(core.lid, "num_prompts", None)
        if isinstance(num_prompts, bool) or not isinstance(num_prompts, int) or num_prompts <= 0:
            raise ValueError("dense-graphed encoder requires a positive integer lid.num_prompts")
    history_frames = int(hf_config.att_context_left)
    if history_frames < 0:
        raise ValueError("encoder attention history must be nonnegative")
    geometry_shapes = tuple(encoder_geometry_shape(core, geometry) for geometry in geometries)
    geometry_shape_by_id = dict(zip(geometries, geometry_shapes, strict=True))
    t_cap = max(shape.out_width + history_frames for shape in geometry_shapes)
    populations = tuple(range(1, maximum_population + 1))
    specialization_budget = len(geometries) * len(populations)
    compiled = torch.compile(
        transition,
        fullgraph=True,
        dynamic=False,
        options={"triton.cudagraphs": False},
    )
    execution = ResolvedEncoderExecution(
        arm=arm,
        transition=transition,
        warmup_geometries=geometries,
        warmup_populations=populations,
        t_cap=t_cap,
        _history_frames=history_frames,
        _geometry_shapes=geometry_shape_by_id,
        _core=core,
        _compiled_transition=compiled,
        _vllm_config=vllm_config,
        _graph_runtime=(graph_runtime or platform_graph_runtime() if arm == "dense-graphed" else None),
    )

    def guarded_transition(
        mel: torch.Tensor,
        caches: EncoderCaches,
        out_offsets: torch.Tensor,
        out_lengths: torch.Tensor,
        out_width: int,
        prompt_index: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        execution._raise_if_failed()
        if not execution._sealed and execution._active_cell is None:
            raise ValueError(f"{arm} encoder pre-seal invocation lacks declared cell authority")
        signature = _encoder_signature(
            mel,
            caches,
            out_offsets,
            out_lengths,
            out_width,
            prompt_index,
        )
        population = signature.mel.shape[0]
        if population not in execution.warmup_populations:
            raise ValueError(f"{arm} encoder population {population} was not declared")
        required = int(out_width) + execution._history_frames
        if required > execution.t_cap:
            raise ValueError(
                "encoder.pos_enc.pe capacity is smaller than the transition window: "
                f"required={required}, declared={execution.t_cap}"
            )
        if int(caches.left_context) != execution._history_frames:
            raise ValueError("encoder transition history differs from declared resident history")
        active_cell = execution._active_cell
        if active_cell is not None:
            geometry = active_cell[0]
            declared_shape = execution._geometry_shapes[geometry]
            if int(mel.shape[-1]) != declared_shape.mel_width or int(out_width) != declared_shape.out_width:
                raise ValueError(
                    "encoder transition shape differs from declared geometry "
                    f"{geometry}: mel_width={int(mel.shape[-1])}, "
                    f"out_width={int(out_width)}, "
                    f"declared_mel_width={declared_shape.mel_width}, "
                    f"declared_out_width={declared_shape.out_width}"
                )
        if execution._sealed and signature not in execution._allowed_signatures:
            raise ValueError(f"{arm} encoder signature was not warmed")
        if execution._sealed:
            if execution.arm == "dense-graphed":
                staging_cell = execution._staging_cell
                if staging_cell is not None:
                    execution._stage_graph_entry(
                        cell=staging_cell,
                        signature=signature,
                        mel=mel,
                        caches=caches,
                        out_offsets=out_offsets,
                        out_lengths=out_lengths,
                        out_width=out_width,
                        prompt_index=prompt_index,
                    )
                    result = compiled(
                        mel,
                        caches,
                        out_offsets,
                        out_lengths,
                        out_width,
                        prompt_index,
                    )
                else:
                    result = execution._replay_graph_entry(
                        signature=signature,
                        mel=mel,
                        caches=caches,
                        out_offsets=out_offsets,
                        out_lengths=out_lengths,
                        prompt_index=prompt_index,
                    )
            else:
                result = compiled(
                    mel,
                    caches,
                    out_offsets,
                    out_lengths,
                    out_width,
                    prompt_index,
                )
        else:
            if active_cell is None:
                raise ValueError(f"{arm} encoder pre-seal invocation lacks declared cell authority")
            if population != active_cell[1]:
                raise ValueError(f"{arm} encoder population differs from declared cell")
            execution._materialize_preprofile_state(out_width)
            if execution._runner_device is None:
                raise ValueError(f"{arm} encoder runner device is unavailable")
            _validate_transition_tensor_contract(
                mel,
                caches,
                out_offsets,
                out_lengths,
                prompt_index,
                runner_device=execution._runner_device,
            )
            with _compiler_specialization_budget(specialization_budget):
                result = compiled(
                    mel,
                    caches,
                    out_offsets,
                    out_lengths,
                    out_width,
                    prompt_index,
                )
        execution._invocations += 1
        execution._last_signature = signature
        return result

    execution.transition = guarded_transition
    return execution


__all__ = [
    "EncoderCaches",
    "EncoderGeometryShape",
    "EncoderTransition",
    "ResolvedEncoderExecution",
    "build_encoder_execution",
    "encoder_geometry_shape",
    "execute_encoder_transition",
]
