# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Feature-gated execution for the static streaming encoder transition.

Both arms execute the same encoder, language-conditioning, and padded-row
zeroing function. The candidate changes only the execution mechanism: it
lets Inductor specialize one full graph for each exact host-derived tensor
shape. Compiler CUDA graphs stay disabled because session cache scratch is
mutable and its addresses are not a stable graph-owned input contract.
"""

from __future__ import annotations

from collections.abc import Callable, Generator
from contextlib import contextmanager
from dataclasses import dataclass, field
from threading import RLock
from typing import TYPE_CHECKING, Any, NamedTuple, Protocol, cast

import torch

from vllm_omni.model_executor.models.nemotron_asr.encoder import (
    StreamingCaches,
    stream_step,
)

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
    )


# Dynamo's cache limits are process-global. Nemotron enters this lock only
# during startup-owned, pre-admission compilation so nested or overlapping
# Nemotron scopes cannot restore stale values.
_COMPILER_BUDGET_LOCK = RLock()


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

    @property
    def ready(self) -> bool:
        """Whether the selected arm is safe to admit served work."""
        return self.arm == "eager" or (self._sealed and not self._failed)

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
        if self.arm != "compiled-static":
            raise ValueError("encoder warmup cells require compiled-static")
        self._raise_if_failed()
        if self._sealed:
            raise ValueError("compiled-static encoder execution is already sealed")
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
        invoke: Callable[[], Any],
    ) -> Any:
        """Authorize and attest the sole activation-memory profile cell."""
        if self.arm != "compiled-static":
            return invoke()
        self._raise_if_failed()
        if self._sealed:
            raise ValueError("compiled-static encoder execution is already sealed")
        if self._profile_signature is not None:
            raise ValueError("compiled-static encoder memory profile was repeated")
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
        invoke: Callable[[], Any],
    ) -> Any:
        if cell[0] not in self.warmup_geometries:
            raise ValueError(f"encoder geometry {cell[0]} was not declared")
        if cell[1] not in self.warmup_populations:
            raise ValueError(f"encoder population {cell[1]} was not declared")
        if self._active_cell is not None:
            raise ValueError("compiled-static encoder cell invocation overlapped")
        self._active_cell = cell
        try:
            with torch.inference_mode():
                return invoke()
        finally:
            self._active_cell = None

    def _raise_if_failed(self) -> None:
        if self._failed:
            raise ValueError("compiled-static encoder authority is failed and discarded")

    def _discard(self) -> None:
        self._failed = True
        self._sealed = False
        self._warmup_signatures.clear()
        self._profile_signature = None
        self._allowed_signatures = frozenset()

    def _materialize_preprofile_state(
        self,
        out_width: int,
    ) -> None:
        if self._core is None or self.t_cap <= 0:
            raise ValueError("compiled-static encoder positional capacity is unavailable")
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
            raise ValueError("compiled-static encoder model state was not materialized")
        changed = _changed_model_state_name(
            self._model_state,
            _compiled_transition_model_state(self._core),
        )
        if changed is not None:
            raise ValueError(f"compiled-static encoder model state changed at {changed}")

    # @spec PORT-PERF-009
    def warmup_domain(
        self,
        *,
        expected_cells: tuple[tuple[int, int], ...],
        invoke: Callable[[int, int], Any],
    ) -> None:
        """Compile and seal the complete finite specialization domain."""
        if self.arm != "compiled-static":
            raise ValueError("encoder warmup domain requires compiled-static")
        self._raise_if_failed()
        if self._sealed:
            raise ValueError("compiled-static encoder execution is already sealed")
        if self._warmup_signatures:
            raise ValueError("compiled-static encoder warmup already started")
        cells = tuple((int(geometry), int(population)) for geometry, population in expected_cells)
        if not cells:
            raise ValueError("compiled-static encoder warmup domain is empty")
        if len(set(cells)) != len(cells):
            raise ValueError("compiled-static encoder warmup cells repeat")
        expected_domain = {
            (geometry, population) for geometry in self.warmup_geometries for population in self.warmup_populations
        }
        if set(cells) != expected_domain:
            missing = sorted(expected_domain - set(cells))
            unexpected = sorted(set(cells) - expected_domain)
            raise ValueError(
                "compiled-static encoder warmup domain differs from authority: "
                f"missing={missing}, unexpected={unexpected}"
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
        except Exception:
            self._discard()
            raise

    def _seal(self) -> None:
        """Seal the exact signatures proven by product-owned warmup."""
        if self.arm != "compiled-static":
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
                "compiled-static encoder warmup cells differ from authority: "
                f"missing={missing}, unexpected={unexpected}"
            )
        self._allowed_signatures = frozenset(self._warmup_signatures.values())
        self._sealed = True

    def ready_receipt(self) -> dict[str, Any]:
        """Return the machine-readable resolved-arm readiness receipt."""
        return {
            "arm": self.arm,
            "ready": self.ready,
            "warmup_cells": [[geometry, population] for geometry, population in sorted(self._warmup_signatures)],
            "warmup_geometries": list(self.warmup_geometries),
            "warmup_populations": list(self.warmup_populations),
        }


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


def build_encoder_execution(
    core: NemotronASRCore,
    hf_config: Any,
    *,
    maximum_population: int,
    warmup_geometries: tuple[int, ...],
) -> ResolvedEncoderExecution:
    """Resolve the analysis-only encoder execution arm at startup.

    ``eager`` is the compatibility baseline when the field is absent.
    ``compiled-static`` uses a full, static graph and deliberately has no
    eager fallback: a graph break or compilation failure invalidates that
    experimental arm. ``dynamic=False`` may cache multiple exact-shape
    specializations; the profiling warmup must cover the measured shapes.
    """
    if isinstance(maximum_population, bool) or not isinstance(maximum_population, int) or maximum_population <= 0:
        raise ValueError("maximum encoder population must be positive")
    geometries = tuple(int(geometry) for geometry in warmup_geometries)
    if not geometries:
        raise ValueError("encoder warmup geometries must not be empty")
    if any(geometry < 0 for geometry in geometries) or len(set(geometries)) != len(geometries):
        raise ValueError("encoder warmup geometries must be unique nonnegative ids")
    arm = getattr(hf_config, "encoder_execution_arm", None) or "eager"
    if arm not in {"eager", "compiled-static"}:
        raise ValueError(f"unknown encoder_execution_arm {arm!r} (known: ['compiled-static', 'eager'])")

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
            raise ValueError("compiled-static encoder pre-seal invocation lacks declared cell authority")
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
            raise ValueError(f"compiled-static encoder population {population} was not declared")
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
            raise ValueError("compiled-static encoder signature was not warmed")
        if execution._sealed:
            result: tuple[torch.Tensor, torch.Tensor] = compiled(
                mel,
                caches,
                out_offsets,
                out_lengths,
                out_width,
                prompt_index,
            )
        else:
            if active_cell is None:
                raise ValueError("compiled-static encoder pre-seal invocation lacks declared cell authority")
            if population != active_cell[1]:
                raise ValueError("compiled-static encoder population differs from declared cell")
            execution._materialize_preprofile_state(out_width)
            if execution._runner_device is None:
                raise ValueError("compiled-static encoder runner device is unavailable")
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
