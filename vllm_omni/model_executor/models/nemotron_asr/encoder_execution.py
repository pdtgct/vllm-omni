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
    )


def _tensor_group_signature(value: Any) -> tuple[TensorSignature, ...]:
    if isinstance(value, torch.Tensor):
        return (_tensor_signature(value),)
    return tuple(_tensor_signature(tensor) for tensor in value)


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
    _invocations: int = 0
    _last_signature: EncoderSignature | None = None
    _warmup_signatures: dict[tuple[int, int], EncoderSignature] = field(default_factory=dict)
    _profile_signature: tuple[tuple[int, int], EncoderSignature] | None = None
    _allowed_signatures: frozenset[EncoderSignature] = frozenset()
    _active_cell: tuple[int, int] | None = None
    _sealed: bool = False

    @property
    def ready(self) -> bool:
        """Whether the selected arm is safe to admit served work."""
        return self.arm == "eager" or self._sealed

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
        if self._sealed:
            raise ValueError("compiled-static encoder execution is already sealed")
        cell = (int(geometry), int(population))
        if cell in self._warmup_signatures:
            raise ValueError(f"encoder warmup cell {cell} was repeated")
        if cell[0] not in self.warmup_geometries:
            raise ValueError(f"encoder warmup geometry {cell[0]} was not declared")
        if cell[1] not in self.warmup_populations:
            raise ValueError(f"encoder warmup population {cell[1]} was not declared")
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
        self._warmup_signatures[cell] = self._last_signature

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
        if self._sealed:
            raise ValueError("compiled-static encoder execution is already sealed")
        if self._profile_signature is not None:
            raise ValueError("compiled-static encoder memory profile was repeated")
        cell = (int(geometry), int(population))
        before = self._invocations
        result = self._invoke_declared_cell(cell=cell, invoke=invoke)
        if self._invocations != before + 1 or self._last_signature is None:
            raise ValueError("encoder memory profile must execute exactly one transition")
        self._profile_signature = (cell, self._last_signature)
        return result

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
            return invoke()
        finally:
            self._active_cell = None

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
            self._warmup_signatures.clear()
            raise

    def _seal(self) -> None:
        """Seal the exact signatures proven by product-owned warmup."""
        if self.arm != "compiled-static":
            return
        expected = {
            (geometry, population)
            for geometry in self.warmup_geometries
            for population in self.warmup_populations
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
    )

    def guarded_transition(
        mel: torch.Tensor,
        caches: EncoderCaches,
        out_offsets: torch.Tensor,
        out_lengths: torch.Tensor,
        out_width: int,
        prompt_index: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
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
            if execution._active_cell is None:
                raise ValueError("compiled-static encoder pre-seal invocation lacks declared cell authority")
            if population != execution._active_cell[1]:
                raise ValueError("compiled-static encoder population differs from declared cell")
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
    "EncoderTransition",
    "ResolvedEncoderExecution",
    "build_encoder_execution",
    "execute_encoder_transition",
]
