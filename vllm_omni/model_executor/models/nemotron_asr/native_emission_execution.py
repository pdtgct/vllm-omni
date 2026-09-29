# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Opt-in, startup-sealed compilation of native emission tensor operations."""

from typing import Any, NamedTuple, cast

import torch

from vllm_omni.model_executor.models.nemotron_asr.advance import (
    ROLE_CHUNK,
    ROLE_EOU,
    ROLE_FLUSH,
    ROLE_REPLAY,
    EmissionContext,
    EmissionProjection,
)
from vllm_omni.model_executor.models.nemotron_asr.encoder_execution import _tensor_signature
from vllm_omni.model_executor.models.nemotron_asr.manifests import BOOK_FIELDS, SESSION_LIMITS
from vllm_omni.model_executor.models.nemotron_asr.native_burst import (
    NativeBurstProjection,
    finalize_native_burst,
    native_burst_invariant_rows,
)


class _Context(NamedTuple):
    roles: torch.Tensor
    book: torch.Tensor


class _Source(NamedTuple):
    queue: torch.Tensor
    book: torch.Tensor
    row_status: torch.Tensor


class _Payload(NamedTuple):
    queue: torch.Tensor
    book: torch.Tensor
    row_status: torch.Tensor
    sampled_token_ids: torch.Tensor
    num_sampled: torch.Tensor


def _finalize_tensors(rows, queue, book, status, roles, previous_book, *, park_id, blank_id):
    if not torch.compiler.is_compiling():
        raise RuntimeError("native emission compiled entrypoint cannot execute eagerly")
    # Only fields consumed by the existing function enter its specialization.
    return finalize_native_burst(
        EmissionProjection(rows, queue, book, status),
        cast(EmissionContext, _Context(roles, previous_book)),
        park_id=park_id,
        blank_id=blank_id,
    )


def _invariant_tensors(
    queue,
    book,
    status,
    roles,
    previous_book,
    native_queue,
    native_book,
    native_status,
    tokens,
    count,
    *,
    park_id,
    blank_id,
    eou_token_id,
):
    if not torch.compiler.is_compiling():
        raise RuntimeError("native emission compiled entrypoint cannot execute eagerly")
    return native_burst_invariant_rows(
        cast(EmissionProjection, _Source(queue, book, status)),
        cast(EmissionContext, _Context(roles, previous_book)),
        cast(NativeBurstProjection, _Payload(native_queue, native_book, native_status, tokens, count)),
        park_id=park_id,
        blank_id=blank_id,
        eou_token_id=eou_token_id,
    )


def _producer_inputs(source, context):
    return source.rows, source.queue, source.book, source.row_status, context.roles, context.book


def _validator_inputs(source, context, native):
    return (
        source.queue,
        source.book,
        source.row_status,
        context.roles,
        context.book,
        native.queue,
        native.book,
        native.row_status,
        native.sampled_token_ids,
        native.num_sampled,
    )


def _signature(tensors):
    if any(type(t) is not torch.Tensor or t.layout != torch.strided for t in tensors):
        raise ValueError("native emission signature requires ordinary strided tensors")
    # Storage identity is used only to canonicalize alias relationships; no
    # address is retained and no tensor value or device scalar is read.
    groups = {}
    aliases = tuple(groups.setdefault((str(t.device), t.untyped_storage().data_ptr()), len(groups)) for t in tensors)
    return tuple(_tensor_signature(t) for t in tensors), aliases, torch.is_inference_mode_enabled()


class NativeEmissionExecution:
    """Two pure compiled entrypoints; no request, state, reservation or graph owner."""

    def __init__(self, *, populations, hidden_size, park_id, blank_id, eou_token_id, max_tokens):
        self.populations = tuple(populations)
        self._hidden_size = hidden_size
        self._identity = (park_id, blank_id, eou_token_id, max_tokens)
        self._functions: dict[str, Any] = {}
        self._signatures: dict[tuple[str, int], Any] = {}
        self._calls: dict[tuple[str, int], int] = {}
        self._device: torch.device | None = None
        self._ready = False
        self._failed = False

    @property
    def ready(self):
        return self._ready and not self._failed

    def _require_ready(self):
        if self._failed:
            raise RuntimeError("native emission compilation failed; a fresh worker is required")
        if not self._ready:
            raise RuntimeError("native emission execution is not ready")

    def _check_identity(self, *, park_id, blank_id, max_tokens, eou_token_id=None, validator=False):
        expected = self._identity
        if (park_id, blank_id, max_tokens) != (expected[0], expected[1], expected[3]) or (
            validator and eou_token_id != expected[2]
        ):
            raise ValueError("native emission execution identity changed")

    def _invoke(self, operation, population, tensors):
        if _signature(tensors) != self._signatures[(operation, population)]:
            raise ValueError(f"native emission {operation} signature differs from startup warmup")
        kwargs = dict(park_id=self._identity[0], blank_id=self._identity[1])
        if operation == "invariant":
            kwargs["eou_token_id"] = self._identity[2]
        try:
            # Full-graph compilation must never specialize or silently fall
            # back to eager after the finite domain has been warmed.
            with (
                torch._dynamo.config.patch(error_on_recompile=True, suppress_errors=False),
                torch.compiler.set_stance("fail_on_recompile"),
            ):
                result = self._functions[operation](*tensors, **kwargs)
        except Exception:
            self._failed = True
            raise
        self._calls[(operation, population)] += 1
        return result

    # @spec PORT-DEC-011, PORT-DEC-012, PORT-PERF-011
    def finalize(self, source, context, *, park_id, blank_id, max_tokens):
        self._require_ready()
        self._check_identity(park_id=park_id, blank_id=blank_id, max_tokens=max_tokens)
        population = source.queue.shape[0]
        if population not in self.populations:
            return finalize_native_burst(source, context, park_id=park_id, blank_id=blank_id)
        return self._invoke("finalize", population, _producer_inputs(source, context))

    # @spec PORT-ADV-004, PORT-DEC-012, PORT-PERF-011
    def invariant_rows(self, source, context, native, *, park_id, blank_id, eou_token_id, max_tokens):
        self._require_ready()
        self._check_identity(
            park_id=park_id, blank_id=blank_id, max_tokens=max_tokens, eou_token_id=eou_token_id, validator=True
        )
        population = source.queue.shape[0]
        if population not in self.populations:
            return native_burst_invariant_rows(
                source, context, native, park_id=park_id, blank_id=blank_id, eou_token_id=eou_token_id
            )
        return self._invoke("invariant", population, _validator_inputs(source, context, native))

    def _warm_inputs(self, population, variant, device):
        capacity = SESSION_LIMITS["queue_capacity"]
        queue = (
            (torch.arange(capacity, device=device, dtype=torch.int32) % self._identity[1])
            .expand(population, -1)
            .clone()
        )
        book = torch.zeros(population, len(BOOK_FIELDS), device=device, dtype=torch.int32)
        length = (
            0 if variant == "blank" else capacity if variant == "full" else capacity + 1 if variant == "overflow" else 2
        )
        book[:, 0] = int(length > 0)
        book[:, 1] = length
        book[:, 5] = int(length > 0)
        status = torch.zeros(population, device=device, dtype=torch.int32)
        role = {"replay": ROLE_REPLAY, "flush": ROLE_FLUSH, "eou": ROLE_EOU}.get(variant, ROLE_CHUNK)
        roles = torch.full((population,), role, device=device, dtype=torch.int64)
        if variant == "failed":
            status[::2] = 512
        if variant == "eou":
            queue[:, 0] = self._identity[2]
            book[:, 1] = 1
        source = EmissionProjection(
            torch.zeros(population, self._hidden_size, device=device, dtype=torch.float32), queue, book, status
        )
        return source, _Context(roles, book.clone())

    # @spec PORT-PERF-009, PORT-PERF-011
    @torch.inference_mode()
    def warmup(self, device):
        if self._failed:
            raise RuntimeError("native emission compilation failed; a fresh worker is required")
        device = torch.device(device)
        if self._ready:
            if self._device != device:
                raise ValueError("native emission warmup device changed")
            return
        try:
            with (
                torch._dynamo.config.patch(error_on_recompile=False, suppress_errors=False),
                torch.compiler.set_stance("default"),
            ):
                self._functions = {
                    name: torch.compile(fn, fullgraph=True, dynamic=False, options={"triton.cudagraphs": False})
                    for name, fn in (("finalize", _finalize_tensors), ("invariant", _invariant_tensors))
                }
                retained = []
                for population in self.populations:
                    variants = ("blank", "full", "replay", "failed", "flush", "overflow")
                    if self._identity[2] is not None:
                        variants += ("eou",)
                    for variant in variants:
                        source, context = self._warm_inputs(population, variant, device)
                        producer_inputs = _producer_inputs(source, context)
                        expected = finalize_native_burst(
                            source, context, park_id=self._identity[0], blank_id=self._identity[1]
                        )
                        actual = self._functions["finalize"](
                            *producer_inputs, park_id=self._identity[0], blank_id=self._identity[1]
                        )
                        for field in actual.__dataclass_fields__:
                            torch.testing.assert_close(getattr(actual, field), getattr(expected, field), rtol=0, atol=0)
                            if any(
                                getattr(actual, field).untyped_storage().data_ptr() == t.untyped_storage().data_ptr()
                                for t in producer_inputs
                            ):
                                raise ValueError("compiled native emission output aliases input storage")
                        validator_inputs = _validator_inputs(source, context, actual)
                        expected_bad = native_burst_invariant_rows(
                            source,
                            context,
                            actual,
                            park_id=self._identity[0],
                            blank_id=self._identity[1],
                            eou_token_id=self._identity[2],
                        )
                        bad = self._functions["invariant"](
                            *validator_inputs,
                            park_id=self._identity[0],
                            blank_id=self._identity[1],
                            eou_token_id=self._identity[2],
                        )
                        torch.testing.assert_close(bad, expected_bad, rtol=0, atol=0)
                        for operation, tensors in (("finalize", producer_inputs), ("invariant", validator_inputs)):
                            key = operation, population
                            signature = _signature(tensors)
                            if self._signatures.setdefault(key, signature) != signature:
                                raise ValueError("native emission startup cases have inconsistent signatures")
                            self._calls[key] = 0
                        retained.append(
                            (actual, tuple(getattr(actual, f).clone() for f in actual.__dataclass_fields__))
                        )
                for actual, saved in retained:
                    for field, expected in zip(actual.__dataclass_fields__, saved, strict=True):
                        torch.testing.assert_close(getattr(actual, field), expected, rtol=0, atol=0)
            self._device = device
            self._ready = True
        except Exception:
            self._failed = True
            raise

    def require_profile_coverage(self):
        self._require_ready()
        if any(
            self._calls.get((operation, population), 0) == 0
            for operation in ("finalize", "invariant")
            for population in self.populations
        ):
            raise RuntimeError("native emission compiled profile coverage is incomplete")

    def receipt(self):
        return dict(
            arm="experimental-compiled-native-emission",
            ready=self.ready,
            populations=list(self.populations),
            eager_other_populations=True,
            fullgraph=True,
            dynamic=False,
            cudagraphs=False,
            hidden_size=self._hidden_size,
            queue_capacity=SESSION_LIMITS["queue_capacity"],
            identity=dict(zip(("park_id", "blank_id", "eou_token_id", "max_tokens"), self._identity)),
            compiled_calls={f"{op}/{pop}": calls for (op, pop), calls in self._calls.items()},
        )


def build_native_emission_execution(config, *, maximum_population, park_id, blank_id, max_tokens):
    enabled = getattr(config, "experimental_native_emission_compile", False)
    if not isinstance(enabled, bool):
        raise ValueError("experimental native emission compile selection must be a boolean")
    if not enabled:
        return None
    lookaheads = getattr(config, "supported_num_lookahead_tokens", ())
    if (
        getattr(config, "experimental_native_burst", False) is not True
        or not isinstance(lookaheads, (tuple, list))
        or tuple(lookaheads) != (1,)
    ):
        raise ValueError("experimental native emission compilation requires native burst and the 160-ms profile")
    eou = getattr(config, "eou_token_id", None)
    if (
        any(isinstance(value, bool) or not isinstance(value, int) for value in (park_id, blank_id, max_tokens))
        or blank_id <= 0
        or park_id <= blank_id
        or not 1 <= max_tokens <= SESSION_LIMITS["queue_capacity"] + 1
        or (
            eou is not None and (isinstance(eou, bool) or not isinstance(eou, int) or eou <= blank_id or eou == park_id)
        )
    ):
        raise ValueError("native emission identity requires valid integer token ids and a bounded token budget")
    hidden = getattr(config, "hidden_size", 0)
    if isinstance(maximum_population, bool) or not isinstance(maximum_population, int) or maximum_population < 1:
        raise ValueError("native emission maximum population must be positive")
    if isinstance(hidden, bool) or not isinstance(hidden, int) or hidden <= 0:
        raise ValueError("native emission hidden size must be positive")
    return NativeEmissionExecution(
        populations=tuple(range(1, min(2, maximum_population) + 1)),
        hidden_size=hidden,
        park_id=park_id,
        blank_id=blank_id,
        eou_token_id=eou,
        max_tokens=max_tokens,
    )
