# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Bounded, opt-in graph of the shared MRV1 projection and its independent checks."""

import threading
from dataclasses import dataclass
from typing import Any

import torch

from vllm_omni.model_executor.models.nemotron_asr.advance import (
    ROLE_CHUNK,
    ROLE_FLUSH,
    AdvanceResult,
    EmissionContext,
    EmissionProjection,
    _project_and_validate_emission,
    make_mrv1_adapter,
)
from vllm_omni.model_executor.models.nemotron_asr.decode_graph import GraphRuntime, platform_graph_runtime
from vllm_omni.model_executor.models.nemotron_asr.encoder_execution import _tensor_signature
from vllm_omni.model_executor.models.nemotron_asr.rnnt import BOOK_GEOMETRY, QUEUE_LAST_LABEL


def _inputs(
    merged: AdvanceResult, context: EmissionContext, geometry: torch.Tensor, endpoint: torch.Tensor | None
) -> tuple[torch.Tensor, ...]:
    assert merged.row_status is not None
    values = (
        merged.token_ids,
        merged.token_lengths,
        merged.row_status,
        context.roles,
        context.input_ids,
        context.chunk_rows,
        context.queue,
        context.book,
        context.prompt_index,
        context.row_status,
        geometry,
    )
    return values if endpoint is None else (*values, endpoint)


def _unpack(
    values: tuple[torch.Tensor, ...],
) -> tuple[AdvanceResult, EmissionContext, torch.Tensor, torch.Tensor | None]:
    return (
        AdvanceResult(values[0], values[1], row_status=values[2]),
        EmissionContext(*values[3:10]),
        values[10],
        values[11] if len(values) == 12 else None,
    )


def _outputs(result: tuple[EmissionProjection, torch.Tensor, torch.Tensor | None]) -> tuple[torch.Tensor, ...]:
    projection, status, endpoint = result
    values = (projection.rows, projection.queue, projection.book, projection.row_status, status)
    return values if endpoint is None else (*values, endpoint)


def _stream_identity(device: torch.device) -> int | None:
    # The experimental domain is CUDA only in production. CPU serves boundary tests.
    return torch.cuda.current_stream(device).cuda_stream if device.type == "cuda" else None


@dataclass
class _Entry:
    scratch: tuple[torch.Tensor, ...]
    owned: tuple[torch.Tensor, ...]
    signature: tuple[Any, ...]
    call: Any
    replay_count: int = 0


class EmissionGraphBinding:
    """One model generation's pre-captured exact N/B/K cells, never a lazy cache.

    The factory-created adapter is retained by identity. Custom adapter calls and
    unsupported metadata execute the unchanged eager region. Only fixed inputs
    and independent outputs belong to graph storage; resident pages never do.
    """

    def __init__(
        self,
        *,
        hidden_size: int,
        park_id: int,
        blank_id: int,
        queue_capacity: int,
        token_widths: tuple[int, ...],
        maximum_population: int,
        eou_token_id: int | None,
        vllm_config: Any,
        runtime: GraphRuntime | None = None,
    ):
        if hidden_size <= 0 or queue_capacity <= 0 or maximum_population <= 0:
            raise ValueError("emission graph requires positive dimensions")
        if not token_widths or any(type(k) is not int or k <= 0 for k in token_widths):
            raise ValueError("emission graph requires explicit positive token widths")
        self._hidden, self._park, self._blank = hidden_size, park_id, blank_id
        self._capacity, self._eou = queue_capacity, eou_token_id
        self._config, self._runtime = vllm_config, runtime
        self._adapter = make_mrv1_adapter(hidden_size=hidden_size, park_id=park_id, blank_id=blank_id)
        self._keys = frozenset(
            (n, b, k)
            for n in range(1, min(2, maximum_population) + 1)
            for b in range(n + 1)
            for k in ((0,) if b == 0 else sorted(set(token_widths)))
        )
        self._entries: dict[tuple[int, int, int], _Entry] = {}
        self._fallbacks: dict[tuple[int, int, int], int] = {}
        self._storages: frozenset[int] = frozenset()
        self._failed = False
        self._thread: int | None = None
        self._stream: int | None = None
        self._guard = threading.Lock()

    @property
    def adapter(self) -> Any:
        return self._adapter

    @property
    def ready(self) -> bool:
        return not self._failed and self._entries.keys() == self._keys

    @property
    def captured_keys(self) -> tuple[tuple[int, int, int], ...]:
        return tuple(sorted(self._entries))

    def _kwargs(self, geometry: torch.Tensor, endpoint: torch.Tensor | None) -> dict[str, Any]:
        return dict(
            adapter=self._adapter,
            hidden=self._hidden,
            rows_dtype=torch.float32,
            park_id=self._park,
            blank_id=self._blank,
            plan_geometry=geometry,
            endpoint_book=endpoint,
            eou_token_id=self._eou,
        )

    def _example(self, key: tuple[int, int, int], device: torch.device) -> tuple[torch.Tensor, ...]:
        n, b, k = key
        merged = AdvanceResult(
            torch.zeros(b, k, dtype=torch.int32, device=device),
            torch.zeros(b, dtype=torch.int32, device=device),
            row_status=torch.zeros(b, dtype=torch.int32, device=device),
        )
        book = torch.zeros(n, 7, dtype=torch.int32, device=device)
        book[:, QUEUE_LAST_LABEL] = self._blank
        book[:, BOOK_GEOMETRY] = 1
        roles = torch.full((n,), ROLE_FLUSH, dtype=torch.long, device=device)
        roles[:b] = ROLE_CHUNK
        context = EmissionContext(
            roles,
            torch.zeros(n, dtype=torch.long, device=device),
            torch.arange(b, device=device),
            torch.zeros(n, self._capacity, dtype=torch.int32, device=device),
            book,
            torch.zeros(n, dtype=torch.long, device=device),
            torch.zeros(n, dtype=torch.int32, device=device),
        )
        endpoint = torch.zeros(n, 6, dtype=torch.int32, device=device) if self._eou is not None else None
        return _inputs(merged, context, torch.ones(n, dtype=torch.long, device=device), endpoint)

    @torch.inference_mode()
    def warmup(self, device: torch.device) -> None:
        if self._failed:
            raise RuntimeError("emission graph startup failed; a fresh binding is required")
        if self.ready:
            return
        runtime = self._runtime or platform_graph_runtime()
        # Capture contexts may switch to side streams; admission retains the
        # calling worker stream, after capture has returned to that stream.
        owner_thread, owner_stream = threading.get_ident(), _stream_identity(device)
        pending = {}
        try:
            for key in sorted(self._keys):
                pending[key] = self._capture(self._example(key, device), runtime)
            runtime.synchronize(device)
        except Exception:
            self._failed = True
            self._entries.clear()
            raise
        self._runtime = runtime
        self._thread, self._stream = owner_thread, owner_stream
        self._storages = frozenset(
            t.untyped_storage().data_ptr()
            for entry in pending.values()
            for t in (*entry.scratch, *entry.owned)
            if t.numel()
        )
        self._entries = pending

    def _capture(self, sources: tuple[torch.Tensor, ...], runtime: GraphRuntime) -> _Entry:
        scratch = tuple(t.clone() for t in sources)
        merged, context, geometry, endpoint = _unpack(scratch)
        kwargs = self._kwargs(geometry, endpoint)
        probe = _project_and_validate_emission(merged, context, **kwargs)
        owned = tuple(torch.empty_like(t) for t in _outputs(probe))

        def body():
            result = _project_and_validate_emission(merged, context, **kwargs)
            for destination, source in zip(owned, _outputs(result), strict=True):
                destination.copy_(source)
            return owned

        wrapper = runtime.wrapper_factory(body, self._config, runtime_mode=runtime.graph_mode)
        descriptor = runtime.descriptor_factory(int(context.roles.shape[0]))

        def call(mode):
            with runtime.forward_context(None, self._config, cudagraph_runtime_mode=mode, batch_descriptor=descriptor):
                wrapper()
            return owned

        for _ in range(3):
            call(runtime.eager_mode)
        expected = tuple(t.clone() for t in call(runtime.eager_mode))
        runtime.synchronize(context.book.device)
        runtime.set_capture_enabled(True)
        try:
            with runtime.capture_context(context.book.device):
                call(runtime.graph_mode)
            actual = call(runtime.graph_mode)
            for left, right in zip(actual, expected, strict=True):
                torch.testing.assert_close(left, right, atol=0, rtol=0)
            runtime.synchronize(context.book.device)
        finally:
            runtime.set_capture_enabled(False)
        return _Entry(scratch, owned, tuple(_tensor_signature(t) for t in sources), call)

    @torch.inference_mode()
    def project(
        self,
        merged: AdvanceResult,
        context: EmissionContext,
        *,
        adapter: Any,
        hidden: int,
        rows_dtype: torch.dtype,
        park_id: int,
        blank_id: int,
        plan_geometry: torch.Tensor,
        endpoint_book: torch.Tensor | None = None,
        eou_token_id: int | None = None,
    ):
        if not self.ready:
            raise RuntimeError("emission graph inventory is incomplete")
        values = _inputs(merged, context, plan_geometry, endpoint_book)
        key = (int(context.roles.shape[0]), int(context.chunk_rows.shape[0]), int(merged.token_ids.shape[1]))
        entry = self._entries.get(key)
        if any(t.numel() and t.untyped_storage().data_ptr() in self._storages for t in values):
            raise ValueError("emission caller aliases graph-owned storage")
        eligible = (
            entry is not None
            and adapter is self._adapter
            and hidden == self._hidden
            and rows_dtype == torch.float32
            and park_id == self._park
            and blank_id == self._blank
            and eou_token_id == self._eou
            and (endpoint_book is not None) == (self._eou is not None)
            and tuple(_tensor_signature(t) for t in values) == entry.signature
        )
        if not eligible:
            result = _project_and_validate_emission(
                merged,
                context,
                adapter=adapter,
                hidden=hidden,
                rows_dtype=rows_dtype,
                park_id=park_id,
                blank_id=blank_id,
                plan_geometry=plan_geometry,
                endpoint_book=endpoint_book,
                eou_token_id=eou_token_id,
            )
            self._fallbacks[key] = self._fallbacks.get(key, 0) + 1
            return result
        if threading.get_ident() != self._thread:
            raise RuntimeError("emission graph must run on its owning worker thread")
        if _stream_identity(context.book.device) != self._stream:
            raise RuntimeError("emission graph must run on its owning worker stream")
        if not self._guard.acquire(blocking=False):
            raise RuntimeError("concurrent or reentrant emission graph execution")
        try:
            for destination, source in zip(entry.scratch, values, strict=True):
                destination.copy_(source)
            assert self._runtime is not None
            entry.call(self._runtime.graph_mode)
            escaped = tuple(t.clone() for t in entry.owned)
            entry.replay_count += 1
        finally:
            # Same-stream order protects scratch until all escape copies complete.
            self._guard.release()
        return EmissionProjection(*escaped[:4]), escaped[4], escaped[5] if len(escaped) == 6 else None

    def receipt(self) -> dict[str, Any]:
        return dict(
            ready=self.ready,
            cells=[
                dict(rows=n, chunk_rows=b, token_width=k, successful_replays=entry.replay_count)
                for (n, b, k), entry in sorted(self._entries.items())
            ],
            fallbacks=[
                dict(rows=n, chunk_rows=b, token_width=k, successful_calls=count)
                for (n, b, k), count in sorted(self._fallbacks.items())
            ],
        )
