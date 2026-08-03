# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Direct MRv2 projection for Nemotron's manager-owned persistent state."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import torch
from vllm.config.compilation import CUDAGraphMode
from vllm.v1.core.sched.output import NewRequestData
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.worker.gpu.input_batch import InputBatch
from vllm.v1.worker.gpu.model_states.interface import ModelState
from vllm.v1.worker.gpu.states import RequestState
from vllm.v1.worker.utils import AttentionGroup


@dataclass(frozen=True)
class ProjectionSnapshot:
    """One complete same-epoch runner view, preserving source identity."""

    epoch: int
    req_ids: tuple[str, ...]
    scheduled_encoder_inputs: dict[str, list[int]]
    encoder_features: object
    bindings: tuple[object, ...]
    block_ids: tuple[tuple[int, ...], ...]
    input_batch: object
    req_states: object
    model_kwargs: dict[str, Any]
    request_metadata: dict[str, object] | None = None
    dummy_run: bool = False
    is_profile: bool = False
    no_page_io: bool = False


@dataclass
class _ProjectionRecord:
    dummy_run: bool
    is_profile: bool
    mm_req_ids: tuple[str, ...] | None = None
    scheduled_encoder_inputs: dict[str, list[int]] | None = None
    encoder_features: object | None = None
    binding_req_ids: tuple[str, ...] | None = None
    bindings: tuple[object, ...] | None = None
    block_ids: tuple[tuple[int, ...], ...] | None = None


class ProjectionJoin:
    """Bounded one-execution join over scheduler, MM, and block-table views."""

    def __init__(self) -> None:
        self._next_epoch = 1
        self._active_epoch: int | None = None
        self._records: dict[int, _ProjectionRecord] = {}

    def _record(self, epoch: int) -> _ProjectionRecord:
        if epoch != self._active_epoch or epoch not in self._records:
            raise RuntimeError("projection epoch is not active")
        return self._records[epoch]

    def begin(self, *, dummy_run: bool, is_profile: bool) -> int:
        if is_profile and not dummy_run:
            raise ValueError("persistent-state profile execution requires a dummy run")
        if self._active_epoch is not None:
            raise RuntimeError("projection epoch is already active")
        epoch = self._next_epoch
        self._next_epoch += 1
        self._active_epoch = epoch
        self._records[epoch] = _ProjectionRecord(dummy_run, is_profile)
        return epoch

    def is_dummy(self, epoch: int) -> bool:
        """Return the active record's invocation kind."""

        return self._record(epoch).dummy_run

    def record_mm(
        self,
        epoch: int,
        *,
        req_ids: tuple[str, ...],
        scheduled_encoder_inputs: dict[str, list[int]],
        encoder_features: object,
    ) -> None:
        record = self._record(epoch)
        record.mm_req_ids = req_ids
        record.scheduled_encoder_inputs = scheduled_encoder_inputs
        record.encoder_features = encoder_features

    def record_bindings(
        self,
        epoch: int,
        *,
        req_ids: tuple[str, ...],
        bindings: tuple[object, ...],
        block_ids: tuple[tuple[int, ...], ...],
    ) -> None:
        record = self._record(epoch)
        record.binding_req_ids = req_ids
        record.bindings = bindings
        record.block_ids = block_ids

    def complete(
        self,
        epoch: int,
        *,
        input_batch: object,
        req_states: object,
        model_kwargs: dict[str, Any],
    ) -> ProjectionSnapshot:
        record = self._record(epoch)
        required = (
            record.mm_req_ids,
            record.scheduled_encoder_inputs,
            record.encoder_features,
            record.binding_req_ids,
            record.bindings,
            record.block_ids,
        )
        if any(value is None for value in required):
            raise RuntimeError("persistent-state projection is incomplete")
        scheduled_encoder_inputs = record.scheduled_encoder_inputs
        bindings = record.bindings
        block_ids = record.block_ids
        assert scheduled_encoder_inputs is not None
        assert bindings is not None
        assert block_ids is not None
        batch_req_ids = tuple(getattr(input_batch, "req_ids"))
        if not (
            record.mm_req_ids == record.binding_req_ids == batch_req_ids
        ):
            raise ValueError("persistent-state projection request order differs")
        return ProjectionSnapshot(
            epoch=epoch,
            req_ids=batch_req_ids,
            scheduled_encoder_inputs=scheduled_encoder_inputs,
            encoder_features=record.encoder_features,
            bindings=bindings,
            block_ids=block_ids,
            input_batch=input_batch,
            req_states=req_states,
            model_kwargs=model_kwargs,
        )

    def complete_dummy(
        self,
        epoch: int,
        *,
        input_batch: object,
        req_states: object,
        model_kwargs: dict[str, Any],
    ) -> ProjectionSnapshot:
        record = self._record(epoch)
        if not record.dummy_run:
            raise RuntimeError("real projection cannot use dummy completion")
        return ProjectionSnapshot(
            epoch=epoch,
            req_ids=tuple(getattr(input_batch, "req_ids")),
            scheduled_encoder_inputs={},
            encoder_features=None,
            bindings=(),
            block_ids=(),
            input_batch=input_batch,
            req_states=req_states,
            model_kwargs=model_kwargs,
            dummy_run=True,
            is_profile=record.is_profile,
            no_page_io=True,
        )

    def end(self, epoch: int) -> None:
        self._record(epoch)
        self._records.pop(epoch)
        self._active_epoch = None


class _RequestMetadata:
    def __init__(self) -> None:
        self.records: dict[str, NewRequestData] = {}

    def add(self, data: NewRequestData) -> None:
        self.records[data.req_id] = data

    def prune_finished(self, req_ids: list[str]) -> None:
        for req_id in req_ids:
            self.records.pop(req_id, None)

    def snapshot(self, req_ids: tuple[str, ...]) -> dict[str, object]:
        try:
            return {req_id: self.records[req_id] for req_id in req_ids}
        except KeyError as exc:
            raise RuntimeError(
                "persistent-state request metadata is incomplete"
            ) from exc


class NemotronASRModelState(ModelState):  # type: ignore[misc]
    """Model-specific runner projection, never slot-allocation authority."""

    num_new_sampled_tokens_per_step = 1

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._projection = ProjectionJoin()
        self._projection_epoch: int | None = None
        self._scheduler_output: object | None = None
        self._request_metadata = _RequestMetadata()
        self._projected_binding_keys: set[tuple[str, int]] = set()
        self._initialized_binding_keys: set[tuple[str, int]] = set()

    def add_request(self, req_index: int, new_req_data: NewRequestData) -> None:
        del req_index
        self._request_metadata.add(new_req_data)

    def remove_request(self, req_id: str) -> None:
        # MRv2 removes every streaming request before re-adding its next turn;
        # only post-update residency reconciliation can distinguish terminality.
        del req_id

    def begin_omni_projection(
        self,
        scheduler_output: object,
        *,
        dummy_run: bool,
        is_profile: bool,
    ) -> None:
        epoch = self._projection.begin(
            dummy_run=dummy_run,
            is_profile=is_profile,
        )
        self._scheduler_output = scheduler_output
        self._projection_epoch = epoch

    def end_omni_projection(self) -> None:
        if self._projection_epoch is not None:
            self._projection.end(self._projection_epoch)
        self._projection_epoch = None
        self._scheduler_output = None
        self._projected_binding_keys.clear()

    def reconcile_omni_lifecycle(
        self,
        *,
        finished_req_ids: frozenset[str],
        preempted_req_ids: frozenset[str],
        resident_req_ids: frozenset[str],
    ) -> None:
        if preempted_req_ids:
            raise RuntimeError(
                "no-recompute persistent state cannot survive preemption"
            )
        terminal = sorted(finished_req_ids - resident_req_ids)
        self._request_metadata.prune_finished(terminal)
        terminal_set = set(terminal)
        initialized: set[tuple[str, int]] = getattr(
            self,
            "_initialized_binding_keys",
            set(),
        )
        self._initialized_binding_keys = {
            key
            for key in initialized
            if key[0] not in terminal_set
        }

    def commit_omni_projection(self) -> None:
        """Publish freshness only after the selected execution succeeds."""

        self._initialized_binding_keys.update(self._projected_binding_keys)

    def get_mm_embeddings(
        self,
        scheduled_encoder_inputs: dict[str, list[int]],
        input_batch: InputBatch,
        req_states: RequestState,
    ) -> torch.Tensor | None:
        if not self.supports_mm_inputs:
            features = None
        else:
            mm_hashes, mm_kwargs = self.encoder_runner.prepare_mm_inputs(
                scheduled_encoder_inputs
            )
            if mm_kwargs:
                encoded = self.encoder_runner.execute_mm_encoder(mm_kwargs)
                self.encoder_cache.encoder_outputs.update(zip(mm_hashes, encoded))
            mm_embeds, is_mm_embed = self.gather_mm_embeddings(input_batch)
            features = self.encoder_runner.get_inputs_embeds(
                input_batch.input_ids[: input_batch.num_tokens],
                mm_embeds,
                is_mm_embed,
            )[: input_batch.num_tokens_after_padding]
        if self._projection_epoch is None:
            raise RuntimeError("multimodal projection has no active epoch")
        self._projection.record_mm(
            self._projection_epoch,
            req_ids=tuple(input_batch.req_ids),
            scheduled_encoder_inputs=scheduled_encoder_inputs,
            encoder_features=features,
        )
        return features

    def prepare_inputs(
        self, input_batch: InputBatch, req_states: RequestState
    ) -> dict[str, Any]:
        if self._projection_epoch is None:
            raise RuntimeError("model input projection has no active epoch")
        if self._projection.is_dummy(self._projection_epoch):
            snapshot = self._projection.complete_dummy(
                self._projection_epoch,
                input_batch=input_batch,
                req_states=req_states,
                model_kwargs={},
            )
        else:
            snapshot = self._projection.complete(
                self._projection_epoch,
                input_batch=input_batch,
                req_states=req_states,
                model_kwargs={},
            )
            snapshot = replace(
                snapshot,
                request_metadata=self._request_metadata.snapshot(snapshot.req_ids),
            )
        return {"persistent_state_projection": snapshot}

    def prepare_dummy_inputs(self, num_reqs: int, num_tokens: int) -> dict[str, Any]:
        del num_reqs, num_tokens
        raise RuntimeError(
            "Nemotron persistent state does not support CUDA graph capture dummy inputs"
        )

    def prepare_attn(
        self,
        input_batch: InputBatch,
        cudagraph_mode: CUDAGraphMode,
        block_tables: tuple[torch.Tensor, ...],
        slot_mappings: torch.Tensor,
        attn_groups: list[list[AttentionGroup]],
        kv_cache_config: KVCacheConfig,
        for_capture: bool = False,
    ) -> dict[str, Any]:
        del slot_mappings, attn_groups, for_capture
        if cudagraph_mode == CUDAGraphMode.FULL:
            raise RuntimeError("persistent state does not support CUDA graph capture")
        if self._projection_epoch is None or self._scheduler_output is None:
            raise RuntimeError("state binding projection has no active epoch")
        del block_tables, kv_cache_config
        bindings_by_id = getattr(
            self._scheduler_output, "persistent_state_bindings", {}
        )
        req_ids = tuple(input_batch.req_ids)
        try:
            source_bindings = tuple(
                bindings_by_id[req_id] for req_id in req_ids
            )
        except KeyError as exc:
            raise RuntimeError(
                "scheduler omitted a persistent-state binding"
            ) from exc
        bindings = tuple(
            replace(
                binding,
                fresh=(binding.request_id, binding.generation)
                not in self._initialized_binding_keys,
            )
            for binding in source_bindings
        )
        if any(binding.request_id != req_id for req_id, binding in zip(req_ids, bindings)):
            raise RuntimeError("persistent-state binding request mismatch")
        block_ids = tuple((int(binding.slot_id),) for binding in bindings)
        self._projected_binding_keys = {
            (binding.request_id, binding.generation) for binding in bindings
        }
        self._projection.record_bindings(
            self._projection_epoch,
            req_ids=req_ids,
            bindings=bindings,
            block_ids=block_ids,
        )
        # Nemotron's attention is model-local and advances through its state
        # page; there is no core attention backend metadata to construct.
        return {}
