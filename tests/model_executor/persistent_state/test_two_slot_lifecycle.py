# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""B3 exploratory CPU lifecycle probes; NOT a qualified mixed MRv2 run.

The actual MRv2 initialization rejects mixed caches (first test). Remaining
tests deliberately bypass that worker gate to isolate scheduler/manager risks.
They use core's mixed KVCacheManager and byte allocator, real lifecycle methods,
and real ModelState projection. GPU execution, scheduler dispatch, utility
transport and client disappearance are simulated; no lease service is present.
Four drain/reconcile calls are an experimental bound, not a production timeout.
"""

import asyncio
from collections import deque
from dataclasses import asdict, replace
from types import SimpleNamespace

import pytest
import torch
from vllm.config.compilation import CUDAGraphMode
from vllm.sampling_params import SamplingParams
from vllm.v1.core.encoder_cache_manager import EncoderCacheManager
from vllm.v1.core.kv_cache_manager import KVCacheManager
from vllm.v1.core.kv_cache_utils import get_kv_cache_config_from_groups
from vllm.v1.core.sched.output import NewRequestData
from vllm.v1.core.sched.request_queue import SchedulingPolicy, create_request_queue
from vllm.v1.core.single_type_kv_cache_manager import register_all_kvcache_specs
from vllm.v1.engine import UtilityOutput
from vllm.v1.engine.core import EngineCoreProc
from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheGroupSpec
from vllm.v1.kv_cache_layout import KVCacheLayout
from vllm.v1.request import Request, RequestStatus
from vllm.v1.worker.utils import allocate_kv_cache

from tests.model_executor.models.nemotron_asr.test_plan_local import _bind, _row
from tests.model_executor.persistent_state._helpers import make_generic_spec
from vllm_omni.model_executor import persistent_state as ps
from vllm_omni.model_executor.models.nemotron_asr.model_state_v2 import (
    NemotronASRModelState,
    ProjectionJoin,
    _RequestMetadata,
)
from vllm_omni.model_executor.models.nemotron_asr.plan import SessionRegistry
from vllm_omni.model_executor.models.nemotron_asr.scheduler import NemotronASRScheduler
from vllm_omni.platforms.interface import OmniPlatform
from vllm_omni.worker.gpu_ar_model_runner_v2 import GPUARModelRunnerV2

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]
_DRAIN_BOUND = 4


class _ReclamationError(AssertionError):
    """Only the named missing reclamation assertion may be xfailed."""


class _MixedCacheUnsupportedError(ValueError):
    """Only the exact worker mixed-cache guard may be xfailed."""


def _mixed_config():
    state_spec = make_generic_spec(ps)
    attention_spec = FullAttentionSpec(block_size=8, num_kv_heads=1, head_size=1, dtype=torch.float32)
    config = SimpleNamespace(
        attention_config=SimpleNamespace(hisparse_config=None),
        cache_config=SimpleNamespace(
            get_resolved_kv_cache_layout=lambda: KVCacheLayout.LBHNC,
            num_gpu_blocks_override=None,
            prefix_cache_retention_interval=None,
        ),
    )
    # One null block plus four usable blocks: two state + two ordinary KV.
    return get_kv_cache_config_from_groups(
        config,
        [KVCacheGroupSpec(["state"], state_spec), KVCacheGroupSpec(["attention"], attention_spec)],
        available_memory=5 * 64,
    )


@pytest.mark.xfail(strict=True, raises=_MixedCacheUnsupportedError, reason="B3: MRv2 rejects persistent + ordinary KV")
def test_mixed_cache_reaches_mrv2_initialization():
    """B3 gate: a mixed config must pass the actual worker partition seam."""
    config = _mixed_config()
    assert config.num_blocks == 5
    runner = object.__new__(GPUARModelRunnerV2)
    try:
        runner.initialize_kv_cache(config)
    except ValueError as exc:
        assert str(exc) == "persistent-state allocation requires a persistent-only cache layout"
        raise _MixedCacheUnsupportedError(str(exc)) from exc


class _Lifecycle:
    """Manually drive production lifecycle seams, without a GPU dispatch loop."""

    def __init__(self):
        self.config = _mixed_config()
        register_all_kvcache_specs(None)
        OmniPlatform.register_custom_kv_cache_specs(None)
        self.kv = KVCacheManager(self.config, 256, 8, 8, enable_caching=False)
        self.manager, self.ordinary = self.kv.coordinator.single_type_managers
        self.manager.configure_capacity(safety_reserve_slots=0, max_resident_sessions=2)
        # Exercise core's shared allocation, then expose state-only bytes.
        # This is explicitly NOT the fork's rejected worker partition path.
        self.views = allocate_kv_cache(self.config, "cpu", KVCacheLayout.LBHNC)
        assert self.views["state"].untyped_storage().data_ptr() == self.views["attention"].untyped_storage().data_ptr()
        self.storage = ps.persistent_state_storage_from_raw(
            self.manager.persistent_state_spec, self.views["state"].reshape(-1)
        )
        self.scheduler = s = object.__new__(NemotronASRScheduler)
        s.kv_cache_manager = self.kv
        s.encoder_cache_manager = EncoderCacheManager(0)
        s.vllm_config = SimpleNamespace(kv_transfer_config=None, model_config=SimpleNamespace())
        s.requests = {}
        s.running = []
        s.waiting = create_request_queue(SchedulingPolicy.FCFS)
        s.skipped_waiting = create_request_queue(SchedulingPolicy.FCFS)
        s.finished_req_ids = set()
        s.finished_req_ids_dict = None
        s.num_waiting_for_streaming_input = 0
        s._new_prompt_len_snapshot = {}
        s._claimed_state_bindings = {}
        s._omits_kv_transfer_cache = {}
        s.connector = s.ec_connector = s.chunk_transfer_adapter = None
        s.log_stats = s.defer_block_free = False
        s.spec_decode_metrics_level = "none"
        s.sched_step_seq = s.processed_step_seq = 0
        s.deferred_frees = deque()
        self.state = object.__new__(NemotronASRModelState)
        self.state._projection = ProjectionJoin()
        self.state._projection_epoch = None
        self.state._scheduler_output = None
        self.state._projected_binding_keys = set()
        self.state._initialized_binding_keys = set()
        self.state._request_metadata = _RequestMetadata()

    @staticmethod
    def request(request_id):
        return Request(
            request_id=request_id,
            prompt_token_ids=[101],
            sampling_params=SamplingParams(max_tokens=142),
            pooling_params=None,
            block_hasher=None,
            resumable=True,
        )

    def admit(self, request_id):
        # Proposed utility body, simulated because no such core utility exists.
        # Allocate once, insert one resumable request; no API lease service.
        self.manager.allocate_new_blocks(request_id, 1, 1)
        request = self.request(request_id)
        assert self.kv.allocate_slots(request, 1) is not None
        binding = self.manager.get_state_binding(request_id)
        request.additional_information = {"persistent_state_binding": asdict(binding)}
        self.scheduler.add_request(request)
        self.scheduler._claimed_state_bindings[request_id] = binding
        self.storage.initialize_fresh_state_slot(binding.slot_id, binding.generation)
        self.storage.views["generic.float_state"][binding.slot_id].fill_(binding.generation + 0.25)
        self.storage.views["generic.integer_state"][binding.slot_id].fill_(binding.generation * 17)
        for block in self.ordinary.req_to_blocks[request_id]:
            self.views["attention"][block.block_id].fill_(binding.generation + 40)
        self.install_metadata(request)
        return request

    def install_metadata(self, request):
        data = NewRequestData.from_request(request, self.kv.get_blocks(request.request_id).get_block_ids())
        self.state.add_request(0, data)

    def start(self, request):
        # Simulate dispatch only; lifecycle methods themselves remain real.
        self.scheduler.waiting.remove_requests((request,))
        request.status = RequestStatus.RUNNING
        request.num_in_flight_tokens = 1
        request.last_sched_seq = 1
        self.scheduler.sched_step_seq = 1
        self.scheduler.running.append(request)

    def snapshot(self, request_id):
        slot = self.manager.get_state_binding(request_id).slot_id
        pages = [self.storage.raw[slot * 64 : (slot + 1) * 64]]
        pages.extend(
            self.storage.raw[b.block_id * 64 : (b.block_id + 1) * 64] for b in self.ordinary.req_to_blocks[request_id]
        )
        return torch.cat(pages).clone()

    def project(self, request_ids):
        bindings = {r: self.manager.get_state_binding(r) for r in request_ids}
        batch = SimpleNamespace(req_ids=request_ids)
        self.state.begin_omni_projection(
            SimpleNamespace(persistent_state_bindings=bindings), dummy_run=False, is_profile=False
        )
        try:
            self.state._projection.record_mm(
                self.state._projection_epoch,
                req_ids=tuple(request_ids),
                scheduled_encoder_inputs={},
                encoder_features=(),
            )
            self.state.prepare_attn(batch, CUDAGraphMode.NONE, (), torch.empty(0), [], self.config)
            projection = self.state.prepare_inputs(batch, SimpleNamespace())["persistent_state_projection"]
            self.state.commit_omni_projection()
            return projection
        finally:
            self.state.end_omni_projection()

    def drain(self):
        # Simulate completion of one execution, exercise actual drain/reconcile.
        self.scheduler.processed_step_seq += 1
        self.scheduler._drain_deferred_frees()
        self.state.reconcile_omni_lifecycle(
            finished_req_ids=frozenset(self.scheduler.finished_req_ids),
            preempted_req_ids=frozenset(),
            resident_req_ids=frozenset(self.scheduler.requests),
        )


@pytest.fixture
def lifecycle():
    return _Lifecycle()


def test_two_concurrent_callers_fill_capacity_and_third_is_refused(lifecycle):
    """B3 admission: concurrent callers, serialized single-engine execution."""

    async def connect(request_id):
        await asyncio.sleep(0)
        return lifecycle.admit(request_id)

    async def connect_all():
        return await asyncio.gather(*(connect(r) for r in ("a", "b", "c")), return_exceptions=True)

    a, b, rejected = asyncio.run(connect_all())
    assert isinstance(a, Request) and isinstance(b, Request)
    assert isinstance(rejected, ValueError) and str(rejected) == "persistent-state capacity exhausted"
    assert set(lifecycle.scheduler.requests) == {"a", "b"}
    assert lifecycle.kv.block_pool.get_num_free_blocks() == 0
    assert all(len(group) == 1 for r in ("a", "b") for group in lifecycle.kv.get_blocks(r).blocks)
    before = lifecycle.snapshot("a"), lifecycle.snapshot("b")
    projection = lifecycle.project(("a", "b"))
    assert [b.fresh for b in projection.bindings] == [True, True]
    assert torch.equal(lifecycle.snapshot("a"), before[0])
    assert torch.equal(lifecycle.snapshot("b"), before[1])


def test_park_resume_preserves_both_sessions_and_binding(lifecycle):
    """@spec PORT-STATE-010 / PORT-STATE-020; B3 park is not retirement."""
    a, b = lifecycle.admit("a"), lifecycle.admit("b")
    lifecycle.project(("a", "b"))
    lifecycle.start(a)
    lifecycle.start(b)
    before_a, before_b = lifecycle.snapshot("a"), lifecycle.snapshot("b")
    binding = lifecycle.manager.get_state_binding("a")
    a.num_in_flight_tokens = 0
    a.status = RequestStatus.FINISHED_STOPPED
    assert not lifecycle.scheduler._handle_stopped_request(a)
    lifecycle.scheduler.running.remove(a)
    lifecycle.state.remove_request("a")
    lifecycle.drain()
    assert a.status == RequestStatus.WAITING_FOR_STREAMING_REQ
    assert b.status == RequestStatus.RUNNING
    lifecycle.project(("b",))
    # Stand in for one successful model update while a is parked. Production
    # batch validation/scatter runs; arithmetic and ordinary-KV writes are CPU
    # sentinels, not an acoustic forward pass or attention kernel.
    live = lifecycle.manager.get_state_binding("b")
    row = {**asdict(live), "binding": live, "request_key": "b", "order": 0, "role": "resident"}
    batch = ps.PersistentStateBatch([row])
    current = batch.gather(
        lifecycle.manager, lambda row: lifecycle.storage.views["generic.float_state"][row["slot_id"]].clone()
    )[0]

    def write(row, *, state):
        lifecycle.storage.views["generic.float_state"][row["slot_id"]].copy_(state)

    batch.scatter(lifecycle.manager, [current + 1], ["clean"], write)
    for block in lifecycle.ordinary.req_to_blocks["b"]:
        lifecycle.views["attention"][block.block_id].fill_(97)
    after_b = lifecycle.snapshot("b")
    assert not torch.equal(after_b, before_b)
    assert torch.equal(lifecycle.snapshot("a"), before_a)
    update = lifecycle.request("a")
    update.additional_information = a.additional_information
    lifecycle.scheduler.add_request(update)
    assert lifecycle.scheduler.requests["a"] is a
    assert lifecycle.scheduler._verify_streaming_readd(a) == binding
    lifecycle.install_metadata(a)
    projection = lifecycle.project(("a", "b"))
    assert not any(binding.fresh for binding in projection.bindings)
    assert torch.equal(lifecycle.snapshot("a"), before_a)
    assert torch.equal(lifecycle.snapshot("b"), after_b)
    assert lifecycle.kv.block_pool.get_num_free_blocks() == 0


@pytest.mark.xfail(
    strict=True,
    raises=_ReclamationError,
    reason="P1(b): lost successful utility reply retains admission; no engine reclamation",
)
def test_lost_utility_reply_reclaims_admission(lifecycle):
    """B3 P1(b): successful utility result is discarded by a vanished caller."""
    lifecycle.admit("survivor")
    before = lifecycle.snapshot("survivor")
    output = UtilityOutput(7)
    EngineCoreProc._invoke_utility_method(
        "admit", lambda: (lifecycle.admit("lost"), "allocated")[1], output, lambda reply: None
    )
    assert output.failure_message is None
    for _ in range(_DRAIN_BOUND):
        lifecycle.drain()
    assert torch.equal(lifecycle.snapshot("survivor"), before)
    if lifecycle.manager.get_state_binding("lost") is not None:
        raise _ReclamationError("lost reply retains slot after 4 drain steps")
    lifecycle.admit("replacement")


@pytest.mark.xfail(
    strict=True, raises=_ReclamationError, reason="P1(b): core utility error handling does not roll back allocation"
)
def test_failed_utility_body_rolls_back_admission(lifecycle):
    """B3 P1(b): core reports an error but does not undo a partial utility."""
    lifecycle.admit("survivor")
    before = lifecycle.snapshot("survivor")

    def fail_after_allocation():
        lifecycle.manager.allocate_new_blocks("orphan", 1, 1)
        raise RuntimeError("insertion failed after allocation")

    replies = []
    EngineCoreProc._invoke_utility_method("admit", fail_after_allocation, UtilityOutput(8), replies.append)
    assert "insertion failed after allocation" in replies[0].failure_message
    for _ in range(_DRAIN_BOUND):
        lifecycle.drain()
    assert torch.equal(lifecycle.snapshot("survivor"), before)
    if lifecycle.manager.get_state_binding("orphan") is not None:
        raise _ReclamationError("utility error retains uninserted allocation")


@pytest.mark.xfail(
    strict=True, raises=_ReclamationError, reason="P1(a): manager.free retains terminal binding without drop_lease"
)
def test_abort_inflight_reclaims_slot_and_reopens_admission(lifecycle):
    """B3 P1(a): no consuming connector and no external lease release."""
    a, b = lifecycle.admit("a"), lifecycle.admit("b")
    lifecycle.start(a)
    lifecycle.start(b)
    before = lifecycle.snapshot("b")
    assert lifecycle.scheduler._request_blocks_can_be_freed(a)
    lifecycle.scheduler.finish_requests("a", RequestStatus.FINISHED_ABORTED)
    assert "a" not in lifecycle.scheduler.requests
    assert lifecycle.manager.is_terminal("a")
    assert lifecycle.ordinary.req_to_blocks.get("a", []) == []
    assert lifecycle.kv.block_pool.get_num_free_blocks() == 1
    for _ in range(_DRAIN_BOUND):
        lifecycle.drain()
    assert torch.equal(lifecycle.snapshot("b"), before)
    if lifecycle.manager.get_state_binding("a") is not None:
        raise _ReclamationError("abort retains state binding after 4 drain steps")
    lifecycle.admit("replacement")


def test_no_recompute_preemption_fails_closed(lifecycle):
    """@spec PORT-STATE-005; P1(c) cannot be solved by token recompute."""
    a, b = lifecycle.admit("a"), lifecycle.admit("b")
    lifecycle.start(a)
    lifecycle.start(b)
    before = lifecycle.snapshot("b")
    a.num_computed_tokens = 1
    lifecycle.scheduler.running.remove(a)  # Core schedule pops before this hook.
    lifecycle.scheduler._preempt_request(a, 0.0)
    assert a.status == RequestStatus.FINISHED_ERROR
    assert a.num_computed_tokens == 1
    assert lifecycle.manager.is_terminal("a")
    assert torch.equal(lifecycle.snapshot("b"), before)
    with pytest.raises(RuntimeError, match="no-recompute"):
        lifecycle.state.reconcile_omni_lifecycle(
            finished_req_ids=frozenset(), preempted_req_ids=frozenset({"a"}), resident_req_ids=frozenset({"b"})
        )


def test_explicit_release_control_rejects_stale_scatter_after_reuse(lifecycle):
    """B3 control only: explicit drop_lease is NOT autonomous reclamation."""
    a, b = lifecycle.admit("a"), lifecycle.admit("b")
    lifecycle.start(a)
    lifecycle.start(b)
    old = lifecycle.manager.get_state_binding("a")
    before = lifecycle.snapshot("b")
    registry = SessionRegistry()
    old_worker, survivor_worker = _bind(
        registry,
        [
            _row("a", old.slot_id, chunk=True, prior=False),
            _row("b", lifecycle.manager.get_state_binding("b").slot_id, chunk=True, prior=False),
        ],
    )
    survivor_progress = registry.publish_progress(
        "b", survivor_worker.admission_generation, (1, 1, 1), starts_chunk=True
    )
    lifecycle.scheduler.finish_requests("a", RequestStatus.FINISHED_ABORTED)
    lifecycle.drain()  # Assumed completion boundary, not a CUDA execution fence.
    lifecycle.manager.drop_lease("a")
    lifecycle.admit("replacement")
    new = lifecycle.manager.get_state_binding("replacement")
    assert new.slot_id == old.slot_id and new.generation > old.generation
    replacement_before = lifecycle.snapshot("replacement")
    for stale in (old, replace(new, generation=old.generation)):
        row = {**asdict(stale), "binding": stale, "request_key": stale.request_id, "order": 0, "role": "resident"}
        batch = ps.PersistentStateBatch([row])
        writes = []
        with pytest.raises(ValueError, match="stale or non-authoritative"):
            batch.scatter(lifecycle.manager, [torch.ones(2, 3)], ["clean"], lambda *args, **kwargs: writes.append(args))
        assert writes == []
    assert torch.equal(lifecycle.snapshot("replacement"), replacement_before)
    assert torch.equal(lifecycle.snapshot("b"), before)
    assert lifecycle.kv.block_pool.get_num_free_blocks() == 0

    registry.prune({"b"})
    new_worker = _bind(registry, [_row("replacement", new.slot_id, chunk=True, prior=False)])[0]
    assert new_worker.admission_generation > old_worker.admission_generation
    assert registry.publish_progress("a", old_worker.admission_generation, (2, 99, 3), starts_chunk=False) is None
    assert (
        registry.publish_progress("replacement", old_worker.admission_generation, (2, 99, 3), starts_chunk=False)
        is None
    )
    assert registry.publish_progress("replacement", new_worker.admission_generation, (1, 1, 1), starts_chunk=True) == (
        0,
        1,
        1,
    )
    assert registry._sessions["b"].progress_key == survivor_progress


@pytest.mark.xfail(strict=True, raises=_ReclamationError, reason="P1(a): replacement remains refused after abort/drain")
def test_immediate_replacement_without_external_release(lifecycle):
    a = lifecycle.admit("a")
    lifecycle.admit("survivor")
    before = lifecycle.snapshot("survivor")
    lifecycle.start(a)
    lifecycle.scheduler.finish_requests("a", RequestStatus.FINISHED_ABORTED)
    for _ in range(_DRAIN_BOUND):
        lifecycle.drain()
    assert torch.equal(lifecycle.snapshot("survivor"), before)
    try:
        lifecycle.admit("replacement")
    except ValueError as exc:
        assert str(exc) == "persistent-state capacity exhausted"
        raise _ReclamationError("replacement refused after 4 drain steps without drop_lease") from exc
