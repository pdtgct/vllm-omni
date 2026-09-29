# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Regional graph boundaries; fake runtime tests make no CUDA qualification claim."""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace

import pytest
import torch
from test_encoder_execution import _graph_runtime

from vllm_omni.model_executor.models.nemotron_asr import advance, rnnt
from vllm_omni.model_executor.models.nemotron_asr.emission_graph import EmissionGraphBinding

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _binding(runtime=None):
    binding = EmissionGraphBinding(
        hidden_size=16,
        park_id=99,
        blank_id=12,
        queue_capacity=8,
        token_widths=(4,),
        maximum_population=2,
        eou_token_id=98,
        vllm_config=None,
        runtime=runtime or _graph_runtime(),
    )
    binding.warmup(torch.device("cpu"))
    return binding


def _inputs(binding, n, b):
    result = advance.AdvanceResult(
        torch.zeros(b, 4 if b else 0, dtype=torch.int32),
        torch.zeros(b, dtype=torch.int32),
        row_status=torch.zeros(b, dtype=torch.int32),
    )
    book = torch.zeros(n, 7, dtype=torch.int32)
    book[:, rnnt.QUEUE_LAST_LABEL] = 12
    book[:, rnnt.BOOK_GEOMETRY] = 1
    context = advance.EmissionContext(
        torch.full((n,), advance.ROLE_FLUSH, dtype=torch.long),
        torch.zeros(n, dtype=torch.long),
        torch.arange(b),
        torch.zeros(n, 8, dtype=torch.int32),
        book,
        torch.zeros(n, dtype=torch.long),
        torch.zeros(n, dtype=torch.int32),
    )
    context.roles[:b] = advance.ROLE_CHUNK
    kwargs = dict(
        adapter=binding.adapter,
        hidden=16,
        rows_dtype=torch.float32,
        park_id=99,
        blank_id=12,
        plan_geometry=torch.ones(n, dtype=torch.long),
        endpoint_book=torch.zeros(n, 6, dtype=torch.int32),
        eou_token_id=98,
    )
    return result, context, kwargs


def _tensors(output):
    projection, status, endpoint = output
    return projection.rows, projection.queue, projection.book, projection.row_status, status, endpoint


@torch.inference_mode()
@pytest.mark.parametrize("n,b", [(1, 0), (1, 1), (2, 0), (2, 1), (2, 2)])
def test_exact_cells_dynamic_values_and_owned_outputs(n, b):
    binding = _binding()
    assert len(binding.captured_keys) == 5
    held = None
    for step in range(3):
        result, context, kwargs = _inputs(binding, n, b)
        if b and step:
            result.token_ids[:, 0] = step
            result.token_lengths[:] = 1
            context.book[:b, rnnt.QUEUE_LAST_LABEL] = step
        if step == 2:
            context.row_status[-1] = advance.ROW_STATUS_DECODE_INVARIANT
        expected = advance._project_and_validate_emission(result, context, **kwargs)
        actual = binding.project(result, context, **kwargs)
        for x, y in zip(_tensors(actual), _tensors(expected), strict=True):
            torch.testing.assert_close(x, y, atol=0, rtol=0)
        if held:
            for x, y in zip(held[0], held[1], strict=True):
                torch.testing.assert_close(x, y, atol=0, rtol=0)
        held = _tensors(actual), tuple(x.clone() for x in _tensors(actual))
    cell = next(c for c in binding.receipt()["cells"] if (c["rows"], c["chunk_rows"]) == (n, b))
    assert cell["successful_replays"] == 3
    assert not binding.receipt()["fallbacks"]


@torch.inference_mode()
def test_custom_adapter_snapshots_and_validator_remain_independent():
    binding = _binding()
    result, context, kwargs = _inputs(binding, 1, 1)
    context.row_status[:] = advance.ROW_STATUS_BOOK_INVARIANT
    original = context.book.clone()

    def malicious(result, context):
        context.row_status.zero_()
        context.book.fill_(0)
        return binding.adapter(result, context)

    kwargs["adapter"] = malicious
    projection, status, _ = binding.project(result, context, **kwargs)
    assert status[0] & advance.ROW_STATUS_BOOK_INVARIANT
    assert status[0] & advance.ROW_STATUS_DECODE_INVARIANT
    torch.testing.assert_close(context.book, original)
    assert sum(c["successful_replays"] for c in binding.receipt()["cells"]) == 0
    assert binding.receipt()["fallbacks"][0]["successful_calls"] == 1


@torch.inference_mode()
def test_startup_atomic_failure_and_thread_guard():
    binding = EmissionGraphBinding(
        hidden_size=16,
        park_id=99,
        blank_id=12,
        queue_capacity=8,
        token_widths=(4,),
        maximum_population=2,
        eou_token_id=98,
        vllm_config=None,
        runtime=_graph_runtime(fail_graph_call=3),
    )
    with pytest.raises(RuntimeError, match="synthetic"):
        binding.warmup(torch.device("cpu"))
    assert not binding.ready and not binding.captured_keys
    with pytest.raises(RuntimeError, match="failed"):
        binding.warmup(torch.device("cpu"))
    binding = _binding()
    result, context, kwargs = _inputs(binding, 1, 0)
    with ThreadPoolExecutor(max_workers=1) as pool:
        with pytest.raises(RuntimeError, match="thread"):
            pool.submit(binding.project, result, context, **kwargs).result()


@torch.inference_mode()
@pytest.mark.parametrize("n,b", [(1, 0), (1, 1), (2, 0), (2, 1), (2, 2)])
def test_real_transaction_layouts_with_configured_mode_zero_endpoint(n, b):
    from test_advance_model_rows_local import (
        CAP,
        CARRIER_HIDDEN,
        PARK_ID,
        PLACEHOLDER_ID,
        _assert_pools_equal,
        _clone_pools,
        _CommitRecorder,
        _envelope,
        _fixed_resolver,
        _fresh_pools,
        _plan,
        _tiny_core,
    )

    core = _tiny_core()
    binding = EmissionGraphBinding(
        hidden_size=CARRIER_HIDDEN,
        park_id=PARK_ID,
        blank_id=core.blank_id,
        queue_capacity=CAP,
        token_widths=(21,),
        maximum_population=2,
        eou_token_id=9002,
        vllm_config=None,
        runtime=_graph_runtime(),
    )
    binding.warmup(torch.device("cpu"))
    pools = _fresh_pools(n + 1)
    pools["endpoint_history_pool"] = torch.zeros(n + 1, 8, dtype=torch.int32)
    pools["endpoint_book_pool"] = torch.zeros(n + 1, 6, dtype=torch.int32)
    reference = _clone_pools(pools)
    carrier = torch.stack([_envelope(torch.zeros(2560), final=False, seq=0, geometry=1) for _ in range(n)])
    ids = torch.tensor([PLACEHOLDER_ID] * b + [PARK_ID] * (n - b))
    plan = _plan(
        prefills=list(range(1, n + 1)), num_pool_blocks=n + 1, geometries=[1] * n, chunk=[True] * b + [False] * (n - b)
    )
    sinks = [_CommitRecorder(), _CommitRecorder()]
    kwargs = dict(
        adapter=binding.adapter,
        decode_resolver=_fixed_resolver(rnnt.decode_dense_masked_frames),
        placeholder_id=PLACEHOLDER_ID,
        park_id=PARK_ID,
        eou_token_id=9002,
    )
    expected = advance.advance_model_rows(core, ids, carrier, plan, **reference, **kwargs, commit_sink=sinks[0])
    actual = advance.advance_model_rows(
        core, ids, carrier, plan, **pools, **kwargs, commit_sink=sinks[1], emission_binding=binding
    )
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    _assert_pools_equal(pools, reference)
    torch.testing.assert_close(sinks[0].staged[-1], sinks[1].staged[-1], atol=0, rtol=0)
    assert not torch.count_nonzero(sinks[1].staged[-1])
    assert sinks[0].log == sinks[1].log == ["reserve", "stage"]
    receipt = binding.receipt()
    assert not receipt["fallbacks"], receipt
    assert sum(c["successful_replays"] for c in receipt["cells"]) == 1
    assert next(c for c in receipt["cells"] if c["successful_replays"])["token_width"] == (21 if b else 0)


@torch.inference_mode()
def test_replay_flush_eou_values_and_stream_alias_reentrant_guards(monkeypatch):
    from vllm_omni.model_executor.models.nemotron_asr import emission_graph as module

    binding = _binding()
    for role in (advance.ROLE_REPLAY, advance.ROLE_FLUSH, advance.ROLE_EOU):
        result, context, kwargs = _inputs(binding, 2, 0)
        context.roles[:] = role
        if role == advance.ROLE_REPLAY:
            context.queue[:, :2] = torch.tensor([2, 3])
            context.book[:, rnnt.QUEUE_HEAD] = 1
            context.book[:, rnnt.QUEUE_LEN] = 2
            context.book[:, rnnt.QUEUE_LAST_LABEL] = 3
            context.book[:, rnnt.BOOK_PENDING_ECHO] = 1
            context.book[:, rnnt.BOOK_EXPECTED_LABEL] = 2
        if role == advance.ROLE_EOU:
            kwargs["endpoint_book"][:, 4] = 1
        expected = advance._project_and_validate_emission(result, context, **kwargs)
        actual = binding.project(result, context, **kwargs)
        for x, y in zip(_tensors(actual), _tensors(expected), strict=True):
            torch.testing.assert_close(x, y, atol=0, rtol=0)
        assert not actual[1].any()
    entry = binding._entries[(2, 0, 0)]
    context.queue = entry.scratch[6]
    with pytest.raises(ValueError, match="aliases"):
        binding.project(result, context, **kwargs)
    result, context, kwargs = _inputs(binding, 2, 0)
    binding._guard.acquire()
    try:
        with pytest.raises(RuntimeError, match="reentrant"):
            binding.project(result, context, **kwargs)
    finally:
        binding._guard.release()
    monkeypatch.setattr(module, "_stream_identity", lambda _device: 71)
    with pytest.raises(RuntimeError, match="stream"):
        binding.project(result, context, **kwargs)


@torch.inference_mode()
def test_corrupted_producer_inside_graph_keeps_independent_oracle(monkeypatch):
    from vllm_omni.model_executor.models.nemotron_asr import emission_graph as module

    original = module.make_mrv1_adapter

    def corrupt_factory(**kwargs):
        real = original(**kwargs)

        def corrupt(result, context):
            projection = real(result, context)
            projection.rows[:, 0] = 11
            # Attempts to rewrite the producer's copies cannot rewrite the oracle.
            context.queue.fill_(11)
            context.row_status.zero_()
            return projection

        return corrupt

    monkeypatch.setattr(module, "make_mrv1_adapter", corrupt_factory)
    binding = _binding()
    result, context, kwargs = _inputs(binding, 1, 0)
    actual = binding.project(result, context, **kwargs)
    assert actual[1][0] & advance.ROW_STATUS_DECODE_INVARIANT
    assert not context.queue.any()
    assert not binding.receipt()["fallbacks"]


@torch.inference_mode()
def test_replay_metadata_failure_precedes_reservation_and_all_resident_stores():
    from types import SimpleNamespace

    from test_advance_model_rows_local import (
        CARRIER_HIDDEN,
        PARK_ID,
        PLACEHOLDER_ID,
        _assert_pools_equal,
        _clone_pools,
        _CommitRecorder,
        _envelope,
        _fixed_resolver,
        _fresh_pools,
        _plan,
        _tiny_core,
    )

    core = _tiny_core()
    pools = _fresh_pools()
    before = _clone_pools(pools)
    adapter = advance.make_mrv1_adapter(hidden_size=CARRIER_HIDDEN, park_id=PARK_ID, blank_id=core.blank_id)

    def corrupt(merged, context, **kwargs):
        projection, status, endpoint = advance._project_and_validate_emission(merged, context, **kwargs)
        return (
            advance.EmissionProjection(
                projection.rows[:, 1:], projection.queue, projection.book, projection.row_status
            ),
            status,
            endpoint,
        )

    sink = _CommitRecorder()
    with pytest.raises(ValueError, match="adapter returned rows"):
        advance.advance_model_rows(
            core,
            torch.tensor([PLACEHOLDER_ID]),
            _envelope(torch.zeros(2560), final=False, seq=0, geometry=1).unsqueeze(0),
            _plan(prefills=[1], geometries=[1]),
            **pools,
            adapter=adapter,
            decode_resolver=_fixed_resolver(),
            placeholder_id=PLACEHOLDER_ID,
            park_id=PARK_ID,
            emission_binding=SimpleNamespace(project=corrupt),
            commit_sink=sink,
        )
    _assert_pools_equal(pools, before)
    assert not sink.log


@torch.inference_mode()
@pytest.mark.parametrize("change", ["population", "width", "stride"])
def test_unsupported_metadata_falls_back_without_growing_inventory(change):
    binding = _binding()
    result, context, kwargs = _inputs(binding, 3 if change == "population" else 2, 1)
    if change == "width":
        result = replace(result, token_ids=torch.zeros(1, 5, dtype=torch.int32))
    if change == "stride":
        context.queue = torch.zeros(2, 16, dtype=torch.int32)[:, ::2]
    keys = binding.captured_keys
    expected = advance._project_and_validate_emission(result, context, **kwargs)
    actual = binding.project(result, context, **kwargs)
    for x, y in zip(_tensors(actual), _tensors(expected), strict=True):
        torch.testing.assert_close(x, y, atol=0, rtol=0)
    assert binding.captured_keys == keys
    assert sum(c["successful_replays"] for c in binding.receipt()["cells"]) == 0
    assert binding.receipt()["fallbacks"][0]["successful_calls"] == 1
