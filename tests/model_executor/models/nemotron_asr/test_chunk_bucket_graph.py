# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""CPU boundary/ownership checks; real CUDA replay remains a separate gate."""

import pytest
import torch
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
from test_advance_session_local import _fresh_state
from test_encoder_execution import _graph_runtime

from vllm_omni.model_executor.models.nemotron_asr import advance, rnnt
from vllm_omni.model_executor.models.nemotron_asr.chunk_bucket_graph import (
    _clone_state,
    _outputs,
    _state_tensors,
    capture_chunk_bucket,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _fixture():
    core = _tiny_core()
    env = torch.stack([_envelope(torch.randn(2560), final=False, seq=0, geometry=1) for _ in range(32)])
    state = _fresh_state(32)
    args = dict(
        geometry=1,
        admitted_prompt=torch.zeros(32, dtype=torch.long),
        incoming_status=torch.zeros(32, dtype=torch.int32),
        queue_capacity=48,
    )
    return core, env, state, args


@torch.inference_mode()
def test_bucket_replay_changed_contents_lifetime_and_fail_closed():
    core, env, state, args = _fixture()
    initial = _clone_state(state)
    transition = capture_chunk_bucket(core, env, state, **args, vllm_config=None, runtime=_graph_runtime())
    for actual, expected in zip(_state_tensors(state), _state_tensors(initial), strict=True):
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    held = None
    held_copy = ()
    for step in range(3):
        env[:, advance.ENV_CHUNK_SEQUENCE] = step
        env[:, advance.ENVELOPE_HEADER_SLOTS :] *= 0.71
        if step == 2:
            args["incoming_status"][0] = advance.ROW_STATUS_DECODE_INVARIANT
            env[1, advance.ENV_PROMPT_INDEX] = 1
        expected_state = _clone_state(state)
        expected = advance.advance_chunk_bucket(
            core, env, expected_state, **args, decode_fn=rnnt.decode_dense_masked_frames
        )
        actual = transition(core, env, state, **args, decode_fn=rnnt.decode_dense_masked_frames)
        for left, right in zip(
            (*_outputs(actual), *_state_tensors(state)),
            (*_outputs(expected), *_state_tensors(expected_state)),
            strict=True,
        ):
            torch.testing.assert_close(left, right, atol=0, rtol=0)
        if held is not None:
            for left, right in zip(_outputs(held), held_copy, strict=True):
                torch.testing.assert_close(left, right, atol=0, rtol=0)
        held, held_copy = actual, tuple(t.clone() for t in _outputs(actual))
    before = tuple(t.clone() for t in _state_tensors(state))
    with pytest.raises(ValueError, match="captured cell"):
        transition(core, env, state, **(args | {"geometry": 0}), decode_fn=rnnt.decode_dense_masked_frames)
    for left, right in zip(_state_tensors(state), before, strict=True):
        torch.testing.assert_close(left, right, atol=0, rtol=0)


@torch.inference_mode()
def test_bucket_full_transaction_preserves_mixed_status_and_resident_commit():
    core, env, state, args = _fixture()
    transition = capture_chunk_bucket(core, env, state, **args, vllm_config=None, runtime=_graph_runtime())
    # Recycled pages must remain unread for fresh rows. A malformed row must
    # preserve its sentinel page even when its neighbors commit successfully.
    pools = _fresh_pools(33)
    for value in pools.values():
        for tensor in value if isinstance(value, list) else [value]:
            tensor.fill_(123)
    expected_pools = _clone_pools(pools)
    env[1, advance.ENV_PROMPT_INDEX] = 1
    plan = _plan(prefills=list(range(1, 33)), num_pool_blocks=33, geometries=[1] * 32)
    kwargs = dict(
        adapter=advance.make_mrv1_adapter(hidden_size=CARRIER_HIDDEN, park_id=PARK_ID, blank_id=core.blank_id),
        decode_resolver=_fixed_resolver(rnnt.decode_dense_masked_frames),
        placeholder_id=PLACEHOLDER_ID,
        park_id=PARK_ID,
    )
    sink, reference_sink = _CommitRecorder(), _CommitRecorder()
    ids = torch.full((32,), PLACEHOLDER_ID, dtype=torch.long)
    expected = advance.advance_model_rows(core, ids, env, plan, **expected_pools, **kwargs, commit_sink=reference_sink)
    actual = advance.advance_model_rows(
        core, ids, env, plan, **pools, **kwargs, commit_sink=sink, bucket_transition=transition
    )
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    _assert_pools_equal(pools, expected_pools)
    torch.testing.assert_close(sink.staged[0], reference_sink.staged[0], atol=0, rtol=0)
    assert int(sink.staged[0][0]) == 0
    assert int(sink.staged[0][1]) != 0
    assert torch.all(pools["book_pool"][2] == 123)


@torch.inference_mode()
def test_full_turn_timing_fixture_uses_valid_checkpoint():
    from p6c_chunk_bucket_screen import run_full_turn_screen

    core, env, state, args = _fixture()
    for step in range(6):
        env[:, advance.ENV_CHUNK_SEQUENCE] = step
        bucket = advance.advance_chunk_bucket(core, env, state, **args, decode_fn=rnnt.decode_dense_masked_frames)
        assert not torch.count_nonzero(bucket.result.row_status)
    report = run_full_turn_screen(
        core,
        state,
        bucket.batch,
        baseline_encoder=None,
        baseline_decode=rnnt.decode_dense_masked_frames,
        vllm_config=None,
        runtime=_graph_runtime(),
    )
    assert report["full_transaction_exact"]
    assert report["clean_statuses"]
    assert len(report["timings"]) == 3


@torch.inference_mode()
def test_endpoint_overflow_masks_only_delta_predicate(monkeypatch):
    from dataclasses import replace

    from vllm_omni.model_executor.models.nemotron_asr import endpointing

    core, env, _state, _args = _fixture()
    original_observe = endpointing.observe_chunk_tensors

    def overflow(**kwargs):
        output = original_observe(**kwargs)
        forced = torch.zeros_like(output.overflow)
        forced[:2] = True
        return replace(output, overflow=forced)

    def pending_delta(*args, **kwargs):
        bucket = advance.advance_chunk_bucket(*args, **kwargs)
        invariant = torch.zeros_like(bucket.counter_invariant_bad)
        invariant[1] = True
        return replace(
            bucket, counter_invariant_bad=invariant, counter_delta_bad=torch.ones_like(bucket.counter_delta_bad)
        )

    monkeypatch.setattr(endpointing, "observe_chunk_tensors", overflow)
    pools = _fresh_pools(33)
    before = _clone_pools(pools)
    sink = _CommitRecorder()
    advance.advance_model_rows(
        core,
        torch.full((32,), PLACEHOLDER_ID, dtype=torch.long),
        env,
        _plan(prefills=list(range(1, 33)), num_pool_blocks=33, geometries=[1] * 32),
        **pools,
        endpoint_history_pool=torch.zeros(33, 8, dtype=torch.int32),
        endpoint_book_pool=torch.zeros(33, 6, dtype=torch.int32),
        eou_token_id=core.blank_id + 3,
        adapter=advance.make_mrv1_adapter(hidden_size=CARRIER_HIDDEN, park_id=PARK_ID, blank_id=core.blank_id),
        decode_resolver=_fixed_resolver(rnnt.decode_dense_masked_frames),
        placeholder_id=PLACEHOLDER_ID,
        park_id=PARK_ID,
        commit_sink=sink,
        bucket_transition=pending_delta,
    )
    status = sink.staged[0]
    assert int(status[0]) == advance.ROW_STATUS_DECODE_INVARIANT
    assert int(status[1]) == advance.ROW_STATUS_DECODE_INVARIANT | advance.ROW_STATUS_BOOK_INVARIANT
    assert torch.all(status[2:] == advance.ROW_STATUS_BOOK_INVARIANT)
    _assert_pools_equal(pools, before)
