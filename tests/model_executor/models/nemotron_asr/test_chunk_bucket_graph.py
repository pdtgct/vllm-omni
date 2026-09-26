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


def _fixture(population=32):
    core = _tiny_core()
    env = torch.stack([_envelope(torch.randn(2560), final=False, seq=0, geometry=1) for _ in range(population)])
    state = _fresh_state(population)
    args = dict(
        geometry=1,
        admitted_prompt=torch.zeros(population, dtype=torch.long),
        incoming_status=torch.zeros(population, dtype=torch.int32),
        queue_capacity=48,
    )
    return core, env, state, args


@torch.inference_mode()
@pytest.mark.parametrize("population,tier", [(31, 32), (63, 64)])
def test_exact_encoder_population_keeps_split_decoder_padding_and_retained_outputs(population, tier):
    from vllm_omni.model_executor.models.nemotron_asr.decode_graph import DenseGraphBinding

    core, env, state, args = _fixture(population)
    encoder_populations = []
    hook = core.encoder.pre_encode.register_forward_pre_hook(
        lambda _module, inputs: encoder_populations.append(int(inputs[0].shape[0]))
    )
    decoder_populations = []

    def binding(decode_fn):
        result = DenseGraphBinding(
            decode_fn=decode_fn,
            predictor=core.predictor,
            joint=core.joint,
            vllm_config=None,
            frame_widths=(None, 2),
            tiers=(tier,),
            encoder_hidden=state.channel[0].shape[-1],
            predictor_layers=2,
            predictor_hidden=16,
            blank_id=core.blank_id,
            runtime=_graph_runtime(),
        )
        result.warmup(torch.device("cpu"), torch.float32)
        return result

    # Startup uses full tiers; padding assertions apply only after it completes.
    candidate_decoder = binding(rnnt.decode_dense_masked_frames)
    reference_decoder = binding(rnnt.decode_dense_masked_frames)
    # The eager tier wrapper keeps its original raw callable; inspect its staging
    # through an independent eager tier, while both paths use identical arithmetic.
    graph_decode = candidate_decoder.decode_fn(geometry=1, tier=tier)
    eager_decode = candidate_decoder.uncaptured_decode_fn(geometry=1, tier=tier)

    def checked_decode(frames, lengths, predictor, joint, decoder_state):
        result = eager_decode(frames, lengths, predictor, joint, decoder_state)
        entry = candidate_decoder._entries[(1, tier)]
        decoder_populations.append(int(entry.enc_frames.shape[0]))
        assert torch.count_nonzero(entry.enc_lengths[population:]) == 0
        assert torch.count_nonzero(entry.enc_frames[population:]) == 0
        assert torch.count_nonzero(entry.h[:, population:]) == 0
        assert torch.count_nonzero(entry.c[:, population:]) == 0
        assert torch.all(entry.last_label[population:] == core.blank_id)
        return result

    encoder_binding = object()
    transition = capture_chunk_bucket(
        core,
        env,
        state,
        **args,
        vllm_config=None,
        runtime=_graph_runtime(),
        capture_decode_fn=checked_decode,
        admitted_decode_fn=graph_decode,
        admitted_encoder_transition=encoder_binding,
        decoder_tier=tier,
    )
    assert set(encoder_populations) == {population}
    held, held_copy = None, ()
    for step in range(3):
        env[:, advance.ENV_CHUNK_SEQUENCE] = step
        env[:, advance.ENVELOPE_HEADER_SLOTS :] *= 0.71
        if step == 2:
            args["incoming_status"][0] = advance.ROW_STATUS_DECODE_INVARIANT
        expected_state = _clone_state(state)
        expected = advance.advance_chunk_bucket(
            core, env, expected_state, **args, decode_fn=reference_decoder.decode_fn(geometry=1, tier=tier)
        )
        # Every padding slot must be overwritten before it reaches the decoder.
        entry = candidate_decoder._entries[(1, tier)]
        entry.enc_frames.fill_(float("nan"))
        entry.h.fill_(float("nan"))
        entry.c.fill_(float("nan"))
        entry.enc_lengths.fill_(99)
        entry.last_label.fill_(999999)
        actual = transition(core, env, state, **args, decode_fn=graph_decode, encoder_transition=encoder_binding)
        assert transition.replay_count == step + 1
        assert encoder_populations[-1] == population
        for left, right in zip(
            (*_outputs(actual), *_state_tensors(state)),
            (*_outputs(expected), *_state_tensors(expected_state)),
            strict=True,
        ):
            torch.testing.assert_close(left, right, atol=0, rtol=0)
        # The fallback exact encoder cell uses the SAME decoder tier workspace.
        # Its writes must not modify any escaped prior CHUNK outputs.
        fallback_env = torch.stack([_envelope(torch.randn(2560), final=False, seq=0, geometry=1) for _ in range(tier)])
        advance.advance_chunk_bucket(
            core,
            fallback_env,
            _fresh_state(tier),
            geometry=1,
            admitted_prompt=torch.zeros(tier, dtype=torch.long),
            incoming_status=torch.zeros(tier, dtype=torch.int32),
            queue_capacity=48,
            decode_fn=graph_decode,
        )
        assert encoder_populations[-1] == tier
        if held is not None:
            for left, right in zip(_outputs(held), held_copy, strict=True):
                torch.testing.assert_close(left, right, atol=0, rtol=0)
        held, held_copy = actual, tuple(t.clone() for t in _outputs(actual))
    hook.remove()
    assert set(encoder_populations) == {population, tier}
    assert set(decoder_populations) == {tier}


@torch.inference_mode()
def test_chunk_binding_resolution_failure_precedes_resident_gather(monkeypatch):
    core, env, _state, _args = _fixture(31)

    class UnreadyBinding:
        def resolve(self, **_kwargs):
            raise ValueError("CHUNK graph inventory is incomplete")

    def forbidden(*_args, **_kwargs):
        pytest.fail("unready CHUNK binding read resident state")

    monkeypatch.setattr(advance, "_gather_initialized_rows", forbidden)
    with pytest.raises(ValueError, match="CHUNK graph inventory"):
        advance.advance_model_rows(
            core,
            torch.full((31,), PLACEHOLDER_ID, dtype=torch.long),
            env,
            _plan(prefills=list(range(1, 32)), num_pool_blocks=32, geometries=[1] * 31),
            **_fresh_pools(32),
            adapter=advance.make_mrv1_adapter(hidden_size=CARRIER_HIDDEN, park_id=PARK_ID, blank_id=core.blank_id),
            decode_resolver=_fixed_resolver(rnnt.decode_dense_masked_frames),
            placeholder_id=PLACEHOLDER_ID,
            park_id=PARK_ID,
            bucket_transition=UnreadyBinding(),
        )


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
    assert transition.replay_count == 3
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
