# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Scratch ownership checks, including an opt-in real CUDA transaction test."""

import pytest
import torch
from test_advance_model_rows_local import (
    _BOOK,
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
from test_encoder_execution import _graph_runtime, _real_cuda_graph_runtime

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
@pytest.mark.parametrize("population,tier", [(1, 1), (31, 32), (63, 64)])
@pytest.mark.parametrize("borrowed", [False, True])
def test_exact_encoder_population_keeps_split_decoder_padding_and_retained_outputs(
    population, tier, borrowed, monkeypatch
):
    from vllm_omni.model_executor.models.nemotron_asr.decode_graph import DenseGraphBinding
    from vllm_omni.model_executor.models.nemotron_asr.encoder_execution import EncoderCacheStorage

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
        if borrowed:
            storage = EncoderCacheStorage(tuple(state.channel), tuple(state.time), tuple(state.window_valid))
            with transition._encoder_scratch.hold(torch.device("cpu")) as transaction:
                borrow = transition._borrow_cache(transaction, (1, population), storage, _state_tensors(state))
                for destination_family, source_family in zip(borrow.storage, storage, strict=True):
                    for destination, source in zip(destination_family, source_family, strict=True):
                        destination.copy_(source)
                working = advance.SessionStateBatch(
                    raw_tail=state.raw_tail,
                    mel_tail=state.mel_tail,
                    frontend_counters=state.frontend_counters,
                    channel=list(borrow.storage.channel),
                    time=list(borrow.storage.time),
                    window_valid=list(borrow.storage.valid),
                    h=state.h,
                    c=state.c,
                    last_label=state.last_label,
                )
                cache_ids = {id(tensor) for family in borrow.storage for tensor in family}
                original_copy = torch.Tensor.copy_

                def checked_copy(destination, source, *args, **kwargs):
                    assert not (id(destination) in cache_ids and destination is source), "redundant cache staging"
                    return original_copy(destination, source, *args, **kwargs)

                with monkeypatch.context() as patch:
                    patch.setattr(torch.Tensor, "copy_", checked_copy)
                    actual = transition._borrowed_transition(
                        borrow, core, env, working, **args, decode_fn=graph_decode, encoder_transition=encoder_binding
                    )
                # Stand in for the last consumer: retain the checkpoint before
                # returning storage to the next transaction.
                state = _clone_state(working)
        else:
            actual = transition(core, env, state, **args, decode_fn=graph_decode, encoder_transition=encoder_binding)
        assert transition.replay_count == step + 1
        assert encoder_populations[-1] == population
        for left, right in zip(
            (*_outputs(actual), *_state_tensors(state)),
            (*_outputs(expected), *_state_tensors(expected_state)),
            strict=True,
        ):
            assert torch.equal(left, right)
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
                assert torch.equal(left, right)
        held, held_copy = actual, tuple(t.clone() for t in _outputs(actual))
    hook.remove()
    assert set(encoder_populations) == {population, tier}
    assert set(decoder_populations) == {tier}


@pytest.mark.parametrize("tier", [None, 2])
def test_single_chunk_requires_existing_sealed_decoder_tier(tier):
    core, env, state, args = _fixture(1)
    with pytest.raises(ValueError, match="decoder"):
        capture_chunk_bucket(core, env, state, **args, vllm_config=None, decoder_tier=tier)


@torch.inference_mode()
@pytest.mark.parametrize("population,tier", [(1, 1), (31, 32), (63, 64)])
def test_chunk_warmup_scratch_matches_real_gather_layout(monkeypatch, population, tier):
    from types import SimpleNamespace

    from vllm_omni.model_executor.models.nemotron_asr import chunk_bucket_graph as graph
    from vllm_omni.model_executor.models.nemotron_asr.configuration_nemotron_asr import NemotronASRConfig
    from vllm_omni.model_executor.models.nemotron_asr.profile_execution import build_profile_invocation

    core = _tiny_core()
    config = NemotronASRConfig(
        num_asr_labels=12,
        vocab_size=17,
        eos_token_id=13,
        audio_chunk_token_id=14,
        eou_token_id=15,
        flush_token_id=16,
        n_mels=16,
        d_model=32,
        n_layers=2,
        conv_kernel=5,
        att_context_left=8,
        att_context_right=1,
        pred_hidden=16,
        joint_hidden=16,
        num_prompts=4,
        prompt_dictionary={"en-US": 0},
        supported_num_lookahead_tokens=[1],
    )
    pools = build_profile_invocation(config, num_rows=population, device=torch.device("cpu"), geometry_id=1).pools
    source_pools = (
        pools.frontend_raw,
        pools.frontend_mel,
        pools.frontend_counters,
        *pools.channel,
        *pools.valid_length,
        *pools.convolution,
        pools.predictor_h,
        pools.predictor_c,
    )
    rows = torch.arange(1, population + 1)
    observed = []

    def capture(_core, _env, state, **_kwargs):
        for tensor, pool in zip(_state_tensors(state)[:-1], source_pools, strict=True):
            for fresh in (False, True):
                gathered = advance._gather_initialized_rows(pool, rows, torch.full((population,), fresh))
                assert graph._tensor_signature(tensor) == graph._tensor_signature(gathered)
            assert torch.count_nonzero(tensor) == 0
            if population > 1:
                assert graph._tensor_signature(tensor) == graph._tensor_signature(torch.zeros_like(pool[1:]))
        observed.append(population)
        return lambda *_args, **_kwargs: None

    execution: SimpleNamespace = SimpleNamespace(
        ready=False,
        _sealed=True,
        _chunk_graph_entries={},
        _vllm_config=None,
        _graph_runtime=_graph_runtime(),
        transition=object(),
        reserve_chunk_graph_cells=lambda _cells: None,
        _record_memory_diagnostic=lambda *_args, **_kwargs: None,
        publish_chunk_graphs=lambda entries: execution._chunk_graph_entries.update(entries),
        _validate_scratch_inventory=lambda **_kwargs: None,
        _discard=lambda: None,
    )
    decoder = SimpleNamespace(
        execution_tier=lambda _n: tier,
        uncaptured_decode_fn=lambda **_kw: object(),
        decode_fn=lambda **_kw: object(),
        _entries={},
    )
    monkeypatch.setattr(graph, "capture_chunk_bucket", capture)
    binding = graph.ExactChunkGraphBinding(core, config, execution, decoder, [population])
    binding.warmup(torch.device("cpu"))
    assert observed == [population]


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
            assert torch.equal(left, right)
        if held is not None:
            for left, right in zip(_outputs(held), held_copy, strict=True):
                assert torch.equal(left, right)
        held, held_copy = actual, tuple(t.clone() for t in _outputs(actual))
    before = tuple(t.clone() for t in _state_tensors(state))
    with pytest.raises(ValueError, match="captured cell"):
        transition(core, env, state, **(args | {"geometry": 0}), decode_fn=rnnt.decode_dense_masked_frames)
    assert transition.replay_count == 3
    for left, right in zip(_state_tensors(state), before, strict=True):
        torch.testing.assert_close(left, right, atol=0, rtol=0)


@torch.inference_mode()
def test_bucket_full_transaction_preserves_mixed_status_and_resident_commit(monkeypatch):
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
    destinations = []
    original_gather = advance._gather_initialized_rows
    original_scatter = advance._execute_masked_page_scatter_

    def gather(pool, blocks, fresh, **kwargs):
        destination = kwargs.get("destination")
        if destination is not None:
            destinations.append(destination)
        return original_gather(pool, blocks, fresh, **kwargs)

    def scatter(*args):
        # Even the final scatter must execute before the transaction releases
        # scratch. Reentry cannot steal it during validation/reservation/commit.
        with pytest.raises(ValueError, match="active|overlap|reentrant"):
            with transition._encoder_scratch.hold(torch.device("cpu")):
                pass
        return original_scatter(*args)

    monkeypatch.setattr(advance, "_gather_initialized_rows", gather)
    monkeypatch.setattr(advance, "_execute_masked_page_scatter_", scatter)
    actual = advance.advance_model_rows(
        core, ids, env, plan, **pools, **kwargs, commit_sink=sink, bucket_transition=transition
    )
    assert len(destinations) == 6  # both layers, all three cache families
    with transition._encoder_scratch.hold(torch.device("cpu")):
        pass
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    _assert_pools_equal(pools, expected_pools, raw_bytes=True)
    torch.testing.assert_close(sink.staged[0], reference_sink.staged[0], atol=0, rtol=0)
    assert int(sink.staged[0][0]) == 0
    assert int(sink.staged[0][1]) != 0
    assert torch.all(pools["book_pool"][2] == 123)


@torch.inference_mode()
@pytest.mark.parametrize("failure", ["compute", "adapter", "descriptor", "reserve"])
def test_chunk_borrow_failure_preserves_resident_bytes_and_recovers(monkeypatch, failure):
    core, env, state, args = _fixture(32)
    transition = capture_chunk_bucket(core, env, state, **args, vllm_config=None, runtime=_graph_runtime())
    pools = _fresh_pools(33)
    for value in pools.values():
        for tensor in value if isinstance(value, list) else [value]:
            tensor.fill_(123)
    before = _clone_pools(pools)
    plan = _plan(prefills=list(range(1, 33)), num_pool_blocks=33, geometries=[1] * 32)
    kwargs = dict(
        adapter=advance.make_mrv1_adapter(hidden_size=CARRIER_HIDDEN, park_id=PARK_ID, blank_id=core.blank_id),
        decode_resolver=_fixed_resolver(rnnt.decode_dense_masked_frames),
        placeholder_id=PLACEHOLDER_ID,
        park_id=PARK_ID,
        bucket_transition=transition,
    )
    ids = torch.full((32,), PLACEHOLDER_ID, dtype=torch.long)

    def fail(*_args, **_kwargs):
        raise RuntimeError("injected failure")

    with monkeypatch.context() as patch:
        candidate = dict(kwargs)
        if failure == "compute":
            patch.setattr(advance, "advance_session", fail)
        elif failure == "adapter":
            candidate["adapter"] = fail
        elif failure == "descriptor":
            patch.setattr(advance, "validate_masked_page_scatter", fail)
        else:
            candidate["commit_sink"] = _CommitRecorder(fail_reserve=True)
        with pytest.raises(RuntimeError, match="failure|capacity"):
            advance.advance_model_rows(core, ids, env, plan, **pools, **candidate)
    _assert_pools_equal(pools, before, raw_bytes=True)
    advance.advance_model_rows(core, ids, env, plan, **pools, **kwargs)
    assert not torch.equal(pools["channel_pools"][0][1], before["channel_pools"][0][1])


@torch.inference_mode()
def test_chunk_capability_rejects_public_alias_stale_and_foreign_capture():
    from vllm_omni.model_executor.models.nemotron_asr.encoder_execution import EncoderCacheStorage

    core, env, state, args = _fixture(32)
    transition = capture_chunk_bucket(core, env, state, **args, vllm_config=None, runtime=_graph_runtime())
    other = capture_chunk_bucket(core, env, state, **args, vllm_config=None, runtime=_graph_runtime())
    storage = EncoderCacheStorage(tuple(state.channel), tuple(state.time), tuple(state.window_valid))
    with transition._encoder_scratch.hold(torch.device("cpu")) as transaction:
        with pytest.raises(ValueError, match="cell"):
            transition._borrow_cache(transaction, (1, 31), storage, ())
        borrow = transition._borrow_cache(transaction, (1, 32), storage, _state_tensors(state))
        state.channel, state.time, state.window_valid = map(list, borrow.storage)
        for family in borrow.storage:
            for tensor in family:
                tensor.zero_()
        with pytest.raises(ValueError, match="stale|capability"):
            other._borrowed_transition(borrow, core, env, state, **args, decode_fn=rnnt.decode_dense_masked_frames)
        transition._borrowed_transition(borrow, core, env, state, **args, decode_fn=rnnt.decode_dense_masked_frames)
    with pytest.raises(ValueError, match="stale|capability"):
        transition._borrowed_transition(borrow, core, env, state, **args, decode_fn=rnnt.decode_dense_masked_frames)
    with pytest.raises(ValueError, match="alias"):
        transition(core, env, state, **args, decode_fn=rnnt.decode_dense_masked_frames)


@torch.inference_mode()
@pytest.mark.parametrize("mutation", ["set", "resize", "stride", "data"])
@pytest.mark.parametrize("boundary", ["borrow", "replay"])
def test_chunk_borrow_checks_same_object_mutation(monkeypatch, mutation, boundary):
    from test_encoder_execution import _mutate_scratch_tensor

    from vllm_omni.model_executor.models.nemotron_asr.encoder_execution import EncoderCacheStorage

    core, env, state, args = _fixture(32)
    transition = capture_chunk_bucket(core, env, state, **args, vllm_config=None, runtime=_graph_runtime())
    pools = EncoderCacheStorage(tuple(state.channel), tuple(state.time), tuple(state.window_valid))
    resident = tuple(t for family in pools for t in family)
    before = tuple(t.clone() for t in resident)
    tensor = transition._cache_storage.channel[0]
    original_identity = id(tensor)
    with transition._encoder_scratch.hold(torch.device("cpu")) as transaction:
        if boundary == "replay":
            borrow = transition._borrow_cache(transaction, (1, 32), pools, resident)
            state.channel, state.time, state.window_valid = map(list, borrow.storage)
        _mutate_scratch_tensor(tensor, mutation)
        monkeypatch.setattr(
            torch.Tensor, "copy_", lambda *_args, **_kwargs: pytest.fail("mutated scratch staged CHUNK graph inputs")
        )
        with pytest.raises(ValueError, match="storage|cell"):
            if boundary == "borrow":
                transition._borrow_cache(transaction, (1, 32), pools, resident)
            else:
                transition._borrowed_transition(
                    borrow, core, env, state, **args, decode_fn=rnnt.decode_dense_masked_frames
                )
        assert id(tensor) == original_identity
    assert transition.replay_count == 0
    assert all(torch.equal(left, right) for left, right in zip(resident, before, strict=True))


@torch.inference_mode()
def test_wrapped_chunk_extension_keeps_public_copy_path(monkeypatch):
    from functools import wraps

    core, env, state, args = _fixture(32)
    transition = capture_chunk_bucket(core, env, state, **args, vllm_config=None, runtime=_graph_runtime())

    @wraps(transition)
    def extension(*args, **kwargs):
        return transition(*args, **kwargs)

    def forbidden(*_args, **_kwargs):
        pytest.fail("extension wrapper acquired private graph scratch")

    monkeypatch.setattr(extension, "_borrow_cache", forbidden)
    advance.advance_model_rows(
        core,
        torch.full((32,), PLACEHOLDER_ID, dtype=torch.long),
        env,
        _plan(prefills=list(range(1, 33)), num_pool_blocks=33, geometries=[1] * 32),
        **_fresh_pools(33),
        adapter=advance.make_mrv1_adapter(hidden_size=CARRIER_HIDDEN, park_id=PARK_ID, blank_id=core.blank_id),
        decode_resolver=_fixed_resolver(rnnt.decode_dense_masked_frames),
        placeholder_id=PLACEHOLDER_ID,
        park_id=PARK_ID,
        bucket_transition=extension,
    )
    assert transition.replay_count == 1


def _mixed_inventory_config():
    from vllm_omni.model_executor.models.nemotron_asr.configuration_nemotron_asr import NemotronASRConfig

    return NemotronASRConfig(
        num_asr_labels=12,
        vocab_size=17,
        eos_token_id=13,
        audio_chunk_token_id=14,
        eou_token_id=15,
        flush_token_id=16,
        n_mels=16,
        d_model=32,
        n_layers=2,
        conv_kernel=5,
        att_context_left=8,
        att_context_right=1,
        pred_hidden=16,
        joint_hidden=16,
        num_prompts=4,
        prompt_dictionary={"en-US": 0},
        supported_num_lookahead_tokens=[1],
        encoder_execution_arm="eager-graphed",
    )


@torch.inference_mode()
def test_mixed_chunk_inventory_borrows_native_fallback_and_shared_decoder_safely(monkeypatch):
    from types import SimpleNamespace

    from vllm_omni.model_executor.models.nemotron_asr import encoder_execution as encoder
    from vllm_omni.model_executor.models.nemotron_asr.chunk_bucket_graph import ExactChunkGraphBinding
    from vllm_omni.model_executor.models.nemotron_asr.decode_graph import DenseGraphBinding
    from vllm_omni.model_executor.models.nemotron_asr.manifests import SESSION_LIMITS

    core = _tiny_core()
    core.policy = SimpleNamespace(dtype_for=lambda _name: torch.float32)
    config = _mixed_inventory_config()
    execution = encoder.build_encoder_execution(
        core, config, maximum_population=2, warmup_geometries=(1,), vllm_config=object(), graph_runtime=_graph_runtime()
    )
    decoder = DenseGraphBinding(
        decode_fn=rnnt.decode_dense_masked_frames,
        predictor=core.predictor,
        joint=core.joint,
        vllm_config=None,
        frame_widths=(None, 2),
        tiers=(1, 2),
        encoder_hidden=32,
        predictor_layers=2,
        predictor_hidden=16,
        blank_id=core.blank_id,
        runtime=_graph_runtime(),
    )
    decoder.warmup(torch.device("cpu"), torch.float32)
    binding = ExactChunkGraphBinding(core, config, execution, decoder, (1,))

    def invoke(geometry, population):
        shape = execution._geometry_shapes[geometry]
        return execution.transition(
            torch.zeros(population, 16, shape.mel_width),
            advance._GatheredCaches(_fresh_state(population)),
            torch.zeros(population, dtype=torch.long),
            torch.full((population,), shape.out_width, dtype=torch.long),
            shape.out_width,
            torch.zeros(population, dtype=torch.long),
        )

    execution.warmup_domain(expected_cells=((1, 1), (1, 2)), invoke=invoke)
    binding.warmup(torch.device("cpu"))
    assert binding.ready
    assert set(execution._graph_entries) == {(1, 2)}
    chunk = execution._chunk_graph_entries[(1, 1)]
    assert chunk._encoder_scratch is execution._scratch
    # Exercise CHUNK -> native fallback -> CHUNK against the same decoder
    # inventory. Each result and resident checkpoint survives the next replay.
    retained: list[tuple[torch.Tensor, torch.Tensor]] = []
    original_gather = advance._gather_initialized_rows
    destinations = []

    def gather(*args, **kwargs):
        if kwargs.get("destination") is not None:
            destinations.append(kwargs["destination"])
        return original_gather(*args, **kwargs)

    monkeypatch.setattr(advance, "_gather_initialized_rows", gather)
    monkeypatch.setattr(encoder, "_copy_cache_storage_", lambda *_args: pytest.fail("native fallback staged caches"))
    for population in (1, 2, 1):
        env = torch.stack(
            [
                _envelope(torch.randn(2560), final=False, seq=0, geometry=1, hidden=config.hidden_size)
                for _ in range(population)
            ]
        )
        pools = _fresh_pools(population + 1)
        pools["queue_pool"] = torch.zeros(population + 1, SESSION_LIMITS["queue_capacity"], dtype=torch.int32)
        expected_pools = _clone_pools(pools)
        kwargs = dict(
            adapter=advance.make_mrv1_adapter(hidden_size=config.hidden_size, park_id=PARK_ID, blank_id=core.blank_id),
            decode_resolver=_fixed_resolver(decoder.decode_fn(geometry=1, tier=population)),
            placeholder_id=PLACEHOLDER_ID,
            park_id=PARK_ID,
        )
        plan = _plan(
            prefills=list(range(1, population + 1)), num_pool_blocks=population + 1, geometries=[1] * population
        )
        ids = torch.full((population,), PLACEHOLDER_ID, dtype=torch.long)
        expected = advance.advance_model_rows(core, ids, env, plan, **expected_pools, **kwargs)
        destinations.clear()
        sink = _CommitRecorder()
        actual = advance.advance_model_rows(
            core,
            ids,
            env,
            plan,
            **pools,
            **kwargs,
            encoder_transition=execution.transition,
            bucket_transition=binding,
            commit_sink=sink,
        )
        assert len(destinations) == 6
        assert torch.count_nonzero(sink.staged[0]) == 0
        assert torch.equal(actual, expected)
        _assert_pools_equal(pools, expected_pools, raw_bytes=True)
        for old, snapshot in retained:
            assert torch.equal(old, snapshot)
        retained.append((actual, actual.clone()))
    assert chunk.replay_count == 2
    assert binding.receipt()["fallbacks"] == [{"geometry": 1, "encoder_population": 2, "successful_calls": 1}]


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires real CUDA capture and resident scatter")
@torch.inference_mode()
def test_real_cuda_borrowed_transactions_preserve_queued_commit_and_recover(monkeypatch):
    # @spec PORT-ADV-004, PORT-STATE-008, PORT-PERF-009
    from types import SimpleNamespace

    from vllm.config import VllmConfig

    from vllm_omni.model_executor.models.nemotron_asr import encoder_execution as encoder
    from vllm_omni.model_executor.models.nemotron_asr.chunk_bucket_graph import ExactChunkGraphBinding
    from vllm_omni.model_executor.models.nemotron_asr.decode_graph import DenseGraphBinding
    from vllm_omni.model_executor.models.nemotron_asr.manifests import SESSION_LIMITS

    device = torch.device("cuda", torch.accelerator.current_device_index())
    core = _tiny_core()
    for value in vars(core).values():
        if isinstance(value, torch.nn.Module):
            value.to(device).eval()
    core.policy = SimpleNamespace(dtype_for=lambda _name: torch.float32)
    config = _mixed_inventory_config()
    runtime = _real_cuda_graph_runtime(device)
    vllm_config = VllmConfig()
    execution = encoder.build_encoder_execution(
        core, config, maximum_population=2, warmup_geometries=(1,), vllm_config=vllm_config, graph_runtime=runtime
    )
    decoder = DenseGraphBinding(
        decode_fn=rnnt.decode_dense_masked_frames,
        predictor=core.predictor,
        joint=core.joint,
        vllm_config=vllm_config,
        frame_widths=(None, 2),
        tiers=(1, 2),
        encoder_hidden=32,
        predictor_layers=2,
        predictor_hidden=16,
        blank_id=core.blank_id,
        runtime=runtime,
    )
    decoder.warmup(device, torch.float32)
    binding = ExactChunkGraphBinding(core, config, execution, decoder, (1,))

    def warm(geometry, population):
        shape = execution._geometry_shapes[geometry]
        state = _fresh_state(population)
        for name, value in vars(state).items():
            setattr(state, name, [t.to(device) for t in value] if isinstance(value, list) else value.to(device))
        return execution.transition(
            torch.zeros(population, 16, shape.mel_width, device=device),
            advance._GatheredCaches(state),
            torch.zeros(population, dtype=torch.long, device=device),
            torch.full((population,), shape.out_width, dtype=torch.long, device=device),
            shape.out_width,
            torch.zeros(population, dtype=torch.long, device=device),
        )

    execution.warmup_domain(expected_cells=((1, 1), (1, 2)), invoke=warm)
    binding.warmup(device)
    chunk = execution._chunk_graph_entries[(1, 1)]
    assert binding.ready and set(execution._graph_entries) == {(1, 2)}
    assert chunk._encoder_scratch is execution._scratch
    pools = _fresh_pools(3)
    pools["queue_pool"] = torch.zeros(3, SESSION_LIMITS["queue_capacity"], dtype=torch.int32)
    # Poison recycled pages before upload; fresh rows must not read them.
    for value in pools.values():
        for tensor in value if isinstance(value, list) else [value]:
            tensor.fill_(123)
    pools = {
        key: [t.to(device) for t in value] if isinstance(value, list) else value.to(device)
        for key, value in pools.items()
    }
    reference = _clone_pools(pools)
    initial = _clone_pools(pools)
    advance.warmup_advance_model_rows_scatter(**pools)

    calls = []
    # Both native calls mix continuing block 1 with fresh block 2. The first
    # masks block 2; the next admits it cleanly, still ignoring its poison.
    for step, population in enumerate((1, 2, 2, 1)):
        env = torch.stack(
            [
                _envelope(
                    torch.randn(2560),
                    final=False,
                    seq=step if row == 0 else 0,
                    geometry=1,
                    hidden=config.hidden_size,
                    prompt=int(step == 1 and row == 1),
                )
                for row in range(population)
            ]
        ).to(device)
        plan = _plan(
            prefills=list(range(1, population + 1)),
            num_pool_blocks=3,
            geometries=[1] * population,
            has_initial=[step > 0] + [False] * (population - 1),
        )
        ids = torch.full((population,), PLACEHOLDER_ID, dtype=torch.long, device=device)
        calls.append((ids, env, plan))
    adapter = advance.make_mrv1_adapter(hidden_size=config.hidden_size, park_id=PARK_ID, blank_id=core.blank_id)

    def invoke(target, call, *, borrowed, sink, emission=adapter):
        ids, env, plan = call
        population = ids.shape[0]
        # Public CHUNK requires the exact admitted encoder/decode identities.
        # Wrapping only its outer bucket call disables private borrowing while
        # preserving those bindings. Native uses the public copy-in/copyback API.
        if borrowed:
            bucket, transition = binding, execution.transition
        elif population == 1:
            bucket, transition = lambda *a, **kw: chunk(*a, **kw), execution.transition
        else:
            bucket, transition = None, lambda *a: execution.transition(*a)
        return advance.advance_model_rows(
            core,
            ids,
            env,
            plan,
            **target,
            adapter=emission,
            commit_sink=sink,
            decode_resolver=_fixed_resolver(decoder.decode_fn(geometry=1, tier=population)),
            placeholder_id=PLACEHOLDER_ID,
            park_id=PARK_ID,
            encoder_transition=transition,
            bucket_transition=bucket,
        )

    def drain(target, step):
        # Reuse the existing legal drained-checkpoint convention. Only queued
        # device writes: no status readback or CPU-dependent decoder loop.
        book = target["book_pool"][1 : 3 if step >= 2 else 2]
        book[:, _BOOK["queue_head"]].copy_(book[:, _BOOK["queue_length"]])
        book[:, _BOOK["pending_echo"]].zero_()
        book[:, _BOOK["expected_label"]].zero_()

    copies = []
    original_copy = encoder._copy_cache_storage_

    def copied(*args):
        copies.append(True)
        return original_copy(*args)

    monkeypatch.setattr(encoder, "_copy_cache_storage_", copied)
    expected = []
    reference_sink = _CommitRecorder()
    # Independent caller-owned pools and public staging, completed before the
    # candidate sequence; no baseline loader or alternate numerical kernel.
    for step, call in enumerate(calls):
        output = invoke(reference, call, borrowed=False, sink=reference_sink)
        expected.append((output, _clone_pools(reference)))
        drain(reference, step)
    assert len(copies) == 4  # two native calls, each with copy-in and copyback
    copies.clear()

    retained: list[tuple[torch.Tensor, torch.Tensor]] = []
    replayed = []
    original_native = execution._borrowed_transition
    original_chunk = chunk._borrowed_transition

    def retain(values):
        retained.extend((value, value.clone()) for value in values)

    def native_replay(*args):
        outputs = original_native(*args)
        retain(outputs)
        replayed.append("native")
        return outputs

    def chunk_replay(*args, **kwargs):
        output = original_chunk(*args, **kwargs)
        retain(_outputs(output))
        replayed.append("chunk")
        return output

    monkeypatch.setattr(execution, "_borrowed_transition", native_replay)
    monkeypatch.setattr(chunk, "_borrowed_transition", chunk_replay)
    original_gather = advance._gather_initialized_rows
    original_scatter = advance._execute_masked_page_scatter_
    gathers, scatters, aborts = [], [], []
    reject_stream = False

    def gather(*args, **kwargs):
        if reject_stream:
            pytest.fail("different-stream transaction reached resident gather")
        if kwargs.get("destination") is not None:
            gathers.append(kwargs["destination"])
        return original_gather(*args, **kwargs)

    def scatter(*args):
        # Observe the real executor; no replacement copy or fake scatter.
        original_scatter(*args)
        scatters.append(args[1])

    monkeypatch.setattr(advance, "_gather_initialized_rows", gather)
    monkeypatch.setattr(advance, "_execute_masked_page_scatter_", scatter)
    sink = _CommitRecorder()  # capture=False: records empty; stage only clones status

    def abort_after_replay(*_args):
        aborts.append((tuple(replayed), len(sink.plans), len(scatters)))
        raise RuntimeError("abort after queued replay before reservation")

    other_stream = torch.cuda.Stream(device=device)
    actual, failed = [], []
    chunk_count = chunk.replay_count
    # Graph capture/parity and scatter compilation synchronize during SETUP.
    # No explicit waits, tensor assertions/readbacks or oracle calls occur in
    # the candidate sequence. PyTorch's sync guard rejects supported implicit
    # synchronization sites too; it is not a complete CUDA API trace.
    runtime.synchronize(device)
    sync_debug = torch.cuda.get_sync_debug_mode()
    torch.cuda.set_sync_debug_mode("error")
    try:
        for step, call in enumerate(calls):
            if step in (1, 3):
                before = _clone_pools(pools)
                with pytest.raises(RuntimeError, match="abort after queued replay"):
                    invoke(pools, call, borrowed=True, sink=sink, emission=abort_after_replay)
                after = _clone_pools(pools)
                failed.append((before, after))
                reject_stream = True
                with torch.cuda.stream(other_stream), pytest.raises(ValueError, match="serialized stream"):
                    invoke(pools, call, borrowed=True, sink=sink)
                reject_stream = False
            output = invoke(pools, call, borrowed=True, sink=sink)
            retain((output,))
            actual.append((output, _clone_pools(pools)))
            drain(pools, step)
    finally:
        torch.cuda.set_sync_debug_mode(sync_debug)
        runtime.synchronize(device)

    assert not copies  # both native staging copies remain absent
    assert replayed == ["chunk", "native", "native", "native", "chunk", "chunk"]
    assert chunk.replay_count - chunk_count == 3
    assert len(gathers) == 6 * 6  # six cache tensors per successful/aborted replay
    pointers = [tuple(t.data_ptr() for t in gathers[i : i + 6]) for i in range(0, len(gathers), 6)]
    assert pointers[0] == pointers[4] == pointers[5]
    assert pointers[1] == pointers[2] == pointers[3]
    assert len(scatters) == 4 * 13  # all resident families, only successful calls
    assert [(events[-1], reservations, stores) for events, reservations, stores in aborts] == [
        ("native", 1, 13),
        ("chunk", 3, 39),
    ]
    assert len(sink.plans) == 4 and sink.cancels == 0
    for (output, checkpoint), (oracle, reference_checkpoint) in zip(actual, expected, strict=True):
        torch.testing.assert_close(output, oracle, rtol=0, atol=0)
        _assert_pools_equal(checkpoint, reference_checkpoint, raw_bytes=True)
    for before, after in failed:
        _assert_pools_equal(before, after, raw_bytes=True)
    for key, value in actual[1][1].items():
        original = initial[key]
        pairs = zip(value, original, strict=True) if isinstance(value, list) else [(value, original)]
        for tensor, poison in pairs:
            assert torch.equal(tensor[2].contiguous().view(torch.uint8), poison[2].contiguous().view(torch.uint8))
    for step, (status, reference_status) in enumerate(zip(sink.staged, reference_sink.staged, strict=True)):
        torch.testing.assert_close(status, reference_status, rtol=0, atol=0)
        assert status[0] == 0
        if step == 1:
            assert status[1] != 0
        else:
            assert torch.count_nonzero(status) == 0
    for output, snapshot in retained:
        torch.testing.assert_close(output, snapshot, rtol=0, atol=0)


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
