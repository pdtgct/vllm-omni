# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""CPU boundary/ownership checks; real CUDA replay remains a separate gate."""

from types import SimpleNamespace

import pytest
import torch
from test_advance_model_rows_local import (
    CARRIER_HIDDEN,
    PARK_ID,
    PLACEHOLDER_ID,
    _assert_native_burst_transaction,
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
@pytest.mark.parametrize("population,tier", [(1, 1), (31, 32), (63, 64)])
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
        _discard=lambda: None,
    )
    decoder = SimpleNamespace(
        execution_tier=lambda _n: tier, uncaptured_decode_fn=lambda **_kw: object(), decode_fn=lambda **_kw: object()
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
@pytest.mark.parametrize("mode", [0, 1])
def test_endpoint_overflow_masks_only_delta_predicate(monkeypatch, mode):
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

    monkeypatch.setattr(endpointing, "observe_chunk_tensors" if mode else "observe_disabled_chunk_tensors", overflow)
    pools = _fresh_pools(33)
    pools["endpoint_history_pool"] = torch.full((33, 8), 2, dtype=torch.int32)
    pools["endpoint_book_pool"] = torch.full((33, 6), 7, dtype=torch.int32)
    before = _clone_pools(pools)
    sink = _CommitRecorder()
    advance.advance_model_rows(
        core,
        torch.full((32,), PLACEHOLDER_ID, dtype=torch.long),
        env,
        replace(
            _plan(prefills=list(range(1, 33)), num_pool_blocks=33, geometries=[1] * 32),
            endpoint_mode=torch.full((32,), mode, dtype=torch.int64),
        ),
        **pools,
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


@torch.inference_mode()
def _run_chunk_native_handoff(population, *, device="cpu", runtime=None, vllm_config=None):
    """Tiny transaction oracle; a caller-supplied runtime enables real CUDA capture.

    This checks handoff composition, not trained numerical qualification or the
    standalone native encoder inventory used by fallback serving cells.
    """
    from vllm_omni.model_executor.models.nemotron_asr.chunk_bucket_graph import ExactChunkGraphBinding
    from vllm_omni.model_executor.models.nemotron_asr.decode_graph import DenseGraphBinding
    from vllm_omni.model_executor.models.nemotron_asr.native_burst import NativeBurstHandoff

    device = torch.device(device)
    if runtime is None:
        assert device.type == "cpu", "CUDA requires the actual platform graph runtime"
        runtime = _graph_runtime()
    core = _tiny_core(seed=1)
    for module in vars(core).values():
        if isinstance(module, torch.nn.Module):
            module.to(device)
    tier = 1 if population == 1 else population + 1
    decoder = DenseGraphBinding(
        decode_fn=rnnt.decode_dense_masked_frames,
        predictor=core.predictor,
        joint=core.joint,
        vllm_config=vllm_config,
        frame_widths=(None, 2),
        tiers=(1, 2) if population == 1 else (tier,),
        encoder_hidden=32,
        predictor_layers=2,
        predictor_hidden=16,
        blank_id=core.blank_id,
        runtime=runtime,
    )
    decoder.warmup(device, torch.float32)
    # Use the real serving resolver and captured CHUNK callable; the small
    # fixture supplies only the sealed inventory fields that they consume.
    execution = SimpleNamespace(
        ready=True,
        transition=None,
        _chunk_graph_entries={},
        reserve_chunk_graph_cells=lambda _cells: None,
    )
    binding = ExactChunkGraphBinding(core, None, execution, decoder, [population])
    state = advance.SessionStateBatch(
        **{
            name: [t.to(device) for t in value] if isinstance(value, list) else value.to(device)
            for name, value in vars(_fresh_state(population)).items()
        }
    )
    env = torch.stack([_envelope(torch.zeros(2560), final=False, seq=0, geometry=1) for _ in range(population)])
    execution._chunk_graph_entries[(1, population)] = capture_chunk_bucket(
        core,
        env.to(device),
        state,
        geometry=1,
        admitted_prompt=torch.zeros(population, dtype=torch.long, device=device),
        incoming_status=torch.zeros(population, dtype=torch.int32, device=device),
        queue_capacity=48,
        vllm_config=vllm_config,
        runtime=runtime,
        capture_decode_fn=decoder.uncaptured_decode_fn(geometry=1, tier=tier),
        admitted_decode_fn=decoder.decode_fn(geometry=1, tier=tier),
        decoder_tier=tier,
    )

    def resolve(request):
        return advance.ResolvedDecode(
            arm="dense-graphed",
            decode_fn=decoder.decode_fn(
                geometry=request.geometry, tier=decoder.execution_tier(request.execution_batch_size)
            ),
        )

    replay_pools = {
        name: [t.to(device) for t in value] if isinstance(value, list) else value.to(device)
        for name, value in _fresh_pools(population + 2).items()
    }
    native_pools = _clone_pools(replay_pools)
    if device.type == "cuda":
        advance.warmup_advance_model_rows_scatter(**replay_pools)
        advance.warmup_advance_model_rows_scatter(**native_pools)
    handoff = NativeBurstHandoff(max_tokens=22)
    sequences = {}
    retained = []
    receipts = []
    torch.manual_seed(5)
    for step, blocks in enumerate(
        (
            list(range(1, population + 1)),
            list(range(population + 1, 0, -1)),
            list(range(1, population + 1)),
        )
    ):
        final = step == 2
        prompts = [(block - 1) % 4 for block in blocks]
        carrier = torch.stack(
            [
                _envelope(
                    torch.randn(1920 if final else 2560) * 0.01,
                    final=final,
                    seq=sequences.get(block, 0),
                    geometry=1,
                    prompt=prompt,
                )
                for block, prompt in zip(blocks, prompts, strict=True)
            ]
        ).to(device)
        plan = _plan(
            prefills=blocks,
            has_initial=[block in sequences for block in blocks],
            num_pool_blocks=population + 2,
            geometries=[1] * len(blocks),
            prompts=prompts,
            request_ids=tuple(f"session-{block}" for block in blocks),
            generations=[1] * len(blocks),
        )
        failed = (0,) if final and population > 1 else ()
        if failed:
            carrier[0, advance.ENV_PROMPT_INDEX] = (prompts[0] + 1) % 4
        payload = _assert_native_burst_transaction(
            core,
            replay_pools,
            native_pools,
            carrier,
            plan,
            handoff=handoff,
            epoch=step + 1,
            resolver=resolve,
            bucket_transition=binding,
            failed_rows=failed,
        )
        # Later calls reuse both CHUNK-owned and fallback decoder scratch.
        tensors = (payload.sampled_token_ids, payload.num_sampled, payload.row_status, payload.queue, payload.book)
        for previous, copies in retained:
            for actual, expected in zip(previous, copies, strict=True):
                torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        retained.append((tensors, tuple(t.clone() for t in tensors)))
        receipts.append(
            dict(
                population=len(blocks),
                final=final,
                fresh_rows=sum(block not in sequences for block in blocks),
                block_order=blocks,
                counts_including_park=payload.num_sampled.tolist(),
                row_status=payload.row_status.tolist(),
                token_ids=[payload.sampled_token_ids[i, : int(n)].tolist() for i, n in enumerate(payload.num_sampled)],
            )
        )
        for block in blocks:
            sequences[block] = sequences.get(block, 0) + 1
    emitted = {token for step in receipts for tokens in step["token_ids"] for token in tokens if token != PARK_ID}
    assert len(emitted) > 1, "fixture must distinguish label order, not just repeated labels"
    receipt = binding.receipt()
    assert receipt["cells"][0]["successful_replays"] == 4
    assert receipt["fallbacks"] == [{"geometry": 1, "encoder_population": population + 1, "successful_calls": 2}]
    return dict(
        selected_population=population,
        transactions=receipts,
        chunk_binding=receipt,
        all_replayed_pools_equal=True,
        retained_outputs_unchanged=True,
        scope="Random tiny fixture, handoff composition only; not trained qualification or serving",
    )


@pytest.mark.parametrize("population", [1, 31, 63])
def test_chunk_native_handoff_matches_replay_through_population_changes(population):
    _run_chunk_native_handoff(population)


@torch.inference_mode()
@pytest.mark.parametrize(
    "modes,geometries,expected_calls",
    [
        ([0], [1], [("disabled", 1)]),
        ([0, 0], [1, 1], [("disabled", 2)]),
        ([1, 1], [1, 1], [("generic", 2)]),
        ([0, 1], [1, 1], [("generic", 2)]),
        ([1, 0, 0], [0, 1, 1], [("generic", 1), ("disabled", 2)]),
    ],
)
def test_endpoint_selection_uses_actual_cpu_bucket_and_preserves_atomic_pools(
    monkeypatch, modes, geometries, expected_calls
):
    # @spec PORT-SEG-002, PORT-SEG-003, PORT-SEG-007
    from dataclasses import replace

    from vllm_omni.model_executor.models.nemotron_asr import endpointing

    population = len(modes)
    core = _tiny_core(seed=29)
    env = torch.stack(
        [
            _envelope(torch.randn(1280 if geometry == 0 else 2560), final=False, seq=0, geometry=geometry)
            for geometry in geometries
        ]
    )
    pools = _fresh_pools(population + 1)
    pools["endpoint_history_pool"] = torch.zeros(population + 1, 12, dtype=torch.int32)
    pools["endpoint_book_pool"] = torch.zeros(population + 1, 6, dtype=torch.int32)
    reference_pools = _clone_pools(pools)
    plan = replace(
        _plan(prefills=list(range(1, population + 1)), num_pool_blocks=population + 1, geometries=geometries),
        endpoint_mode=torch.tensor(modes, dtype=torch.int64),
    )
    generic = endpointing.observe_chunk_tensors
    disabled = endpointing.observe_disabled_chunk_tensors
    calls = []

    def checked(which, fn, **kwargs):
        calls.append((which, kwargs["mode"].numel()))
        if which == "disabled":
            assert torch.all(kwargs["mode"] == 0)
        before = {key: value.clone() for key, value in kwargs.items() if isinstance(value, torch.Tensor)}
        actual, expected = fn(**kwargs), generic(**kwargs)
        for name in vars(actual):
            torch.testing.assert_close(getattr(actual, name), getattr(expected, name), atol=0, rtol=0)
        for key, value in before.items():
            torch.testing.assert_close(kwargs[key], value, atol=0, rtol=0)
        return actual

    monkeypatch.setattr(
        endpointing, "observe_disabled_chunk_tensors", lambda **kwargs: checked("disabled", disabled, **kwargs)
    )
    monkeypatch.setattr(endpointing, "observe_chunk_tensors", lambda **kwargs: checked("generic", generic, **kwargs))
    kwargs = dict(
        eou_token_id=core.blank_id + 3,
        adapter=advance.make_mrv1_adapter(hidden_size=CARRIER_HIDDEN, park_id=PARK_ID, blank_id=core.blank_id),
        decode_resolver=_fixed_resolver(rnnt.decode_dense_masked_frames),
        placeholder_id=PLACEHOLDER_ID,
        park_id=PARK_ID,
    )
    ids = torch.full((population,), PLACEHOLDER_ID, dtype=torch.long)
    sink, reference_sink = _CommitRecorder(), _CommitRecorder()
    actual = advance.advance_model_rows(core, ids, env, plan, **pools, **kwargs, commit_sink=sink)
    assert calls == expected_calls
    monkeypatch.setattr(endpointing, "observe_disabled_chunk_tensors", generic)
    monkeypatch.setattr(endpointing, "observe_chunk_tensors", generic)
    expected = advance.advance_model_rows(core, ids, env, plan, **reference_pools, **kwargs, commit_sink=reference_sink)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    torch.testing.assert_close(sink.staged[0], reference_sink.staged[0], atol=0, rtol=0)
    _assert_pools_equal(pools, reference_pools)
