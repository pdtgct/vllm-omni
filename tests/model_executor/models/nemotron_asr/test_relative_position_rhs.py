# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""CPU contracts for the default-off eager relative-position experiment."""

from types import SimpleNamespace
from typing import Any

import pytest
import torch

from vllm_omni.model_executor.models.nemotron_asr.encoder import (
    FastConformerEncoder,
    StreamingCaches,
    stream_step,
)
from vllm_omni.model_executor.models.nemotron_asr.encoder_execution import build_encoder_execution
from vllm_omni.model_executor.models.nemotron_asr.lid import PromptConditioner

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _encoder():
    torch.manual_seed(73)
    return FastConformerEncoder(
        feat_in=16,
        d_model=32,
        d_ff=64,
        n_layers=2,
        n_heads=4,
        conv_kernel=5,
        subsampling_channels=16,
        att_context=(8, 1),
    ).eval()


def _prepare(encoder, *, maximum=7, widths=(2, 4)):
    encoder.prepare_stream_relative_position_rhs(
        out_widths=widths,
        cache_len=8,
        maximum_population=maximum,
        reference=torch.empty((), dtype=torch.float32),
    )


@torch.inference_mode()
@pytest.mark.parametrize("batch", [1, 2, 3, 7, 31, 63, 128])
@pytest.mark.parametrize("width", [2, 4])
def test_rhs_exact_full_window_flattened_geometry_and_shared_prefix(batch, width):
    encoder = _encoder()
    _prepare(encoder, maximum=128)
    reference = torch.empty(batch, width, 32)
    projected = encoder.stream_relative_position_projections(out_width=width, cache_len=8, reference=reference)
    rhs = encoder.stream_relative_position_rhs(out_width=width, cache_len=8, reference=reference)
    pointers = []
    for idx, layer in enumerate(encoder.layers):
        attn = layer.self_attn
        pos = encoder.pos_enc(torch.empty(1, width + 8, 32))
        native = attn.linear_pos(pos).view(1, -1, attn.h, attn.d_k).transpose(1, 2)
        assert torch.equal(projected[idx], native)
        assert projected[idx].stride() == native.stride()
        if batch == 1:
            assert rhs is None  # Preserve native noncontiguous B1 operand.
            continue
        buffer = getattr(attn, f"_stream_relative_rhs_{width + 8}")
        assert rhs[idx].untyped_storage().data_ptr() == buffer.untyped_storage().data_ptr()
        pointers.append(buffer.data_ptr())
        positions = 2 * (width + 8) - 1
        old_flat = (
            native.transpose(-2, -1)
            .expand(batch, attn.h, attn.d_k, positions)
            .reshape(batch * attn.h, attn.d_k, positions)
        )
        new_flat = rhs[idx].view(batch * attn.h, attn.d_k, positions)
        assert old_flat.stride() == new_flat.stride() == (attn.d_k * positions, positions, 1)
        # old_flat[b*H+h,d,p] = native[0,h,p,d] = new_flat[b*H+h,d,p].
        assert torch.equal(old_flat, new_flat)
        q = torch.randn(batch, attn.h, width, attn.d_k)
        assert torch.equal(torch.matmul(q, native.transpose(-2, -1)), torch.matmul(q, rhs[idx]))
    assert len(set(pointers)) == len(pointers)  # Learned weights are layer-owned.
    before = {name: value.data_ptr() for name, value in encoder.named_buffers()}
    _prepare(encoder, maximum=128)
    assert before == {name: value.data_ptr() for name, value in encoder.named_buffers()}
    assert not any("_stream_relative" in key for key in encoder.state_dict())


@torch.inference_mode()
@pytest.mark.parametrize("width,offset", [(2, 2), (4, 0)])
def test_rhs_full_stream_output_and_state_are_exact_across_reverse_populations(width, offset):
    encoder, native = _encoder(), _encoder()
    _prepare(encoder)
    pointers = {name: value.data_ptr() for name, value in encoder.named_buffers()}
    originals = {name: value.clone() for name, value in encoder.named_buffers()}
    for batch in (7, 3, 1, 2, 7):
        caches = [
            StreamingCaches(
                n_layers=2, batch=batch, d_model=32, left_context=8, conv_kernel=5, device=torch.device("cpu")
            )
            for _ in range(2)
        ]
        for step in range(3):
            mel = torch.randn(batch, 16, 25)
            lengths = torch.full((batch,), width, dtype=torch.long)
            lengths[-1] = 0 if step == 1 else 1
            args = dict(out_offsets=torch.full((batch,), offset), out_lengths=lengths, out_width=width)
            actual = stream_step(encoder, mel, caches[0], **args)
            expected = stream_step(native, mel, caches[1], **args)
            assert torch.equal(actual, expected)
            for field in ("channel", "time", "valid"):
                assert torch.equal(getattr(caches[0], field), getattr(caches[1], field))
    assert pointers == {name: value.data_ptr() for name, value in encoder.named_buffers()}
    assert all(torch.equal(originals[name], value) for name, value in encoder.named_buffers())


@pytest.mark.parametrize("mutation", ["train", "dtype", "load"])
@torch.inference_mode()
def test_rhs_model_mutations_invalidate_all_derived_buffers(mutation):
    encoder = _encoder()
    _prepare(encoder)
    if mutation == "train":
        encoder.train()
    elif mutation == "dtype":
        encoder.to(torch.float64)
    else:
        encoder.load_state_dict(encoder.state_dict())
    assert encoder._stream_relative_rhs_maximum_population == 0
    assert encoder._stream_relative_position_lengths == ()
    assert not any("_stream_relative" in name for name, _ in encoder.named_buffers())


@torch.inference_mode()
def test_rhs_rejects_changed_geometry_population_dtype_and_autocast():
    encoder = _encoder()
    _prepare(encoder)
    with pytest.raises(ValueError, match="population"):
        _prepare(encoder, maximum=8)
    with pytest.raises(ValueError, match="geometry"):
        _prepare(encoder, widths=(2,))
    for reference, width in (
        (torch.empty(8, 2, 32), 2),
        (torch.empty(2, 3, 32), 3),
        (torch.empty(2, 2, 32, dtype=torch.float64), 2),
    ):
        with pytest.raises(ValueError):
            encoder.stream_relative_position_rhs(out_width=width, cache_len=8, reference=reference)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        with pytest.raises(ValueError, match="autocast"):
            _prepare(_encoder())
        with pytest.raises(ValueError, match="autocast"):
            encoder.stream_relative_position_rhs(out_width=2, cache_len=8, reference=torch.empty(2, 2, 32))


@torch.inference_mode()
def test_rhs_partial_allocation_failure_clears_projections_and_rhs(monkeypatch):
    encoder = _encoder()
    original = torch.Tensor.contiguous
    calls = 0

    def fail_second(tensor, *args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise torch.OutOfMemoryError("injected RHS allocation failure")
        return original(tensor, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "contiguous", fail_second)
    with pytest.raises(torch.OutOfMemoryError, match="injected"):
        _prepare(encoder)
    assert encoder._stream_relative_position_lengths == ()
    assert encoder._stream_relative_rhs_maximum_population == 0
    assert not any("_stream_relative" in name for name, _ in encoder.named_buffers())


def _execution(*, enabled=False, arm="eager-graphed", maximum=7):
    encoder = _encoder()
    lid = PromptConditioner(enc_hidden=32, num_prompts=1).eval()
    core = SimpleNamespace(encoder=encoder, lid=lid, policy=SimpleNamespace(dtype_for=lambda _: torch.float32))
    config = SimpleNamespace(encoder_execution_arm=arm, encoder_relative_rhs_preparation=enabled, att_context_left=8)
    return build_encoder_execution(
        core, config, maximum_population=maximum, warmup_geometries=(1,), vllm_config=object(), graph_runtime=object()
    )


@torch.inference_mode()
def test_rhs_startup_opt_in_accounting_snapshot_and_discard():
    execution = _execution(enabled=True)
    execution._materialize_preprofile_state(out_width=execution._geometry_shapes[1].out_width)
    encoder = execution._core.encoder
    buffers = [tensor for name, tensor in encoder.named_buffers() if "_stream_relative" in name]
    receipt = execution.ready_receipt()["relative_position_rhs"]
    assert receipt["enabled"]
    assert receipt["maximum_population"] == 7
    # Geometry1 is a two-frame logical cadence but four padded output frames.
    assert execution._geometry_shapes[1].out_width == 4
    assert receipt["window_lengths"] == [12]
    assert receipt["projection_bytes"] == 2 * 23 * 32 * 4
    assert receipt["rhs_bytes"] == 7 * receipt["projection_bytes"]
    assert receipt["resident_bytes"] == sum(t.numel() * t.element_size() for t in buffers)
    assert receipt["expected_bytes"] == receipt["resident_bytes"]
    assert any("_stream_relative_rhs" in name for name, _ in execution._model_state.buffers)
    execution._sealed = True
    execution._assert_model_state()
    encoder.to(torch.float64)
    with pytest.raises(ValueError, match="changed|prepared"):
        execution._assert_model_state()
    execution._discard()
    assert execution.ready_receipt()["relative_position_rhs"]["resident_bytes"] == 0
    assert not any("_stream_relative" in name for name, _ in encoder.named_buffers())


@torch.inference_mode()
def test_rhs_default_off_keeps_native_projection_guard():
    execution = _execution()
    execution._materialize_preprofile_state(out_width=2)
    assert "relative_position_rhs" not in execution.ready_receipt()
    _prepare(execution._core.encoder)
    with pytest.raises(ValueError, match="prepared.*projection"):
        execution._assert_native_projection_state()


@pytest.mark.parametrize("arm", ["eager", "compiled-static", "dense-graphed"])
def test_rhs_option_rejects_other_execution_arms(arm):
    with pytest.raises(ValueError, match="requires eager-graphed"):
        _execution(enabled=True, arm=arm)


def test_rhs_option_rejects_nonboolean():
    with pytest.raises(ValueError, match="must be boolean"):
        _execution(enabled="true")


@torch.inference_mode()
def test_rhs_profile_and_fake_graph_inventory_share_frozen_buffers(monkeypatch):
    from test_encoder_execution import _graph_runtime

    from vllm_omni.model_executor.models.nemotron_asr.advance import _GatheredCaches
    from vllm_omni.model_executor.models.nemotron_asr.encoder_execution import execute_encoder_transition

    execution = _execution(enabled=True, maximum=3)
    baseline = _execution()._core  # Independent unprepared control, created before startup.
    execution._graph_runtime = _graph_runtime()
    shape = execution._geometry_shapes[1]

    def args(batch):
        state = SimpleNamespace(
            channel=[torch.zeros(batch, 8, 32) for _ in range(2)],
            time=[torch.zeros(batch, 32, 4) for _ in range(2)],
            window_valid=[torch.zeros(batch, 1, dtype=torch.int32) for _ in range(2)],
        )
        return (
            torch.full((batch, 16, shape.mel_width), 0.3),
            _GatheredCaches(state),
            torch.zeros(batch, dtype=torch.long),
            torch.full((batch,), shape.out_width, dtype=torch.long),
            shape.out_width,
            torch.zeros(batch, dtype=torch.long),
        )

    execution.profile_cell(geometry=1, population=3, invoke=lambda: execution.transition(*args(3)))
    buffers = {name: value for name, value in execution._core.encoder.named_buffers()}

    def forbidden(*_args, **_kwargs):
        pytest.fail("startup-prepared projections must never be recomputed")

    for layer in execution._core.encoder.layers:
        monkeypatch.setattr(layer.self_attn.linear_pos, "forward", forbidden)
    execution.warmup_domain(
        expected_cells=((1, 1), (1, 2), (1, 3)), invoke=lambda _, batch: execution.transition(*args(batch))
    )
    assert execution.ready
    execution.profile_ready_cell(geometry=1, population=3, invoke=lambda: execution.transition(*args(3)))
    for batch in (3, 1, 2, 3):
        actual_args, expected_args = args(batch), args(batch)
        actual = execution.transition(*actual_args)
        expected = execute_encoder_transition(baseline, *expected_args)
        assert all(torch.equal(a, b) for a, b in zip(actual, expected, strict=True))
        for a_family, b_family in zip(actual_args[1].graph_storage(), expected_args[1].graph_storage(), strict=True):
            assert all(torch.equal(a, b) for a, b in zip(a_family, b_family, strict=True))
    assert all(value is buffers[name] for name, value in execution._core.encoder.named_buffers())
    assert baseline.encoder._stream_relative_position_lengths == ()
    with torch.autocast("cpu", dtype=torch.bfloat16):
        with pytest.raises(ValueError, match="autocast"):
            execution.transition(*args(1))
    execution._core.encoder.train()
    with pytest.raises(ValueError, match="prepared"):
        execution.transition(*args(1))


@torch.inference_mode()
def test_rhs_preparation_failure_discards_execution_authority(monkeypatch):
    execution = _execution(enabled=True)

    def fail(*_args, **_kwargs):
        raise torch.OutOfMemoryError("injected startup allocation failure")

    monkeypatch.setattr(execution._core.encoder, "prepare_stream_relative_position_rhs", fail)
    with pytest.raises(torch.OutOfMemoryError, match="injected"):
        execution._materialize_preprofile_state(out_width=4)
    assert execution._failed
    assert execution._model_state is None
    assert execution.ready_receipt()["relative_position_rhs"]["resident_bytes"] == 0
    assert execution.ready_receipt()["relative_position_rhs"]["expected_bytes"] > 0
    with pytest.raises(ValueError, match="failed"):
        execution._raise_if_failed()


def test_rhs_config_default_and_opt_in_serialize_distinctly():
    from vllm_omni.model_executor.models.nemotron_asr.configuration_nemotron_asr import NemotronASRConfig

    default = NemotronASRConfig().to_dict()
    enabled = NemotronASRConfig(encoder_relative_rhs_preparation=True).to_dict()
    assert default["encoder_relative_rhs_preparation"] is False
    assert enabled["encoder_relative_rhs_preparation"] is True


@torch.inference_mode()
def test_rhs_single_population_has_no_expanded_storage():
    execution = _execution(enabled=True, maximum=1)
    execution._materialize_preprofile_state(out_width=4)
    receipt = execution.ready_receipt()["relative_position_rhs"]
    assert receipt["rhs_bytes"] == 0
    assert receipt["resident_bytes"] == receipt["projection_bytes"]
    assert not any("_stream_relative_rhs" in name for name, _ in execution._core.encoder.named_buffers())


@torch.inference_mode()
def test_rhs_prepared_generation_cannot_be_replaced_after_seal():
    execution = _execution(enabled=True)
    execution._materialize_preprofile_state(out_width=4)
    encoder = execution._core.encoder
    encoder.invalidate_stream_relative_position_projections()
    _prepare(encoder, widths=(4,))
    with pytest.raises(ValueError, match="changed"):
        execution._assert_native_projection_state()


@torch.inference_mode()
def test_rhs_unsupported_precision_is_rejected_before_preparation():
    execution = _execution(enabled=True)
    execution._core.policy.dtype_for = lambda _: torch.bfloat16
    with pytest.raises(ValueError, match="FP32"):
        execution._materialize_preprofile_state(out_width=4)
    assert execution._model_state is None
    assert execution._core.encoder._stream_relative_position_lengths == ()


def _mixed_rhs_inventory(*, maximum=128, populations=(1, 31, 63)):
    from test_advance_model_rows_local import CARRIER_HIDDEN, _tiny_core
    from test_advance_session_local import _fresh_state
    from test_encoder_execution import _graph_runtime

    from vllm_omni.model_executor.models.nemotron_asr.advance import _GatheredCaches
    from vllm_omni.model_executor.models.nemotron_asr.chunk_bucket_graph import ExactChunkGraphBinding
    from vllm_omni.model_executor.models.nemotron_asr.configuration_nemotron_asr import NemotronASRConfig
    from vllm_omni.model_executor.models.nemotron_asr.decode_graph import DenseGraphBinding
    from vllm_omni.model_executor.models.nemotron_asr.rnnt import decode_dense_masked_frames

    core, baseline = _tiny_core(), _tiny_core()
    core.policy = SimpleNamespace(dtype_for=lambda _: torch.float32)
    config = NemotronASRConfig(
        num_asr_labels=12,
        hidden_size=CARRIER_HIDDEN,
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
        encoder_relative_rhs_preparation=True,
    )
    execution = build_encoder_execution(
        core,
        config,
        maximum_population=maximum,
        warmup_geometries=(1,),
        vllm_config=object(),
        graph_runtime=_graph_runtime(),
    )

    def decoder(value):
        return DenseGraphBinding(
            decode_fn=decode_dense_masked_frames,
            predictor=value.predictor,
            joint=value.joint,
            vllm_config=object(),
            frame_widths=(None, 2),
            tiers=(1, 32, 64),
            encoder_hidden=32,
            predictor_layers=2,
            predictor_hidden=16,
            blank_id=12,
            runtime=_graph_runtime(),
        )

    decode, reference_decode = decoder(core), decoder(baseline)
    binding = ExactChunkGraphBinding(core, config, execution, decode, list(populations))
    shape = execution._geometry_shapes[1]

    def invoke(_geometry, batch):
        return execution.transition(
            torch.full((batch, 16, shape.mel_width), 0.3),
            _GatheredCaches(_fresh_state(batch)),
            torch.zeros(batch, dtype=torch.long),
            torch.full((batch,), shape.out_width, dtype=torch.long),
            shape.out_width,
            torch.zeros(batch, dtype=torch.long),
        )

    execution.profile_cell(geometry=1, population=maximum, invoke=lambda: invoke(1, maximum))
    before = {name: tensor for name, tensor in core.encoder.named_buffers()}
    execution.warmup_domain(expected_cells=tuple((1, batch) for batch in range(1, maximum + 1)), invoke=invoke)
    assert not execution.ready  # Reserved CHUNK cells must actually capture.
    decode.warmup(torch.device("cpu"), torch.float32)
    reference_decode.warmup(torch.device("cpu"), torch.float32)
    binding.warmup(torch.device("cpu"))
    assert binding.ready
    assert all(before[name] is tensor for name, tensor in core.encoder.named_buffers())
    return core, baseline, execution, binding, decode, reference_decode


@torch.inference_mode()
def test_rhs_actual_mixed_startup_captures_all_125_native_and_three_chunk_cells():
    from test_advance_model_rows_local import _envelope
    from test_advance_session_local import _fresh_state

    from vllm_omni.model_executor.models.nemotron_asr.advance import advance_chunk_bucket
    from vllm_omni.model_executor.models.nemotron_asr.chunk_bucket_graph import (
        _outputs,
        _state_tensors,
        capture_chunk_bucket,
    )
    from vllm_omni.model_executor.models.nemotron_asr.manifests import SESSION_LIMITS

    core, baseline, execution, binding, decode, reference_decode = _mixed_rhs_inventory()
    assert len(execution._graph_entries) == 125
    assert set(execution._chunk_graph_entries) == {(1, 1), (1, 31), (1, 63)}
    assert baseline.encoder._stream_relative_position_lengths == ()
    assert binding.receipt()["relative_position_rhs"] == execution.ready_receipt()["relative_position_rhs"]
    held: list[tuple[Any, tuple[torch.Tensor, ...]]] = []
    for batch in (63, 31, 1, 63):
        state, expected_state = _fresh_state(batch), _fresh_state(batch)
        env = torch.stack([_envelope(torch.full((2560,), 0.03), final=False, seq=0, geometry=1)] * batch)
        tier = decode.execution_tier(batch)
        kwargs = dict(
            geometry=1,
            admitted_prompt=torch.zeros(batch, dtype=torch.long),
            incoming_status=torch.zeros(batch, dtype=torch.int32),
            queue_capacity=SESSION_LIMITS["queue_capacity"],
        )
        decoder = decode.decode_fn(geometry=1, tier=tier)
        transition = binding.resolve(
            geometry=1, population=batch, decode_fn=decoder, encoder_transition=execution.transition, capture=False
        )
        expected = advance_chunk_bucket(
            baseline,
            env,
            expected_state,
            **kwargs,
            decode_fn=reference_decode.uncaptured_decode_fn(geometry=1, tier=tier),
        )
        actual = transition(core, env, state, **kwargs, decode_fn=decoder, encoder_transition=execution.transition)
        for a, b in zip(
            (*_outputs(actual), *_state_tensors(state)),
            (*_outputs(expected), *_state_tensors(expected_state)),
            strict=True,
        ):
            assert torch.equal(a, b)
        for output, original in held:
            assert all(torch.equal(a, b) for a, b in zip(_outputs(output), original, strict=True))
        held.append((actual, tuple(t.clone() for t in _outputs(actual))))
        with pytest.raises(ValueError, match="unprepared native"):
            capture_chunk_bucket(
                core, env, state, **kwargs, vllm_config=object(), capture_decode_fn=decoder, decoder_tier=tier
            )
        with pytest.raises(ValueError, match="sealed encoder authority"):
            capture_chunk_bucket(
                core,
                env,
                state,
                **kwargs,
                vllm_config=object(),
                capture_decode_fn=decoder,
                decoder_tier=tier,
                prepared_encoder_execution=execution,
                admitted_encoder_transition=None,
            )


@pytest.mark.parametrize("mutation", ["train", "dtype", "load", "regenerate"])
@torch.inference_mode()
def test_rhs_chunk_supported_mutation_rejects_before_gather_or_replay_stage(monkeypatch, mutation):
    from test_advance_model_rows_local import (
        CARRIER_HIDDEN,
        PARK_ID,
        PLACEHOLDER_ID,
        _envelope,
        _fixed_resolver,
        _fresh_pools,
        _plan,
    )
    from test_advance_session_local import _fresh_state

    from vllm_omni.model_executor.models.nemotron_asr import advance
    from vllm_omni.model_executor.models.nemotron_asr.chunk_bucket_graph import _state_tensors

    core, _, execution, binding, decode, _ = _mixed_rhs_inventory(maximum=2, populations=(1,))
    decoder = decode.decode_fn(geometry=1, tier=1)
    transition = binding.resolve(
        geometry=1, population=1, decode_fn=decoder, encoder_transition=execution.transition, capture=False
    )
    env = _envelope(torch.full((2560,), 0.03), final=False, seq=0, geometry=1).unsqueeze(0)
    state = _fresh_state(1)
    before = tuple(t.clone() for t in _state_tensors(state))
    if mutation == "train":
        core.encoder.train()
    elif mutation == "dtype":
        core.encoder.to(torch.float64)
    elif mutation == "load":
        core.encoder.load_state_dict(core.encoder.state_dict())
    else:
        core.encoder.invalidate_stream_relative_position_projections()
        _prepare(core.encoder, maximum=2, widths=(4,))
    with pytest.raises(ValueError, match="prepared"):
        transition(
            core,
            env,
            state,
            geometry=1,
            admitted_prompt=torch.zeros(1, dtype=torch.long),
            incoming_status=torch.zeros(1, dtype=torch.int32),
            queue_capacity=48,
            decode_fn=decoder,
            encoder_transition=execution.transition,
        )
    assert all(torch.equal(a, b) for a, b in zip(_state_tensors(state), before, strict=True))

    def forbidden(*_args, **_kwargs):
        pytest.fail("invalid prepared CHUNK binding must reject before resident gather")

    monkeypatch.setattr(advance, "_gather_initialized_rows", forbidden)
    with pytest.raises(ValueError, match="prepared"):
        advance.advance_model_rows(
            core,
            torch.full((1,), PLACEHOLDER_ID, dtype=torch.long),
            env,
            _plan(prefills=[1], num_pool_blocks=2, geometries=[1]),
            **_fresh_pools(2),
            adapter=advance.make_mrv1_adapter(hidden_size=CARRIER_HIDDEN, park_id=PARK_ID, blank_id=core.blank_id),
            decode_resolver=_fixed_resolver(decoder),
            placeholder_id=PLACEHOLDER_ID,
            park_id=PARK_ID,
            encoder_transition=execution.transition,
            bucket_transition=binding,
        )
