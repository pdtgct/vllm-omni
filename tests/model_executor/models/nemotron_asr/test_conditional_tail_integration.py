# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Binding ownership on CPU; actual IF, split and CHUNK capture on CUDA."""

import gc
import weakref
from contextlib import contextmanager
from dataclasses import replace

import pytest
import torch
from test_conditional_tail import _CountingPredictor, _equal, _outputs, _state, _ThresholdJoint
from test_decode_graph_binding import _runtime

from vllm_omni.model_executor.models.nemotron_asr.conditional_tail import ConditionalTailDecoder
from vllm_omni.model_executor.models.nemotron_asr.decode_graph import DenseGraphBinding, platform_graph_runtime
from vllm_omni.model_executor.models.nemotron_asr.rnnt import DecodeState, decode_dense_masked_frames

pytestmark = [pytest.mark.core_model]


def _binding(decoder, predictor, joint, *, runtime, frames=(1, 2), tiers=(1, 2, 4), hidden=1, config=None):
    return DenseGraphBinding(
        decode_fn=decoder,
        predictor=predictor,
        joint=joint,
        vllm_config=config,
        frame_widths=frames,
        tiers=tiers,
        encoder_hidden=hidden,
        predictor_layers=2,
        predictor_hidden=hidden,
        blank_id=predictor.blank_id,
        runtime=runtime,
    )


@pytest.mark.cpu
def test_binding_owns_each_key_and_forwards_scope_to_enclosing_chunk():
    class Owner:
        if_nodes = 0

    class Decoder:
        active = None
        prepared = None

        def prepare(self, device):
            self.prepared = device

        @contextmanager
        def capture_scope(self):
            self.active = owner = Owner()
            try:
                yield owner
            finally:
                self.active = None

        def __call__(self, *args):
            if self.active is not None:
                self.active.if_nodes = args[0].shape[1]
            return decode_dense_masked_frames(*args)

    decoder, predictor, joint = Decoder(), _CountingPredictor(), _ThresholdJoint()
    binding = _binding(decoder, predictor, joint, runtime=_runtime(), tiers=(1, 2))
    binding.warmup(torch.device("cpu"), torch.float32)
    assert decoder.prepared == torch.device("cpu")
    assert binding.conditional_capture_counts == {(0, 1): 1, (0, 2): 1, (1, 1): 2, (1, 2): 2}
    owner_ref = weakref.ref(binding._entries[(0, 1)].capture_resources)
    enclosing_decode = binding.uncaptured_decode_fn(geometry=0, tier=1)
    with enclosing_decode.capture_scope() as enclosing_owner:
        enclosing_decode(torch.zeros(1, 1, 1), torch.ones(1, dtype=torch.long), predictor, joint, _state(1))
        assert enclosing_owner.if_nodes == 1
        assert enclosing_owner is not owner_ref()
    del enclosing_decode, enclosing_owner
    binding._entries.pop((0, 1))
    binding._decode_fns.pop((0, 1))
    gc.collect()
    assert owner_ref() is None
    assert binding.conditional_capture_counts == {(0, 2): 1, (1, 1): 2, (1, 2): 2}


@pytest.mark.cpu
def test_capture_scope_failure_is_not_published_or_retried_as_dense():
    class Decoder:
        @contextmanager
        def capture_scope(self):
            raise RuntimeError("injected conditional capture failure")
            yield  # pragma: no cover

        def __call__(self, *args):
            return decode_dense_masked_frames(*args)

    binding = _binding(Decoder(), _CountingPredictor(), _ThresholdJoint(), runtime=_runtime(), tiers=(1,))
    with pytest.raises(RuntimeError, match="injected conditional capture failure"):
        binding.warmup(torch.device("cpu"), torch.float32)
    assert binding.captured_keys == ()
    assert binding.conditional_capture_counts == {}


def _cuda_runtime(device):
    @contextmanager
    def capture_stream(_device):
        # Existing real-platform test_encoder_execution capture fixture.
        parent = torch.cuda.current_stream(device)
        stream = torch.cuda.Stream(device=device)
        stream.wait_stream(parent)
        with torch.cuda.stream(stream):
            yield
        parent.wait_stream(stream)

    return replace(platform_graph_runtime(), capture_context=capture_stream)


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires real CUDA conditional nodes")
@torch.inference_mode()
def test_cuda_thresholds_alternating_replays_padding_recurrence_and_earlier_keys():
    from vllm.config import VllmConfig

    device = torch.device("cuda", torch.accelerator.current_device_index())
    predictor, joint = _CountingPredictor(), _ThresholdJoint()
    runtime = _cuda_runtime(device)
    common = dict(predictor=predictor, joint=joint, runtime=runtime, config=VllmConfig())
    control = _binding(decode_dense_masked_frames, **common)
    candidate = _binding(ConditionalTailDecoder(), **common)
    control.warmup(device, torch.float32)
    candidate.warmup(device, torch.float32)
    assert control.conditional_capture_counts == {}
    assert candidate.conditional_capture_counts == {(g, b): g + 1 for g in (0, 1) for b in (1, 2, 4)}
    # Visit early keys after all later captures, then reverse the order. Mixed
    # live3/tier4 and changing lengths exercise padding without merging batches.
    keys = list(candidate.captured_keys)
    for key in keys + list(reversed(keys)):
        geometry, tier = key
        frames = geometry + 1
        live = 3 if tier == 4 else tier
        decode_a = control.decode_fn(geometry=geometry, tier=tier)
        decode_b = candidate.decode_fn(geometry=geometry, tier=tier)
        state = _state(live)
        state = DecodeState(state.h.to(device), state.c.to(device), state.last_label.to(device))
        for emissions in (0, 3, 4, 5, 10, 0, 4):
            # Independent initial state and cumulative thresholds create exactly
            # the requested emissions in each valid frame, including recurrence.
            encoded = (state.h[0, :, 0, None] + torch.arange(1, frames + 1, device=device) * emissions).unsqueeze(-1)
            lengths = torch.full((live,), frames, dtype=torch.int64, device=device)
            if live > 1:
                lengths[-1] = 0
            before = tuple(t.clone() for t in (encoded, lengths, state.h, state.c, state.last_label))
            expected = decode_a(encoded, lengths, predictor, joint, state)
            expected_values = tuple(t.clone() for t in _outputs(expected))
            actual = decode_b(encoded, lengths, predictor, joint, state)
            for value, reference in zip(_outputs(actual), expected_values, strict=True):
                torch.testing.assert_close(value, reference, atol=0, rtol=0)
            assert actual.frame_emission_counts[0].tolist() == [emissions] * frames
            for value, saved in zip((encoded, lengths, state.h, state.c, state.last_label), before, strict=True):
                torch.testing.assert_close(value, saved, atol=0, rtol=0)
            state = DecodeState(actual.state.h.clone(), actual.state.c.clone(), actual.state.last_label.clone())
        # A skipped frame followed by an emitting frame must retain the prefix
        # predictor lookahead even when no tail wrote its merge buffers.
        if frames == 2:
            encoded = state.h[0, :, 0, None] + torch.tensor([0, 5], device=device)
            encoded = encoded.unsqueeze(-1)
            lengths = torch.full((live,), 2, dtype=torch.int64, device=device)
            expected = decode_a(encoded, lengths, predictor, joint, state)
            actual = decode_b(encoded, lengths, predictor, joint, state)
            _equal(actual, expected)
            assert actual.frame_emission_counts[0].tolist() == [0, 5]
    runtime.synchronize(device)


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires real CUDA conditional nodes")
@pytest.mark.parametrize("population,chunk", [(1, True), (31, True), (63, True), (2, False), (32, False), (64, False)])
@torch.inference_mode()
def test_cuda_split_and_accepted_chunk_exact_outputs_and_owned_capture(population, chunk):
    from dataclasses import fields

    from test_chunk_bucket_graph import _fixture
    from vllm.config import VllmConfig

    from vllm_omni.model_executor.models.nemotron_asr.advance import (
        ENV_FINAL_TAIL,
        ENV_VALID_SAMPLES,
        advance_chunk_bucket,
    )
    from vllm_omni.model_executor.models.nemotron_asr.chunk_bucket_graph import (
        _clone_state,
        _state_tensors,
        capture_chunk_bucket,
    )
    from vllm_omni.model_executor.models.nemotron_asr.chunk_bucket_graph import (
        _outputs as chunk_outputs,
    )

    device = torch.device("cuda", torch.accelerator.current_device_index())
    core, env, initial_state, args = _fixture(population)
    for module in vars(core).values():
        if isinstance(module, torch.nn.Module):
            module.to(device)
    core.joint.joint_net[-1].weight.zero_()
    core.joint.joint_net[-1].bias.zero_()
    core.joint.joint_net[-1].bias[0] = 1
    initial_state = replace(
        initial_state,
        **{
            f.name: [t.to(device) for t in value] if isinstance(value, list) else value.to(device)
            for f in fields(initial_state)
            for value in [getattr(initial_state, f.name)]
        },
    )
    env = env.to(device)
    args = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in args.items()}
    tier = next(t for t in (1, 2, 4, 8, 16, 32, 64) if t >= population)
    runtime, config = _cuda_runtime(device), VllmConfig()
    common = dict(
        predictor=core.predictor,
        joint=core.joint,
        vllm_config=config,
        frame_widths=(None, 2),
        tiers=(tier,),
        encoder_hidden=32,
        predictor_layers=2,
        predictor_hidden=16,
        blank_id=core.blank_id,
        runtime=runtime,
    )
    control = DenseGraphBinding(decode_fn=decode_dense_masked_frames, **common)
    candidate = DenseGraphBinding(decode_fn=ConditionalTailDecoder(), **common)
    for binding in (control, candidate):
        binding.warmup(device, torch.float32)
    assert control.conditional_capture_counts == {}
    assert candidate.conditional_capture_counts == {(1, tier): 2}
    decodes = [binding.decode_fn(geometry=1, tier=tier) for binding in (control, candidate)]
    transitions = [advance_chunk_bucket, advance_chunk_bucket]
    if chunk:
        transitions = [
            capture_chunk_bucket(
                core,
                env,
                initial_state,
                **args,
                vllm_config=config,
                runtime=runtime,
                capture_decode_fn=binding.uncaptured_decode_fn(geometry=1, tier=tier),
                admitted_decode_fn=decode,
                decoder_tier=tier,
            )
            for binding, decode in zip((control, candidate), decodes, strict=True)
        ]
        assert not hasattr(transitions[0], "capture_resources")
        assert transitions[1].capture_resources.if_nodes == 2
        assert transitions[1].capture_resources is not candidate._entries[(1, tier)].capture_resources
    for blank, final in ((True, False), (False, False), (True, True), (False, False)):
        core.joint.joint_net[-1].bias[core.blank_id] = 2 if blank else 0
        env[:, ENV_FINAL_TAIL] = int(final)
        env[:, ENV_VALID_SAMPLES] = 0 if final else 2560
        states = [_clone_state(initial_state), _clone_state(initial_state)]
        expected = transitions[0](core, env, states[0], **args, decode_fn=decodes[0])
        # Immutable control witness before any later binding/CHUNK replay.
        reference = tuple(t.clone() for t in (*chunk_outputs(expected), *_state_tensors(states[0])))
        actual = transitions[1](core, env, states[1], **args, decode_fn=decodes[1])
        for value, saved in zip((*chunk_outputs(actual), *_state_tensors(states[1])), reference, strict=True):
            torch.testing.assert_close(value, saved, atol=0, rtol=0)
    runtime.synchronize(device)
