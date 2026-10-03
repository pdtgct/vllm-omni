# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Target CUDA startup test; synthetic networks, no serving performance claim."""

from contextlib import contextmanager
from dataclasses import replace

import pytest
import torch

from vllm_omni.model_executor.models.nemotron_asr.decode_graph import DenseGraphBinding, platform_graph_runtime
from vllm_omni.model_executor.models.nemotron_asr.rnnt import DecodeState, Joint, Predictor, decode_dense_masked_frames
from vllm_omni.model_executor.models.nemotron_asr.rnnt_selection import select_committed, select_predicted

pytestmark = [
    pytest.mark.core_model,
    pytest.mark.cuda,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA and Torch 2.13"),
]


@torch.inference_mode()
def test_two_bindings_warm_capture_and_revisit_all_selection_tiers(monkeypatch) -> None:
    from torch._dynamo import config
    from torch._inductor.compile_fx import compile_fx
    from vllm.config import VllmConfig

    device = torch.device("cuda", 0)
    real_compile = torch.compile
    compiled_helpers = []
    graphs = []
    sealed = False

    def backend(graph, inputs, **kwargs):
        assert not sealed
        assert not torch.cuda.is_current_stream_capturing()
        operations = [node for node in graph.graph.nodes if node.op not in ("placeholder", "output")]
        assert len(operations) in (2, 3)
        assert all(node.op == "call_function" and node.target is torch.where for node in operations)
        graphs.append(graph)
        return compile_fx(graph, inputs, config_patches=kwargs["options"])

    def compile_counted(fn, **kwargs):
        assert kwargs["isolate_recompiles"]
        compiled = real_compile(fn, backend=backend, **kwargs)
        compiled_helpers.append(compiled)
        return compiled

    monkeypatch.setattr(torch, "compile", compile_counted)
    assert config.recompile_limit == 8
    accumulated_limit = config.accumulated_recompile_limit

    @contextmanager
    def capture_stream(_device):
        # Same stream adapter as the existing real-CUDA encoder binding test.
        current = torch.cuda.current_stream(device)
        stream = torch.cuda.Stream(device=device)
        stream.wait_stream(current)
        with torch.cuda.stream(stream):
            yield
        current.wait_stream(stream)

    runtime = replace(platform_graph_runtime(), capture_context=capture_stream)
    predictor = Predictor(vocab_size=13087, pred_hidden=640, pred_rnn_layers=2).to(device).eval()
    joint = Joint(enc_hidden=1024, pred_hidden=640, joint_hidden=640, vocab_size=13087).to(device).eval()
    # Force all ten attempts to emit, while retaining the real predictor math.
    joint.joint_net[1].weight.zero_()
    joint.joint_net[1].bias.fill_(-100)
    joint.joint_net[1].bias[0] = 0
    tiers = (1, 2, 4, 8, 16, 32, 64, 128)
    bindings = [
        DenseGraphBinding(
            decode_fn=decode_dense_masked_frames,
            predictor=predictor,
            joint=joint,
            vllm_config=VllmConfig(),
            frame_widths=(2,),
            tiers=tiers,
            encoder_hidden=1024,
            predictor_layers=2,
            predictor_hidden=640,
            blank_id=13087,
            runtime=runtime,
        )
        for _ in range(2)
    ]
    for index, binding in enumerate(bindings):
        binding.warmup(device, torch.float32)
        assert len(graphs) == 16 * (index + 1)
    assert len(compiled_helpers) == 4
    retained = []
    for binding in bindings:
        for entry in binding._entries.values():
            captured = entry.wrapper.cudagraph_wrapper.concrete_cudagraph_entries
            assert len(captured) == 1
            graph = next(iter(captured.values())).cudagraph
            assert isinstance(graph, torch.cuda.CUDAGraph)
            retained.append(graph)
    assert len({id(graph) for graph in retained}) == 16
    sealed = True
    with config.patch(error_on_recompile=True):
        # Exercise the compiled helpers with the decoder's distinct, contiguous
        # FP32 inputs and all mask classes, including non-finite payload bits.
        bits = torch.tensor(
            [0, -2147483648, 1, -2147483647, 2139095040, -8388608, 2143289345, -4194302],
            dtype=torch.int32,
            device=device,
        )
        for tier in reversed(tiers):
            values = [
                bits.roll(index).repeat(2 * tier * 640 // 8).view(torch.float32).reshape(2, tier, 640)
                for index in range(4)
            ]
            outputs = [
                bits.roll(index).repeat(tier * 640 // 8).view(torch.float32).reshape(tier, 640) for index in (4, 5)
            ]
            rows = torch.arange(tier, device=device)
            for emit in (rows < 0, rows >= 0, rows == 0, rows == tier - 1, rows % 2 == 0, rows % 3 == 1):
                before = (emit.view(1, tier, 1), *values)
                after = (emit.unsqueeze(-1), before[0], *outputs, *values)
                for pre, post in zip(compiled_helpers[::2], compiled_helpers[1::2]):
                    saved = [value.clone() for value in (*values, *outputs)]
                    result = (*pre(*before), *post(*after))
                    expected = (*select_committed(*before), *select_predicted(*after))
                    for left, right in zip(result, expected):
                        assert torch.equal(left.view(torch.int32), right.view(torch.int32))
                    pointers = {value.untyped_storage().data_ptr() for value in result}
                    assert len(pointers) == 5
                    assert not pointers.intersection(
                        value.untyped_storage().data_ptr() for value in (*values, *outputs)
                    )
                    for left, right in zip((*values, *outputs), saved):
                        assert torch.equal(left.view(torch.int32), right.view(torch.int32))
        for binding in reversed(bindings):
            for tier in reversed(tiers):
                frames = torch.randn(tier, 2, 1024, device=device)
                lengths = torch.arange(tier, device=device) % 3
                state = DecodeState(
                    h=torch.randn(2, tier, 640, device=device),
                    c=torch.randn(2, tier, 640, device=device),
                    last_label=torch.full((tier,), 13087, dtype=torch.int64, device=device),
                )
                expected_decode = decode_dense_masked_frames(frames, lengths, predictor, joint, state)
                state_before = (state.h.clone(), state.c.clone(), state.last_label.clone())
                decode = binding.decode_fn(geometry=0, tier=tier)
                actual = decode(frames, lengths, predictor, joint, state)
                expected_tuple = (
                    expected_decode.token_ids,
                    expected_decode.token_lengths,
                    expected_decode.state.h,
                    expected_decode.state.c,
                    expected_decode.state.last_label,
                    expected_decode.frame_emission_counts,
                    expected_decode.frame_final_labels,
                )
                entry = binding._entries[(0, tier)]
                for left, right in zip(expected_tuple, entry.output_tuple()):
                    assert torch.equal(left.view(torch.uint8), right.view(torch.uint8))
                for left, right in zip((state.h, state.c, state.last_label), state_before):
                    assert torch.equal(left, right)
                snapshot = actual.token_lengths.clone()
                decode(frames, torch.full_like(lengths, 2), predictor, joint, state)
                assert torch.equal(actual.token_lengths, torch.full_like(lengths, 20).to(torch.int32))
                assert torch.equal(snapshot, lengths.to(torch.int32) * 10)
                # CHUNK captures this same binding through the eager wrapper.
                uncaptured = binding.uncaptured_decode_fn(geometry=0, tier=tier)
                chunk = uncaptured(frames, lengths, predictor, joint, state)
                assert torch.equal(chunk.token_lengths, snapshot)
    assert len(graphs) == 32
    assert config.recompile_limit == 8
    assert config.accumulated_recompile_limit == accumulated_limit
