# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Selection exactness and storage ownership, independent of decoder arithmetic."""

import inspect

import pytest
import torch

from vllm_omni.model_executor.models.nemotron_asr.rnnt_selection import (
    compile_dense_selections,
    select_committed,
    select_predicted,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _values(shape: tuple[int, ...], offset: int) -> torch.Tensor:
    # Include both zeros, infinities, subnormals and distinct NaN payloads.
    bits = torch.tensor(
        [0, -2147483648, 1, -2147483647, 2139095040, -8388608, 2143289345, -4194302, 1065353216],
        dtype=torch.int32,
    )
    count = 1
    for dimension in shape:
        count *= dimension
    return bits[(torch.arange(count) + offset) % len(bits)].reshape(shape).view(torch.float32)


@pytest.mark.parametrize("batch", [1, 2, 4, 8, 16, 32, 64, 128])
@pytest.mark.parametrize("mask_kind", ["none", "all", "first", "last", "alternating", "mixed"])
def test_selections_preserve_bits_inputs_and_independent_storage(batch: int, mask_kind: str) -> None:
    rows = torch.arange(batch)
    masks = {
        "none": rows < 0,
        "all": rows >= 0,
        "first": rows == 0,
        "last": rows == batch - 1,
        "alternating": rows % 2 == 0,
        "mixed": rows % 3 == 1,
    }
    mask = masks[mask_kind].unsqueeze(-1)
    gate = mask.unsqueeze(0)
    states = [_values((2, batch, 5), offset) for offset in range(4)]
    outputs = [_values((batch, 5), offset) for offset in (3, 6)]
    inputs = [gate, mask, *states, *outputs]
    snapshots = [value.clone() for value in inputs]
    committed = select_committed(gate, *states)
    predicted = select_predicted(mask, gate, *outputs, *states)
    pairs = [(states[0], states[1]), (states[2], states[3])]
    predicted_pairs = [(outputs[0], outputs[1]), *pairs]
    for actual, (yes, no) in zip((*committed, *predicted), (*pairs, *predicted_pairs)):
        selector = (mask if actual.ndim == 2 else gate).expand_as(actual)
        expected_bits = no.view(torch.int32).clone()
        expected_bits[selector] = yes.view(torch.int32)[selector]
        assert torch.equal(actual.view(torch.int32), expected_bits)
        assert actual.dtype == yes.dtype
        assert actual.shape == yes.shape
        assert actual.stride() == yes.stride()
    all_outputs = (*committed, *predicted)
    pointers = [value.untyped_storage().data_ptr() for value in all_outputs]
    assert len(set(pointers)) == 5
    assert not set(pointers).intersection(value.untyped_storage().data_ptr() for value in inputs)
    for value, snapshot in zip(inputs, snapshots):
        assert torch.equal(value.view(torch.uint8), snapshot.view(torch.uint8))


def test_selections_do_not_alias_even_when_source_branches_alias() -> None:
    state = _values((2, 4, 5), 0)
    out = state[-1]
    gate = torch.ones((1, 4, 1), dtype=torch.bool)
    results = (
        *select_committed(gate, state, state, state, state),
        *select_predicted(gate[0], gate, out, out, state, state, state, state),
    )
    assert len({value.untyped_storage().data_ptr() for value in results}) == 5
    assert all(value.untyped_storage().data_ptr() != state.untyped_storage().data_ptr() for value in results)


@torch.inference_mode()
def test_compiled_regions_isolate_eight_tiers_and_keep_default_limit(monkeypatch) -> None:
    """Dynamo ownership only; the real CUDA test exercises Inductor and capture."""
    if "isolate_recompiles" not in inspect.signature(torch.compile).parameters:
        pytest.skip("requires Torch with public isolate_recompiles support")
    from torch._dynamo import config
    from torch._dynamo.exc import FailOnRecompileLimitHit

    real_compile = torch.compile
    graphs = []

    def backend(graph, _inputs):
        graphs.append(graph)
        return graph.forward

    def compile_counted(fn, **kwargs):
        assert kwargs.pop("options") == {"triton.cudagraphs": False}
        return real_compile(fn, backend=backend, **kwargs)

    monkeypatch.setattr(torch, "compile", compile_counted)
    assert config.recompile_limit == 8
    accumulated_limit = config.accumulated_recompile_limit
    pairs = [compile_dense_selections(), compile_dense_selections()]
    arguments = []
    for width, pair in zip((5, 7), pairs):
        cells = []
        for batch in (1, 2, 4, 8, 16, 32, 64, 128):
            state = torch.ones(2, batch, width)
            out = torch.ones(batch, width)
            gate = torch.ones(1, batch, 1, dtype=torch.bool)
            before = (gate, state, state, state, state)
            after = (gate[0], gate, out, out, state, state, state, state)
            pair.committed(*before)
            pair.predicted(*after)
            cells.append((before, after))
        arguments.append(cells)
    assert len(graphs) == 32
    with config.patch(error_on_recompile=True):
        for pair, cells in reversed(list(zip(pairs, arguments))):
            for before, after in reversed(cells):
                pair.committed(*before)
                pair.predicted(*after)
    assert len(graphs) == 32
    # A ninth specialization must fail fullgraph, never run an eager fallback.
    state = torch.ones(2, 256, 5)
    gate = torch.ones(1, 256, 1, dtype=torch.bool)
    with pytest.raises(FailOnRecompileLimitHit, match="recompile_limit"):
        pairs[0].committed(gate, state, state, state, state)
    assert len(graphs) == 32
    assert config.recompile_limit == 8
    assert config.accumulated_recompile_limit == accumulated_limit
