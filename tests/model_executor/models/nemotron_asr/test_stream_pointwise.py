# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Pointwise lowering retains module customization and streaming state semantics."""

import copy
from unittest.mock import patch

import pytest
import torch
from torch import nn

from vllm_omni.model_executor.models.nemotron_asr.encoder import (
    ConformerLayer,
    _stream_conv,
    _stream_pointwise_linear,
)

pytestmark = [pytest.mark.core_model]


def layer_inputs(device="cpu", dtype=torch.float32):
    torch.manual_seed(61)
    layer = ConformerLayer(d_model=16, d_ff=32, n_heads=4, conv_kernel=5, conv_norm_type="layer_norm")
    layer = layer.to(device=device, dtype=dtype).eval()
    # Noncontiguous caller input, mixed logical lengths, and independent cache.
    x = torch.randn(3, 16, 4, device=device, dtype=dtype).transpose(1, 2)
    cache = torch.randn(3, 16, 4, device=device, dtype=dtype)
    lengths = torch.tensor([0, 1, 4], device=device)
    return layer, x, cache, lengths


@pytest.mark.cpu
def test_cpu_dtype_and_grad_contexts_keep_module_calls():
    for dtype in (torch.float32, torch.float64):
        layer, x, cache, lengths = layer_inputs(dtype=dtype)
        seen = []
        handles = [
            m.register_forward_hook(lambda *args: seen.append(True))
            for m in (layer.conv.pointwise_conv1, layer.conv.pointwise_conv2)
        ]
        x.requires_grad_()
        output, updated = _stream_conv(layer, x, cache, new_lengths=lengths)
        output.sum().backward()
        assert len(seen) == 2 and x.grad is not None
        assert torch.equal(updated[0], cache[0])
        with torch.no_grad():
            assert not _stream_pointwise_linear(layer.conv.pointwise_conv1, x)
        for handle in handles:
            handle.remove()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA pointwise path")
@pytest.mark.parametrize(
    "custom",
    [
        "none",
        "hook",
        "pre_hook",
        "forward",
        "subclass",
        "parametrized",
        "training",
        "grad",
        "double",
        "autocast",
        "global_hook",
    ],
)
@pytest.mark.parametrize("site", ["pointwise_conv1", "pointwise_conv2"])
def test_cuda_stream_pointwise_matches_original_calls(custom, site):
    layer, x, cache, lengths = layer_inputs("cuda")
    original = copy.deepcopy(layer)
    handles, seen = [], []
    pointwise = getattr(layer.conv, site)
    if custom == "hook":
        handles.append(pointwise.register_forward_hook(lambda module, args, output: output + 0.25))
        handles.append(getattr(original.conv, site).register_forward_hook(lambda module, args, output: output + 0.25))
    elif custom == "pre_hook":
        for model in (layer, original):
            handles.append(getattr(model.conv, site).register_forward_pre_hook(lambda module, args: (args[0] + 0.25,)))
    elif custom == "forward":
        for model in (layer, original):
            module = getattr(model.conv, site)
            forward = module.forward
            module.forward = lambda value, forward=forward: forward(value) + 0.25
    elif custom == "subclass":

        class OffsetConv(nn.Conv1d):
            def forward(self, value):
                return super().forward(value) + 0.25

        for model in (layer, original):
            getattr(model.conv, site).__class__ = OffsetConv
    elif custom == "parametrized":

        class Scale(nn.Module):
            def forward(self, value):
                return value * 0.9

        for model in (layer, original):
            nn.utils.parametrize.register_parametrization(getattr(model.conv, site), "weight", Scale())
    elif custom == "training":
        layer.train()
        original.train()
    elif custom == "double":
        layer.double()
        original.double()
        x, cache = x.double(), cache.double()
    elif custom == "global_hook":
        handles.append(nn.modules.module.register_module_forward_hook(lambda *args: seen.append(args[0])))
    # A harmless local hook forces the independent copy through original Conv1d.
    for module in (original.conv.pointwise_conv1, original.conv.pointwise_conv2):
        handles.append(module.register_forward_hook(lambda *args: None))
    params = {name: (id(value), value.data_ptr(), value.clone()) for name, value in layer.named_parameters()}
    try:
        with torch.set_grad_enabled(custom == "grad"), torch.autocast("cuda", enabled=custom == "autocast"):
            assert _stream_pointwise_linear(pointwise, x) is (custom == "none")
            actual, updated = _stream_conv(layer, x, cache, new_lengths=lengths)
            expected, expected_cache = _stream_conv(original, x, cache, new_lengths=lengths)
        torch.testing.assert_close(actual, expected, atol=5e-5, rtol=5e-5)
        torch.testing.assert_close(updated, expected_cache, atol=5e-5, rtol=5e-5)
        assert actual.shape == expected.shape
        assert actual.untyped_storage().data_ptr() != x.untyped_storage().data_ptr()
        if custom == "autocast":
            # The original path casts cache through the autocast compute dtype;
            # fallback must preserve that result, including its rounding.
            assert torch.equal(updated, expected_cache)
        else:
            assert torch.equal(updated[0], cache[0])
        for name, value in layer.named_parameters():
            identity, pointer, before = params[name]
            assert (id(value), value.data_ptr()) == (identity, pointer)
            assert torch.equal(value, before)
        if custom == "global_hook":
            assert pointwise in seen
    finally:
        for handle in handles:
            handle.remove()


@pytest.mark.cpu
@pytest.mark.parametrize("noncontiguous", [False, True])
def test_pointwise_math_and_cache_match_module_reference_on_cpu(noncontiguous):
    # Exercise lowering arithmetic independently of CUDA admission. This is not
    # evidence that the actual CUDA kernels or graph capture meet the gates.
    layer, x, cache, lengths = layer_inputs()
    if not noncontiguous:
        x = x.contiguous()
    original = copy.deepcopy(layer)
    original_cache = cache.clone()
    params = {name: (id(value), value.data_ptr()) for name, value in layer.named_parameters()}
    for _ in range(3):
        with torch.no_grad():
            expected, original_cache = _stream_conv(original, x, original_cache, new_lengths=lengths)
            with patch(_stream_conv.__module__ + "._stream_pointwise_linear", return_value=True):
                actual, cache = _stream_conv(layer, x, cache, new_lengths=lengths)
        torch.testing.assert_close(actual, expected, atol=5e-5, rtol=5e-5)
        torch.testing.assert_close(cache, original_cache, atol=5e-5, rtol=5e-5)
        x = x + 0.1
    assert {name: (id(value), value.data_ptr()) for name, value in layer.named_parameters()} == params


@pytest.mark.cpu
def test_compilation_retains_original_calls():
    layer, x, _, _ = layer_inputs()
    with torch.no_grad(), patch("torch.compiler.is_compiling", return_value=True):
        assert not _stream_pointwise_linear(layer.conv.pointwise_conv1, x)
