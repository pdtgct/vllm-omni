# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Actual predicate-cache/graph-owner classes with CPU fake CUDA APIs.

These tests prove bounded ownership and cleanup ordering. They do not explain
the unreproduced GPU capture segfault or replace real CUDA validation.
"""

import gc
import importlib
import sys
import threading
import weakref
from concurrent.futures import ThreadPoolExecutor
from types import ModuleType, SimpleNamespace

import pytest
import torch
from test_conditional_tail import _CountingPredictor, _ThresholdJoint
from test_decode_graph_binding import _runtime

from vllm_omni.model_executor.models.nemotron_asr.conditional_tail import ConditionalCapture, ConditionalTailDecoder
from vllm_omni.model_executor.models.nemotron_asr.decode_graph import DenseGraphBinding

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.fixture
def cuda_api(monkeypatch):
    module = importlib.import_module("vllm_omni.model_executor.models.nemotron_asr.conditional_tail")
    state = SimpleNamespace(
        device=0,
        context_device=0,
        visible_devices=2,
        context=7,
        context_id=101,
        capturing=False,
        loads=0,
        unloads=[],
        programs=0,
        destroys=0,
        lookup_error=0,
        destroy_error=0,
        unload_error=0,
        compile_hook=lambda: None,
    )

    def load(_ptx):
        state.loads += 1
        return 0, state.loads

    def unload(handle):
        state.unloads.append((handle, state.capturing))
        return (state.unload_error,)

    def create(*_args):
        state.programs += 1
        return 0, state.programs

    def destroy(program):
        state.destroys += 1
        return (state.destroy_error,)

    def compile_program(*_args):
        state.compile_hook()
        return (0,)

    driver = SimpleNamespace(
        cuDeviceGetCount=lambda: (0, state.visible_devices),
        cuCtxGetDevice=lambda: (0, state.context_device),
        cuCtxGetCurrent=lambda: (0, state.context),
        cuCtxGetId=lambda context: (0, state.context_id),
        cuModuleLoadData=load,
        cuModuleUnload=unload,
        cuModuleGetFunction=lambda handle, name: (state.lookup_error, 42),
    )
    nvrtc = SimpleNamespace(
        nvrtcCreateProgram=create,
        nvrtcCompileProgram=compile_program,
        nvrtcGetPTXSize=lambda program: (0, 16),
        nvrtcGetPTX=lambda program, data: (0,),
        nvrtcDestroyProgram=destroy,
    )
    cuda, bindings = ModuleType("cuda"), ModuleType("cuda.bindings")
    bindings.__dict__.update(__version__="13.4.1", driver=driver, nvrtc=nvrtc, runtime=SimpleNamespace())
    setattr(cuda, "bindings", bindings)
    monkeypatch.setitem(sys.modules, "cuda", cuda)
    monkeypatch.setitem(sys.modules, "cuda.bindings", bindings)
    monkeypatch.setattr(module, "_MODULE_CACHE", {})
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: state.capturing)
    monkeypatch.setattr(torch.accelerator, "current_device_index", lambda: state.device)
    real_empty = torch.empty

    def cpu_empty(*args, **kwargs):
        if "device" in kwargs and torch.device(kwargs["device"]).type == "cuda":
            kwargs["device"] = "cpu"
        return real_empty(*args, **kwargs)

    monkeypatch.setattr(torch, "empty", cpu_empty)
    state.module = module
    return state


def _prepare():
    decoder = ConditionalTailDecoder()
    decoder.prepare(torch.device("cuda", 0))
    return decoder


@pytest.mark.parametrize("workers", [1, 4])
def test_actual_cache_loads_one_module_for_repeated_and_concurrent_prepares(cuda_api, workers):
    ready = threading.Barrier(workers + 1)
    compiling, release = threading.Event(), threading.Event()

    def compile_hook():
        compiling.set()
        assert release.wait(5), "test did not release the compile witness"

    def prepare():
        ready.wait(5)
        return _prepare()

    cuda_api.compile_hook = compile_hook
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(prepare) for _ in range(workers)]
        ready.wait(5)
        assert compiling.wait(5)
        release.set()
        decoders = [future.result(timeout=5) for future in futures]
    for decoder in decoders:
        decoder.prepare(torch.device("cuda", 0))
    assert cuda_api.loads == cuda_api.programs == cuda_api.destroys == 1
    assert len({id(decoder._compiled) for decoder in decoders}) == 1
    assert list(cuda_api.module._MODULE_CACHE) == [0]
    assert cuda_api.unloads == []
    owners = []
    for decoder in decoders:
        with decoder.capture_scope() as owner:
            owner.keep(torch.zeros(1))
            owners.append(owner)
    assert len({id(owner.tensors) for owner in owners}) == workers
    assert len({id(owner.nodes) for owner in owners}) == workers
    assert len({id(owner.tensors[0]) for owner in owners}) == workers
    assert all(owner.compiled is decoders[0]._compiled for owner in owners)


@pytest.mark.parametrize("same_decoder", [False, True])
def test_context_id_replacement_fails_even_when_handle_address_is_reused(cuda_api, same_decoder):
    decoder = _prepare()
    cuda_api.context_id = 102  # Same opaque handle 7, new context generation.
    target = decoder if same_decoder else ConditionalTailDecoder()
    with pytest.raises(RuntimeError, match="context changed; restart"):
        target.prepare(torch.device("cuda", 0))
    assert cuda_api.loads == 1
    assert cuda_api.module._MODULE_CACHE[0][0] == 101


def test_cache_admits_only_one_slot_per_visible_device(cuda_api):
    first = _prepare()
    cuda_api.device = cuda_api.context_device = 1
    cuda_api.context_id = 102
    second = ConditionalTailDecoder()
    second.prepare(torch.device("cuda", 1))
    assert first._compiled is not second._compiled
    assert set(cuda_api.module._MODULE_CACHE) == {0, 1}
    cuda_api.device = cuda_api.context_device = 2
    with pytest.raises(ValueError, match="not a visible CUDA ordinal"):
        ConditionalTailDecoder().prepare(torch.device("cuda", 2))
    assert cuda_api.loads == 2 and set(cuda_api.module._MODULE_CACHE) == {0, 1}


@pytest.mark.parametrize("failure,error", [("lookup_error", 17), ("destroy_error", 29)])
def test_failed_load_cleans_module_and_poisoned_slot_prevents_another_attempt(cuda_api, failure, error):
    setattr(cuda_api, failure, error)
    with pytest.raises(RuntimeError, match=str(error)):
        _prepare()
    assert cuda_api.loads == cuda_api.destroys == 1
    assert cuda_api.unloads == [(1, False)]
    assert cuda_api.module._MODULE_CACHE == {0: (101, None)}
    setattr(cuda_api, failure, 0)
    with pytest.raises(RuntimeError, match="initialization previously failed; restart"):
        _prepare()
    assert cuda_api.loads == 1


def test_failed_cleanup_preserves_initial_error_and_still_bounds_abandoned_modules(cuda_api, caplog):
    cuda_api.lookup_error, cuda_api.unload_error = 17, 23
    with pytest.raises(RuntimeError, match="17"):
        _prepare()
    assert "Module cleanup failed" in caplog.text and "23" in caplog.text
    with pytest.raises(RuntimeError, match="initialization previously failed; restart"):
        _prepare()
    assert cuda_api.loads == 1 and cuda_api.unloads == [(1, False)]


def test_prepare_guards_precede_cache_and_compilation(cuda_api):
    cuda_api.capturing = True
    with pytest.raises(RuntimeError, match="before capture"):
        _prepare()
    cuda_api.capturing = False
    with pytest.raises(ValueError, match="current CUDA device"):
        ConditionalTailDecoder().prepare(torch.device("cuda", 1))
    cuda_api.context_device = 1
    with pytest.raises(ValueError, match="context belongs to another device"):
        _prepare()
    assert cuda_api.loads == 0 and cuda_api.module._MODULE_CACHE == {}


@pytest.mark.parametrize("during_other_capture", [False, True])
def test_actual_binding_cycle_releases_graph_owners_without_unloading_module(cuda_api, during_other_capture):
    destroyed = []

    class GraphWitness:
        def __del__(self):
            destroyed.append("graph")

    class StreamWitness:
        pass

    decoder = _prepare()
    runtime = _runtime()
    binding = DenseGraphBinding(
        decode_fn=decoder,
        predictor=_CountingPredictor(),
        joint=_ThresholdJoint(),
        vllm_config=None,
        frame_widths=(1,),
        tiers=(1,),
        encoder_hidden=1,
        predictor_layers=2,
        predictor_hidden=1,
        blank_id=1,
        runtime=runtime,
    )
    # Use the real factory and binding closures: no restatement of their cycle.
    entry = binding._new_entry(0, 1, device=torch.device("cpu"), dtype=torch.float32, runtime=runtime)
    binding._entries[(0, 1)] = entry
    binding._decode_fns[(0, 1)] = binding._bind_decode(entry, runtime)
    owner = ConditionalCapture(decoder._compiled)
    tensor, stream, graph = torch.zeros(1), StreamWitness(), GraphWitness()
    owner.keep(tensor)
    owner.nodes.append((stream,))
    entry.capture_resources = owner
    entry.wrapper.graph = graph
    references = [weakref.ref(value) for value in (owner, tensor, stream, graph)]
    compiled_ref = weakref.ref(decoder._compiled)
    del binding, entry, owner, tensor, stream, graph, decoder
    cuda_api.capturing = during_other_capture
    gc.collect()
    assert destroyed == ["graph"] and all(reference() is None for reference in references)
    assert compiled_ref() is cuda_api.module._MODULE_CACHE[0][1]
    assert cuda_api.unloads == []
    # Retention is exactly the immutable context module, with no per-graph data.
    assert not hasattr(compiled_ref(), "nodes") and not hasattr(compiled_ref(), "tensors")


def test_failed_capture_releases_owner_but_preserves_usable_context_module(cuda_api):
    decoder = _prepare()
    with pytest.raises(RuntimeError, match="capture refused"):
        with decoder.capture_scope() as owner:
            tensor = torch.zeros(1)
            owner.keep(tensor)
            owner_ref, tensor_ref = weakref.ref(owner), weakref.ref(tensor)
            raise RuntimeError("capture refused")
    del owner, tensor
    gc.collect()
    assert owner_ref() is None and tensor_ref() is None
    assert decoder._capture is None and cuda_api.unloads == []
    assert _prepare()._compiled is decoder._compiled
