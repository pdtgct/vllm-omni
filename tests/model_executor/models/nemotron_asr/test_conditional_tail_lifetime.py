# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Actual native-IF owner and GC guard with a CPU fake of PyTorch's API.

The fake records scope/ownership contracts, not CUDA allocator behavior. Real
CUDA tests and broader earlier-key replay remain separate validation gates.
"""

import gc
import importlib
import threading
import weakref
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch
from test_conditional_tail import _CountingPredictor, _state, _ThresholdJoint
from test_decode_graph_binding import _runtime

from vllm_omni.model_executor.models.nemotron_asr.conditional_tail import ConditionalTailDecoder
from vllm_omni.model_executor.models.nemotron_asr.decode_graph import DenseGraphBinding

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.fixture
def cuda_api(monkeypatch):
    module = importlib.import_module("vllm_omni.model_executor.models.nemotron_asr.conditional_tail")
    state = SimpleNamespace(
        device=0,
        capturing=False,
        stream="parent",
        lookups=0,
        calls=[],
        predicates=[],
        begin_error=None,
        end_error=None,
        module=module,
    )

    class NativeGraph:
        @staticmethod
        def get_currently_capturing_graph():
            state.lookups += 1
            if not state.capturing:
                raise RuntimeError("no current graph capture")
            return state.graph

        def begin_capture_to_if_node(self, predicate):
            state.calls.append("begin")
            assert predicate.ndim == 0 and predicate.dtype == torch.bool
            assert state.stream == "parent"
            if state.begin_error:
                raise RuntimeError(state.begin_error)
            state.predicates.append(weakref.ref(predicate))
            state.stream = "child"

        def end_capture_to_conditional_node(self):
            state.calls.append("end")
            if state.end_error:
                raise RuntimeError(state.end_error)
            assert state.stream == "child"
            state.stream = "parent"

    state.graph = NativeGraph()
    state.graph_class = NativeGraph
    monkeypatch.setattr(torch.cuda, "CUDAGraph", NativeGraph)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: state.capturing)
    monkeypatch.setattr(torch.accelerator, "current_device_index", lambda: state.device)
    monkeypatch.setattr(torch.version, "cuda", "13.0")
    return state


def _prepare():
    decoder = ConditionalTailDecoder()
    decoder.prepare(torch.device("cuda", 0))
    return decoder


def test_eager_path_never_prepares_or_invokes_native_api(cuda_api):
    decoder = ConditionalTailDecoder()
    output = decoder(
        torch.ones(1, 1, 1), torch.ones(1, dtype=torch.long), _CountingPredictor(), _ThresholdJoint(), _state(1)
    )
    assert output.frame_emission_counts.tolist() == [[1]]
    assert decoder._device is None and decoder._capture is None
    assert cuda_api.lookups == 0 and cuda_api.calls == []


def test_native_if_retains_fresh_scalar_and_one_parent_per_owner(cuda_api):
    decoder = _prepare()
    with decoder.capture_scope() as owner:
        cuda_api.capturing = True
        for values in ([False, False], [False, True]):
            active = torch.tensor(values)

            def body():
                assert cuda_api.stream == "child"
                cuda_api.calls.append("body")

            owner.run_if(active, body)
            predicate = cuda_api.predicates[-1]()
            assert predicate is not None and predicate.device == active.device
            assert predicate.item() is any(values)
            assert any(tensor is predicate for tensor in owner.tensors)
            assert any(tensor is active for tensor in owner.tensors)
            assert cuda_api.stream == "parent"
        assert owner.if_nodes == 2 and owner._graph is cuda_api.graph
        assert cuda_api.predicates[0]() is not cuda_api.predicates[1]()
        assert cuda_api.calls == ["begin", "body", "end"] * 2
        cuda_api.graph = cuda_api.graph_class()
        with pytest.raises(RuntimeError, match="cannot cross parent graphs"):
            owner.run_if(torch.ones(2, dtype=torch.bool), lambda: pytest.fail("wrong-parent body"))
        assert owner.if_nodes == 2 and len(cuda_api.calls) == 6


@pytest.mark.parametrize("failure", ["lookup", "begin", "body", "body_and_end", "end"])
def test_native_if_failures_propagate_without_duplicate_end_or_receipt(cuda_api, failure, caplog):
    decoder = _prepare()
    with decoder.capture_scope() as owner:
        cuda_api.capturing = failure != "lookup"
        cuda_api.begin_error = "begin failed" if failure == "begin" else None
        cuda_api.end_error = "end failed" if failure in ("end", "body_and_end") else None

        def body():
            cuda_api.calls.append("body")
            if failure in ("body", "body_and_end"):
                raise ValueError("body failed")

        error = ValueError if failure in ("body", "body_and_end") else RuntimeError
        with pytest.raises(error, match="body failed" if error is ValueError else "failed|no current graph"):
            owner.run_if(torch.ones(2, dtype=torch.bool), body)
        assert owner.if_nodes == 0
        if failure == "lookup":
            assert cuda_api.calls == []
        elif failure == "begin":
            assert cuda_api.calls == ["begin"]
        else:
            assert cuda_api.calls == ["begin", "body", "end"]
        if failure == "body":
            assert cuda_api.stream == "parent"
        if failure == "body_and_end":
            assert "cleanup failed" in caplog.text and "end failed" in caplog.text
    assert decoder._capture is None


@pytest.mark.parametrize(
    "missing", ["get_currently_capturing_graph", "begin_capture_to_if_node", "end_capture_to_conditional_node"]
)
def test_prepare_requires_each_native_capability_without_invoking_it(cuda_api, monkeypatch, missing):
    monkeypatch.setattr(cuda_api.graph_class, missing, None)
    decoder = ConditionalTailDecoder()
    with pytest.raises(RuntimeError, match="native PyTorch 2.13 conditional graph API"):
        decoder.prepare(torch.device("cuda", 0))
    assert decoder._device is None and cuda_api.lookups == 0 and cuda_api.calls == []


@pytest.mark.parametrize("cuda_version", [None, "12.3", "12.4", "13.0"])
def test_prepare_checks_cuda_build_and_only_prepares_device_identity(cuda_api, monkeypatch, cuda_version):
    monkeypatch.setattr(torch.version, "cuda", cuda_version)
    decoder = ConditionalTailDecoder()
    if cuda_version in (None, "12.3"):
        with pytest.raises(RuntimeError, match="CUDA build of 12.4"):
            decoder.prepare(torch.device("cuda", 0))
        assert decoder._device is None
    else:
        decoder.prepare(torch.device("cuda"))
        decoder.prepare(torch.device("cuda", 0))
        assert decoder._device == torch.device("cuda", 0)
    assert cuda_api.lookups == 0 and cuda_api.calls == []


def test_prepare_rejects_capture_wrong_current_device_and_device_migration(cuda_api):
    decoder = _prepare()
    cuda_api.capturing = True
    with pytest.raises(RuntimeError, match="before capture"):
        decoder.prepare(torch.device("cuda", 0))
    cuda_api.capturing = False
    with pytest.raises(ValueError, match="current CUDA device"):
        ConditionalTailDecoder().prepare(torch.device("cuda", 1))
    cuda_api.device = 1
    with pytest.raises(ValueError, match="cannot cross CUDA devices"):
        decoder.prepare(torch.device("cuda", 1))
    assert cuda_api.lookups == 0 and cuda_api.calls == []


@pytest.mark.parametrize("during_other_capture", [False, True])
def test_actual_binding_cycle_releases_native_graph_owner_and_buffers(cuda_api, during_other_capture):
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
    entry = binding._new_entry(0, 1, device=torch.device("cpu"), dtype=torch.float32, runtime=runtime)
    binding._entries[(0, 1)] = entry
    binding._decode_fns[(0, 1)] = binding._bind_decode(entry, runtime)
    with decoder.capture_scope() as owner:
        cuda_api.capturing = True
        owner.run_if(torch.ones(2, dtype=torch.bool), lambda: None)
    tensor, graph = owner.tensors[-1], cuda_api.graph
    entry.capture_resources = owner
    entry.wrapper.graph = graph
    references = [weakref.ref(value) for value in (owner, tensor, graph)]
    del binding, entry, owner, tensor, graph, decoder
    cuda_api.graph = None
    cuda_api.capturing = during_other_capture
    gc.collect()
    assert all(reference() is None for reference in references)


def test_failed_capture_releases_native_graph_owner_and_buffers(cuda_api):
    decoder = _prepare()
    with pytest.raises(RuntimeError, match="capture refused"):
        with decoder.capture_scope() as owner:
            cuda_api.capturing = True
            owner.run_if(torch.ones(2, dtype=torch.bool), lambda: None)
            tensor = owner.tensors[-1]
            owner_ref, tensor_ref, graph_ref = weakref.ref(owner), weakref.ref(tensor), weakref.ref(cuda_api.graph)
            raise RuntimeError("capture refused")
    cuda_api.graph = None
    del owner, tensor
    gc.collect()
    assert owner_ref() is None and tensor_ref() is None and graph_ref() is None
    assert decoder._capture is None


@pytest.mark.parametrize("initially_enabled", [False, True])
def test_capture_scope_restores_gc_with_nested_scopes_and_rejected_overlap(cuda_api, initially_enabled):
    original = gc.isenabled()
    decoders = [_prepare(), _prepare()]
    try:
        (gc.enable if initially_enabled else gc.disable)()
        with decoders[0].capture_scope():
            assert not gc.isenabled()
            with pytest.raises(RuntimeError, match="cannot overlap"):
                with decoders[0].capture_scope():
                    pytest.fail("same-decoder overlap was accepted")
            with decoders[1].capture_scope():
                assert not gc.isenabled()
            assert not gc.isenabled()
        assert gc.isenabled() is initially_enabled
        assert cuda_api.module._CAPTURE_GC_USERS == 0
    finally:
        (gc.enable if original else gc.disable)()


def test_unprepared_capture_does_not_change_gc_state(cuda_api):
    original = gc.isenabled()
    with pytest.raises(RuntimeError, match="not prepared"):
        with ConditionalTailDecoder().capture_scope():
            pytest.fail("unprepared capture was accepted")
    assert gc.isenabled() is original
    assert cuda_api.module._CAPTURE_GC_USERS == 0


def test_overlapping_capture_scopes_restore_gc_after_last_thread_exits(cuda_api):
    original = gc.isenabled()
    first, second = _prepare(), _prepare()
    entered, finish = threading.Event(), threading.Event()

    def overlap():
        with second.capture_scope():
            entered.set()
            assert finish.wait(timeout=5)
            assert not gc.isenabled()

    try:
        gc.enable()
        with ThreadPoolExecutor(max_workers=1) as executor:
            try:
                with first.capture_scope():
                    future = executor.submit(overlap)
                    assert entered.wait(timeout=5)
                    assert not gc.isenabled()
                # The scope that originally disabled GC has already exited.
                assert not gc.isenabled()
            finally:
                finish.set()
            future.result(timeout=5)
        assert gc.isenabled()
        assert cuda_api.module._CAPTURE_GC_USERS == 0
    finally:
        finish.set()
        (gc.enable if original else gc.disable)()


@pytest.mark.parametrize("fail_capture_end", [False, True])
@pytest.mark.parametrize("initially_enabled", [False, True])
def test_binding_scope_protects_parent_capture_end_and_restores_gc(
    cuda_api, monkeypatch, fail_capture_end, initially_enabled
):
    original = gc.isenabled()
    decoder = _prepare()
    monkeypatch.setattr(decoder, "prepare", lambda device: None)
    phases = []
    collect = gc.collect

    @contextmanager
    def parent_graph():
        assert not gc.isenabled()
        assert gc.collect is collect
        gc.collect()  # Explicit pre-capture collection remains usable.
        phases.append("begin")
        try:
            yield
        finally:
            assert not gc.isenabled()
            phases.append("end")
            if fail_capture_end:
                raise RuntimeError("parent capture_end failed")

    class Wrapper:
        def __init__(self, run, *args, **kwargs):
            self.run, self.calls = run, 0

        def __call__(self):
            self.calls += 1
            if self.calls == 2:
                with parent_graph():
                    return self.run()
            assert gc.isenabled() is initially_enabled  # Eager and replay.
            return self.run()

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
        runtime=replace(_runtime(), wrapper_factory=Wrapper),
    )
    try:
        (gc.enable if initially_enabled else gc.disable)()
        if fail_capture_end:
            with pytest.raises(RuntimeError, match="parent capture_end failed"):
                binding.warmup(torch.device("cpu"), torch.float32)
            assert binding.captured_keys == ()
        else:
            binding.warmup(torch.device("cpu"), torch.float32)
            assert binding.captured_keys == ((0, 1),)
        assert phases == ["begin", "end"]
        assert gc.isenabled() is initially_enabled
        assert decoder._capture is None
        assert cuda_api.module._CAPTURE_GC_USERS == 0
    finally:
        (gc.enable if original else gc.disable)()
