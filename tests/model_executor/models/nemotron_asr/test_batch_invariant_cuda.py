# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""CUDA qualification discriminators for the opt-in batch-invariant mode.

Run in a fresh process: installing vLLM overrides is process-wide. The synthetic
matrix uses the production optimized graph owner and real encoder dimensions.
It is a necessary numerical falsifier, never a complete qualification verdict.

NeMo and cost tests consume raw lab evidence, not synthetic substitutes:
NEMOTRON_BI_PARITY_MANIFEST names JSON with fingerprint, immutable golden hashes
and tensor pairs (actual/golden safetensors paths and tensor keys).
NEMOTRON_BI_COST_MANIFEST names JSON with matched identity, a discarded warmup
per arm, four or more alternating paired repetitions, raw replay durations and
ready/park/audio/wall observations, reported medians, and the side-effect list.
Missing evidence is a failure on CUDA, never an xfail or a newly pinned golden.
"""

from __future__ import annotations

import copy
import hashlib
import importlib
import json
import os
import statistics
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from test_encoder_batch_invariance import HISTORY, REAL_DIMENSIONS, build_core

from vllm_omni.model_executor.models.nemotron_asr.advance import _GatheredCaches
from vllm_omni.model_executor.models.nemotron_asr.decode_graph import platform_graph_runtime
from vllm_omni.model_executor.models.nemotron_asr.encoder_execution import (
    build_encoder_execution,
    encoder_geometry_shape,
    execute_encoder_transition,
)

pytestmark = [
    pytest.mark.core_model,
    pytest.mark.cuda,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA qualification only"),
]
POPULATIONS = (1, 2, 31, 63, 64, 128)
GEOMETRIES = (0, 1, 2, 3, 4)


def _api():
    return importlib.import_module("vllm_omni.model_executor.models.nemotron_asr.batch_invariance")


@pytest.fixture
def mode_on(isolated_batch_invariance):
    # Must precede weights, capture, or reservations. The adapter owns the
    # successful-return witness; the environment alone is not the record.
    os.environ["VLLM_BATCH_INVARIANT"] = "1"
    from vllm.model_executor.determinism import batch_invariant

    record = _api().installed_batch_invariant_mode(
        module=batch_invariant, initialize=batch_invariant.init_batch_invariance
    )
    assert record.enabled
    return record


def _bits(tensor):
    return tensor.detach().contiguous().view(torch.uint8).cpu().clone()


def _snapshot(raw, state, row, length):
    # Conditioned frames, predictor state, labels and joint logits are outside
    # the cross-batch promise. Valid raw frames and FULL floating caches are in.
    return {
        "encoder_raw": _bits(raw[row, :length]),
        **{f"encoder_window/{i}": _bits(t[row]) for i, t in enumerate(state.channel)},
        **{f"conv_tail/{i}": _bits(t[row]) for i, t in enumerate(state.time)},
    }


def _assert_snapshot(actual, expected, context):
    assert actual.keys() == expected.keys(), context
    for name in expected:
        assert torch.equal(actual[name], expected[name]), f"{context}: {name}"


def _state(population, device, history, composition, row):
    gen = torch.Generator(device=device).manual_seed(751)
    dims = REAL_DIMENSIONS
    probe_c = torch.randn(HISTORY, dims["d_model"], generator=gen, device=device) * 0.1
    probe_t = torch.randn(dims["d_model"], 8, generator=gen, device=device) * 0.1
    channels, times, valids = [], [], []
    for _ in range(dims["n_layers"]):
        c = probe_c.expand(population, -1, -1).clone()
        t = probe_t.expand(population, -1, -1).clone()
        v = torch.full((population, 1), history, dtype=torch.int32, device=device)
        if composition == "mixed":
            c.normal_(generator=gen)
            t.normal_(generator=gen)
            v[:, 0] = torch.arange(population, device=device) % (HISTORY + 1)
        c[row], t[row], v[row] = probe_c, probe_t, history
        channels.append(c)
        times.append(t)
        valids.append(v)
    return SimpleNamespace(channel=channels, time=times, window_valid=valids)


def _inputs(core, geometry, population, row, composition, chunk, final_tail, device):
    shape = encoder_geometry_shape(core, geometry)
    gen = torch.Generator(device=device).manual_seed(993 + chunk)
    probe = torch.randn(128, shape.mel_width, generator=gen, device=device)
    mel = probe.expand(population, -1, -1).clone()
    offsets = torch.full((population,), 2, dtype=torch.long, device=device)
    lengths = torch.full((population,), shape.out_width, dtype=torch.long, device=device)
    prompts = torch.full((population,), 3, dtype=torch.long, device=device)
    if composition == "mixed":
        mel.normal_(generator=gen)
        rows = torch.arange(population, device=device)
        offsets = rows % 3
        lengths = rows % (shape.out_width + 1)  # includes idle co-rows
        prompts = rows % 128
    mel[row], offsets[row], prompts[row] = probe, 2, 3
    valid = min(final_tail, shape.out_width) if chunk == 2 else shape.out_width
    lengths[row] = valid
    if composition == "repeated":
        lengths.fill_(valid)
    # Zero-work and partial final tails exercise clipping, not padded outputs.
    if chunk == 2:
        mel[row, :, min(shape.mel_width, 9 + 8 * valid) :] = 0
        if composition == "repeated":
            mel[:] = mel[row].clone()
    return mel, offsets, lengths, prompts, shape.out_width, valid


@contextmanager
def _capture_stream(device):
    current = torch.cuda.current_stream(device)
    side = torch.cuda.Stream(device=device)
    side.wait_stream(current)
    with torch.cuda.stream(side):
        yield
    current.wait_stream(side)


@pytest.mark.parametrize("policy", ["fp32", "fp16-encoder"])
@pytest.mark.parametrize("padded", [False, True], ids=["exact", "padded"])
@torch.inference_mode()
def test_prec020_023_graph_row_invariance(mode_on, policy, padded):
    """@spec PORT-PREC-020, PORT-PREC-023, PORT-PREC-026, PORT-PREC-027.

    Every geometry, population, first/middle/last position, repeated/mixed rows,
    empty/nonempty histories, final tails and independent B128 resets. The
    deployment cap is deliberately irrelevant to the B128 qualification envelope.
    """
    from vllm.config import VllmConfig

    device = torch.device("cuda", torch.accelerator.current_device_index())
    gpu = torch.cuda.get_device_name(device)
    assert "A100" in gpu or "A40" in gpu, "qualification partition must be A100 or A40"
    core = build_core(policy, device, mode=mode_on)
    torch._dynamo.reset()
    execution = build_encoder_execution(
        core,
        SimpleNamespace(
            encoder_execution_arm="dense-graphed", encoder_population_bucketing=padded, att_context_left=56
        ),
        maximum_population=128,
        warmup_geometries=GEOMETRIES,
        vllm_config=VllmConfig(),
        graph_runtime=replace(platform_graph_runtime(), capture_context=_capture_stream),
    )

    def warmup(geometry, population):
        state = _state(population, device, 37, "mixed", 0)
        mel, offsets, lengths, prompts, width, _ = _inputs(core, geometry, population, 0, "mixed", 0, 1, device)
        return execution.transition(mel, _GatheredCaches(state), offsets, lengths, width, prompts)

    cells = tuple((g, p) for g in GEOMETRIES for p in execution.warmup_populations)
    execution.warmup_domain(expected_cells=cells, invoke=warmup)
    assert execution.ready
    assert execution.ready_receipt()["batch_invariant_mode"] == "on"

    def sequence(geometry, population, row, composition, history, tail):
        # Each call allocates a fresh independent state, including the second B128 run.
        state = _state(population, device, history, composition, row)
        eager_state = copy.deepcopy(state)
        snapshots = []
        for chunk in range(3):
            mel, offsets, lengths, prompts, width, valid = _inputs(
                core, geometry, population, row, composition, chunk, tail, device
            )
            raw, conditioned = execution.transition(mel, _GatheredCaches(state), offsets, lengths, width, prompts)
            eager_raw, eager_conditioned = execute_encoder_transition(
                core, mel, _GatheredCaches(eager_state), offsets, lengths, width, prompts
            )
            # Same-batch graph/eager obligations remain broader than raw-only BI.
            assert torch.equal(_bits(raw), _bits(eager_raw))
            assert torch.equal(_bits(conditioned), _bits(eager_conditioned))
            for name in ("channel", "time", "window_valid"):
                for actual, expected in zip(getattr(state, name), getattr(eager_state, name), strict=True):
                    assert torch.equal(_bits(actual), _bits(expected))
            selected = range(population) if composition == "repeated" else (row,)
            snapshots.append([_snapshot(raw, state, r, valid) for r in selected])
        return snapshots

    for geometry in GEOMETRIES:
        width = encoder_geometry_shape(core, geometry).out_width
        for history in (0, 37):
            for tail in sorted({0, 1, max(1, width - 1)}):
                reference = sequence(geometry, 1, 0, "repeated", history, tail)
                for population in POPULATIONS:
                    for composition in ("repeated", "mixed"):
                        for row in sorted({0, population // 2, population - 1}):
                            actual = sequence(geometry, population, row, composition, history, tail)
                            context = (policy, padded, geometry, population, composition, row, history, tail)
                            for chunk, (outputs, baseline) in enumerate(zip(actual, reference, strict=True)):
                                for output in outputs:
                                    _assert_snapshot(output, baseline[0], (context, chunk))
                            if population == 128:
                                repeat = sequence(geometry, population, row, composition, history, tail)
                                for chunk_a, chunk_b in zip(actual, repeat, strict=True):
                                    for a, b in zip(chunk_a, chunk_b, strict=True):
                                        _assert_snapshot(a, b, (context, "independent reset"))


def _manifest(variable):
    value = os.environ.get(variable)
    assert value, f"provide pre-existing lab evidence via {variable}; no goldens are generated here"
    path = Path(value)
    assert path.name != ".env"
    return json.loads(path.read_text()), path.parent


def _file(base, relative):
    path = (base / relative).resolve()
    assert path.name != ".env"
    return path


@torch.inference_mode()
def test_prec024_fp32_nemo_parity(mode_on):
    """@spec PORT-PREC-024: unchanged goldens, complete shapes, exact 5e-5 gate."""
    from safetensors.torch import load_file

    manifest, base = _manifest("NEMOTRON_BI_PARITY_MANIFEST")
    assert manifest["fingerprint"]["batch_invariant_mode"] == "on"
    assert manifest["fingerprint"]["precision"] == "fp32"
    assert manifest["golden_revision"] == manifest["mode_off_golden_revision"]
    assert set(manifest["geometries"]) == set(GEOMETRIES)
    assert manifest["pairs"]
    covered = set()
    for pair in manifest["pairs"]:
        actual_path, golden_path = _file(base, pair["actual"]), _file(base, pair["golden"])
        assert actual_path != golden_path
        before = golden_path.read_bytes()
        assert hashlib.sha256(before).hexdigest() == pair["mode_off_golden_sha256"]
        assert hashlib.sha256(actual_path.read_bytes()).hexdigest() == pair["actual_sha256"]
        actual, golden = load_file(str(actual_path)), load_file(str(golden_path))
        assert set(pair["keys"]) == set(golden), "no selective tensor comparison"
        for key in pair["keys"]:
            torch.testing.assert_close(actual[key], golden[key], atol=5e-5, rtol=5e-5, equal_nan=False)
        assert golden_path.read_bytes() == before
        covered.add(pair["geometry"])
    assert covered == set(GEOMETRIES)


# Exact installed aten overrides and other process-wide mutations at 98dff2a81.
SIDE_EFFECTS = {
    "aten::mm",
    "aten::addmm",
    "aten::matmul",
    "aten::linear",
    "aten::bmm",
    "aten::softmax",
    "aten::_softmax",
    "aten::_log_softmax",
    "aten::mean.dim",
    "torch.bmm",
    "torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction",
    "torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction",
    "torch.backends.cuda.preferred_blas_library",
    "torch.backends.cuda.matmul.fp32_precision",
    "torch.backends.cudnn.conv.fp32_precision",
    "torch.backends.cudnn.rnn.fp32_precision",
    "VLLM_ALLREDUCE_USE_SYMM_MEM",
    "CUBLAS_WORKSPACE_CONFIG",
    "NCCL_LAUNCH_MODE",
    "NCCL_COLLNET_ENABLE",
    "NCCL_NVLS_ENABLE",
    "NCCL_P2P_NET_DISABLE",
    "NCCL_MIN_NCHANNELS",
    "NCCL_MAX_NCHANNELS",
    "NCCL_PROTO",
    "NCCL_ALGO",
    "NCCL_NTHREADS",
    "NCCL_SOCKET_NTHREADS",
    "VLLM_USE_AOT_COMPILE",
}


# Port-owned schema compatibility is additional to every pinned vLLM effect.
SIDE_EFFECTS.update(
    {
        "nemotron_bi.aten_schema_adapters",
        "aten::mm.out",
        "aten::mm.dtype",
        "aten::mm.dtype_out",
        "aten::addmm.out",
        "aten::addmm.dtype",
        "aten::addmm.dtype_out",
        "aten::bmm.out",
        "aten::bmm.dtype",
        "aten::bmm.dtype_out",
        "aten::matmul.out",
        "aten::linear.out",
        "aten::softmax.int",
        "aten::softmax.int_out",
        "aten::_softmax.out",
        "aten::_log_softmax.out",
        "aten::mean.out",
    }
)


def test_prec022_disclosure_manifest():
    """@spec PORT-PREC-022: matched unprofiled alternating pairs, no speed threshold."""
    manifest, _ = _manifest("NEMOTRON_BI_COST_MANIFEST")
    assert set(manifest["side_effects"]) == SIDE_EFFECTS
    assert manifest["threshold"] is None
    assert manifest["default_changed"] is False and manifest["supported_changed"] is False
    arms = manifest["arms"]
    assert set(arms) == {"off", "on"}
    off_id, on_id = copy.deepcopy(arms["off"]["identity"]), copy.deepcopy(arms["on"]["identity"])
    assert "batch_invariant_mode" not in off_id
    assert on_id.pop("batch_invariant_mode") == "on"
    assert off_id == on_id
    for field in ("gpu", "audio_sha256", "cadence", "concurrency", "software", "precision", "optimizations"):
        assert field in off_id
    pairs = manifest["pairs"]
    assert len(pairs) >= 4
    samples: dict[str, dict[str, list[float]]] = {
        arm: {key: [] for key in ("replay_ms", "service_ms", "throughput")} for arm in arms
    }
    for arm in arms:
        assert len(arms[arm]["warmup"]) == 1
        assert arms[arm]["warmup"][0]["discarded"] is True
    for index, pair in enumerate(pairs):
        assert pair["order"] == (["off", "on"] if index % 2 == 0 else ["on", "off"])
        for arm in pair["order"]:
            run = pair[arm]
            assert not run["profiled"] and not run["discarded"]
            assert run["encoder_replay_ns"] and run["chunks"]
            samples[arm]["replay_ms"].extend(value / 1e6 for value in run["encoder_replay_ns"])
            audio_seconds = 0.0
            for chunk in run["chunks"]:
                assert chunk["park_ns"] >= chunk["ready_ns"]
                samples[arm]["service_ms"].append((chunk["park_ns"] - chunk["ready_ns"]) / 1e6)
                audio_seconds += chunk["audio_seconds"]
            start, end = run["window_start_ns"], run["window_end_ns"]
            assert end > start
            assert all(start <= c["ready_ns"] <= c["park_ns"] <= end for c in run["chunks"])
            samples[arm]["throughput"].append(audio_seconds / ((end - start) / 1e9))
    for arm, metrics in samples.items():
        for metric, values in metrics.items():
            assert arms[arm]["reported"][metric] == pytest.approx(statistics.median(values))


@pytest.mark.parametrize("ndim", [1, 2])
@torch.inference_mode()
def test_prec021_compiled_biased_pointwise(mode_on, ndim):
    """@spec PORT-PREC-021: Inductor retains addmm and its pre-cast bias."""
    device = torch.device("cuda", torch.accelerator.current_device_index())
    shape = (2, 2, 5) if ndim == 1 else (2, 2, 3, 5)
    x = torch.empty(shape, device=device, dtype=torch.float16)
    x[:, 0], x[:, 1] = 1.0, 2**-11
    weight = torch.ones((3, 2) + (1,) * ndim, device=device, dtype=x.dtype)
    bias = torch.full((3,), -1.0, device=device, dtype=x.dtype)
    compiled = torch.compile(_api().pointwise_conv_as_linear, backend="inductor", fullgraph=True)
    actual = compiled(x, weight, bias)
    # A post-store bias addition would lose the half-ULP and return zero.
    assert torch.equal(actual, torch.full((2, 3) + shape[2:], 2**-11, device=device, dtype=x.dtype))
    assert torch.equal(actual, _api().pointwise_conv_as_linear(x, weight, bias))
