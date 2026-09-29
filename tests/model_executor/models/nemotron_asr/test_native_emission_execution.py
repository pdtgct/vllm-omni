# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Finite native-emission compilation, preserving the independent validator."""

from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from vllm_omni.model_executor.models.nemotron_asr.advance import (
    ROLE_CHUNK,
    ROLE_EOU,
    ROLE_FLUSH,
    ROLE_REPLAY,
    EmissionContext,
    EmissionProjection,
)
from vllm_omni.model_executor.models.nemotron_asr.native_burst import (
    finalize_native_burst,
    native_burst_invariant_rows,
)
from vllm_omni.model_executor.models.nemotron_asr.native_emission_execution import (
    build_native_emission_execution,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

PARK, BLANK, EOU, MAX_TOKENS = 9000, 12, 13, 22


def _config(**changes):
    return SimpleNamespace(
        **{
            "experimental_native_emission_compile": True,
            "experimental_native_burst": True,
            "supported_num_lookahead_tokens": [1],
            "hidden_size": 8,
            "eou_token_id": EOU,
            **changes,
        }
    )


def _binding(config=None, maximum_population=128):
    return build_native_emission_execution(
        _config() if config is None else config,
        maximum_population=maximum_population,
        park_id=PARK,
        blank_id=BLANK,
        max_tokens=MAX_TOKENS,
    )


def _inputs(rows, *, role=ROLE_CHUNK, failed=False, length=2):
    queue = torch.zeros(rows, 141, dtype=torch.int32)
    queue[:, :2] = torch.tensor([2, 3], dtype=torch.int32)
    book = torch.zeros(rows, 7, dtype=torch.int32)
    book[:, 0] = 1 if length else 0
    book[:, 1] = length
    book[:, 2] = 3
    book[:, 5] = int(bool(length))
    book[:, 6] = 2
    status = torch.full((rows,), int(failed) * 512, dtype=torch.int32)
    roles = torch.full((rows,), role, dtype=torch.int64)
    source = EmissionProjection(torch.zeros(rows, 8), queue, book, status)
    context = EmissionContext(roles, roles, roles, queue.clone(), book.clone(), roles, status.clone())
    return source, context


def _finalize(binding, source, context):
    return binding.finalize(source, context, park_id=PARK, blank_id=BLANK, max_tokens=MAX_TOKENS)


def _validate(binding, source, context, payload):
    return binding.invariant_rows(
        source, context, payload, park_id=PARK, blank_id=BLANK, eou_token_id=EOU, max_tokens=MAX_TOKENS
    )


@pytest.fixture
def actual_dynamo(monkeypatch):
    """Capture actual graphs and execute real tensor ops without a C++ toolchain."""
    original = torch.compile
    torch._dynamo.reset()
    graphs, options = [], []

    def backend(graph, _inputs):
        graphs.append(graph)
        return graph.forward

    def compile_for_cpu(fn, **kwargs):
        options.append(kwargs)
        assert kwargs == dict(fullgraph=True, dynamic=False, options={"triton.cudagraphs": False})
        return original(fn, backend=backend, fullgraph=True, dynamic=False)

    monkeypatch.setattr(torch, "compile", compile_for_cpu)
    yield graphs, options
    torch._dynamo.reset()


def test_default_does_not_construct_compiler(monkeypatch):
    # @spec PORT-DEC-011
    monkeypatch.setattr(torch, "compile", lambda *_a, **_k: pytest.fail("default initialized compiler"))
    assert _binding(_config(experimental_native_emission_compile=False)) is None
    assert _binding(SimpleNamespace()) is None
    assert (
        build_native_emission_execution(
            SimpleNamespace(), maximum_population=1, park_id=None, blank_id=None, max_tokens=0
        )
        is None
    )


@pytest.mark.parametrize(
    "change",
    [
        {"experimental_native_emission_compile": 1},
        {"experimental_native_burst": False},
        {"supported_num_lookahead_tokens": [3]},
        {"supported_num_lookahead_tokens": None},
        {"eou_token_id": True},
    ],
)
def test_compile_selection_rejects_unsafe_config(change):
    # @spec PORT-DEC-011
    with pytest.raises(ValueError, match="native emission"):
        _binding(_config(**change))


@pytest.mark.parametrize("changes", [{"park_id": None}, {"blank_id": 0}, {"max_tokens": 143}])
def test_compile_identity_is_validated_only_for_the_selected_experiment(changes):
    values = dict(maximum_population=128, park_id=PARK, blank_id=BLANK, max_tokens=MAX_TOKENS)
    values.update(changes)
    with pytest.raises(ValueError, match="native emission identity"):
        build_native_emission_execution(_config(), **values)


@torch.inference_mode()
def test_declared_warmup_dynamic_values_and_owned_outputs(actual_dynamo):
    # @spec PORT-PERF-009, PORT-PERF-011, PORT-DEC-012
    graphs, options = actual_dynamo
    binding = _binding()
    assert binding.populations == (1, 2)
    with pytest.raises(RuntimeError, match="not ready"):
        _finalize(binding, *_inputs(1))
    binding.warmup(torch.device("cpu"))
    assert binding.ready and len(options) == 2
    captures = len(graphs)
    retained = []
    for rows in (1, 2, 1, 2):
        for role, failed, length in (
            (ROLE_CHUNK, False, 0),
            (ROLE_CHUNK, False, 2),
            (ROLE_REPLAY, False, 2),
            (ROLE_FLUSH, False, 0),
            (ROLE_EOU, False, 1),
            (ROLE_CHUNK, True, 2),
        ):
            source, context = _inputs(rows, role=role, failed=failed, length=length)
            if role == ROLE_EOU:
                source.queue[:, 0] = EOU
            expected = finalize_native_burst(source, context, park_id=PARK, blank_id=BLANK)
            actual = _finalize(binding, source, context)
            for field in ("rows", "queue", "book", "row_status", "sampled_token_ids", "num_sampled"):
                value = getattr(actual, field)
                torch.testing.assert_close(value, getattr(expected, field), rtol=0, atol=0)
                assert all(
                    value.untyped_storage().data_ptr() != tensor.untyped_storage().data_ptr()
                    for tensor in (source.rows, source.queue, source.book, source.row_status)
                )
            expected_bad = native_burst_invariant_rows(
                source, context, expected, park_id=PARK, blank_id=BLANK, eou_token_id=EOU
            )
            torch.testing.assert_close(_validate(binding, source, context, actual), expected_bad, rtol=0, atol=0)
            retained.append((actual, tuple(getattr(actual, field).clone() for field in actual.__dataclass_fields__)))
    assert len(graphs) == captures
    for payload, original in retained:
        for field, expected in zip(payload.__dataclass_fields__, original, strict=True):
            torch.testing.assert_close(getattr(payload, field), expected, rtol=0, atol=0)
    binding.require_profile_coverage()


@pytest.mark.parametrize("mutation", ["blank", "park", "padding", "count", "book", "status", "queue"])
@torch.inference_mode()
def test_compiled_validator_retains_corruption_sensitivity(actual_dynamo, mutation):
    # @spec PORT-ADV-004, PORT-DEC-012
    binding = _binding(maximum_population=1)
    binding.warmup(torch.device("cpu"))
    source, context = _inputs(1)
    payload = _finalize(binding, source, context)
    if mutation == "blank":
        payload.sampled_token_ids[0, 0] = BLANK
    elif mutation == "park":
        payload.sampled_token_ids[0, 2] = 2
    elif mutation == "padding":
        payload.sampled_token_ids[0, 3] = 2
    elif mutation == "count":
        payload.num_sampled[0] = 2
    elif mutation == "book":
        payload.book[0, 5] = 1
    elif mutation == "status":
        payload.row_status[0] = 512
    else:
        payload.queue[0, 0] = 7
    assert _validate(binding, source, context, payload).tolist() == [True]


@torch.inference_mode()
def test_unknown_selected_signature_and_policy_do_not_recompile(actual_dynamo):
    # @spec PORT-PERF-011
    binding = _binding()
    binding.warmup(torch.device("cpu"))
    with pytest.raises(RuntimeError, match="profile"):
        binding.require_profile_coverage()
    source, context = _inputs(1)
    with pytest.raises(ValueError, match="signature"):
        _finalize(binding, replace(source, book=source.book.to(torch.int64)), context)
    with pytest.raises(ValueError, match="identity"):
        binding.finalize(source, context, park_id=PARK + 1, blank_id=BLANK, max_tokens=MAX_TOKENS)
    with pytest.raises(ValueError, match="signature"):
        _finalize(binding, source, replace(context, book=source.book))
    with pytest.raises(ValueError, match="tensor contract|signature"):
        _validate(
            binding,
            source,
            context,
            replace(_finalize(binding, source, context), sampled_token_ids=torch.zeros(1, 140, dtype=torch.int64)),
        )


@torch.inference_mode()
def test_other_populations_explicitly_retain_eager_execution(actual_dynamo):
    # @spec PORT-DEC-011, PORT-PERF-011
    graphs, _ = actual_dynamo
    binding = _binding()
    binding.warmup(torch.device("cpu"))
    captures = len(graphs)
    source, context = _inputs(3)
    actual = _finalize(binding, source, context)
    assert not _validate(binding, source, context, actual).any()
    assert len(graphs) == captures
    assert binding.receipt()["eager_other_populations"] is True


@torch.inference_mode()
def test_lost_compiler_cache_fails_instead_of_compiling_during_serving(actual_dynamo):
    # @spec PORT-PERF-011
    binding = _binding()
    binding.warmup(torch.device("cpu"))
    torch._dynamo.reset()
    with pytest.raises(RuntimeError, match="recompile"):
        _finalize(binding, *_inputs(1))
    assert not binding.ready


def test_compile_failure_is_not_eager_fallback(monkeypatch):
    # @spec PORT-PERF-011
    def fail(*_a, **_k):
        raise RuntimeError("compiler unavailable")

    monkeypatch.setattr(torch, "compile", fail)
    binding = _binding()
    with pytest.raises(RuntimeError, match="compiler unavailable"):
        binding.warmup(torch.device("cpu"))
    assert not binding.ready
    with pytest.raises(RuntimeError, match="fresh worker"):
        binding.warmup(torch.device("cpu"))


@pytest.mark.parametrize("mode", ["disabled", "ignored"])
def test_disabled_or_ignored_compiler_cannot_publish_readiness(monkeypatch, mode):
    # @spec PORT-DEC-011
    if mode == "ignored":
        monkeypatch.setattr(torch, "compile", lambda fn, **_kwargs: fn)
    binding = _binding()
    with torch._dynamo.config.patch(disable=mode == "disabled"):
        with pytest.raises(RuntimeError, match="cannot execute eagerly"):
            binding.warmup(torch.device("cpu"))
    assert not binding.ready


@torch.inference_mode()
def test_compiled_binding_uses_real_transaction_signatures_and_compatibility_drain(actual_dynamo):
    # @spec PORT-ADV-004, PORT-DEC-011, PORT-DEC-012
    from test_advance_model_rows_local import (
        CARRIER_HIDDEN,
        PARK_ID,
        _assert_native_burst_transaction,
        _clone_pools,
        _envelope,
        _fresh_pools,
        _plan,
        _tiny_core,
    )

    from vllm_omni.model_executor.models.nemotron_asr.native_burst import NativeBurstHandoff

    core = _tiny_core(seed=1)
    binding = build_native_emission_execution(
        _config(hidden_size=CARRIER_HIDDEN, eou_token_id=None),
        maximum_population=128,
        park_id=PARK_ID,
        blank_id=core.blank_id,
        max_tokens=142,
    )
    binding.warmup(torch.device("cpu"))
    graphs_before = len(actual_dynamo[0])
    for rows in (1, 2):
        torch.manual_seed(5)
        carrier = torch.stack(
            [_envelope(torch.randn(2560) * 0.01, final=False, seq=0, geometry=1) for _ in range(rows)]
        )
        pools = _fresh_pools(num_blocks=rows + 1)
        pools["queue_pool"] = torch.zeros(rows + 1, 141, dtype=torch.int32)
        handoff = NativeBurstHandoff(max_tokens=142)
        handoff.emission_execution = binding
        _assert_native_burst_transaction(
            core,
            pools,
            _clone_pools(pools),
            carrier,
            _plan(prefills=list(range(1, rows + 1)), geometries=[1] * rows, num_pool_blocks=rows + 1),
            handoff=handoff,
            epoch=1,
        )
    binding.require_profile_coverage()
    assert len(actual_dynamo[0]) == graphs_before


@torch.inference_mode()
def test_enabled_model_prepares_real_profile_inputs_before_readiness(actual_dynamo, monkeypatch):
    # @spec PORT-DEC-011, PORT-DEC-012, PORT-STATE-003
    pytest.importorskip("vllm", reason="model/profile startup needs the matching vLLM runtime")
    from test_advance_model_rows_local import D_MODEL, FEAT, KERNEL, N_LAYERS, WINDOW, _tiny_core
    from test_persistent_profile_execution import _config, _model_module, _profile_module

    module, profile = _model_module(), _profile_module()
    core = _tiny_core(seed=1)
    config = _config(decode_dispatch_arm="dense-eager")
    for name, value in dict(
        d_model=D_MODEL,
        n_layers=N_LAYERS,
        conv_kernel=KERNEL,
        att_context_left=WINDOW,
        att_context_right=1,
        pred_hidden=16,
        pred_rnn_layers=2,
        n_mels=FEAT,
        num_prompts=4,
        num_asr_labels=core.blank_id,
        supported_num_lookahead_tokens=[1],
        experimental_native_burst=True,
        experimental_native_emission_compile=True,
    ).items():
        setattr(config, name, value)
    vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(hf_config=config, dtype=torch.float32, logits_processors=[]),
        scheduler_config=SimpleNamespace(max_num_seqs=3, async_scheduling=False),
        speculative_config=None,
        parallel_config=SimpleNamespace(
            pipeline_parallel_size=1,
            tensor_parallel_size=1,
            data_parallel_size=1,
            decode_context_parallel_size=1,
            prefill_context_parallel_size=1,
            enable_expert_parallel=False,
        ),
    )
    # Reuse the existing tiny real model operations and actual profile allocator;
    # persistent resident allocation is deliberately forbidden before readiness.
    monkeypatch.setattr(module, "NemotronASRCore", lambda **_kwargs: core)
    monkeypatch.setattr(module, "PersistentStateLayerBase", lambda *_args, **_kwargs: object())
    model = module.NemotronASRForRNNT(vllm_config=vllm_config)
    monkeypatch.setattr(model, "_state_pools", lambda: pytest.fail("startup touched resident state"))
    binding = model._native_emission_execution
    assert binding is model._native_burst_handoff.emission_execution
    assert not binding.ready and not model._execution_memory_prepared
    original_profile = profile.run_persistent_state_profile
    calls = []

    def observe_profile(value, **kwargs):
        assert value is model and binding.ready
        assert kwargs["ready_domain"] and kwargs["geometry_id"] == 1
        calls.append(kwargs["num_rows"])
        return original_profile(value, **kwargs)

    monkeypatch.setattr(profile, "run_persistent_state_profile", observe_profile)
    model.prepare_execution_memory()
    assert calls == [3, 1, 2]
    assert model._execution_memory_prepared and not model._execution_memory_failed
    binding.require_profile_coverage()
    assert model.execution_profile_receipt()["native_emission"]["ready"]
    assert model._native_burst_handoff._snapshot is None
    assert model._native_burst_handoff._payload is None
    model.prepare_execution_memory()
    assert calls == [3, 1, 2]


def _frozen_native_oracle():
    """Independent old bodies, never the candidate's callable wrappers."""
    import ast
    import hashlib
    import subprocess
    import types
    from pathlib import Path

    from vllm_omni.model_executor.models.nemotron_asr import native_burst

    revision = "d10a05a25465682a63ab6c4083328a8d3a09be35"
    path = "vllm_omni/model_executor/models/nemotron_asr/native_burst.py"
    text = subprocess.check_output(
        ["git", "-C", str(Path(__file__).resolve().parents[4]), "show", f"{revision}:{path}"], text=True
    )
    assert (
        hashlib.sha256(text.encode()).hexdigest() == "599da3ea0b757e4d67f5b149f5baa725a7c8b1801e2fc1e3e43425f0801f81d6"
    )
    names = {"finalize_native_burst", "native_burst_invariant_rows"}
    functions = [node for node in ast.parse(text).body if isinstance(node, ast.FunctionDef) and node.name in names]
    assert {node.name for node in functions} == names
    namespace = dict(native_burst.__dict__)
    exec(compile(ast.Module(body=functions, type_ignores=[]), f"{revision}:{path}", "exec"), namespace)
    for name in names:
        assert namespace[name] is not getattr(native_burst, name)
    return types.SimpleNamespace(**{name: namespace[name] for name in names}), hashlib.sha256(text.encode()).hexdigest()


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires real CUDA Inductor compilation")
@torch.inference_mode()
def test_real_cuda_native_emission_matches_frozen_oracle():
    # @spec PORT-DEC-011, PORT-DEC-012, PORT-PERF-009, PORT-PERF-011
    import json

    import torch._inductor.metrics as inductor_metrics

    from vllm_omni.model_executor.models.nemotron_asr.manifests import (
        ENVELOPE_HEADER_FIELDS,
        RAW_SAMPLES_PER_CHUNK,
    )

    # Real shipped carrier geometry and ids; no weights are used by this region.
    hidden = len(ENVELOPE_HEADER_FIELDS) + max(RAW_SAMPLES_PER_CHUNK.values())
    park, blank, eou = 13088, 13087, 13090
    config = _config(hidden_size=hidden, eou_token_id=eou)
    binding = build_native_emission_execution(
        config, maximum_population=128, park_id=park, blank_id=blank, max_tokens=MAX_TOKENS
    )
    oracle, source_hash = _frozen_native_oracle()
    graphs_before = torch._dynamo.utils.counters["stats"]["unique_graphs"]
    inductor_before = dict(torch._dynamo.utils.counters["inductor"])
    kernels_before = inductor_metrics.generated_kernel_count
    binding.warmup(torch.device("cuda", 0))
    captured_graphs = torch._dynamo.utils.counters["stats"]["unique_graphs"]
    warm_graphs = captured_graphs - graphs_before
    assert warm_graphs >= 4, "N1/N2 producer/checker did not compile during warmup"
    warm_inductor = {
        name: count - inductor_before.get(name, 0) for name, count in torch._dynamo.utils.counters["inductor"].items()
    }
    warm_kernels = inductor_metrics.generated_kernel_count - kernels_before
    retained = []
    checked, corruptions = 0, 0
    for rows in (1, 2, 1, 2):
        for role, failed, length in (
            (ROLE_CHUNK, False, 0),
            (ROLE_CHUNK, False, 2),
            (ROLE_CHUNK, False, 141),
            (ROLE_CHUNK, False, 142),
            (ROLE_REPLAY, False, 2),
            (ROLE_FLUSH, False, 0),
            (ROLE_EOU, False, 1),
            (ROLE_CHUNK, True, 2),
        ):
            source, context = _inputs(rows, role=role, failed=failed, length=length)
            source = EmissionProjection(
                torch.zeros(rows, hidden, device="cuda"),
                source.queue.cuda(),
                source.book.cuda(),
                source.row_status.cuda(),
            )
            context = EmissionContext(*(tensor.cuda() for tensor in vars(context).values()))
            if role == ROLE_EOU:
                source.queue[:, 0] = eou
            expected = oracle.finalize_native_burst(source, context, park_id=park, blank_id=blank)
            actual = binding.finalize(source, context, park_id=park, blank_id=blank, max_tokens=MAX_TOKENS)
            for field in actual.__dataclass_fields__:
                value = getattr(actual, field)
                torch.testing.assert_close(value, getattr(expected, field), rtol=0, atol=0)
                assert all(
                    value.untyped_storage().data_ptr() != tensor.untyped_storage().data_ptr()
                    for tensor in (source.rows, source.queue, source.book, source.row_status)
                )
            expected_bad = oracle.native_burst_invariant_rows(
                source, context, expected, park_id=park, blank_id=blank, eou_token_id=eou
            )
            actual_bad = binding.invariant_rows(
                source, context, actual, park_id=park, blank_id=blank, eou_token_id=eou, max_tokens=MAX_TOKENS
            )
            torch.testing.assert_close(actual_bad, expected_bad, rtol=0, atol=0)
            torch.testing.assert_close(
                actual.num_sampled > MAX_TOKENS, expected.num_sampled > MAX_TOKENS, rtol=0, atol=0
            )
            retained.append((actual, tuple(getattr(actual, f).clone() for f in actual.__dataclass_fields__)))
            checked += 1
            if role == ROLE_CHUNK and not failed and length == 2:
                for mutation in ("blank", "park", "padding", "count", "book", "status", "queue"):
                    payload = replace(actual, **{f: getattr(actual, f).clone() for f in actual.__dataclass_fields__})
                    if mutation == "blank":
                        payload.sampled_token_ids[0, 0] = blank
                    elif mutation == "park":
                        payload.sampled_token_ids[0, 2] = 2
                    elif mutation == "padding":
                        payload.sampled_token_ids[0, 3] = 2
                    elif mutation == "count":
                        payload.num_sampled[0] = 2
                    elif mutation == "book":
                        payload.book[0, 5] = 1
                    elif mutation == "status":
                        payload.row_status[0] = 512
                    else:
                        payload.queue[0, 0] = 7
                    expected_bad = oracle.native_burst_invariant_rows(
                        source, context, payload, park_id=park, blank_id=blank, eou_token_id=eou
                    )
                    actual_bad = binding.invariant_rows(
                        source, context, payload, park_id=park, blank_id=blank, eou_token_id=eou, max_tokens=MAX_TOKENS
                    )
                    torch.testing.assert_close(actual_bad, expected_bad, rtol=0, atol=0)
                    assert actual_bad[0].item()
                    corruptions += 1
    for payload, original in retained:
        for field, expected in zip(payload.__dataclass_fields__, original, strict=True):
            torch.testing.assert_close(getattr(payload, field), expected, rtol=0, atol=0)
    assert torch._dynamo.utils.counters["stats"]["unique_graphs"] == captured_graphs
    binding.require_profile_coverage()
    print(
        json.dumps(
            dict(
                real_cuda_inductor=True,
                dtype="fp32 rows/int32 state/int64 roles",
                exact_cases=checked,
                corruption_cases=corruptions,
                post_warmup_compilations=0,
                warm_unique_graphs=warm_graphs,
                warm_inductor_counters=warm_inductor,
                warm_generated_kernel_count=warm_kernels,
                frozen_D10A_native_sha256=source_hash,
                execution=binding.receipt(),
            ),
            sort_keys=True,
        )
    )
