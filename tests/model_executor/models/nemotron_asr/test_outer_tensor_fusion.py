# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Tests first: independent outer predicates and bounded execution selection.

CPU cases establish contracts only. Real compiler cases are explicitly CUDA.
"""

from __future__ import annotations

import importlib
from copy import deepcopy
from types import SimpleNamespace
from typing import Any

import pytest
import torch

from tests.model_executor.models.nemotron_asr import test_native_burst as native
from vllm_omni.model_executor.models.nemotron_asr import advance, frontend, rnnt
from vllm_omni.model_executor.models.nemotron_asr import nemotron_asr as model_module

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

CONSTANTS = {
    "blank_id": 12,
    "placeholder_id": 9001,
    "park_id": 9000,
    "eou_token_id": 9002,
    "queue_capacity": 141,
    "raw_tail_capacity": 1953,
    "mel_tail_capacity": 9,
    "hop_length": 160,
    "n_fft": 512,
    "carrier_width": 17927,
    "max_symbols_per_step": 10,
}
FAMILIES = ("pre", "post-control", "post-g0", "post-g1", "post-g2", "post-g3", "post-g4")


def _symbol(module: Any, name: str) -> Any:
    value = getattr(module, name, None)
    assert callable(value), f"PORT-PERF contract missing {type(module).__name__}.{name}"
    return value


def _fusion_module() -> Any:
    name = "vllm_omni.model_executor.models.nemotron_asr.outer_tensor_fusion"
    try:
        return importlib.import_module(name)
    except ModuleNotFoundError as error:
        if error.name != name:
            raise
        pytest.fail(f"PORT-PERF-015 missing {name}", pytrace=False)


def _binding() -> Any:
    return _symbol(_fusion_module(), "OuterTensorExecution")(constants=dict(CONSTANTS))


def _pre_inputs(device: str = "cpu") -> dict[str, torch.Tensor]:
    book = torch.zeros(1, 7, dtype=torch.int32, device=device)
    book[:, rnnt.QUEUE_LAST_LABEL] = CONSTANTS["blank_id"]
    return {
        "input_ids": torch.tensor([9001], dtype=torch.int64, device=device),
        "is_chunk": torch.tensor([True], device=device),
        "plan_geometry": torch.zeros(1, dtype=torch.int64, device=device),
        "prior_prompt": torch.zeros(1, dtype=torch.int64, device=device),
        "queue": torch.zeros(1, 141, dtype=torch.int32, device=device),
        "book": book,
        "frontend_counters": torch.zeros(1, 8, dtype=torch.int64, device=device),
        "cadence_frames": torch.tensor([8], dtype=torch.int64, device=device),
    }


def _pre_constants() -> dict[str, int]:
    return {k: v for k, v in CONSTANTS.items() if k not in {"carrier_width", "max_symbols_per_step"}}


def _post_inputs(geometry: int | None, device: str = "cpu") -> dict[str, torch.Tensor]:
    width = 0 if geometry is None else (1, 2, 4, 7, 14)[geometry] * 10 + 1
    count = int(geometry is not None)
    book = torch.zeros(1, 7, dtype=torch.int32, device=device)
    book[:, rnnt.QUEUE_LAST_LABEL] = 12
    if geometry is not None:
        book[:, rnnt.BOOK_GEOMETRY] = geometry
    rows = torch.zeros(1, 17927, dtype=torch.float32, device=device)
    rows[:, 0] = 9000
    return {
        "token_ids": torch.zeros(count, width, dtype=torch.int32, device=device),
        "token_lengths": torch.zeros(count, dtype=torch.int32, device=device),
        "roles": torch.tensor([advance.ROLE_FLUSH if geometry is None else advance.ROLE_CHUNK], device=device),
        "chunk_rows": torch.arange(count, dtype=torch.int64, device=device),
        "baseline_queue": torch.zeros(1, 141, dtype=torch.int32, device=device),
        "baseline_book": book,
        "prompt_index": torch.zeros(1, dtype=torch.int64, device=device),
        "incoming_status": torch.zeros(1, dtype=torch.int32, device=device),
        "proposed_rows": rows,
        "proposed_queue": torch.zeros(1, 141, dtype=torch.int32, device=device),
        "proposed_book": book.clone(),
        "proposed_status": torch.zeros(1, dtype=torch.int32, device=device),
        "plan_geometry": torch.tensor([geometry or 0], dtype=torch.int64, device=device),
    }


def _post_constants() -> dict[str, int]:
    return {k: CONSTANTS[k] for k in ("park_id", "blank_id", "eou_token_id")}


def _assert_inputs_untouched(inputs: dict[str, torch.Tensor], before: dict[str, torch.Tensor]) -> None:
    for name, tensor in inputs.items():
        torch.testing.assert_close(tensor, before[name], rtol=0, atol=0, equal_nan=True)


@pytest.mark.parametrize("value", [None, 0, 1, "true", [], {}])
def test_fusion_selector_rejects_non_boolean_before_core(monkeypatch, value):
    """@spec PORT-PERF-012: invalid selector cannot construct core/state."""
    hf = native._native_config(profile=native.FIVE_CADENCE_PROFILE, lookaheads=native.LOOKAHEADS)
    hf.experimental_outer_tensor_fusion = value
    monkeypatch.setattr(model_module, "NemotronASRCore", lambda **kw: pytest.fail("core constructed before rejection"))
    monkeypatch.setattr(
        model_module, "PersistentStateLayerBase", lambda *a, **kw: pytest.fail("state spec constructed")
    )
    with pytest.raises(ValueError, match="outer_tensor_fusion|outer tensor fusion"):
        model_module.NemotronASRForRNNT(vllm_config=native._runtime_config(hf))


@pytest.mark.parametrize(
    "field,value",
    [
        ("encoder_execution_arm", "eager"),
        ("decode_dispatch_arm", "dense-eager"),
        ("experimental_native_burst", False),
        ("experimental_native_burst_profile", native.LEGACY_PROFILE),
        ("supported_num_lookahead_tokens", [1]),
        ("chunk_bucket_graph_populations", [1]),
        ("max_num_seqs", 2),
        ("async_scheduling", True),
        ("speculative_config", object()),
        ("tensor_parallel_size", 2),
        ("dtype", torch.float16),
        ("dtype", None),
        ("experimental_encoder_compute_dtype", "float16"),
    ],
)
def test_enabled_profile_rejects_before_core(monkeypatch, field, value):
    """@spec PORT-PERF-013, PORT-DEC-011: no compatibility/profile substitution."""
    hf = native._native_config(profile=native.FIVE_CADENCE_PROFILE, lookaheads=native.LOOKAHEADS)
    hf.experimental_outer_tensor_fusion = True
    hf.encoder_execution_arm = "eager-graphed"
    hf.decode_dispatch_arm = "dense-graphed"
    runtime = native._runtime_config(hf)
    owner = (
        runtime.scheduler_config
        if field in {"max_num_seqs", "async_scheduling"}
        else runtime.parallel_config
        if field == "tensor_parallel_size"
        else runtime.model_config
        if field == "dtype"
        else runtime
        if field == "speculative_config"
        else hf
    )
    setattr(owner, field, value)
    monkeypatch.setattr(model_module, "NemotronASRCore", lambda **kw: pytest.fail("core constructed before rejection"))
    with pytest.raises(ValueError):
        model_module.NemotronASRForRNNT(vllm_config=runtime)


def test_enabled_profile_rejects_installed_batch_invariance_before_core(monkeypatch):
    """@spec PORT-PERF-013: installed BI authority, not an HF alias, excludes fusion."""
    from vllm_omni.model_executor.models.nemotron_asr import batch_invariance

    monkeypatch.setattr(batch_invariance, "_installed", batch_invariance.BatchInvariantExecution(True))
    monkeypatch.setattr(batch_invariance, "_install_failure", None)
    hf = native._native_config(profile=native.FIVE_CADENCE_PROFILE, lookaheads=native.LOOKAHEADS)
    hf.experimental_outer_tensor_fusion = True
    hf.encoder_execution_arm, hf.decode_dispatch_arm = "eager-graphed", "dense-graphed"
    monkeypatch.setattr(model_module, "NemotronASRCore", lambda **kw: pytest.fail("core constructed with BI enabled"))
    with pytest.raises(ValueError, match="batch|BI|invarian"):
        model_module.NemotronASRForRNNT(vllm_config=native._runtime_config(hf))


def test_off_never_constructs_compiler_and_omits_receipt(monkeypatch):
    """@spec PORT-PERF-012, PORT-PERF-022: off preserves existing construction."""
    monkeypatch.setattr(torch, "compile", lambda *a, **kw: pytest.fail("off selected a compiler"))
    hf = native._native_config(profile=native.FIVE_CADENCE_PROFILE, lookaheads=native.LOOKAHEADS)
    hf.experimental_outer_tensor_fusion = False
    model = native._construct_model(monkeypatch, hf)
    assert "outer_tensor_fusion" not in model.execution_profile_receipt()
    hf.experimental_outer_tensor_fusion = True
    assert "outer_tensor_fusion" not in model.execution_profile_receipt()


def test_receipt_runtime_version_roundtrips_through_strict_worker_transport(monkeypatch):
    """@spec PORT-PERF-022: detached runtime identity crosses the worker transport."""
    from torch.torch_version import TorchVersion
    from vllm.v1.serial_utils import MsgpackDecoder, MsgpackEncoder

    monkeypatch.setenv("VLLM_ALLOW_INSECURE_SERIALIZATION", "0")
    runtime_version = TorchVersion(str(torch.__version__))
    monkeypatch.setattr(torch, "__version__", runtime_version)
    encoder = MsgpackEncoder()
    with pytest.raises(TypeError, match="TorchVersion.*not serializable"):
        encoder.encode_into({"torch": runtime_version}, bytearray())

    receipt = _binding().receipt()
    buffers = encoder.encode_into(receipt, bytearray())
    restored = MsgpackDecoder().decode(buffers)
    assert type(receipt["compiler"]["torch"]) is str
    assert receipt["compiler"]["torch"] == str(runtime_version)
    assert restored == receipt
    assert type(restored["compiler"]["torch"]) is str


@pytest.mark.parametrize("kind", ["parameter", "buffer", "model-parameter", "model-buffer", "integer-buffer"])
def test_load_exit_attests_all_floating_storage(monkeypatch, kind):
    """@spec PORT-PERF-013: copied checkpoint dtype cannot excuse wrong destination."""
    model = object.__new__(model_module.NemotronASRForRNNT)
    torch.nn.Module.__init__(model)
    core = torch.nn.Module()
    core.encoder = torch.nn.Module()
    core.encoder.invalidate_stream_relative_position_projections = lambda: None
    if kind == "parameter":
        core.register_parameter("unexpected_float", torch.nn.Parameter(torch.ones(1, dtype=torch.float16)))
    elif kind in {"buffer", "integer-buffer"}:
        core.register_buffer(
            "unexpected_float", torch.ones(1, dtype=torch.int64 if kind == "integer-buffer" else torch.float64)
        )
    elif kind == "model-parameter":
        model.register_parameter("outside_core", torch.nn.Parameter(torch.ones(1, dtype=torch.float16)))
    else:
        model.register_buffer("outside_core", torch.ones(1, dtype=torch.float64))
    model.core = core
    model.config = SimpleNamespace(experimental_outer_tensor_fusion=True)
    object.__setattr__(model, "_outer_tensor_execution", SimpleNamespace(enabled=True))
    object.__setattr__(model, "_encoder_execution", SimpleNamespace(_model_state=None))
    weights = [(name, tensor.clone()) for name, tensor in core.state_dict().items()]
    if kind == "integer-buffer":
        assert model.load_weights(weights) == {"core.unexpected_float"}
    else:
        with pytest.raises(ValueError, match="FP32|float32|dtype"):
            model.load_weights(weights)


PRE_DEFECTS = (None, "role", "book", "counter", "geometry")
POST_DEFECTS = (None, "nan", "inf", "fraction", "tail-nan", "unknown", "lost", "book")


def _pre_case(defect, device="cpu"):
    inputs = _pre_inputs(device)
    expected = 0
    if defect == "role":
        inputs["input_ids"][0] = 3
        expected = advance.ROW_STATUS_ROLE_MISMATCH
    elif defect == "book":
        inputs["book"][0, rnnt.QUEUE_LAST_LABEL] = 999
        expected = advance.ROW_STATUS_BOOK_INVARIANT
    elif defect == "counter":
        inputs["frontend_counters"][0, frontend.CTR_TOTAL_VALID_SAMPLES] = 100
        expected = advance.ROW_STATUS_BOOK_INVARIANT
    elif defect == "geometry":
        inputs["plan_geometry"][0] = 1
        expected = advance.ROW_STATUS_BOOK_IDENTITY
    return inputs, expected


def _post_case(geometry, defect, device="cpu"):
    inputs = _post_inputs(geometry, device)
    expected = 0
    if defect in {"nan", "inf", "fraction"}:
        inputs["proposed_rows"][0, 0] = {"nan": float("nan"), "inf": float("inf"), "fraction": 1.5}[defect]
        expected = advance.ROW_STATUS_DECODE_INVARIANT
    elif defect == "tail-nan":
        inputs["proposed_rows"][0, 1] = float("nan")
        expected = advance.ROW_STATUS_DECODE_INVARIANT
    elif defect == "unknown":
        inputs["proposed_status"][0] = 1 << 25
        expected = (1 << 25) | advance.ROW_STATUS_DECODE_INVARIANT
    elif defect == "lost":
        inputs["incoming_status"][0] = advance.ROW_STATUS_ENVELOPE
        expected = advance.ROW_STATUS_ENVELOPE | advance.ROW_STATUS_DECODE_INVARIANT
    elif defect == "book":
        inputs["proposed_book"][0, rnnt.QUEUE_HEAD] = -1
        # Proposed book failure makes this a failed row BEFORE MRV1 checks its mutation.
        expected = advance.ROW_STATUS_BOOK_INVARIANT | advance.ROW_STATUS_DECODE_INVARIANT
    return inputs, expected


@pytest.mark.parametrize("defect", PRE_DEFECTS)
def test_pre_region_has_independent_status_and_safe_label_expectations(defect):
    """@spec PORT-PERF-014, PORT-ADV-004, PORT-STATE-008."""
    inputs, expected = _pre_case(defect)
    before = {k: v.clone() for k, v in inputs.items()}
    result = _symbol(advance, "_outer_pre_tensor_predicates")(**inputs, **_pre_constants())
    assert [t.tolist() for t in result] == [[expected], [12], [advance.ROLE_CHUNK], [False]]
    _assert_inputs_untouched(inputs, before)
    for tensor in result:
        tensor.fill_(0)
    _assert_inputs_untouched(inputs, before)


@pytest.mark.parametrize("geometry", [None, 0, 1, 2, 3, 4])
@pytest.mark.parametrize("defect", POST_DEFECTS)
def test_post_region_preserves_full_oracle_order(geometry, defect):
    """@spec PORT-PERF-014, PORT-PERF-024, PORT-ADV-003, PORT-STATE-008."""
    inputs, expected = _post_case(geometry, defect)
    before = {k: v.clone() for k, v in inputs.items()}
    status = _symbol(advance, "_outer_post_tensor_predicates")(**inputs, **_post_constants())
    assert status.dtype == torch.int32 and status.tolist() == [expected]
    _assert_inputs_untouched(inputs, before)
    status.fill_(0)
    _assert_inputs_untouched(inputs, before)


def test_tensor_key_excludes_values_and_absolute_storage_but_retains_alias_partition():
    """@spec PORT-PERF-015: metadata keys cannot depend on slot/lease/value identity."""
    key = _symbol(_fusion_module(), "tensor_metadata_key")
    a, b = torch.zeros(4), torch.ones(4)
    distinct = key((a, b))
    assert distinct == key((torch.ones(4), torch.full((4,), 7.0)))
    assert distinct != key((a, a))
    assert key((a, a)) == key((b, b))
    storage = torch.arange(8.0)
    assert distinct != key((storage[:4], storage[2:6]))
    assert key((storage[:4], storage[2:6])) != key((storage[:4], storage[4:]))
    empty = torch.empty(0)
    assert key((empty, empty)) != key((torch.empty(0), torch.empty(0)))
    assert distinct != key((a.double(), b.double()))


@pytest.mark.parametrize("change", ["default_dtype", "deterministic", "grad", "inference", "autocast"])
def test_context_key_tracks_non_tensor_compiler_guards(change):
    """@spec PORT-PERF-015, PORT-PERF-017: guard context is explicit, not guessed from dtype."""
    context = _symbol(_fusion_module(), "compiler_context_key")
    initial = context()
    if change == "default_dtype":
        old = torch.get_default_dtype()
        try:
            torch.set_default_dtype(torch.float64)
            assert context() != initial
        finally:
            torch.set_default_dtype(old)
    elif change == "deterministic":
        old, warn = torch.are_deterministic_algorithms_enabled(), torch.is_deterministic_algorithms_warn_only_enabled()
        try:
            torch.use_deterministic_algorithms(not old, warn_only=True)
            assert context() != initial
        finally:
            torch.use_deterministic_algorithms(old, warn_only=warn)
    else:
        scope = (
            torch.set_grad_enabled(not torch.is_grad_enabled())
            if change == "grad"
            else torch.inference_mode()
            if change == "inference"
            else torch.autocast("cpu", dtype=torch.bfloat16)
        )
        with scope:
            assert context() != initial
    assert context() == initial


def test_context_key_distinguishes_dispatch_modes_and_determinism_warn_only():
    """@spec PORT-PERF-015, PORT-PERF-017: dispatch and warn-only changes cannot reuse a key."""
    from torch.utils._python_dispatch import TorchDispatchMode

    context = _symbol(_fusion_module(), "compiler_context_key")
    baseline = context()

    class TransparentMode(TorchDispatchMode):
        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            return func(*args, **(kwargs or {}))

    with TransparentMode():
        assert context() != baseline
    old, warn = torch.are_deterministic_algorithms_enabled(), torch.is_deterministic_algorithms_warn_only_enabled()
    try:
        torch.use_deterministic_algorithms(old, warn_only=not warn)
        assert context() != baseline
    finally:
        torch.use_deterministic_algorithms(old, warn_only=warn)
    assert context() == baseline


def test_cpu_region_eligibility_never_enters_compiler(monkeypatch):
    """@spec PORT-PERF-017, PORT-PERF-022: CPU eager dispatch is visible, not fused coverage."""
    binding = _binding()
    monkeypatch.setattr(torch, "compile", lambda *a, **kw: pytest.fail("CPU probe entered compiler"))
    before = deepcopy(binding.receipt())
    assert binding.run_pre(**_pre_inputs())[0].tolist() == [0]
    assert binding.run_post(**_post_inputs(None)).tolist() == [0]
    after = binding.receipt()
    for region in ("pre", "post"):
        assert after["dispatch"][region]["eager_calls"] == before["dispatch"][region]["eager_calls"] + 1
        assert after["dispatch"][region]["fused_calls"] == before["dispatch"][region]["fused_calls"]
    assert after["frames"] == before["frames"]


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires actual CUDA compiler")
def test_cuda_all_families_real_compile_exactness_ownership_and_no_late_frames():
    """@spec PORT-PERF-014, PORT-PERF-015, PORT-PERF-016, PORT-PERF-018, PORT-PERF-020, PORT-PERF-024; actual CUDA, never CPU emulation."""
    binding = _binding()
    with torch.inference_mode():
        binding.warmup(torch.device("cuda"))
        before = deepcopy(binding.receipt())
        assert before["sealed"]  # Compiler inventory alone is not model/profile readiness.
        assert before["prepared"] == list(FAMILIES)
        from tests.model_executor.models.nemotron_asr.test_outer_tensor_startup import _ready_receipt

        expected_declarations = _ready_receipt()["outer_tensor_fusion"]["declared"]
        assert before["declared"] == expected_declarations
        assert before["constants"] == CONSTANTS
        assert before["frames"]["accepted"] == dict.fromkeys(FAMILIES, 1)
        assert before["compiler"]["fullgraph"] is True
        assert before["compiler"]["dynamic"] is False
        assert before["compiler"]["options"] == {"triton.cudagraphs": False}
        retained = binding.run_pre(**_pre_inputs("cuda"))
        snapshots = [tensor.clone() for tensor in retained]
        # The exact same independent corruption cases run on CPU and real CUDA.
        for defect in PRE_DEFECTS:
            inputs, expected = _pre_case(defect, "cuda")
            before_inputs = {k: v.clone() for k, v in inputs.items()}
            result = binding.run_pre(**inputs)
            assert [t.tolist() for t in result] == [[expected], [12], [advance.ROLE_CHUNK], [False]]
            _assert_inputs_untouched(inputs, before_inputs)
        for geometry in (None, 0, 1, 2, 3, 4):
            for defect in POST_DEFECTS:
                inputs, expected = _post_case(geometry, defect, "cuda")
                before_inputs = {k: v.clone() for k, v in inputs.items()}
                result = binding.run_post(**inputs)
                assert result.tolist() == [expected]
                _assert_inputs_untouched(inputs, before_inputs)
        for got, expected in zip(retained, snapshots, strict=True):
            assert torch.equal(got, expected)
        after = binding.receipt()
        assert after["frames"] == before["frames"]
        assert after["dispatch"]["pre"]["fused_calls"] > before["dispatch"]["pre"]["fused_calls"]
        assert after["dispatch"]["post"]["fused_calls"] == before["dispatch"]["post"]["fused_calls"] + 6 * len(
            POST_DEFECTS
        )
        for region in ("pre", "post"):
            assert after["dispatch"][region]["eager_calls"] == before["dispatch"][region]["eager_calls"]


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires actual CUDA compiler")
def test_cuda_valid_alias_miss_is_eager_before_compiler_and_preserves_status():
    """@spec PORT-PERF-015, PORT-PERF-017, PORT-PERF-024: same-object/overlap misses cannot surprise-JIT."""
    binding = _binding()
    with torch.inference_mode():
        binding.warmup(torch.device("cuda"))
        for overlap in (False, True):
            inputs = _post_inputs(None, "cuda")
            if overlap:
                storage = torch.zeros(142, dtype=torch.int32, device="cuda")
                inputs["baseline_queue"] = storage[:141].reshape(1, 141)
                inputs["proposed_queue"] = storage[1:].reshape(1, 141)
            else:
                inputs["proposed_queue"] = inputs["baseline_queue"]
            before = deepcopy(binding.receipt())
            assert binding.run_post(**inputs).tolist() == [0]
            after = binding.receipt()
            assert after["frames"] == before["frames"]
            assert after["dispatch"]["post"]["eager_calls"] == before["dispatch"]["post"]["eager_calls"] + 1
            assert after["dispatch"]["post"]["fused_calls"] == before["dispatch"]["post"]["fused_calls"]


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires actual CUDA compiler")
@pytest.mark.parametrize("unsupported", ["population", "noncontiguous", "endpoint-absent", "result-width"])
def test_cuda_valid_unsupported_metadata_is_eager_before_compiler(unsupported):
    """@spec PORT-PERF-015, PORT-PERF-017, PORT-PERF-024: finite-region eligibility."""
    binding = _binding()
    with torch.inference_mode():
        binding.warmup(torch.device("cuda"))
        pre, post = _pre_inputs("cuda"), _post_inputs(0 if unsupported == "result-width" else None, "cuda")
        options = {}
        rows = 1
        if unsupported == "population":
            rows = 2
            pre = {name: t.repeat((2,) + (1,) * (t.dim() - 1)) for name, t in pre.items()}
            post = {name: t.repeat((2,) + (1,) * (t.dim() - 1)) for name, t in post.items()}
        elif unsupported == "noncontiguous":
            pre["queue"] = torch.zeros(1, 282, dtype=torch.int32, device="cuda")[:, ::2]
            post["proposed_queue"] = torch.zeros(1, 282, dtype=torch.int32, device="cuda")[:, ::2]
        elif unsupported == "endpoint-absent":
            options["endpoint_enabled"] = False
        else:
            post["token_ids"] = torch.zeros(1, 12, dtype=torch.int32, device="cuda")
        before = deepcopy(binding.receipt())
        assert binding.run_pre(**pre, **options)[0].tolist() == [0] * rows
        assert binding.run_post(**post, **options).tolist() == [0] * rows
        after = binding.receipt()
        assert after["frames"] == before["frames"]
        for region in ("pre", "post"):
            selected = "fused_calls" if unsupported == "result-width" and region == "pre" else "eager_calls"
            other = "eager_calls" if selected == "fused_calls" else "fused_calls"
            assert after["dispatch"][region][selected] == before["dispatch"][region][selected] + 1
            assert after["dispatch"][region][other] == before["dispatch"][region][other]


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires actual CUDA compiler")
@pytest.mark.parametrize("defect", ["zero-frame", "second-frame", "runtime-error"])
def test_cuda_compiler_frame_budget_failure_restores_settings_and_never_retries(monkeypatch, defect):
    """@spec PORT-PERF-016, PORT-PERF-019, PORT-PERF-020, PORT-PERF-021, PORT-STATE-008: injected compiler failures."""
    from vllm_omni.model_executor.models.nemotron_asr.encoder_execution import _COMPILER_BUDGET_LOCK

    binding = _binding()
    real_compile = torch.compile
    fail_runtime = False
    invocations = []
    compile_entries = []
    injected = RuntimeError("selected compiled region failed")
    config = torch._dynamo.config
    config_names = (
        "cache_size_limit",
        "accumulated_cache_size_limit",
        "suppress_errors",
        "error_on_recompile",
        "fail_on_recompile_limit_hit",
    )
    original_settings = {name: getattr(config, name) for name in config_names}

    def controlled_compile(fn, *args, **kwargs):
        compile_entries.append(True)
        assert _COMPILER_BUDGET_LOCK._is_owned(), "compile setup must hold the shared lock"
        assert kwargs["fullgraph"] is True and kwargs["dynamic"] is False
        assert kwargs["options"] == {"triton.cudagraphs": False}
        if defect == "zero-frame":
            return fn  # A compiler-shaped eager callable must never satisfy preparation.
        backend = kwargs["backend"]
        assert callable(backend), "actual FX frame counts require the bound counted backend"

        def observed_backend(graph, inputs, **options):
            assert _COMPILER_BUDGET_LOCK._is_owned()
            assert config.fail_on_recompile_limit_hit and not config.suppress_errors
            compiled = backend(graph, inputs, **options)
            if defect == "second-frame":
                backend(graph, inputs, **options)
            return compiled

        compiled = real_compile(fn, *args, **dict(kwargs, backend=observed_backend))

        def invoke(*values, **named):
            assert _COMPILER_BUDGET_LOCK._is_owned(), "sealed config scope must use the same lock"
            invocations.append(True)
            if fail_runtime:
                raise injected
            return compiled(*values, **named)

        return invoke

    monkeypatch.setattr(torch, "compile", controlled_compile)
    with torch.inference_mode():
        if defect != "runtime-error":
            with pytest.raises(RuntimeError, match="one accepted frame|second frame exceeds family budget"):
                binding.warmup(torch.device("cuda"))
            assert not binding.receipt()["ready"]
            attempts, entries = len(invocations), len(compile_entries)
            assert entries > 0
            with pytest.raises(RuntimeError, match="preparation failed; fresh worker required"):
                binding.warmup(torch.device("cuda"))
            assert len(invocations) == attempts
            assert len(compile_entries) == entries  # Also detects zero-frame retries returning plain fn.
        else:
            binding.warmup(torch.device("cuda"))
            before = deepcopy(binding.receipt())
            fail_runtime = True
            count = len(invocations)
            with pytest.raises(RuntimeError) as caught:
                binding.run_pre(**_pre_inputs("cuda"))
            assert caught.value is injected
            assert len(invocations) == count + 1
            after = binding.receipt()
            assert after["dispatch"]["pre"]["eager_calls"] == before["dispatch"]["pre"]["eager_calls"]
            for name in ("arm", "profile", "constants", "compiler", "declared", "prepared"):
                assert after[name] == before[name]
    assert {name: getattr(config, name) for name in config_names} == original_settings


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires actual CUDA compiler")
def test_cuda_receipt_is_detached_immutable_identity_with_live_monotone_counters():
    """@spec PORT-PERF-020, PORT-PERF-022: repeated preparation never resets observable evidence."""
    binding = _binding()
    with torch.inference_mode():
        binding.warmup(torch.device("cuda"))
        initial = deepcopy(binding.receipt())
        returned = binding.receipt()
        returned["constants"]["blank_id"] = -1
        returned["declared"].clear()
        assert binding.receipt() == initial
        binding.run_pre(**_pre_inputs("cuda"))
        binding.warmup(torch.device("cuda"))
        current = binding.receipt()
        for field in ("arm", "profile", "constants", "compiler", "context", "declared", "prepared"):
            assert current[field] == initial[field]
        assert current["frames"] == initial["frames"]
        assert current["dispatch"]["pre"]["fused_calls"] == initial["dispatch"]["pre"]["fused_calls"] + 1


def test_real_dynamo_two_bindings_have_independent_helper_cache_budgets(monkeypatch):
    """@spec PORT-PERF-016, PORT-PERF-020: CPU tracing pins actual binding lifetime.

    Only canonical example allocation is routed to CPU and backend codegen is
    replaced by graph.forward. The real warmup, counted backend, clone use and
    seal run unchanged. This does not establish CUDA correctness or readiness.
    """
    from inspect import signature

    import torch._inductor.compile_fx as compile_fx_module

    module = _fusion_module()
    examples = module.OuterTensorExecution._examples

    def host_examples(self, requested_device):
        assert requested_device.type == "cuda"
        return examples(self, torch.device("cpu"))

    generated = []
    compile_fx_signature = signature(compile_fx_module.compile_fx)

    def host_codegen(*args, **kwargs):
        bound = compile_fx_signature.bind(*args, **kwargs)
        assert bound.arguments["config_patches"] == {"triton.cudagraphs": False}
        graph = bound.arguments["model_"]
        generated.append(graph)
        return graph.forward

    monkeypatch.setattr(module.OuterTensorExecution, "_examples", host_examples)
    monkeypatch.setattr(compile_fx_module, "compile_fx", host_codegen)
    names = (
        "cache_size_limit",
        "accumulated_cache_size_limit",
        "suppress_errors",
        "error_on_recompile",
        "fail_on_recompile_limit_hit",
    )
    original = {name: getattr(torch._dynamo.config, name) for name in names}
    bindings = [_binding(), _binding()]
    with torch.inference_mode():
        for binding in bindings:
            binding.warmup(torch.device("cuda"))
            receipt = binding.receipt()
            assert receipt["prepared"] == list(FAMILIES)
            assert receipt["frames"]["accepted"] == dict.fromkeys(FAMILIES, 1)
            assert receipt["frames"]["backend_attempts"] == 7
            assert receipt["declared"][0]["inputs"][0]["device"] == "cpu"
            assert receipt["sealed"] and not receipt["ready"]
            binding.warmup(torch.device("cuda"))
            assert binding.receipt() == receipt
    assert len(generated) == 14
    assert {name: getattr(torch._dynamo.config, name) for name in names} == original
