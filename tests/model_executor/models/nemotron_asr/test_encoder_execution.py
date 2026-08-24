# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Feature-gated static encoder execution contracts."""

from types import SimpleNamespace
from typing import Any

import pytest
import torch
from torch import nn

from vllm_omni.model_executor.models.nemotron_asr import (
    encoder_execution as encoder_execution_module,
)
from vllm_omni.model_executor.models.nemotron_asr.encoder import (
    RelPositionalEncoding,
)
from vllm_omni.model_executor.models.nemotron_asr.encoder_execution import (
    build_encoder_execution,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class _PreEncode(nn.Module):
    """Small host-shape authority with the production 8x length rule."""

    def output_lengths(self, lengths: torch.Tensor) -> torch.Tensor:
        out = lengths
        for _ in range(3):
            out = torch.div(out, 2, rounding_mode="floor") + 1
        return out.to(torch.int64)


class _CompilerEncoder(nn.Module):
    def __init__(self, *, dtype: torch.dtype) -> None:
        super().__init__()
        self.pre_encode = _PreEncode()
        self.pos_enc = RelPositionalEncoding(8)
        self.weight = nn.Parameter(torch.ones(1, dtype=dtype))
        self.register_buffer("running", torch.zeros(2, 3))


class _CompilerCore(nn.Module):
    def __init__(
        self,
        *,
        activation_dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        self.policy = SimpleNamespace(
            dtype_for=lambda tensor_class: activation_dtype if tensor_class == "activations" else torch.float32
        )
        self.encoder = _CompilerEncoder(dtype=activation_dtype)
        self.lid = nn.Linear(1, 1, bias=False).to(activation_dtype)
        self.predictor = nn.Linear(1, 1, bias=False)


def _compiled_core(
    *,
    activation_dtype: torch.dtype = torch.float32,
) -> _CompilerCore:
    return _CompilerCore(activation_dtype=activation_dtype).eval()


def _compiled_config() -> SimpleNamespace:
    return SimpleNamespace(
        encoder_execution_arm="compiled-static",
        att_context_left=56,
    )


def _transition_args(
    *,
    population: int = 1,
    mel_width: int = 17,
    out_width: int = 3,
    device: torch.device | str = "cpu",
) -> tuple[Any, ...]:
    with torch.inference_mode():
        caches = SimpleNamespace(
            channel=(torch.zeros(population, 2, 3, device=device),),
            time=(torch.zeros(population, 2, 3, device=device),),
            valid=torch.zeros(population, dtype=torch.long, device=device),
            left_context=56,
        )
        return (
            torch.zeros(population, 5, mel_width, device=device),
            caches,
            torch.zeros(population, dtype=torch.long, device=device),
            torch.ones(population, dtype=torch.long, device=device),
            out_width,
            torch.zeros(population, dtype=torch.long, device=device),
        )


def test_compiled_static_declares_capacity_from_shared_geometry_authority(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # @spec PORT-PERF-010
    geometry_shape = getattr(
        encoder_execution_module,
        "encoder_geometry_shape",
        None,
    )
    assert callable(geometry_shape), "PORT-PERF-010 missing encoder_geometry_shape"
    calls: list[int] = []

    def fake_shape(_core: Any, geometry: int) -> Any:
        calls.append(geometry)
        return SimpleNamespace(
            cadence_frames=geometry + 1,
            mel_width=geometry + 10,
            out_width=geometry + 20,
        )

    monkeypatch.setattr(
        encoder_execution_module,
        "encoder_geometry_shape",
        fake_shape,
    )
    monkeypatch.setattr(torch, "compile", lambda fn, **_kwargs: fn)
    execution = build_encoder_execution(
        _compiled_core(),
        _compiled_config(),
        maximum_population=1,
        warmup_geometries=(0, 2, 4),
    )

    assert calls == [0, 2, 4]
    assert execution.t_cap == 4 + 20 + 56
    assert not hasattr(execution, "positional_width")


def test_encoder_geometry_shape_reads_the_canonical_manifest(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # @spec PORT-PERF-010
    from vllm_omni.model_executor.models.nemotron_asr import manifests

    monkeypatch.setattr(manifests, "CADENCES", {"probe": (56, 2)})
    shape = encoder_execution_module.encoder_geometry_shape(
        _compiled_core(),
        0,
    )

    assert shape.cadence_frames == 24
    assert shape.mel_width == 33
    assert shape.out_width == int(_PreEncode().output_lengths(torch.tensor([33]))[0])


def test_encoder_geometry_shape_stays_on_host_under_non_cpu_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # @spec PORT-PERF-010
    from vllm_omni.model_executor.models.nemotron_asr import manifests

    monkeypatch.setattr(manifests, "CADENCES", {"probe": (56, 2)})
    core = _compiled_core()

    with torch.device("meta"):
        shape = encoder_execution_module.encoder_geometry_shape(core, 0)

    assert shape.mel_width == 33
    assert shape.out_width == int(_PreEncode().output_lengths(torch.tensor([33]))[0])


def test_compiled_static_preprofile_materializes_before_compiler_invocation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # @spec PORT-PERF-009, PORT-PERF-010
    core = _compiled_core(activation_dtype=torch.bfloat16)
    compiler_calls = 0
    positional_ids: list[int] = []

    def fake_compile(fn: Any, **_kwargs: Any) -> Any:
        def compiled(*args: Any) -> Any:
            nonlocal compiler_calls
            compiler_calls += 1
            assert torch.is_inference_mode_enabled()
            assert core.encoder.pos_enc.pe.shape == (
                1,
                2 * execution.t_cap - 1,
                8,
            )
            assert args[0].dtype == torch.float32
            assert core.encoder.weight.dtype == torch.bfloat16
            assert core.encoder.pos_enc.pe.dtype == torch.bfloat16
            assert core.encoder.pos_enc.pe.device == args[0].device
            assert args[0].layout == torch.strided
            dispatch_keys = str(torch._C._dispatch_keys(args[0]))
            assert "Autograd" not in dispatch_keys
            assert "ADInplaceOrView" not in dispatch_keys
            positional_ids.append(id(core.encoder.pos_enc.pe))
            return fn(*args)

        return compiled

    monkeypatch.setattr(torch, "compile", fake_compile)
    monkeypatch.setattr(
        encoder_execution_module,
        "execute_encoder_transition",
        lambda *_args: (torch.zeros(1), torch.zeros(1)),
    )
    execution = build_encoder_execution(
        core,
        _compiled_config(),
        maximum_population=1,
        warmup_geometries=(0,),
    )
    args = _transition_args(out_width=execution.t_cap - 56)

    execution.profile_cell(
        geometry=0,
        population=1,
        invoke=lambda: execution.transition(*args),
    )
    execution.warmup_domain(
        expected_cells=((0, 1),),
        invoke=lambda _geometry, _population: execution.transition(*args),
    )

    assert compiler_calls == 2
    assert len(set(positional_ids)) == 1


def test_compiled_static_rejects_finalized_parameter_dtype_mismatch_before_compiler(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # @spec PORT-PERF-010
    compiler_calls = 0

    def fake_compile(fn: Any, **_kwargs: Any) -> Any:
        def compiled(*args: Any) -> Any:
            nonlocal compiler_calls
            compiler_calls += 1
            return fn(*args)

        return compiled

    core = _compiled_core(activation_dtype=torch.bfloat16)
    core.lid.to(torch.float32)
    monkeypatch.setattr(torch, "compile", fake_compile)
    execution = build_encoder_execution(
        core,
        _compiled_config(),
        maximum_population=1,
        warmup_geometries=(0,),
    )

    with pytest.raises(
        ValueError,
        match="parameter dtype differs from activation policy",
    ):
        execution.profile_cell(
            geometry=0,
            population=1,
            invoke=lambda: execution.transition(*_transition_args()),
        )

    assert compiler_calls == 0
    assert not execution.ready


def test_compiled_static_undersized_capacity_fails_before_compiler(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # @spec PORT-PERF-010
    geometry_shape = getattr(
        encoder_execution_module,
        "encoder_geometry_shape",
        None,
    )
    assert callable(geometry_shape), "PORT-PERF-010 missing encoder_geometry_shape"
    compiler_calls = 0

    def fake_shape(_core: Any, _geometry: int) -> Any:
        return SimpleNamespace(
            cadence_frames=1,
            mel_width=8,
            out_width=7,
        )

    def fake_compile(fn: Any, **_kwargs: Any) -> Any:
        def compiled(*args: Any) -> Any:
            nonlocal compiler_calls
            compiler_calls += 1
            return fn(*args)

        return compiled

    monkeypatch.setattr(
        encoder_execution_module,
        "encoder_geometry_shape",
        fake_shape,
    )
    monkeypatch.setattr(torch, "compile", fake_compile)
    execution = build_encoder_execution(
        _compiled_core(),
        _compiled_config(),
        maximum_population=1,
        warmup_geometries=(0,),
    )

    with pytest.raises(
        ValueError,
        match=r"encoder\.pos_enc\.pe.*capacity",
    ):
        execution.profile_cell(
            geometry=0,
            population=1,
            invoke=lambda: execution.transition(
                *_transition_args(out_width=8),
            ),
        )
    assert compiler_calls == 0


def test_compiled_static_rechecks_capacity_after_materialization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # @spec PORT-PERF-010
    compiler_calls = 0

    def fake_compile(fn: Any, **_kwargs: Any) -> Any:
        def compiled(*args: Any) -> Any:
            nonlocal compiler_calls
            compiler_calls += 1
            return fn(*args)

        return compiled

    monkeypatch.setattr(torch, "compile", fake_compile)
    monkeypatch.setattr(
        encoder_execution_module,
        "execute_encoder_transition",
        lambda *_args: (torch.zeros(1), torch.zeros(1)),
    )
    execution = build_encoder_execution(
        _compiled_core(),
        _compiled_config(),
        maximum_population=1,
        warmup_geometries=(0,),
    )
    execution.profile_cell(
        geometry=0,
        population=1,
        invoke=lambda: execution.transition(*_transition_args()),
    )

    with pytest.raises(ValueError, match=r"encoder\.pos_enc\.pe.*capacity"):
        execution.warmup_domain(
            expected_cells=((0, 1),),
            invoke=lambda _geometry, _population: execution.transition(
                *_transition_args(out_width=execution.t_cap),
            ),
        )

    assert compiler_calls == 1
    assert not execution.ready


def test_compiled_static_rejects_mislabeled_geometry_before_compiler(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # @spec PORT-PERF-009, PORT-PERF-010
    compiler_calls = 0

    def fake_compile(fn: Any, **_kwargs: Any) -> Any:
        def compiled(*args: Any) -> Any:
            nonlocal compiler_calls
            compiler_calls += 1
            return fn(*args)

        return compiled

    monkeypatch.setattr(torch, "compile", fake_compile)
    execution = build_encoder_execution(
        _compiled_core(),
        _compiled_config(),
        maximum_population=1,
        warmup_geometries=(0, 1),
    )

    with pytest.raises(ValueError, match=r"declared geometry 1"):
        execution.profile_cell(
            geometry=1,
            population=1,
            invoke=lambda: execution.transition(*_transition_args()),
        )

    assert compiler_calls == 0
    assert not execution.ready


@pytest.mark.parametrize(
    ("mutation", "expected_name"),
    (
        ("device", r"out_offsets.*runner device"),
        ("all-wrong-device", r"mel.*runner device"),
        ("layout", r"caches\.channel\[0\].*strided layout"),
        ("mel-dtype", r"mel.*torch\.float32"),
        ("cache-dtype", r"caches\.channel\[0\].*torch\.float32"),
        ("integer-dtype", r"prompt_index.*torch\.int64"),
        ("dispatch-keys", r"caches\.time\[0\].*inference mode"),
    ),
)
def test_compiled_static_validates_complete_transition_tuple_before_compiler(
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
    expected_name: str,
) -> None:
    # @spec PORT-PERF-010
    compiler_calls = 0

    def fake_compile(fn: Any, **_kwargs: Any) -> Any:
        def compiled(*args: Any) -> Any:
            nonlocal compiler_calls
            compiler_calls += 1
            return fn(*args)

        return compiled

    monkeypatch.setattr(torch, "compile", fake_compile)
    execution = build_encoder_execution(
        _compiled_core(),
        _compiled_config(),
        maximum_population=1,
        warmup_geometries=(0,),
    )
    args = list(_transition_args())
    caches = args[1]
    if mutation == "device":
        args[2] = torch.zeros(1, dtype=torch.long, device="meta")
    elif mutation == "all-wrong-device":
        args = list(_transition_args(device="meta"))
        caches = args[1]
    elif mutation == "layout":
        with torch.inference_mode():
            caches.channel = (torch.zeros(1, 2, 3).to_sparse(),)
    elif mutation == "mel-dtype":
        with torch.inference_mode():
            args[0] = torch.zeros(1, 5, 17, dtype=torch.bfloat16)
    elif mutation == "cache-dtype":
        with torch.inference_mode():
            caches.channel = (torch.zeros(1, 2, 3, dtype=torch.bfloat16),)
    elif mutation == "integer-dtype":
        with torch.inference_mode():
            args[5] = torch.zeros(1, dtype=torch.int32)
    elif mutation == "dispatch-keys":
        caches.time = (torch.zeros(1, 2, 3),)
    else:  # pragma: no cover - the parametrization is closed.
        raise AssertionError(mutation)

    with pytest.raises(ValueError, match=expected_name):
        execution.profile_cell(
            geometry=0,
            population=1,
            invoke=lambda: execution.transition(*args),
        )

    assert compiler_calls == 0
    assert not execution.ready


def test_compiled_static_uses_warmed_signature_after_seal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # @spec PORT-PERF-009, PORT-PERF-010
    monkeypatch.setattr(torch, "compile", lambda fn, **_kwargs: fn)
    monkeypatch.setattr(
        encoder_execution_module,
        "execute_encoder_transition",
        lambda *_args: (torch.zeros(1), torch.zeros(1)),
    )
    execution = build_encoder_execution(
        _compiled_core(),
        _compiled_config(),
        maximum_population=1,
        warmup_geometries=(0,),
    )
    args = _transition_args()
    execution.warmup_domain(
        expected_cells=((0, 1),),
        invoke=lambda _geometry, _population: execution.transition(*args),
    )

    monkeypatch.setattr(
        encoder_execution_module,
        "_validate_transition_tensor_contract",
        lambda *_args, **_kwargs: pytest.fail("sealed serving must use its warmed tensor signature"),
    )

    execution.transition(*args)


@pytest.mark.parametrize(
    ("mutation", "expected_name"),
    (
        ("add-parameter", r"encoder\.late_weight"),
        ("remove-parameter", r"encoder\.weight"),
        ("add-buffer", r"encoder\.late_buffer"),
        ("remove-buffer", r"encoder\.running"),
        ("add-module", r"encoder\.late_module"),
        ("remove-module", r"encoder\.pre_encode"),
        ("replace-buffer", r"encoder\.running"),
        ("reshape-buffer", r"encoder\.running"),
        ("restride-buffer", r"encoder\.running"),
        ("relayout-buffer", r"encoder\.running"),
        ("recast-parameter", r"encoder\.weight"),
        ("requires-grad", r"encoder\.weight"),
        ("move-buffer", r"encoder\.running"),
        ("dispatch-keys", r"encoder\.running"),
        ("training", r"encoder.*training"),
    ),
)
def test_compiled_static_profile_detects_compiler_visible_model_mutation(
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
    expected_name: str,
) -> None:
    # @spec PORT-PERF-010
    core = _compiled_core()

    def mutate() -> None:
        if mutation == "add-parameter":
            core.encoder.register_parameter(
                "late_weight",
                nn.Parameter(torch.ones(1)),
            )
        elif mutation == "remove-parameter":
            del core.encoder.weight
        elif mutation == "add-buffer":
            core.encoder.register_buffer("late_buffer", torch.ones(1))
        elif mutation == "remove-buffer":
            del core.encoder.running
        elif mutation == "add-module":
            core.encoder.add_module("late_module", nn.Identity())
        elif mutation == "remove-module":
            del core.encoder.pre_encode
        elif mutation == "replace-buffer":
            core.encoder.running = torch.ones(2, 3)
        elif mutation == "reshape-buffer":
            core.encoder.running.data = torch.ones(3, 2)
        elif mutation == "restride-buffer":
            core.encoder.running.data = torch.ones(3, 2).t()
            assert core.encoder.running.stride() == (1, 2)
        elif mutation == "relayout-buffer":
            core.encoder.running = torch.ones(2, 3).to_sparse()
            assert core.encoder.running.layout == torch.sparse_coo
        elif mutation == "recast-parameter":
            core.encoder.weight.data = core.encoder.weight.data.to(torch.float64)
        elif mutation == "requires-grad":
            core.encoder.weight.requires_grad_(False)
        elif mutation == "move-buffer":
            core.encoder.running = core.encoder.running.to("meta")
        elif mutation == "dispatch-keys":
            original_id = id(core.encoder.running)
            with torch.inference_mode():
                inference_tensor = torch.ones_like(core.encoder.running)
            core.encoder.running.data = inference_tensor
            assert id(core.encoder.running) == original_id
        elif mutation == "training":
            core.encoder.train()
        else:  # pragma: no cover - the parametrization is closed.
            raise AssertionError(mutation)

    def fake_compile(fn: Any, **_kwargs: Any) -> Any:
        def compiled(*args: Any) -> Any:
            result = fn(*args)
            mutate()
            return result

        return compiled

    monkeypatch.setattr(torch, "compile", fake_compile)
    monkeypatch.setattr(
        encoder_execution_module,
        "execute_encoder_transition",
        lambda *_args: (torch.zeros(1), torch.zeros(1)),
    )
    execution = build_encoder_execution(
        core,
        _compiled_config(),
        maximum_population=1,
        warmup_geometries=(0,),
    )

    with pytest.raises(ValueError, match=expected_name):
        execution.profile_cell(
            geometry=0,
            population=1,
            invoke=lambda: execution.transition(
                *_transition_args(),
            ),
        )


def test_compiled_static_rechecks_model_state_before_sealing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # @spec PORT-PERF-010
    core = _compiled_core()
    monkeypatch.setattr(torch, "compile", lambda fn, **_kwargs: fn)
    monkeypatch.setattr(
        encoder_execution_module,
        "execute_encoder_transition",
        lambda *_args: (torch.zeros(1), torch.zeros(1)),
    )
    execution = build_encoder_execution(
        core,
        _compiled_config(),
        maximum_population=1,
        warmup_geometries=(0,),
    )
    args = _transition_args()
    execution.profile_cell(
        geometry=0,
        population=1,
        invoke=lambda: execution.transition(*args),
    )
    core.encoder.running = torch.ones(2, 3)

    with pytest.raises(ValueError, match=r"encoder\.running"):
        execution.warmup_domain(
            expected_cells=((0, 1),),
            invoke=lambda _geometry, _population: execution.transition(
                *args,
            ),
        )
    assert not execution.ready


def test_compiled_static_ignores_unrelated_model_sibling_mutation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # @spec PORT-PERF-010
    core = _compiled_core()
    monkeypatch.setattr(torch, "compile", lambda fn, **_kwargs: fn)
    monkeypatch.setattr(
        encoder_execution_module,
        "execute_encoder_transition",
        lambda *_args: (torch.zeros(1), torch.zeros(1)),
    )
    execution = build_encoder_execution(
        core,
        _compiled_config(),
        maximum_population=1,
        warmup_geometries=(0,),
    )
    args = _transition_args()
    execution.profile_cell(
        geometry=0,
        population=1,
        invoke=lambda: execution.transition(*args),
    )
    core.predictor.weight.data = core.predictor.weight.data.to(torch.float64)

    execution.warmup_domain(
        expected_cells=((0, 1),),
        invoke=lambda _geometry, _population: execution.transition(*args),
    )

    assert execution.ready


def test_absent_encoder_gate_is_the_unchanged_eager_baseline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # @spec PORT-PERF-009
    def forbidden_compile(*_args: Any, **_kwargs: Any) -> Any:
        pytest.fail("the absent baseline must not initialize torch.compile")

    monkeypatch.setattr(torch, "compile", forbidden_compile)
    resolved = build_encoder_execution(
        SimpleNamespace(),
        SimpleNamespace(),
        maximum_population=4,
        warmup_geometries=(0,),
    )
    assert resolved.arm == "eager"
    assert resolved.ready_receipt() == {
        "arm": "eager",
        "ready": True,
        "warmup_cells": [],
        "warmup_geometries": [],
        "warmup_populations": [],
    }


def test_explicit_eager_encoder_gate_does_not_compile(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # @spec PORT-PERF-009
    def forbidden_compile(*_args: Any, **_kwargs: Any) -> Any:
        pytest.fail("the explicit eager arm must not initialize torch.compile")

    monkeypatch.setattr(torch, "compile", forbidden_compile)
    resolved = build_encoder_execution(
        SimpleNamespace(),
        SimpleNamespace(encoder_execution_arm="eager"),
        maximum_population=4,
        warmup_geometries=(0,),
    )
    assert resolved.arm == "eager"


def test_compiled_static_uses_one_fail_closed_fullgraph_authority(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # @spec PORT-PERF-009
    calls: list[tuple[Any, dict[str, Any]]] = []

    def fake_compile(fn: Any, **kwargs: Any) -> Any:
        calls.append((fn, kwargs))
        return "compiled-transition"

    monkeypatch.setattr(torch, "compile", fake_compile)
    resolved = build_encoder_execution(
        _compiled_core(),
        _compiled_config(),
        maximum_population=4,
        warmup_geometries=(0, 1, 2, 3, 4),
    )
    assert resolved.arm == "compiled-static"
    assert callable(resolved.transition)
    assert len(calls) == 1
    assert resolved.warmup_populations == (1, 2, 3, 4)
    assert calls[0][1] == {
        "fullgraph": True,
        "dynamic": False,
        "options": {"triton.cudagraphs": False},
    }


def test_compiler_initialization_failure_is_not_hidden(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # @spec PORT-PERF-009
    def fail_compile(*_args: Any, **_kwargs: Any) -> Any:
        raise RuntimeError("synthetic compiler failure")

    monkeypatch.setattr(torch, "compile", fail_compile)
    with pytest.raises(RuntimeError, match="synthetic compiler failure"):
        build_encoder_execution(
            _compiled_core(),
            _compiled_config(),
            maximum_population=1,
            warmup_geometries=(0,),
        )


def test_unknown_encoder_execution_arm_fails_closed() -> None:
    # @spec PORT-PERF-009
    with pytest.raises(ValueError, match="unknown encoder_execution_arm"):
        build_encoder_execution(
            SimpleNamespace(),
            SimpleNamespace(encoder_execution_arm="auto-magic"),
            maximum_population=4,
            warmup_geometries=(0,),
        )


def test_compiled_static_derives_complete_population_authority_from_runner_ceiling(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # @spec PORT-PERF-009
    monkeypatch.setattr(torch, "compile", lambda fn, **_kwargs: fn)
    resolved = build_encoder_execution(
        _compiled_core(),
        _compiled_config(),
        maximum_population=4,
        warmup_geometries=(0, 2, 4),
    )
    assert resolved.warmup_populations == (1, 2, 3, 4)
    assert resolved.warmup_geometries == (0, 2, 4)


def test_compiled_static_seals_observed_signatures_and_rejects_lazy_compile(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # @spec PORT-PERF-009
    compiled_calls: list[tuple[int, int]] = []

    def fake_compile(fn: Any, **_kwargs: Any) -> Any:
        def compiled(*args: Any) -> Any:
            compiled_calls.append((int(args[0].shape[0]), int(args[4])))
            return fn(*args)

        return compiled

    monkeypatch.setattr(torch, "compile", fake_compile)
    execution = build_encoder_execution(
        _compiled_core(),
        _compiled_config(),
        maximum_population=2,
        warmup_geometries=(0,),
    )

    def run(population: int, *, mel_width: int = 17) -> None:
        with torch.inference_mode():
            caches = SimpleNamespace(
                channel=(torch.zeros(population, 2, 3),),
                time=(torch.zeros(population, 2, 3),),
                valid=torch.zeros(population, dtype=torch.long),
                left_context=56,
            )
            execution.transition(
                torch.zeros(population, 5, mel_width),
                caches,
                torch.zeros(population, dtype=torch.long),
                torch.ones(population, dtype=torch.long),
                3,
                torch.zeros(population, dtype=torch.long),
            )

    monkeypatch.setattr(
        encoder_execution_module,
        "execute_encoder_transition",
        lambda *_args: (torch.zeros(1), torch.zeros(1)),
    )
    execution.warmup_domain(
        expected_cells=((0, 1), (0, 2)),
        invoke=lambda _geometry, population: run(population),
    )
    run(1)
    assert compiled_calls == [(1, 3), (2, 3), (1, 3)]
    assert execution.ready_receipt() == {
        "arm": "compiled-static",
        "ready": True,
        "warmup_cells": [[0, 1], [0, 2]],
        "warmup_geometries": [0],
        "warmup_populations": [1, 2],
    }

    with pytest.raises(ValueError, match="was not warmed"):
        run(1, mel_width=18)
    assert compiled_calls == [(1, 3), (2, 3), (1, 3)]

    with pytest.raises(ValueError, match="population 3 was not declared"):
        execution.transition(
            torch.zeros(3, 5, 17),
            SimpleNamespace(
                channel=(torch.zeros(3, 2, 3),),
                time=(torch.zeros(3, 2, 3),),
                valid=torch.zeros(3, dtype=torch.long),
                left_context=56,
            ),
            torch.zeros(3, dtype=torch.long),
            torch.ones(3, dtype=torch.long),
            3,
            torch.zeros(3, dtype=torch.long),
        )
    assert compiled_calls == [(1, 3), (2, 3), (1, 3)]


def test_compiled_static_warmup_scopes_cache_budget_to_declared_domain(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # @spec PORT-PERF-009
    observed_limits: list[tuple[int, int]] = []

    def fake_compile(fn: Any, **_kwargs: Any) -> Any:
        def compiled(*args: Any) -> Any:
            observed_limits.append(
                (
                    int(torch._dynamo.config.cache_size_limit),
                    int(torch._dynamo.config.accumulated_cache_size_limit),
                )
            )
            return fn(*args)

        return compiled

    monkeypatch.setattr(torch, "compile", fake_compile)
    monkeypatch.setattr(torch._dynamo.config, "cache_size_limit", 8)
    monkeypatch.setattr(
        torch._dynamo.config,
        "accumulated_cache_size_limit",
        12,
    )
    monkeypatch.setattr(
        encoder_execution_module,
        "execute_encoder_transition",
        lambda *_args: (torch.zeros(1), torch.zeros(1)),
    )
    execution = build_encoder_execution(
        _compiled_core(),
        _compiled_config(),
        maximum_population=4,
        warmup_geometries=(0, 1, 2, 3, 4),
    )
    cells = tuple((geometry, population) for geometry in range(5) for population in range(1, 5))

    def invoke(geometry: int, population: int) -> None:
        shape = execution._geometry_shapes[geometry]
        caches = SimpleNamespace(
            channel=(torch.zeros(population, 2, 3),),
            time=(torch.zeros(population, 2, 3),),
            valid=torch.zeros(population, dtype=torch.long),
            left_context=56,
        )
        execution.transition(
            torch.zeros(population, 5, shape.mel_width),
            caches,
            torch.zeros(population, dtype=torch.long),
            torch.ones(population, dtype=torch.long),
            shape.out_width,
            torch.zeros(population, dtype=torch.long),
        )

    execution.profile_cell(
        geometry=4,
        population=4,
        invoke=lambda: invoke(4, 4),
    )
    assert observed_limits == [(20, 20)]
    assert torch._dynamo.config.cache_size_limit == 8
    assert torch._dynamo.config.accumulated_cache_size_limit == 12

    execution.warmup_domain(expected_cells=cells, invoke=invoke)

    assert observed_limits == [(20, 20)] * 21
    assert torch._dynamo.config.cache_size_limit == 8
    assert torch._dynamo.config.accumulated_cache_size_limit == 12
    assert execution.ready


def test_compiled_static_preseal_invocation_requires_declared_cell_authority(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # @spec PORT-PERF-009
    compiled_calls = 0

    def fake_compile(fn: Any, **_kwargs: Any) -> Any:
        def compiled(*args: Any) -> Any:
            nonlocal compiled_calls
            compiled_calls += 1
            return fn(*args)

        return compiled

    monkeypatch.setattr(torch, "compile", fake_compile)
    execution = build_encoder_execution(
        _compiled_core(),
        _compiled_config(),
        maximum_population=1,
        warmup_geometries=(0,),
    )
    caches = SimpleNamespace(
        channel=(torch.zeros(1, 2, 3),),
        time=(torch.zeros(1, 2, 3),),
        valid=torch.zeros(1, dtype=torch.long),
        left_context=56,
    )

    with pytest.raises(ValueError, match="lacks declared cell authority"):
        execution.transition(
            torch.zeros(1, 5, 17),
            caches,
            torch.zeros(1, dtype=torch.long),
            torch.ones(1, dtype=torch.long),
            3,
            torch.zeros(1, dtype=torch.long),
        )
    with pytest.raises(ValueError, match="geometry 1 was not declared"):
        execution.warmup_cell(
            geometry=1,
            population=1,
            invoke=lambda: None,
        )
    with torch.inference_mode():
        execution._materialize_preprofile_state(out_width=3)
    with pytest.raises(ValueError, match=r"missing=\[\(0, 1\)\]"):
        execution._seal()
    assert compiled_calls == 0


def test_compiled_static_failure_restores_budget_and_discards_authority(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # @spec PORT-PERF-009
    compile_instances: list[dict[str, Any]] = []

    def fake_compile(fn: Any, **_kwargs: Any) -> Any:
        instance = {
            "calls": 0,
            "fail_second": not compile_instances,
        }
        compile_instances.append(instance)

        def compiled(*args: Any) -> Any:
            instance["calls"] += 1
            if instance["fail_second"] and instance["calls"] == 2:
                raise RuntimeError("synthetic specialization failure")
            return fn(*args)

        return compiled

    monkeypatch.setattr(torch, "compile", fake_compile)
    monkeypatch.setattr(torch._dynamo.config, "cache_size_limit", 1)
    monkeypatch.setattr(
        torch._dynamo.config,
        "accumulated_cache_size_limit",
        1,
    )
    execution = build_encoder_execution(
        _compiled_core(),
        _compiled_config(),
        maximum_population=1,
        warmup_geometries=(0, 1),
    )
    monkeypatch.setattr(
        encoder_execution_module,
        "execute_encoder_transition",
        lambda *_args: (torch.zeros(1), torch.zeros(1)),
    )

    def invoke(
        authority: Any,
        geometry: int,
        _population: int,
    ) -> Any:
        shape = authority._geometry_shapes[geometry]
        return authority.transition(
            torch.zeros(1, 5, shape.mel_width),
            SimpleNamespace(
                channel=(torch.zeros(1, 2, 3),),
                time=(torch.zeros(1, 2, 3),),
                valid=torch.zeros(1, dtype=torch.long),
                left_context=56,
            ),
            torch.zeros(1, dtype=torch.long),
            torch.ones(1, dtype=torch.long),
            shape.out_width,
            torch.zeros(1, dtype=torch.long),
        )

    with pytest.raises(RuntimeError, match="synthetic specialization failure"):
        execution.warmup_domain(
            expected_cells=((0, 1), (1, 1)),
            invoke=lambda geometry, population: invoke(
                execution,
                geometry,
                population,
            ),
        )

    assert torch._dynamo.config.cache_size_limit == 1
    assert torch._dynamo.config.accumulated_cache_size_limit == 1
    assert not execution.ready
    assert execution.ready_receipt()["warmup_cells"] == []
    failed_calls = compile_instances[0]["calls"]

    with pytest.raises(ValueError, match=r"failed|discarded|invalid"):
        execution.warmup_domain(
            expected_cells=((0, 1), (1, 1)),
            invoke=lambda geometry, population: invoke(
                execution,
                geometry,
                population,
            ),
        )
    assert compile_instances[0]["calls"] == failed_calls

    fresh_execution = build_encoder_execution(
        _compiled_core(),
        _compiled_config(),
        maximum_population=1,
        warmup_geometries=(0, 1),
    )
    fresh_execution.warmup_domain(
        expected_cells=((0, 1), (1, 1)),
        invoke=lambda geometry, population: invoke(
            fresh_execution,
            geometry,
            population,
        ),
    )
    assert fresh_execution.ready
    assert compile_instances[1]["calls"] == 2
