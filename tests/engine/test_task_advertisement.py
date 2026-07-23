"""Unit tests for declared-capability task advertisement (PORT-CAP-001).

Note: Uses importlib to load the derivation module directly, bypassing the
vllm_omni package __init__ which requires the vllm base package. The
StageRuntimeInfo-backed test is skipped where vllm is unavailable.
"""

import importlib.util
import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

_ENGINE_DIR = Path(__file__).resolve().parents[2] / "vllm_omni" / "engine"


def _load_module(name: str, filepath: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, filepath)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


_ta_mod = _load_module(
    "vllm_omni.engine.task_advertisement",
    _ENGINE_DIR / "task_advertisement.py",
)

derive_supported_tasks = _ta_mod.derive_supported_tasks
StageTaskMetadata = _ta_mod.StageTaskMetadata


@dataclass(frozen=True)
class _Meta:
    """Structural stand-in for StageRuntimeInfo."""

    final_output_type: str | None = None
    declared_tasks: tuple[str, ...] = ()


# @spec PORT-CAP-001
def test_empty_defaults_reproduce_generate_fallback() -> None:
    """No comprehension stage, no audio output, no declaration."""
    assert derive_supported_tasks([], has_comprehension_stage=False) == ("generate",)
    assert derive_supported_tasks(
        [_Meta(), _Meta(final_output_type="text")],
        has_comprehension_stage=False,
    ) == ("generate",)


# @spec PORT-CAP-001
def test_empty_defaults_reproduce_comprehension_and_audio_rules() -> None:
    """The historical generate/speech rules are unchanged by the union."""
    assert derive_supported_tasks(
        [_Meta(final_output_type="text")],
        has_comprehension_stage=True,
    ) == ("generate",)
    assert derive_supported_tasks(
        [_Meta(final_output_type="audio")],
        has_comprehension_stage=False,
    ) == ("speech",)
    assert set(
        derive_supported_tasks(
            [_Meta(), _Meta(final_output_type="audio")],
            has_comprehension_stage=True,
        )
    ) == {"generate", "speech"}


# @spec PORT-CAP-001
def test_declared_task_is_advertised() -> None:
    """A stage-declared task rides through to the advertised set."""
    assert derive_supported_tasks(
        [_Meta(declared_tasks=("transcription",))],
        has_comprehension_stage=False,
    ) == ("transcription",)
    assert set(
        derive_supported_tasks(
            [_Meta(final_output_type="audio", declared_tasks=("transcription",))],
            has_comprehension_stage=True,
        )
    ) == {"generate", "speech", "transcription"}


# @spec PORT-CAP-001
def test_declarations_union_across_stages_without_duplication() -> None:
    """The mechanism is a capability list: any string, unioned, deduplicated."""
    tasks = derive_supported_tasks(
        [
            _Meta(declared_tasks=("alpha", "beta")),
            _Meta(declared_tasks=("beta", "gamma")),
            _Meta(declared_tasks=("generate",)),
        ],
        has_comprehension_stage=True,
    )
    assert sorted(tasks) == ["alpha", "beta", "gamma", "generate"]
    assert len(tasks) == len(set(tasks))


# @spec PORT-CAP-001
def test_stage_runtime_info_satisfies_the_metadata_surface() -> None:
    """The engine's own metadata record carries the default-empty field."""
    pytest.importorskip("vllm")
    from vllm_omni.engine.stage_runtime import StageRuntimeInfo

    info = StageRuntimeInfo(
        final_output=True,
        final_output_type=None,
        stage_type="llm",
    )
    assert info.declared_tasks == ()
    assert isinstance(info, StageTaskMetadata)
    assert derive_supported_tasks([info], has_comprehension_stage=True) == ("generate",)

    declaring = StageRuntimeInfo(
        final_output=True,
        final_output_type=None,
        stage_type="llm",
        declared_tasks=("transcription",),
    )
    assert set(derive_supported_tasks([declaring], has_comprehension_stage=True)) == {"generate", "transcription"}


# @spec PORT-CAP-001
def test_producer_extracts_declared_tasks_from_stage_config() -> None:
    """The producer side: ``extract_stage_metadata`` must read ``declared_tasks``
    off a stage config and carry it through to the advertised set.

    Before this test, the consumer side (``StageRuntimeInfo`` /
    ``derive_supported_tasks``) was pinned but nothing populated
    ``declared_tasks`` on the way in, so the union was unreachable in
    production. This exercises the real ``extract_stage_metadata`` function
    end-to-end into ``derive_supported_tasks``.

    Gated on vllm: ``vllm_omni.engine.stage_init_utils`` imports
    ``vllm.sampling_params``, ``vllm.tokenizers``, ``vllm.v1.engine.input_processor``,
    and ``vllm.v1.executor`` unconditionally at module scope, so it cannot be
    loaded via the direct-module-loader pattern used for
    ``task_advertisement.py`` above (which is deliberately kept vllm-free).
    """
    pytest.importorskip("vllm")
    from vllm_omni.engine.stage_init_utils import extract_stage_metadata

    declaring_config = SimpleNamespace(
        stage_id=0,
        stage_type="llm",
        engine_args=SimpleNamespace(),
        declared_tasks=("transcription",),
    )
    metadata = extract_stage_metadata(declaring_config)
    assert metadata.declared_tasks == ("transcription",)
    assert isinstance(metadata, StageTaskMetadata)
    assert derive_supported_tasks([metadata], has_comprehension_stage=False) == ("transcription",)

    silent_config = SimpleNamespace(
        stage_id=0,
        stage_type="llm",
        engine_args=SimpleNamespace(),
    )
    silent_metadata = extract_stage_metadata(silent_config)
    assert silent_metadata.declared_tasks == ()
    assert derive_supported_tasks([silent_metadata], has_comprehension_stage=False) == ("generate",)
