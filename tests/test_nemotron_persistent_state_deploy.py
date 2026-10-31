# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Deploy-profile contracts for Nemotron persistent-state serving."""

from __future__ import annotations

import importlib.resources
import re
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, NoReturn

import pytest

import vllm_omni.config.config_factory as config_factory
from vllm_omni.config.config_factory import StageConfigFactory
from vllm_omni.config.stage_config import PipelineConfig, load_deploy_config

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


_ROOT = Path(__file__).parent.parent
_HARD_CAP_PERSISTENT_STATE = """      persistent_state_admission_policy: hard_cap
      persistent_state_admission_retry_floor_ms: 100"""
_PROFILE_ONLY_KEYS = frozenset(
    {
        "persistent_state_service_profile_trailing_rounds",
        "persistent_state_startup_priming_round_timeout_s",
        "persistent_state_startup_priming_timeout_s",
        "persistent_state_service_profile_derating_factor",
    }
)


def _fail(message: str) -> NoReturn:
    pytest.fail(message, pytrace=False)
    raise AssertionError(message)


def _pipeline() -> PipelineConfig:
    pipeline = StageConfigFactory.resolve_pipeline_config("nemotron_asr")
    if pipeline is None:
        _fail("ENV-MIG-011 missing registered Nemotron pipeline")
    return pipeline


def _persistent_state(stage: Any) -> dict[str, Any]:
    """Return the typed persistent-state deployment block or fail explicitly."""
    value = getattr(stage, "persistent_state", None)
    if value is None:
        _fail("ENV-MIG-011 missing typed StageDeployConfig.persistent_state")
    if not is_dataclass(value) or type(value).__name__ != "PersistentStateDeployConfig":
        _fail("ENV-MIG-011 persistent_state is not PersistentStateDeployConfig")
    return {key: item for key, item in asdict(value).items() if item is not None}


def _resolved_stage(
    deploy_path: Path,
    cli_overrides: dict[str, Any] | None = None,
) -> Any:
    stages, _ = StageConfigFactory._create_from_registry(
        "nemotron_asr",
        _pipeline(),
        cli_overrides or {},
        deploy_config_path=str(deploy_path),
    )
    if len(stages) != 1:
        _fail("ENV-MIG-011 expected exactly one Nemotron serving stage")
    return stages[0]


def _resolved_default_stage() -> Any:
    stages, _ = StageConfigFactory._create_from_registry(
        "nemotron_asr",
        _pipeline(),
        {},
    )
    if len(stages) != 1:
        _fail("ENV-MIG-011 expected exactly one default Nemotron stage")
    return stages[0]


def _runtime_config_for(stage: Any) -> Any:
    """Resolve a lowered stage through the engine's runtime envelope resolver."""
    from types import SimpleNamespace

    from vllm_omni.engine.persistent_state_config import PersistentStateRuntimeConfig
    from vllm_omni.entrypoints.stage_utils import _to_dict
    from vllm_omni.model_executor.models.nemotron_asr.identity import ARCHITECTURE

    # Same conversion the engine applies to stage engine_args before boot.
    engine_args = _to_dict(stage.to_omegaconf().engine_args)
    vllm_config = SimpleNamespace(
        additional_config=engine_args["additional_config"],
        scheduler_config=SimpleNamespace(max_num_seqs=int(engine_args["max_num_seqs"])),
        model_config=SimpleNamespace(architectures=(ARCHITECTURE,)),
    )
    return PersistentStateRuntimeConfig.from_vllm_config(vllm_config)


def _write_deploy(
    tmp_path: Path,
    persistent_state: str = _HARD_CAP_PERSISTENT_STATE,
    extra: str = "",
) -> Path:
    path = tmp_path / "nemotron.yaml"
    path.write_text(
        "\n".join(
            (
                "pipeline: nemotron_asr",
                "async_chunk: false",
                "stages:",
                "  - stage_id: 0",
                "    max_num_seqs: 8",
                "    persistent_state:",
                *persistent_state.splitlines(),
                extra,
            )
        ),
        encoding="utf-8",
    )
    return path


class TestNemotronPersistentStateDeploy:
    """@spec ENV-MIG-011/012: typed deploy and package startup contracts."""

    def test_persistent_state_is_reserved_typed_stage_config(self, tmp_path: Path) -> None:
        """@spec ENV-MIG-011: persistent state is never an engine extra."""
        deploy_path = _write_deploy(
            tmp_path,
            _HARD_CAP_PERSISTENT_STATE,
        )

        deploy = load_deploy_config(deploy_path)
        state = _persistent_state(deploy.stages[0])

        assert state["persistent_state_admission_policy"] == "hard_cap"
        assert not _PROFILE_ONLY_KEYS & state.keys()
        assert "persistent_state" not in deploy.stages[0].engine_extras

    def test_lowering_materializes_canonical_additional_config_before_omegaconf(self, tmp_path: Path) -> None:
        """@spec ENV-MIG-011: typed state lowers before legacy OmegaConf conversion."""
        deploy_path = _write_deploy(
            tmp_path,
            _HARD_CAP_PERSISTENT_STATE,
        )

        stage = _resolved_stage(deploy_path)
        additional = stage.yaml_engine_args.get("additional_config")
        if not isinstance(additional, dict):
            _fail("ENV-MIG-011 missing canonical additional_config before to_omegaconf")
        assert additional["persistent_state_admission_policy"] == "hard_cap"
        assert not _PROFILE_ONLY_KEYS & additional.keys()

        resolved = stage.to_omegaconf()
        assert resolved.engine_args.additional_config == additional

    def test_explicit_additional_config_key_merges_over_typed_state(self, tmp_path: Path) -> None:
        """@spec ENV-MIG-011: explicit same-key values win without dropping peers."""
        deploy_path = _write_deploy(
            tmp_path,
            _HARD_CAP_PERSISTENT_STATE,
        )
        stage = _resolved_stage(
            deploy_path,
            {
                "additional_config": {
                    "persistent_state_admission_retry_floor_ms": 125,
                    "unrelated_explicit_key": "survives",
                }
            },
        )

        additional = stage.yaml_engine_args.get("additional_config")
        if not isinstance(additional, dict):
            _fail("ENV-MIG-011 missing merged canonical additional_config")
        assert additional["persistent_state_admission_policy"] == "hard_cap"
        assert additional["persistent_state_admission_retry_floor_ms"] == 125
        assert additional["unrelated_explicit_key"] == "survives"
        assert stage.to_omegaconf().engine_args.additional_config == additional

    def test_lowered_envelope_participates_in_vllm_execution_identity(
        self,
        tmp_path: Path,
    ) -> None:
        """@spec ENV-MIG-011: API/Core consume one hash-bearing map."""

        from vllm.config import VllmConfig

        stage = _resolved_stage(_write_deploy(tmp_path, _HARD_CAP_PERSISTENT_STATE))
        lowered = stage.to_omegaconf().engine_args.additional_config
        api_config = VllmConfig(additional_config=dict(lowered))
        core_config = VllmConfig(additional_config=dict(lowered))
        changed = VllmConfig(
            additional_config={
                **dict(lowered),
                "persistent_state_admission_retry_floor_ms": 125,
            }
        )

        assert api_config.compute_hash() == core_config.compute_hash()
        assert changed.compute_hash() != api_config.compute_hash()

    @pytest.mark.parametrize(
        "persistent_state",
        (
            "      null",
            "      persistent_state_unknown_key: 1",
            "      persistent_state_admission_policy: hard_cap\n      persistent_state_max_tombstones: 0",
            "      persistent_state_admission_policy: profile\n"
            "      persistent_state_service_profile_trailing_rounds: 3",
        ),
        ids=("null", "unknown", "tombstones", "partial"),
    )
    def test_invalid_persistent_state_profile_fails_closed(
        self,
        tmp_path: Path,
        persistent_state: str,
    ) -> None:
        """@spec ENV-MIG-011: null, unknown, and partial profiles are invalid."""
        deploy_path = _write_deploy(tmp_path, persistent_state)
        with pytest.raises(ValueError):
            _resolved_stage(deploy_path)

    def test_advanced_policy_override_requires_complete_profile_arm(self, tmp_path: Path) -> None:
        """@spec ENV-MIG-011: a profile override cannot borrow hard-cap defaults."""
        deploy_path = _write_deploy(tmp_path)
        with pytest.raises(ValueError, match="profile"):
            _resolved_stage(
                deploy_path,
                {"additional_config": {"persistent_state_admission_policy": "profile"}},
            )

    def test_packaged_nemotron_deploy_selects_profile_at_57_sessions(self) -> None:
        """@spec ENV-MIG-011/012: zero-argument registry boot selects profile admission."""
        stage = _resolved_default_stage()
        additional = stage.yaml_engine_args.get("additional_config")
        if not isinstance(additional, dict):
            _fail("ENV-MIG-011 packaged deploy was not lowered to additional_config")

        assert stage.yaml_engine_args["max_num_seqs"] == 57
        assert additional["persistent_state_admission_policy"] == "profile"
        assert additional["max_resident_sessions"] == 57
        assert _PROFILE_ONLY_KEYS <= additional.keys()
        # The tombstone budget must be derived from the priming plan, never pinned.
        assert "persistent_state_max_tombstones" not in additional

    def test_packaged_nemotron_deploy_resolves_through_runtime_config(self) -> None:
        """@spec ENV-MIG-011/012: the packaged deploy resolves a complete runtime envelope.

        Lowering alone is not enough: the engine resolves the envelope again
        through ``PersistentStateRuntimeConfig.from_vllm_config`` with the
        model's priming plan, and a packaged default that fails there fails
        every zero-argument boot.
        """
        from vllm_omni.engine.persistent_state_config import PersistentStateRuntimeConfig

        stage = _resolved_default_stage()
        runtime = _runtime_config_for(stage)

        assert isinstance(runtime, PersistentStateRuntimeConfig)
        assert runtime.admission_policy == "profile"
        assert runtime.max_resident_sessions == 57
        assert runtime.priming_configured_population_ceiling == 57
        assert runtime.bootstrap_operation_budget > 0
        assert runtime.max_tombstones >= runtime.bootstrap_operation_budget + runtime.runtime_tombstone_allowance

    def test_minimal_hard_cap_deploy_resolves_through_runtime_config(self, tmp_path: Path) -> None:
        """@spec ENV-MIG-011: a policy-only hard_cap deploy boots on runtime defaults.

        On this lane hard_cap startup is profile-free: it runs no priming, so
        it needs no service-profile inputs and no priming tombstone budget.
        """
        stage = _resolved_stage(_write_deploy(tmp_path, _HARD_CAP_PERSISTENT_STATE))
        runtime = _runtime_config_for(stage)

        assert runtime.admission_policy == "hard_cap"
        assert runtime.max_resident_sessions == 8
        assert runtime.bootstrap_operation_budget == 0
        assert runtime.max_tombstones >= runtime.runtime_tombstone_allowance

    def test_hard_cap_unadmitted_lifetime_default_follows_resolved_wait(self, tmp_path: Path) -> None:
        """@spec ENV-MIG-011: the omitted lifetime is derived from the resolved inputs."""
        stage = _resolved_stage(
            _write_deploy(
                tmp_path,
                _HARD_CAP_PERSISTENT_STATE + "\n      persistent_state_admission_wait_timeout_s: 10.0",
            )
        )
        runtime = _runtime_config_for(stage)

        # 2 * (10 s configuration + 10 s wait) + (100 ms floor + 25 ms jitter)
        assert runtime.unadmitted_connection_timeout_s == pytest.approx(40.125)

    def test_missing_packaged_nemotron_deploy_has_no_generic_fallback(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        """@spec ENV-MIG-011: selected persistent state requires its package profile."""
        monkeypatch.setattr(config_factory, "_DEPLOY_DIR", tmp_path)

        try:
            StageConfigFactory._create_from_registry("nemotron_asr", _pipeline(), {})
        except FileNotFoundError:
            return
        _fail("ENV-MIG-011 missing packaged Nemotron deploy did not fail startup")

    def test_explicit_valid_nemotron_deploy_is_allowed(self, tmp_path: Path) -> None:
        """@spec ENV-MIG-011: an explicit valid profile may replace the package one."""
        deploy_path = _write_deploy(
            tmp_path,
            _HARD_CAP_PERSISTENT_STATE,
        )

        stage = _resolved_stage(deploy_path)
        additional = stage.yaml_engine_args.get("additional_config")
        if not isinstance(additional, dict):
            _fail("ENV-MIG-011 explicit deploy was not lowered to additional_config")
        assert additional["persistent_state_admission_policy"] == "hard_cap"
        assert not _PROFILE_ONLY_KEYS & additional.keys()

    def test_package_data_declares_and_exposes_nemotron_deploy(self) -> None:
        """@spec ENV-MIG-011: wheels retain the selected deploy profile."""
        pyproject = (_ROOT / "pyproject.toml").read_text(encoding="utf-8")
        assert re.search(r'"vllm_omni"\s*=\s*\[[^\]]*"deploy/\*\.yaml"', pyproject, re.DOTALL)

        packaged = importlib.resources.files("vllm_omni").joinpath("deploy/nemotron_asr.yaml")
        assert packaged.is_file()
