# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The qualified deploy profile must survive packaging (PORT-STATE-009).

A wheel built without ``vllm_omni/deploy/*.yaml`` silently drops the
pipeline's qualified profile: the loader logs "Deploy config not found"
and proceeds with defaults, so vLLM's auto policy can select a dtype that
does not name the qualified acceleration fingerprint. Editable installs
never see this, which is exactly why it must be pinned by test.
"""

import sys
from pathlib import Path

import pytest
import yaml

from vllm_omni.config.stage_config import _DEPLOY_DIR

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

_REPO_ROOT = Path(__file__).resolve().parents[4]


# @spec PORT-STATE-009
def test_the_qualified_deploy_profile_exists_where_the_loader_looks() -> None:
    """The loader resolves _DEPLOY_DIR/<pipeline>.yaml; the profile must
    be there in every installation, editable or built."""
    profile = _DEPLOY_DIR / "nemotron_asr.yaml"
    assert profile.is_file(), (
        "the qualified deploy profile is missing from this installation - "
        "a built distribution without deploy data loses the precision pin"
    )


# @spec PORT-STATE-009
def test_the_profile_pins_what_the_model_gates() -> None:
    """The profile's load-bearing values match the model-side gates, so a
    missing profile fails loudly at those gates instead of drifting."""
    profile = yaml.safe_load((_DEPLOY_DIR / "nemotron_asr.yaml").read_text())
    assert profile["dtype"] == "float16"
    assert profile["enable_prefix_caching"] is False
    assert all(stage.get("enforce_eager") is True for stage in profile["stages"])


# @spec PORT-STATE-009
def test_package_data_ships_the_deploy_profiles() -> None:
    """pyproject must declare deploy/*.yaml as package data; without the
    declaration only editable installs carry the profiles."""
    if sys.version_info >= (3, 11):
        import tomllib
    else:  # pragma: no cover - py3.10 qualification lane
        import tomli as tomllib
    pyproject = tomllib.loads((_REPO_ROOT / "pyproject.toml").read_text())
    package_data = pyproject["tool"]["setuptools"]["package-data"]
    assert "deploy/*.yaml" in package_data.get("vllm_omni", []), (
        "vllm_omni/deploy/*.yaml is not declared as package data - wheels "
        "will ship without the qualified deploy profiles"
    )
