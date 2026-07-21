# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""PrecisionPolicy: every dtype declaration delegates to one policy object.

Specs: PORT-PREC-001 (delegation, no hardcoded dtypes), PORT-PREC-002
(engine-dtype conflict fails fast naming the policy), PORT-PREC-003
(fp32 bring-up value), PORT-PREC-005 (recurrent-state dtype is an
independent axis, defaults fp32, never silently reduced).
"""

import pytest
import torch

from vllm_omni.model_executor.models.nemotron_asr.precision import (
    FP32_BRINGUP,
    PrecisionPolicy,
    RecurrentStatePrecisionError,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_bringup_policy_is_fp32_everywhere():
    for tensor_class in (
        "weights",
        "activations",
        "attention_cache",
        "conv_state",
        "lstm_state",
        "queue_state",
    ):
        assert FP32_BRINGUP.dtype_for(tensor_class) == torch.float32


def test_identifier_matches_harness_scheme():
    # The uniform fp32 policy canonicalizes to {"*": "fp32"} and must hash
    # to the same identifier the eval harness records in golden/run
    # provenance (EVAL-GOLD-004) — provenance keys are shared across repos.
    assert FP32_BRINGUP.identifier == "pp-d479361445b4"
    assert FP32_BRINGUP.content_hash == (
        "sha256:d479361445b45b54ce1b0df4cd11b7ee1a06deb235ceeb8cf97c565bed2f5ccd"
    )


def test_identifier_is_order_independent_and_distinct():
    a = PrecisionPolicy({"weights": "bf16", "lstm_state": "fp32"})
    b = PrecisionPolicy({"lstm_state": "fp32", "weights": "bf16"})
    assert a.identifier == b.identifier
    assert a.identifier != FP32_BRINGUP.identifier


def test_unknown_tensor_class_is_an_error():
    with pytest.raises(KeyError):
        FP32_BRINGUP.dtype_for("no_such_class")


def test_wildcard_covers_unlisted_classes():
    # Both recurrent-state classes pinned fp32 — a bf16 wildcard alone
    # would (correctly) trip the PORT-PREC-005 guard via conv_state.
    policy = PrecisionPolicy(
        {"*": "bf16", "conv_state": "fp32", "lstm_state": "fp32"}
    )
    assert policy.dtype_for("weights") == torch.bfloat16
    assert policy.dtype_for("lstm_state") == torch.float32


def test_recurrent_state_below_fp32_requires_explicit_override():
    # PORT-PREC-005: lstm_state/conv_state dtype never silently drops
    # below fp32 — an explicit reviewed-policy flag is the only path.
    with pytest.raises(RecurrentStatePrecisionError):
        PrecisionPolicy({"*": "fp32", "lstm_state": "bf16"})
    with pytest.raises(RecurrentStatePrecisionError):
        PrecisionPolicy({"*": "bf16"})  # wildcard would cover conv_state
    ok = PrecisionPolicy(
        {"*": "bf16", "conv_state": "fp32", "lstm_state": "fp32"}
    )
    assert ok.dtype_for("activations") == torch.bfloat16


def test_engine_dtype_conflict_fails_fast_naming_policy():
    # PORT-PREC-002: --dtype bfloat16 during fp32 bring-up must fail at
    # startup with the policy named, never silently allocate pages.
    with pytest.raises(ValueError, match="PrecisionPolicy"):
        FP32_BRINGUP.assert_engine_dtype(torch.bfloat16)
    FP32_BRINGUP.assert_engine_dtype(torch.float32)
