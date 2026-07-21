# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Per-tensor-class precision policy.

Precision is a value, never structure: every dtype or quantization
declaration in this model (spec pages, weights, compute) delegates to
one ``PrecisionPolicy``; no call site hardcodes a dtype. The policy's
identifier is a content hash over the per-tensor-class mapping and keys
golden/run provenance and parity tolerances in the evaluation harness,
so the hash scheme here must stay byte-compatible with it.

Recurrent per-session state (predictor LSTM ``(h, c)``, depthwise-conv
page) is an independent precision axis that defaults fp32 and is never
silently reduced — recurrent accumulation compounds rounding error
(the same convention as vLLM's ``mamba_ssm_cache_dtype``).
"""

import hashlib
import json
from collections.abc import Mapping
from typing import Final

import torch

TENSOR_CLASSES: Final = (
    "weights",
    "activations",
    "attention_cache",
    "conv_state",
    "lstm_state",
    "queue_state",
    # The frontend raw/mel tails (Phase 6b): golden-boundary audio
    # state, fp32 at bring-up; sub-fp32 policies must scope it
    # explicitly (BF16_COMPUTE pins it fp32 below).
    "frontend_state",
)

RECURRENT_STATE_CLASSES: Final = ("conv_state", "lstm_state")

_DTYPES: Final = {
    "fp32": torch.float32,
    "bf16": torch.bfloat16,
    "fp16": torch.float16,
}

_SUB_FP32: Final = frozenset({"bf16", "fp16"})


class RecurrentStatePrecisionError(ValueError):
    """A recurrent-state dtype below fp32 without an explicit override."""


class PrecisionPolicy:
    """Immutable mapping of tensor classes to dtype names.

    A ``"*"`` key provides the default for unlisted classes. Recurrent
    state classes below fp32 raise unless ``allow_sub_fp32_recurrent``
    is set — the constructor flag is the reviewed-policy-change hook.
    """

    def __init__(
        self,
        mapping: Mapping[str, str],
        *,
        allow_sub_fp32_recurrent: bool = False,
    ) -> None:
        for tensor_class, dtype_name in mapping.items():
            if tensor_class != "*" and tensor_class not in TENSOR_CLASSES:
                raise KeyError(f"unknown tensor class: {tensor_class!r}")
            if dtype_name not in _DTYPES:
                raise KeyError(f"unknown dtype name: {dtype_name!r}")
        self._mapping = dict(mapping)
        if not allow_sub_fp32_recurrent:
            for state_class in RECURRENT_STATE_CLASSES:
                try:
                    resolved = self._name_for(state_class)
                except KeyError:
                    # Partial policy: unresolved classes fail at use time.
                    continue
                if resolved in _SUB_FP32:
                    raise RecurrentStatePrecisionError(
                        f"{state_class} resolves below fp32; recurrent "
                        "state precision is never reduced implicitly "
                        "(pass allow_sub_fp32_recurrent=True only as an "
                        "explicit reviewed policy change)"
                    )

    def _name_for(self, tensor_class: str) -> str:
        if tensor_class in self._mapping:
            return self._mapping[tensor_class]
        if "*" in self._mapping:
            return self._mapping["*"]
        raise KeyError(
            f"policy declares no dtype for {tensor_class!r} and no '*'"
        )

    def dtype_for(self, tensor_class: str) -> torch.dtype:
        """Resolve one tensor class to its torch dtype."""
        if tensor_class != "*" and tensor_class not in TENSOR_CLASSES:
            raise KeyError(f"unknown tensor class: {tensor_class!r}")
        return _DTYPES[self._name_for(tensor_class)]

    @property
    def content_hash(self) -> str:
        """Full canonical mapping digest used in run provenance."""
        canonical = json.dumps(
            {key: self._mapping[key] for key in sorted(self._mapping)},
            separators=(",", ":"),
            ensure_ascii=True,
        )
        digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        return f"sha256:{digest}"

    @property
    def identifier(self) -> str:
        """Short content identifier, byte-compatible with the harness."""
        return f"pp-{self.content_hash.removeprefix('sha256:')[:12]}"

    def assert_engine_dtype(self, engine_dtype: torch.dtype) -> None:
        """Fail fast when the launch dtype conflicts with the policy."""
        expected = self.dtype_for("weights")
        if engine_dtype != expected:
            raise ValueError(
                f"engine dtype {engine_dtype} conflicts with "
                f"PrecisionPolicy {self.identifier} (weights: {expected}); "
                "precision changes are reviewed policy values, not launch "
                "flags"
            )


FP32_BRINGUP: Final = PrecisionPolicy({"*": "fp32"})
"""The bring-up value: fp32 everywhere, matching the fp32-locked oracle."""

BF16_COMPUTE: Final = PrecisionPolicy(
    {
        "weights": "bf16",
        "activations": "bf16",
        "attention_cache": "fp32",
        "conv_state": "fp32",
        "lstm_state": "fp32",
        "queue_state": "fp32",
        # Frontend tails stay fp32 under reduced compute: they are the
        # golden input boundary (adding this key restamps this
        # policy's content-hash identifier; only FP32_BRINGUP's id is
        # pinned by value, and its mapping is unchanged).
        "frontend_state": "fp32",
    }
)
"""The first reviewed sub-fp32 policy (PORT-PREC-003): compute classes
bf16, every state class fp32 — the vLLM mamba mixed-precision idiom
(compute dtype below the recurrent/cache dtype, never the reverse).
Transcript parity vs the fp32-locked oracle stays BLOCKING; tensor
deltas are advisory under this identifier (EVAL-PAR-006)."""
