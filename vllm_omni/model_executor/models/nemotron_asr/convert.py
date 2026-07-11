# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Offline weight-conversion verification (PORT-WGT-001/002/003).

The runtime loads standard HF safetensors; this module is the offline
tool's core: a declarative mapping from checkpoint tensor names to the
port's module tree, enforced by a consume-exactly-once ledger with
post-transform shape checks. Any mismatch is a hard failure carrying a
named tensor-level diff — never warn-and-proceed. Missing LID weights
(``prompt_kernel``) are fatal for this model id: language conditioning
is part of the contract, the model never degrades to unconditioned
transcription (PORT-WGT-003).

The concrete rule table for the shipped checkpoint is data, filled in
against the real safetensors index at P2 bring-up; the enforcement
machinery here is what the GPU-free tests pin.
"""

import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field

import torch

Transform = Callable[[torch.Tensor], torch.Tensor]

TRANSFORMS: dict[str, Transform] = {
    "identity": lambda t: t,
    "transpose01": lambda t: t.transpose(0, 1).contiguous(),
}

LID_REQUIRED_PATTERN = re.compile(r"prompt_kernel")


class ConversionError(ValueError):
    """A conversion contract violation, carrying named tensor detail."""


@dataclass(frozen=True)
class TensorRule:
    """One source→target mapping.

    ``source`` is a regex fully matching checkpoint tensor names; each
    matched tensor maps to ``target`` (with regex group expansion) after
    ``transform``; ``expect_shape``, when given, is asserted on the
    transformed tensor.
    """

    source: str
    target: str
    transform: str = "identity"
    expect_shape: tuple[int, ...] | None = None


@dataclass
class ConversionReport:
    """What the conversion consumed and produced."""

    consumed: dict[str, str] = field(default_factory=dict)
    produced: dict[str, tuple[int, ...]] = field(default_factory=dict)


def convert_state_dict(
    checkpoint: Mapping[str, torch.Tensor],
    rules: Sequence[TensorRule],
) -> tuple[dict[str, torch.Tensor], ConversionReport]:
    """Apply the rule table under the consume-exactly-once contract.

    Raises:
        ConversionError: If any checkpoint tensor is unconsumed, any
            tensor is matched by more than one rule, any rule matches
            nothing, a transformed shape misses its expectation, or a
            required LID tensor is absent (PORT-WGT-002/003). The
            message names every offending tensor.
    """
    compiled = [(rule, re.compile(rule.source)) for rule in rules]
    report = ConversionReport()
    out: dict[str, torch.Tensor] = {}
    unmatched_rules = {rule.source for rule in rules}

    for name, tensor in checkpoint.items():
        hits = [
            (rule, regex) for rule, regex in compiled
            if regex.fullmatch(name)
        ]
        if len(hits) > 1:
            sources = [rule.source for rule, _ in hits]
            raise ConversionError(
                f"tensor {name!r} matched by multiple rules: {sources}"
            )
        if not hits:
            continue  # collected below as unconsumed
        rule, regex = hits[0]
        unmatched_rules.discard(rule.source)
        target = regex.sub(rule.target, name)
        transformed = TRANSFORMS[rule.transform](tensor)
        if (
            rule.expect_shape is not None
            and tuple(transformed.shape) != rule.expect_shape
        ):
            raise ConversionError(
                f"shape mismatch for {name!r} -> {target!r}: expected "
                f"{rule.expect_shape}, got {tuple(transformed.shape)} "
                f"(transform {rule.transform!r})"
            )
        if target in out:
            raise ConversionError(
                f"two tensors map to the same target {target!r}"
            )
        out[target] = transformed
        report.consumed[name] = target
        report.produced[target] = tuple(transformed.shape)

    unconsumed = sorted(set(checkpoint) - set(report.consumed))
    problems: list[str] = []
    if unconsumed:
        problems.append(f"unconsumed checkpoint tensors: {unconsumed}")
    if unmatched_rules:
        problems.append(
            f"rules matching no tensor: {sorted(unmatched_rules)}"
        )
    lid_present = any(
        LID_REQUIRED_PATTERN.search(name) for name in checkpoint
    )
    if not lid_present:
        problems.append(
            "prompt_kernel (LID) weights absent — fatal for this model "
            "id; the model never degrades to unconditioned transcription"
        )
    if problems:
        raise ConversionError("; ".join(problems))
    return out, report


def diff_against_referee(
    converted: Mapping[str, torch.Tensor],
    referee: Mapping[str, torch.Tensor],
    pairs: Mapping[str, str],
    *,
    atol: float = 0.0,
    rtol: float = 0.0,
) -> None:
    """Numerically compare converted tensors against the referee.

    ``pairs`` maps converted-tree names to referee (``.nemo``) names —
    the referee settles layout ambiguities (PORT-WGT-002).

    Raises:
        ConversionError: Naming every differing tensor with max-abs
            difference.
    """
    diffs: list[str] = []
    for target, referee_name in sorted(pairs.items()):
        if target not in converted:
            diffs.append(f"{target}: absent from converted tree")
            continue
        if referee_name not in referee:
            diffs.append(f"{referee_name}: absent from referee")
            continue
        a, b = converted[target], referee[referee_name]
        if a.shape != b.shape:
            diffs.append(
                f"{target}: shape {tuple(a.shape)} != referee "
                f"{tuple(b.shape)}"
            )
        elif not torch.allclose(a, b, atol=atol, rtol=rtol):
            diffs.append(
                f"{target}: max abs diff "
                f"{(a - b).abs().max().item():.3e}"
            )
    if diffs:
        raise ConversionError("referee diff: " + "; ".join(diffs))
