# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Startup decode dispatch-table generation (PORT-DEC-008).

Consumes a ``p6b_decode_profile`` artifact and COMPUTES per-bucket
decode selections under the versioned dispatch-calibration policy —
never eyeballed brackets (decisions/decode-dispatch-regime.md r3.2):

- lanes come from the artifact's ``generator_eligible_lanes`` — a
  correctness-unqualified lane never produces entries;
- compact may be selected only from SELECTIVE cells (policy realized
  within tolerance) and only when it beats the sync-free arm by the
  recorded hysteresis at EVERY dcp activity level (the worst-case
  objective); a nonselective or missing level forces the sync-free
  arm;
- an unmeasured tier resolves to the nearest measured tier, and to
  the sync-free arm when the bracketing tiers disagree (the
  crossover-gap rule — dense's worst case is bounded overcompute,
  compact's adds syncs, host stalls, and capture-ineligibility);
- an execution profile whose engine graph covers decode never
  dispatches the capture-ineligible candidate (sanity invariant, not
  a measured outcome);
- the table carries the measurement fingerprint (device, driver,
  torch, model-shape digest — deliberately CHECKPOINT-independent)
  and a runtime mismatch fails closed for performance-gated
  deployments.

Pure host logic: no torch, importable anywhere. The consumer is
``advance_model_rows``, which resolves the callable per
profile+geometry bucket at batch-composition time.
"""

from __future__ import annotations

import json
import logging
import math
import re
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

#: The recorded hysteresis margin (decisions/decode-dispatch-regime.md
#: r3: chosen 2026-07-18, validated by independent profile runs before
#: any performance-gated deployment).
HYSTERESIS_PCT = 25.0

#: Arms whose valid path issues no host/device synchronization.
SYNC_FREE_ARMS = ("dense-eager", "dense-graphed")
_KNOWN_ARMS = frozenset(
    ("dense-eager", "dense-graphed", "compact-eager")
)

#: Admission allowlists — strict generation fails closed on anything
#: outside them.
SUPPORTED_DCP_VERSIONS = ("dcp-v1",)
SUPPORTED_PROBE_SCHEMAS = ("p6b-decode-profile-v3",)
#: The one benign untracked class on a measurement checkout:
#: root-level model-download lock files (round 7's audit).
_UNTRACKED_ALLOWED = re.compile(r"^[^/]+\.lock$")

#: Fingerprint keys a runtime must match for a table to be valid —
#: the full normative identity: device, driver, torch, model shape,
#: decode-algorithm revision (never the fork commit — an unrelated
#: commit must not invalidate a table), probe schema, lane
#: definitions, math mode, and the calibration-policy version.
_FINGERPRINT_KEYS = (
    "device_name", "driver", "torch", "model_shape_digest",
    "decode_algo_revision", "probe_schema",
    "lane_definitions_digest", "tf32_matmul",
)


class FingerprintMismatchError(RuntimeError):
    """A dispatch table presented to a runtime it was not measured
    on (performance-gated deployments fail closed)."""


@dataclass(frozen=True)
class DispatchTable:
    """Computed decode selections plus the identity that scopes them.

    ``entries`` maps ``(geometry, tier, lane)`` to the EAGER-profile
    arm; graph-covered profiles are resolved structurally in
    :meth:`select`, never stored (compact is capture-ineligible by
    construction there).
    """

    entries: dict[tuple[str, int, str], str]
    fingerprint: dict[str, Any]
    hysteresis_pct: float
    policy_version: str
    lanes: tuple[str, ...]
    #: True when generated under admission="analysis" (comparison and
    #: review use only): such a table NEVER validates for a
    #: performance-gated runtime.
    analysis_only: bool = field(default=False)

    def select(
        self,
        geometry: str,
        batch: int,
        *,
        lane: str,
        graph_covers_decode: bool,
    ) -> str:
        """The decode arm for one profile+geometry bucket.

        Raises:
            KeyError: unknown ``lane`` (not generator-eligible) or a
                geometry with no measured tiers in this table.
        """
        if lane not in self.lanes:
            raise KeyError(
                f"lane {lane!r} is not generator-eligible "
                f"(eligible: {self.lanes})"
            )
        tiers = sorted(
            t
            for (g, t, la) in self.entries
            if g == geometry and la == lane
        )
        if not tiers:
            raise KeyError(f"no measured tiers for {geometry!r}")
        if graph_covers_decode:
            return "dense-graphed"
        if batch in tiers:
            return self.entries[(geometry, batch, lane)]
        lower = max((t for t in tiers if t < batch), default=None)
        upper = min((t for t in tiers if t > batch), default=None)
        if lower is None:
            assert upper is not None
            return self.entries[(geometry, upper, lane)]
        if upper is None:
            return self.entries[(geometry, lower, lane)]
        low_arm = self.entries[(geometry, lower, lane)]
        high_arm = self.entries[(geometry, upper, lane)]
        if low_arm != high_arm:
            # Crossover-gap rule: bracketing tiers disagree — take
            # the sync-free arm of the pair.
            logger.warning(
                "decode dispatch: unmeasured tier %d for %s/%s with "
                "disagreeing brackets (%d=%s, %d=%s) — selecting the "
                "sync-free arm",
                batch, geometry, lane, lower, low_arm, upper,
                high_arm,
            )
            return (
                low_arm if low_arm in SYNC_FREE_ARMS else high_arm
            )
        nearest = (
            lower if (batch - lower) <= (upper - batch) else upper
        )
        logger.warning(
            "decode dispatch: unmeasured tier %d for %s/%s — "
            "resolving to nearest measured tier %d (%s)",
            batch, geometry, lane, nearest,
            self.entries[(geometry, nearest, lane)],
        )
        return self.entries[(geometry, nearest, lane)]

    def validate_runtime(
        self,
        runtime_fingerprint: dict[str, Any],
        *,
        performance_gated: bool,
    ) -> list[str]:
        """Check this table's identity against the runtime.

        Returns:
            The mismatched fingerprint keys (empty when valid); a
            dev-profile caller warns on them and regenerates.

        Raises:
            FingerprintMismatchError: any mismatch while
                ``performance_gated`` — the table fails closed.
        """
        if self.analysis_only and performance_gated:
            raise FingerprintMismatchError(
                "analysis-only table (admission='analysis') can "
                "never serve a performance-gated runtime"
            )
        mismatched = [
            key
            for key in _FINGERPRINT_KEYS
            if runtime_fingerprint.get(key)
            != self.fingerprint.get(key)
        ]
        if mismatched and performance_gated:
            raise FingerprintMismatchError(
                "dispatch table measured on a different runtime: "
                + ", ".join(
                    f"{k}: table={self.fingerprint.get(k)!r} "
                    f"runtime={runtime_fingerprint.get(k)!r}"
                    for k in mismatched
                )
            )
        return mismatched

    def to_json(self) -> str:
        """Serialize (entry keys flatten to 'geometry|tier|lane')."""
        return json.dumps({
            "entries": {
                f"{g}|{t}|{lane}": arm
                for (g, t, lane), arm in sorted(self.entries.items())
            },
            "fingerprint": self.fingerprint,
            "hysteresis_pct": self.hysteresis_pct,
            "policy_version": self.policy_version,
            "lanes": list(self.lanes),
            "analysis_only": self.analysis_only,
        })

    @classmethod
    def from_json(cls, blob: str) -> DispatchTable:
        raw = json.loads(blob)
        entries: dict[tuple[str, int, str], str] = {}
        for key, arm in raw["entries"].items():
            geometry, tier, lane = key.split("|")
            entries[(geometry, int(tier), lane)] = arm
        return cls(
            entries=entries,
            fingerprint=raw["fingerprint"],
            hysteresis_pct=float(raw["hysteresis_pct"]),
            policy_version=raw["policy_version"],
            lanes=tuple(raw["lanes"]),
            analysis_only=bool(raw.get("analysis_only", False)),
        )


def _latency(cell: dict[str, Any]) -> float:
    return float(cell["latency_us"])


def _admit_strict(
    report: dict[str, Any], fingerprint: dict[str, Any]
) -> None:
    """Fail-closed artifact admission: every gate must POSITIVELY
    hold (absent or None never passes)."""
    if report.get("execution_ok") is not True:
        raise ValueError(
            "artifact does not record execution_ok=True — the sweep "
            "may not have completed"
        )
    if fingerprint.get("dirty_tree") is not False:
        raise ValueError(
            "artifact does not record a clean tree (dirty_tree must "
            "be exactly False) — evidence identity is not "
            "reproducible"
        )
    if fingerprint.get("config_asserted") is not True:
        raise ValueError(
            "artifact dims were not asserted against the model "
            "config (config_asserted must be True)"
        )
    schema = fingerprint.get("probe_schema")
    if schema not in SUPPORTED_PROBE_SCHEMAS:
        raise ValueError(
            f"unsupported probe schema {schema!r} "
            f"(supported: {SUPPORTED_PROBE_SCHEMAS})"
        )
    untracked = fingerprint.get("untracked_paths")
    if untracked is None:
        raise ValueError(
            "artifact records no untracked_paths — untracked state "
            "is unauditable"
        )
    offending = [
        p for p in untracked if not _UNTRACKED_ALLOWED.match(p)
    ]
    if offending:
        raise ValueError(
            "untracked files outside the allowlisted root lock-file "
            f"class on the measurement checkout: {offending[:5]}"
        )


def _eligible_lanes(
    report: dict[str, Any], *, strict: bool
) -> tuple[str, ...]:
    """Lanes are DERIVED from lane_results, never trusted from the
    declared list; strict admission requires the two to agree."""
    lane_results = report["lane_results"]
    derived = tuple(sorted(
        lane
        for lane, res in lane_results.items()
        if res.get("dispatch_eligible") is True
        and res.get("performance_qualified") is True
        and res.get("mismatch_cells") == 0
    ))
    declared = tuple(sorted(report["generator_eligible_lanes"]))
    if strict and derived != declared:
        raise ValueError(
            f"generator_eligible_lanes {declared} disagrees with "
            f"lane_results derivation {derived}"
        )
    return derived


def generate_dispatch_table(
    report: dict[str, Any],
    *,
    hysteresis_pct: float = HYSTERESIS_PCT,
    admission: str = "strict",
) -> DispatchTable:
    """Compute the dispatch table from a profile artifact.

    Args:
        report: a ``p6b_decode_profile`` report (probe r8+ schema).
        hysteresis_pct: the recorded sync-free preference margin.
        admission: ``"strict"`` (default) applies the fail-closed
            gates; ``"analysis"`` bypasses ONLY the artifact-hygiene
            gates for comparison/review work and marks the table
            ``analysis_only`` — such a table never validates for a
            performance-gated runtime.

    Raises:
        ValueError: schema, hygiene, consistency, or structural
            defects — generation fails closed on all of them.
    """
    if admission not in ("strict", "analysis"):
        raise ValueError(f"unknown admission mode {admission!r}")
    strict = admission == "strict"
    for required in ("fingerprint", "cells", "dcp"):
        if required not in report:
            raise ValueError(f"artifact missing {required!r}")
    if (
        "lane_results" not in report
        or "generator_eligible_lanes" not in report
    ):
        raise ValueError(
            "artifact predates per-lane qualification "
            "(lane_results/generator_eligible_lanes missing) — "
            "not generator input"
        )
    fingerprint = dict(report["fingerprint"])
    dcp = report["dcp"]
    if dcp.get("version") not in SUPPORTED_DCP_VERSIONS:
        raise ValueError(
            f"unsupported dcp version {dcp.get('version')!r} "
            f"(supported: {SUPPORTED_DCP_VERSIONS})"
        )
    if strict:
        _admit_strict(report, fingerprint)
    lanes = _eligible_lanes(report, strict=strict)
    selective_activities = tuple(dcp["selective_activities"])

    by_key: dict[
        tuple[str, int, str], dict[str, dict[str, Any]]
    ] = {}
    mismatch_recount: dict[str, int] = {}
    seen: set[tuple[str, int, str, str, str]] = set()
    for cell in report["cells"]:
        lane = str(cell["precision"])
        arm = str(cell["arm"])
        if arm not in _KNOWN_ARMS:
            raise ValueError(f"unknown arm {arm!r} in artifact")
        latency = float(cell["latency_us"])
        if not math.isfinite(latency) or latency <= 0.0:
            raise ValueError(
                f"non-finite/non-positive latency in cell "
                f"{cell['geometry']}/{cell['batch']}/{lane}/{arm}"
            )
        if not cell.get("tokens_match", True):
            mismatch_recount[lane] = (
                mismatch_recount.get(lane, 0) + 1
            )
        ident = (
            str(cell["geometry"]), int(cell["batch"]), lane, arm,
            str(cell["activity"]),
        )
        if ident in seen:
            raise ValueError(f"duplicate cell {ident}")
        seen.add(ident)
        if lane not in lanes:
            continue
        key = (str(cell["geometry"]), int(cell["batch"]), lane)
        arms = by_key.setdefault(key, {})
        if arm == "compact-eager":
            # Keep every activity row; selection reads dcp levels.
            arms[f"compact:{cell['activity']}"] = cell
        else:
            arms.setdefault(arm, cell)
    if strict:
        for lane, res in report["lane_results"].items():
            if mismatch_recount.get(lane, 0) != res.get(
                "mismatch_cells"
            ):
                raise ValueError(
                    f"lane_results mismatch_cells for {lane!r} "
                    f"({res.get('mismatch_cells')}) disagrees with "
                    f"the cells ({mismatch_recount.get(lane, 0)})"
                )

    entries: dict[tuple[str, int, str], str] = {}
    factor = 1.0 - hysteresis_pct / 100.0
    for key, arms in by_key.items():
        dense = arms.get("dense-eager")
        if dense is None:
            if strict:
                raise ValueError(
                    f"measured group {key} has no dense-eager row — "
                    "artifact is structurally incomplete"
                )
            continue
        dense_lat = _latency(dense)
        compact_wins = True
        for activity in selective_activities:
            cell = arms.get(f"compact:{activity}")
            if cell is None or not cell.get("selective"):
                # Missing or nonselective dcp level: the worst case
                # is unproven — sync-free forced.
                compact_wins = False
                break
            if _latency(cell) > dense_lat * factor:
                compact_wins = False
                break
        entries[key] = (
            "compact-eager" if compact_wins else "dense-eager"
        )

    return DispatchTable(
        entries=entries,
        fingerprint=fingerprint,
        hysteresis_pct=hysteresis_pct,
        policy_version=str(dcp["version"]),
        lanes=lanes,
        analysis_only=not strict,
    )
