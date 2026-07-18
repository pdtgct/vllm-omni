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
from dataclasses import dataclass
from typing import Any

#: The recorded hysteresis margin (decisions/decode-dispatch-regime.md
#: r3: chosen 2026-07-18, validated by independent profile runs before
#: any performance-gated deployment).
HYSTERESIS_PCT = 25.0

#: Arms whose valid path issues no host/device synchronization.
SYNC_FREE_ARMS = ("dense-eager", "dense-graphed")

#: Fingerprint keys a runtime must match for a table to be valid.
_FINGERPRINT_KEYS = (
    "device_name", "driver", "torch", "model_shape_digest",
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
            return (
                low_arm if low_arm in SYNC_FREE_ARMS else high_arm
            )
        nearest = (
            lower if (batch - lower) <= (upper - batch) else upper
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
        )


def _latency(cell: dict[str, Any]) -> float:
    return float(cell["latency_us"])


def generate_dispatch_table(
    report: dict[str, Any],
    *,
    hysteresis_pct: float = HYSTERESIS_PCT,
) -> DispatchTable:
    """Compute the dispatch table from a profile artifact.

    Args:
        report: a ``p6b_decode_profile`` report (probe r6+ schema:
            per-lane qualification and per-cell policy realization).
        hysteresis_pct: the recorded sync-free preference margin.

    Raises:
        ValueError: pre-lane-schema artifact (no ``lane_results`` /
            ``generator_eligible_lanes``), a dirty-tree artifact, or
            a missing dcp declaration — all fail generation outright.
    """
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
    if fingerprint.get("dirty_tree"):
        raise ValueError(
            "artifact was produced from a dirty tree — evidence "
            "identity is not reproducible"
        )
    dcp = report.get("dcp")
    if not dcp:
        raise ValueError("artifact carries no dcp declaration")
    selective_activities = tuple(dcp["selective_activities"])
    lanes = tuple(report["generator_eligible_lanes"])

    by_key: dict[
        tuple[str, int, str], dict[str, dict[str, Any]]
    ] = {}
    for cell in report["cells"]:
        lane = cell["precision"]
        if lane not in lanes:
            continue
        key = (cell["geometry"], int(cell["batch"]), lane)
        arms = by_key.setdefault(key, {})
        arm = cell["arm"]
        if arm == "compact-eager":
            # Keep every activity row; selection reads dcp levels.
            arms[f"compact:{cell['activity']}"] = cell
        else:
            arms.setdefault(arm, cell)

    entries: dict[tuple[str, int, str], str] = {}
    factor = 1.0 - hysteresis_pct / 100.0
    for key, arms in by_key.items():
        dense = arms.get("dense-eager")
        if dense is None:
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
    )
