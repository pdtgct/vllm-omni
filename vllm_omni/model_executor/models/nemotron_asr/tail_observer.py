# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Bounded, opt-in diagnostic for a two-attempt conditional decoder tail.

Reduction runs after each actual bucket invocation, outside captured graphs.
It owns a copy before the next invocation may reuse decoder workspace. D2H
shares the commit sink's existing event; host inspection happens at collect.
This diagnostic can perturb batching and must be disabled for timing runs.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch


class ConditionalTailObserver:
    def __init__(
        self,
        *,
        max_invocations: int,
        device: torch.device,
        cadences: Mapping[str, tuple[int, int]],
        max_symbols: int,
    ) -> None:
        if max_invocations <= 0:
            raise ValueError("conditional-tail record bound must be positive")
        self._limit = max_invocations
        self._cadences = cadences
        self._max_symbols = max_symbols
        # At most one invocation per cadence bucket per transaction.
        self._max_frames = max(right + 1 for _, right in self._cadences.values())
        shape = (len(self._cadences), self._max_frames + 1, 5)
        self._device = torch.zeros(shape, dtype=torch.int32, device=device)
        self._host = torch.zeros(shape, dtype=torch.int32, pin_memory=device.type == "cuda")
        self._pending: list[dict[str, Any]] = []
        self._records: list[dict[str, Any]] = []
        self._attempted = 0
        self._collected = 0
        self._dropped = 0
        self._invalid = 0
        self._step = 0

    def begin(self) -> None:
        # An earlier transaction may have failed before reserve/stage. Never
        # silently reuse its partial observation as a later transaction's data.
        self._dropped += len(self._pending)
        self._pending = []
        self._step += 1

    def observe(
        self,
        *,
        rows: tuple[int, ...],
        geometry: int,
        tier: int,
        arm: str,
        chunk_graph: bool,
        counts: torch.Tensor | None,
        lengths: torch.Tensor | None,
        final_tail: torch.Tensor,
    ) -> None:
        self._attempted += 1
        if len(self._records) + len(self._pending) >= self._limit or len(self._pending) >= len(self._cadences):
            self._dropped += 1
            return
        live = len(rows)
        record: dict[str, Any] = {
            "invocation": self._attempted,
            "transaction": self._step,
            "rows": list(rows),  # transaction-local membership; no session identity
            "geometry": geometry,
            "cadence_ms": list(self._cadences)[geometry].removesuffix("ms"),
            "live_rows": live,
            "decoder_tier": tier,
            "arm": arm,
            "chunk_graph": chunk_graph,
            "error": None,
        }
        slot = len(self._pending)
        self._pending.append(record)
        if (
            counts is None
            or lengths is None
            or counts.ndim != 2
            or counts.shape[0] != live
            or counts.shape[1] != list(self._cadences.values())[geometry][1] + 1
            or counts.dtype not in (torch.int32, torch.int64)
            or lengths.shape != (live,)
            or lengths.dtype not in (torch.int32, torch.int64)
            or final_tail.shape != (live,)
            or counts.device != self._device.device
            or lengths.device != self._device.device
            or live == 0
            or tier < live
        ):
            record["error"] = "missing-or-malformed-frame-observation"
            return
        width = counts.shape[1]
        record["frame_width"] = width
        valid = torch.arange(width, device=counts.device)[None, :] < lengths.clamp(0, width)[:, None]
        final = final_tail.to(device=counts.device, dtype=torch.bool)
        invalid = (counts < 0) | (counts > self._max_symbols) | (~valid & (counts != 0))
        # These reductions create owned values on the current stream. Copy
        # completes in stream order before any later borrowed-workspace reuse.
        summary = torch.stack(
            (
                valid.sum(0),
                torch.where(valid, counts, 0).amax(0),
                (valid & (counts >= 2)).any(0),
                (valid & final[:, None]).sum(0),
                invalid.sum(0),
            ),
            dim=1,
        )
        self._device[slot, 0, 0].copy_(final.sum())
        self._device[slot, 1 : width + 1].copy_(summary)

    def stage(self) -> None:
        # No allocation or host read. The sink records its existing event AFTER
        # this copy and waits only at its normal status-consumption boundary.
        if self._pending:
            self._host.copy_(self._device, non_blocking=True)

    def collect(self) -> None:
        # Called only after the sink's event completes. Include every lane,
        # even if status/lease processing later rejects its transaction.
        for slot, record in enumerate(self._pending):
            self._collected += 1
            if record["error"] is None:
                width = record["frame_width"]
                values = self._host[slot, : width + 1].tolist()
                record["final_tail_rows"] = values[0][0]
                record["frames"] = [
                    {
                        "position": position,
                        "valid_lanes": row[0],
                        "fully_padded": row[0] == 0,
                        "max_emissions": row[1],
                        "tail_runs": bool(row[2]),
                        "final_valid_lanes": row[3],
                        "invalid_counts": row[4],
                    }
                    for position, row in enumerate(values[1:])
                ]
                if any(row[4] for row in values[1:]):
                    record["error"] = "invalid-emission-counts"
            if record["error"] is not None:
                self._invalid += 1
            self._records.append(record)
        self._pending = []

    def receipt(self) -> dict[str, Any]:
        complete = self._attempted == self._collected and self._dropped == 0 and self._invalid == 0
        return {
            "schema": "nemotron-conditional-tail/1",
            "diagnostic_only": True,
            "prefix_attempts": 2,
            "max_invocations": self._limit,
            "attempted_invocations": self._attempted,
            "collected_invocations": self._collected,
            "dropped_invocations": self._dropped,
            "invalid_invocations": self._invalid,
            "pending_invocations": len(self._pending),
            "coverage_complete": complete,
            "estimate_eligible": complete and self._collected > 0,
            "records": list(self._records),
        }
