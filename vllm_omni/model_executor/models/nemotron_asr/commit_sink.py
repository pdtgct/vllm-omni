# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The production composite commit sink (PORT-HOOK-001 as amended).

One bounded, model/worker-owned :class:`CommitSink` implementation:
``reserve`` admits status capacity and optional capture capacity
atomically, pre-commit, and pins the registry lease for the call's
complete ordered bindings; the returned reservation's single
post-commit ``stage`` is no-fail by construction — the status slot was
preallocated (pinned host memory on CUDA) at sink construction, so
staging is one non-blocking device-to-host copy plus reference stores.
The sink exposes only status-clean committed records once the
asynchronous status copy has completed at the normal scheduling
boundary; durable persistence stays outside the model transaction.

Torch + stdlib only: locally testable under the loader chain.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch

from vllm_omni.model_executor.models.nemotron_asr.advance import (
    CaptureRecord,
    CommitPlan,
    PreparedRowBinding,
)
from vllm_omni.model_executor.models.nemotron_asr.plan import (
    SessionRegistry,
)


@dataclass(frozen=True)
class SessionStatusReport:
    """One row's consumed status, joined to its session identity."""

    request_id: str
    admission_generation: int
    row_status: int


class CommitTicket:
    """One reserved composite status/capture slot (CommitReservation).

    ``stage`` runs post-commit exactly once and cannot fail: it copies
    the device status into the sink's preallocated host slot
    (non-blocking), records the completion event, revalidates the
    registry lease as a RECORDED FACT (never a raise — nothing may
    fail after resident state committed), and stores the prepared
    candidate records. ``cancel`` is idempotent.
    """

    def __init__(self, sink: BoundedCommitSink, plan: CommitPlan) -> None:
        self._sink = sink
        self._plan = plan
        self._done = False

    def stage(
        self,
        row_status: torch.Tensor,
        records: Sequence[CaptureRecord],
    ) -> None:
        if self._done:
            return
        self._done = True
        self._sink._stage(self._plan, row_status, list(records))

    def cancel(self) -> None:
        if self._done:
            return
        self._done = True
        self._sink._release()


class BoundedCommitSink:
    """Bounded composite sink over one engine worker's serial steps.

    The synchronous engine loop runs one transaction at a time, so one
    in-flight reservation is the capacity model; a second ``reserve``
    before the previous step's ``collect`` (or cancel) fails
    pre-commit rather than overwriting staged-but-unconsumed status.
    """

    def __init__(
        self,
        registry: SessionRegistry,
        *,
        max_rows: int,
        max_capture_rows: int = 0,
        max_capture_bytes: int = 0,
        device: torch.device | None = None,
    ) -> None:
        if max_rows <= 0:
            raise ValueError("max_rows must be positive")
        self._registry = registry
        self._max_rows = max_rows
        self._max_capture_rows = max_capture_rows
        self._max_capture_bytes = max_capture_bytes
        self._cuda = device is not None and device.type == "cuda"
        # Preallocated once: staging never allocates (no-fail stage).
        self._status_host = torch.zeros(
            max_rows, dtype=torch.int32, pin_memory=self._cuda
        )
        self._event: torch.cuda.Event | None = (
            torch.cuda.Event() if self._cuda else None
        )
        self._reserved: CommitPlan | None = None
        self._staged: CommitPlan | None = None
        self._staged_records: list[CaptureRecord] = []
        self._staged_lease_ok = False

    def reserve(self, plan: CommitPlan) -> CommitTicket:
        """Pre-commit composite admission (fallible by design)."""
        if self._reserved is not None or self._staged is not None:
            raise ValueError(
                "commit sink already holds an unconsumed reservation — "
                "collect or cancel the previous step first"
            )
        rows = len(plan.bindings)
        if rows > self._max_rows:
            raise ValueError(
                f"{rows} status rows exceed the sink bound "
                f"{self._max_rows}"
            )
        for binding in plan.bindings:
            if not isinstance(binding, PreparedRowBinding):
                raise ValueError("commit plan bindings are malformed")
        if plan.capture is not None:
            if plan.capture.rows > self._max_capture_rows:
                raise ValueError(
                    f"{plan.capture.rows} capture rows exceed the sink "
                    f"bound {self._max_capture_rows}"
                )
            if plan.capture.payload_bytes > self._max_capture_bytes:
                raise ValueError(
                    f"{plan.capture.payload_bytes} capture bytes exceed "
                    f"the sink bound {self._max_capture_bytes}"
                )
        # Pin the lease: every binding must be CURRENT in the registry.
        self._registry.validate_lease(plan.bindings)
        self._reserved = plan
        return CommitTicket(self, plan)

    def _stage(
        self,
        plan: CommitPlan,
        row_status: torch.Tensor,
        records: list[CaptureRecord],
    ) -> None:
        rows = len(plan.bindings)
        self._status_host[:rows].copy_(row_status, non_blocking=True)
        if self._event is not None:
            self._event.record()
        # Stage-time lease REVALIDATION is a recorded fact, never a
        # raise: nothing may fail after resident state committed.
        self._staged_lease_ok = self._registry.lease_is_current(
            plan.bindings
        )
        self._staged_records = records
        self._staged = plan
        self._reserved = None

    def _release(self) -> None:
        self._reserved = None
        self._staged = None
        self._staged_records = []
        self._staged_lease_ok = False

    @property
    def has_staged(self) -> bool:
        return self._staged is not None

    def collect(
        self,
    ) -> tuple[list[SessionStatusReport], list[CaptureRecord], bool]:
        """Consume the staged step at the scheduling boundary.

        Synchronizes the asynchronous status copy (legal here — this is
        the host's normal consumption point, outside the model
        transaction), then returns per-session status reports, the
        status-clean committed capture records only, and whether the
        stage-time lease revalidation held. Releases the slot.
        """
        plan = self._staged
        if plan is None:
            raise ValueError("no staged commit to collect")
        if self._event is not None:
            self._event.synchronize()
        rows = len(plan.bindings)
        status = self._status_host[:rows].tolist()
        reports = [
            SessionStatusReport(
                request_id=binding.request_id,
                admission_generation=binding.admission_generation,
                row_status=int(status[i]),
            )
            for i, binding in enumerate(plan.bindings)
        ]
        by_request = {
            report.request_id: report.row_status for report in reports
        }
        committed = [
            record
            for record in self._staged_records
            if by_request.get(record.request_id, -1) == 0
        ]
        lease_ok = self._staged_lease_ok
        self._release()
        return reports, committed, lease_ok
