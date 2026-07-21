# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Full-turn CUDA probe: sync-free / allocation-free serving turn.

Phase 6c full-turn evidence (ledger §8): unlike ``p6b_cuda_matrix_probe.py``
(``advance_session`` alone, torch-only), this drives the WHOLE outer
transaction — pools, a real :class:`~...plan.SessionRegistry`, a
:class:`~...plan.PlanContext` minted via ``plan.prepare_plan_context``
(``ObservedRow`` lists, no engine), :func:`~...advance.advance_model_rows`
with the MRV1 adapter, a dense-eager fixed resolver, a
:class:`~...commit_sink.BoundedCommitSink`, and
:class:`~...advance.HostStaging` — at both tiny-core scale (arms a-d) and
one production-shape smoke (arm e). POD-TIER: importing
``vllm_omni.model_executor.models.nemotron_asr.nemotron_asr`` pulls
``vllm_omni``, which pulls ``vllm`` — this file cannot be collected on
macOS and must run on the pod venv (do NOT add it to a local pytest run).

Five arms, all evidence into one JSON report:

``sync``       — ``torch.cuda.set_sync_debug_mode(2)`` as a TRIPWIRE
  around a complete turn's dense-eager transaction calls (expected
  CLEAN), paired with Kineto host-trace sync-marker counts, on two
  INDEPENDENT identically-seeded fixtures (tripwire-only, never
  profiled; Kineto-only, never sync-debugged — profiler
  instrumentation can itself sync, so measuring both at once would
  contaminate the tripwire verdict). Covers (i) a mixed batch — 2
  geometry buckets (ids 0 and 2) + one REPLAY row + one FLUSH row —
  and (ii) a follow-up echo turn (the two CHUNK rows' first turn-1
  label, precomputed off-trace via a reference single-row
  ``advance_session`` call so bridging the two turns needs no runtime
  device read). The sink's ``collect()`` is the sole sanctioned host
  sync (via ``torch.cuda.Event.synchronize()``) but is called plainly,
  NOT under the tripwire: confirmed on real hardware (pod evidence,
  2026-07-21) that ``set_sync_debug_mode`` does not cover
  ``Event.synchronize()`` — PyTorch's own docs mark the feature
  "experimental... not all synchronizing operations are currently
  covered" — so a prior "expect it to fire, then retry" design here
  silently succeeded on the dry attempt and released the only staged
  commit, crashing the real attempt on an empty sink. ``collect()``'s
  sync is instead verified via the Kineto fixture's marker-presence
  check (``collectN_kineto``, the full ``SYNC_MARKERS``). The TURN
  body's own strict-empty check (``turnN_kineto``) uses the narrower
  ``TURN_BODY_SYNC_MARKERS`` — ground-truth traced against a real
  exported turn trace (pod evidence, 2026-07-21):
  ``aten::nonzero`` is Kineto category ``cpu_op`` (RowPlan's
  host-authority tensors, never CUDA); ``Memcpy HtoD``/``Memcpy
  DtoH`` pair with ``cudaMemcpyAsync`` (``HostStaging``'s pinned
  copies, ``BoundedCommitSink._stage``'s non-blocking status copy);
  and the sole ``cudaDeviceSynchronize`` was the trace's absolute
  LAST event, past a timing gap after the last real op — the
  profiler's own stop-tracing sync, not code under test. All three
  are legitimate noise for THIS check's purpose, not evidence of a
  blocking transaction.
``allocation`` — ``torch.cuda.memory_stats()["allocation.all.allocated"]``
  snapshotted immediately before the first prevalidated scatter and
  immediately after the commit ticket's ``stage()`` (monkeypatching
  ``advance._execute_masked_page_scatter_`` and
  ``commit_sink.CommitTicket.stage``): the delta across that window
  must be exactly zero. Whole-turn allocation counts are also recorded
  (informational, not asserted).
``warmup``     — after ``warmup_advance_model_rows_scatter``, the Triton
  cache directory's file count is unchanged and the second
  transaction's commit-window wall time stays within 3x of the
  first's across two transactions (no late JIT spike); a deliberately
  unwarmed pool layout is confirmed to raise through
  ``validate_masked_page_scatter``.
``resolver``   — a recording wrapper around the dense-eager fixed
  resolver proves ``ready_decode_buckets == 2`` on every request for a
  genuine two-bucket turn; the same turn through a compact-eager
  DECLARED arm (``nemotron_asr.build_decode_resolver`` over a
  ``SimpleNamespace`` config) is forced to dense-eager by the
  multi-bucket override, with ``override_reason`` recorded on both
  resolutions.
``production`` — ONE transaction at real dims (24 layers, d_model
  1024, window 56, kernel 9, pred 640x2, vocab 13088, n_mels 128,
  geometry 4 = 1120 ms, batch 8) with synthetic (randomly initialized)
  weights via the real ``NemotronASRCore``: wall time, peak allocated
  memory, and a clean (all-zero) row-status report. No perf matrix.

Run (on the pod):
    /opt/venv-port/bin/python tests/model_executor/models/nemotron_asr/\
      p6c_full_turn_probe.py --out /workspace/evidence/p6c-full-turn/report.json
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
import traceback
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch

from vllm_omni.model_executor.models.nemotron_asr import advance, manifests, rnnt, state_scatter
from vllm_omni.model_executor.models.nemotron_asr import commit_sink as commit_sink_mod
from vllm_omni.model_executor.models.nemotron_asr import encoder as encoder_mod
from vllm_omni.model_executor.models.nemotron_asr import featurizer as featurizer_mod
from vllm_omni.model_executor.models.nemotron_asr import frontend as frontend_mod
from vllm_omni.model_executor.models.nemotron_asr import lid as lid_mod
from vllm_omni.model_executor.models.nemotron_asr import plan as plan_mod
from vllm_omni.model_executor.models.nemotron_asr.nemotron_asr import (
    NemotronASRCore,
    build_decode_resolver,
)

#: ---- tiny-core dims (arms a-d) — matches
#: test_advance_model_rows_local.py's proven fixture exactly, so the
#: mixed-geometry/echo behaviors this probe leans on are pinned by an
#: already-green local bar, not reinvented here.
FEAT = 16
D_MODEL = 32
N_LAYERS = 2
KERNEL = 5
WINDOW = 8
PRED_HIDDEN = 16
PARK_ID = 9000
PLACEHOLDER_ID = 9001
VOCAB = 12
NUM_PROMPTS = 4
CARRIER_HIDDEN = 5_127
RAW_TAIL = 1_953
NULL_BLOCK_ID = 0
NUM_GEOMETRIES = len(manifests.CADENCES)
GEOM_REG = 0  # "80ms"
REG_SAMPLES = 1_280
#: A LEGAL final-tail size at GEOM_REG: ROW_STATUS_FINAL_OVERSIZE fires
#: at valid_samples >= cadence*hop == REG_SAMPLES exactly, so a final
#: chunk at this geometry must stay strictly under it.
REG_FINAL_SAMPLES = 1_120
GEOM_FINAL = 2  # "320ms"
FINAL_SAMPLES = 3_840
CAP = 48

_BOOK = {name: i for i, (name, _) in enumerate(manifests.BOOK_FIELDS)}
_CTR = {name: i for i, name in enumerate(manifests.FRONTEND_COUNTER_FIELDS)}
BOOK_WIDTH = len(manifests.BOOK_FIELDS)
CTR_WIDTH = len(manifests.FRONTEND_COUNTER_FIELDS)

#: ---- production dims (arm e) — real NemotronASRCore/FastConformer
#: defaults (d_model 1024, 24 layers, window 56, kernel 9, pred
#: 640x2, n_mels 128); geometry 4 == "1120ms" (56, 13).
PROD_VOCAB = 13_088
PROD_PARK_ID = 13_089
PROD_PLACEHOLDER_ID = 13_090
PROD_WINDOW = 56
PROD_KERNEL = 9
PROD_PRED_HIDDEN = 640
PROD_N_LAYERS = 24
PROD_GEOM = 4  # "1120ms"
PROD_SAMPLES = manifests.RAW_SAMPLES_PER_CHUNK["1120ms"]
PROD_HIDDEN = advance.ENVELOPE_HEADER_SLOTS + PROD_SAMPLES
PROD_BATCH = 8
#: (lookahead + 1) * max_symbols_per_step == 140 exactly for 1120ms —
#: the queue capacity manifests.py's own limit was sized for.
PROD_QUEUE_CAPACITY = manifests.SESSION_LIMITS["queue_capacity"]

#: Reused from p6b's own Kineto convention, plus cudaEventSynchronize
#: (BoundedCommitSink.collect's own sync primitive — a distinct CUDA
#: API from decode's cudaStreamSynchronize/nonzero).
SYNC_MARKERS = (
    "cudaStreamSynchronize",
    "cudaDeviceSynchronize",
    "cudaEventSynchronize",
    "aten::nonzero",
    "Memcpy DtoH",
    "Memcpy HtoD",
)

#: The subset of SYNC_MARKERS that unambiguously means the CODE UNDER
#: TEST blocked the host mid-transaction — used for the TURN body's
#: strict-empty check specifically (turnN_kineto), never for
#: collectN_kineto (which legitimately wants SYNC_MARKERS' full set
#: present, since collect() is the one place a sync is sanctioned).
#: Ground-truth traced against a real turn's exported Kineto trace
#: (pod evidence, 2026-07-21) rather than assumed: SYNC_MARKERS' other
#: three members all turned out to be false positives for a strict-
#: empty turn-body check —
#:   - "aten::nonzero": every match is Kineto category "cpu_op", not a
#:     CUDA op at all. All four .nonzero() call sites in this module
#:     operate on RowPlan's host-authority fields (is_chunk,
#:     geometry_id, has_initial_states_p) — CPU tensors by
#:     construction (plan.py's prepare_plan_context builds every
#:     PlanContext field via bare torch.tensor(list, dtype=...), no
#:     device= kwarg).
#:   - "Memcpy HtoD"/"Memcpy DtoH": every observed instance pairs with
#:     a cudaMemcpyAsync runtime call (confirmed via the trace's event
#:     sequence) — the expected, by-design non-blocking traffic from
#:     HostStaging's pinned per-geometry/per-slot copies and
#:     BoundedCommitSink._stage's non_blocking=True status copy, not a
#:     blocking transfer.
#:   - "cudaDeviceSynchronize": the sole observed instance was the
#:     trace's ABSOLUTE LAST event, ~288us after the last real op,
#:     immediately preceding the trace's own "Record Window End"
#:     marker — torch.profiler's own stop-tracing synchronize, not a
#:     call our code made (advance_model_rows/state_scatter/
#:     commit_sink have exactly two .synchronize() call sites, neither
#:     reachable during a turn: state_scatter's is warmup-only,
#:     already executed before profiling starts; commit_sink's is
#:     inside collect(), never called during turn1()/turn2()).
#: cudaStreamSynchronize/cudaEventSynchronize are kept: unlike the
#: three above, no known benign source in this codebase produces
#: them, and they correspond directly to an explicit .synchronize()
#: call our own code (or a real regression) could make.
TURN_BODY_SYNC_MARKERS = (
    "cudaStreamSynchronize",
    "cudaEventSynchronize",
)


class Check:
    """Failure collector (p6b's convention): one bad cell never hides
    the rest of the report."""

    def __init__(self) -> None:
        self.failures: list[str] = []
        self.count = 0

    def ok(self, cond: bool, ctx: str) -> None:
        self.count += 1
        if not cond:
            self.failures.append(ctx)

    def equal(self, a: Any, b: Any, ctx: str) -> None:
        self.ok(a == b, f"{ctx}: {a!r} != {b!r}")


# ---- tiny-core construction (mirrors p6b's _core()) ------------------------


def _tiny_core(device: str, seed: int = 1) -> Any:
    torch.manual_seed(seed)
    encoder = encoder_mod.FastConformerEncoder(
        feat_in=FEAT,
        d_model=D_MODEL,
        d_ff=64,
        n_layers=N_LAYERS,
        n_heads=4,
        conv_kernel=KERNEL,
        subsampling_channels=16,
        att_context=(WINDOW, 1),
    )
    core = SimpleNamespace(
        encoder=encoder,
        lid=lid_mod.PromptConditioner(enc_hidden=D_MODEL, num_prompts=NUM_PROMPTS),
        predictor=rnnt.Predictor(vocab_size=VOCAB, pred_hidden=PRED_HIDDEN, pred_rnn_layers=2),
        joint=rnnt.Joint(enc_hidden=D_MODEL, pred_hidden=PRED_HIDDEN, joint_hidden=16, vocab_size=VOCAB),
        featurizer=featurizer_mod.MelFeaturizer(
            filterbank=torch.rand(FEAT, 257) * 0.01,
            window=torch.hann_window(400),
        ),
        blank_id=VOCAB,
    )
    for m in (core.encoder, core.lid, core.predictor, core.joint, core.featurizer):
        m.to(device)
        m.eval()
    return core


def _fresh_pools(
    *,
    n_layers: int,
    window: int,
    d_model: int,
    kernel: int,
    pred_layers: int,
    pred_hidden: int,
    cap: int,
    raw_tail: int,
    n_mels: int,
    num_blocks: int,
    device: str,
) -> dict[str, Any]:
    return {
        "channel_pools": [
            torch.zeros(num_blocks, window, d_model, device=device) for _ in range(n_layers)
        ],
        "time_pools": [
            torch.zeros(num_blocks, d_model, kernel - 1, device=device) for _ in range(n_layers)
        ],
        "len_pools": [
            torch.zeros(num_blocks, 1, dtype=torch.int32, device=device) for _ in range(n_layers)
        ],
        "h_pool": torch.zeros(num_blocks, pred_layers, pred_hidden, device=device),
        "c_pool": torch.zeros(num_blocks, pred_layers, pred_hidden, device=device),
        "queue_pool": torch.zeros(num_blocks, cap, dtype=torch.int32, device=device),
        "book_pool": torch.zeros(num_blocks, BOOK_WIDTH, dtype=torch.int32, device=device),
        "frontend_raw_pool": torch.zeros(num_blocks, raw_tail, device=device),
        "frontend_mel_pool": torch.zeros(
            num_blocks, n_mels, frontend_mod.MEL_TAIL_FRAMES, device=device
        ),
        "frontend_counter_pool": torch.zeros(num_blocks, CTR_WIDTH, dtype=torch.int64, device=device),
    }


def _header(
    *,
    valid: int,
    geometry: int,
    final: bool,
    prompt: int = 0,
    seq: int = 0,
    admission_ms_mod: int = 0,
) -> tuple[float, ...]:
    return (
        float(advance.ENVELOPE_VERSION),
        float(valid),
        float(geometry),
        1.0 if final else 0.0,
        float(prompt),
        float(seq),
        float(admission_ms_mod),
    )


def _carrier(
    samples: torch.Tensor,
    *,
    final: bool,
    seq: int,
    geometry: int,
    prompt: int,
    hidden: int,
    admission_ms_mod: int = 0,
) -> torch.Tensor:
    n = samples.shape[0]
    row = torch.zeros(hidden)
    row[advance.ENV_VERSION] = advance.ENVELOPE_VERSION
    row[advance.ENV_VALID_SAMPLES] = n
    row[advance.ENV_GEOMETRY_ID] = geometry
    row[advance.ENV_FINAL_TAIL] = 1.0 if final else 0.0
    row[advance.ENV_PROMPT_INDEX] = prompt
    row[advance.ENV_CHUNK_SEQUENCE] = seq
    row[advance.ENV_ADMISSION_MS_MOD] = admission_ms_mod
    row[advance.ENVELOPE_HEADER_SLOTS : advance.ENVELOPE_HEADER_SLOTS + n] = samples
    return row


def _set_replay_book(pools: dict[str, Any], block: int, *, queue: list[int], head: int, expected: int, geometry: int, prompt: int) -> None:
    book = pools["book_pool"]
    qp = pools["queue_pool"]
    for i, label in enumerate(queue):
        qp[block, i] = label
    book[block, _BOOK["queue_head"]] = head
    book[block, _BOOK["queue_length"]] = len(queue)
    book[block, _BOOK["last_label"]] = queue[-1]
    book[block, _BOOK["pending_echo"]] = 1
    book[block, _BOOK["expected_label"]] = expected
    book[block, _BOOK["geometry"]] = geometry
    book[block, _BOOK["prompt"]] = prompt


def _set_drained_book(pools: dict[str, Any], block: int, *, blank: int, geometry: int, prompt: int) -> None:
    book = pools["book_pool"]
    book[block, _BOOK["queue_head"]] = 0
    book[block, _BOOK["queue_length"]] = 0
    book[block, _BOOK["last_label"]] = blank
    book[block, _BOOK["pending_echo"]] = 0
    book[block, _BOOK["expected_label"]] = 0
    book[block, _BOOK["geometry"]] = geometry
    book[block, _BOOK["prompt"]] = prompt


def _dense_eager_resolver(request: advance.DecodeRequest) -> advance.ResolvedDecode:
    """The fixed dense-eager resolver every arm drives production
    transactions through — no measured dispatch table is needed for a
    sync/allocation/warmup probe (PORT-DEC-008: startup dispatch is a
    SEPARATE, already-tested seam; see ``decode_dispatch``)."""
    del request
    return advance.ResolvedDecode(arm="dense-eager", decode_fn=rnnt.decode_dense_masked)


class _RecordingResolver:
    def __init__(self, inner: advance.DecodeResolver) -> None:
        self.requests: list[advance.DecodeRequest] = []
        self.resolutions: list[advance.ResolvedDecode] = []
        self._inner = inner

    def __call__(self, request: advance.DecodeRequest) -> advance.ResolvedDecode:
        self.requests.append(request)
        resolved = self._inner(request)
        self.resolutions.append(resolved)
        return resolved


def _reference_first_label(
    core: Any, samples: torch.Tensor, *, geometry: int, prompt: int, device: str
) -> int:
    """The label a fresh session-first CHUNK emits first, via a direct
    single-row ``advance_session`` call — computed OFF the measured/
    traced region so bridging turn 1 -> turn 2 needs no device read."""
    n_layers = len(core.encoder.layers)
    state = advance.SessionStateBatch(
        raw_tail=torch.zeros(1, RAW_TAIL, device=device),
        mel_tail=torch.zeros(1, FEAT, frontend_mod.MEL_TAIL_FRAMES, device=device),
        frontend_counters=torch.zeros(1, CTR_WIDTH, dtype=torch.int64, device=device),
        channel=[torch.zeros(1, WINDOW, D_MODEL, device=device) for _ in range(n_layers)],
        window_valid=[torch.zeros(1, 1, dtype=torch.int32, device=device) for _ in range(n_layers)],
        time=[torch.zeros(1, D_MODEL, KERNEL - 1, device=device) for _ in range(n_layers)],
        h=torch.zeros(1, 2, PRED_HIDDEN, device=device),
        c=torch.zeros(1, 2, PRED_HIDDEN, device=device),
        last_label=torch.full((1,), core.blank_id, dtype=torch.long, device=device),
    )
    batch = advance.ChunkBatch(
        samples=samples.unsqueeze(0).to(device),
        valid_samples=torch.tensor([samples.shape[0]], device=device),
        geometry_id=torch.tensor([geometry], device=device),
        final_tail=torch.tensor([True], device=device),
        prompt_index=torch.tensor([prompt], device=device),
        chunk_sequence=torch.tensor([0], device=device),
    )
    result = advance.advance_session(
        core, batch, state, geometry=geometry, decode_fn=rnnt.decode_dense_masked
    )
    torch.accelerator.synchronize()
    length = int(result.token_lengths[0])
    if length == 0:
        return PARK_ID
    return int(result.token_ids[0, 0])


def _register_fresh(
    registry: plan_mod.SessionRegistry,
    *,
    request_id: str,
    block: int,
    geometry: int,
    prompt: int,
    now_ns: int,
    resident: list[str],
) -> None:
    """Register one session's identity WITHOUT running a transition —
    the block/generation authority ``prepare_plan_context`` needs to
    treat a later row as CONTINUING. Pool state (book/queue/counters)
    is seeded separately by the caller.

    ``resident`` names every OTHER session (registered by an earlier
    call) that must survive this call's prune; the row being
    registered here is added automatically — ``bind_rows`` requires
    every row it's handed to also appear in ``resident_request_ids``.
    """
    registry.bind_rows(
        [
            plan_mod.ObservedRow(
                request_id=request_id,
                block_id=block,
                scheduled_token_id=PLACEHOLDER_ID,
                has_prior_state=False,
                envelope_header=_header(valid=1, geometry=geometry, final=True, prompt=prompt),
            )
        ],
        placeholder_id=PLACEHOLDER_ID,
        num_prompts=NUM_PROMPTS,
        num_geometries=NUM_GEOMETRIES,
        now_ns=now_ns,
        resident_request_ids=[request_id, *resident],
    )


def _context(
    registry: plan_mod.SessionRegistry,
    rows: list[plan_mod.ObservedRow],
    *,
    now_ns: int,
    step: int,
    num_prompts: int = NUM_PROMPTS,
) -> plan_mod.PlanContext:
    return plan_mod.prepare_plan_context(
        registry,
        rows,
        resident_request_ids=[r.request_id for r in rows],
        placeholder_id=PLACEHOLDER_ID,
        num_prompts=num_prompts,
        num_geometries=NUM_GEOMETRIES,
        now_ns=now_ns,
        step=step,
    )


# ---- arm (a): sync-free full turn ------------------------------------------


def run_sync_arm(check: Check, device: str, out_dir: Path) -> dict[str, Any]:
    from torch.profiler import ProfilerActivity, profile

    cell: dict[str, Any] = {}
    # The core is inference-only (weights never mutate), so every
    # fixture below shares it; only the RESIDENT state (pools, registry,
    # sink, staging) is rebuilt per run.
    core = _tiny_core(device, seed=1)
    adapter = advance.make_mrv1_adapter(
        hidden_size=CARRIER_HIDDEN, park_id=PARK_ID, blank_id=core.blank_id
    )

    # Reference labels, off-trace and deterministic: what each fresh
    # CHUNK will emit — used to build turn 2's input_ids without a
    # mid-run device read. Identical for every fixture below.
    torch.manual_seed(21)
    chunk_reg_samples = torch.randn(REG_FINAL_SAMPLES) * 0.01
    torch.manual_seed(22)
    chunk_final_samples = torch.randn(FINAL_SAMPLES) * 0.01
    ref_reg = _reference_first_label(core, chunk_reg_samples, geometry=GEOM_REG, prompt=0, device=device)
    ref_final = _reference_first_label(core, chunk_final_samples, geometry=GEOM_FINAL, prompt=0, device=device)
    cell["reference_first_labels"] = {"chunk_reg": ref_reg, "chunk_final": ref_final}

    def build_fixture() -> SimpleNamespace:
        """One complete, independent turn fixture: fresh resident pools,
        registry, sink, and staging, seeded to the pending-echo/drained
        states a real prior CHUNK would have left. Deterministic — every
        call reproduces byte-identical inputs, so the tripwire run and
        the Kineto run measure the SAME transaction on INDEPENDENT state
        (neither retries the other's mutated pools / burned registry
        generation, the reason a single shared run had to combine both
        and could not cleanly separate them)."""
        num_blocks = 8
        pools = _fresh_pools(
            n_layers=N_LAYERS, window=WINDOW, d_model=D_MODEL, kernel=KERNEL,
            pred_layers=2, pred_hidden=PRED_HIDDEN, cap=CAP, raw_tail=RAW_TAIL,
            n_mels=FEAT, num_blocks=num_blocks, device=device,
        )
        advance.warmup_advance_model_rows_scatter(**pools)
        registry = plan_mod.SessionRegistry()
        sink = commit_sink_mod.BoundedCommitSink(registry, max_rows=8, device=torch.device(device))
        staging = advance.HostStaging(8)
        _register_fresh(registry, request_id="replay_sess", block=1, geometry=GEOM_REG, prompt=0, now_ns=1, resident=[])
        _register_fresh(registry, request_id="flush_sess", block=2, geometry=GEOM_FINAL, prompt=0, now_ns=2, resident=["replay_sess"])
        _set_replay_book(pools, 1, queue=[3, 5], head=1, expected=3, geometry=GEOM_REG, prompt=0)
        _set_drained_book(pools, 2, blank=core.blank_id, geometry=GEOM_FINAL, prompt=0)
        pools["frontend_counter_pool"][2, _CTR["finalized"]] = 1
        pools["frontend_counter_pool"][2, _CTR["expected_chunk_sequence"]] = 1
        torch.accelerator.synchronize()

        rows1 = [
            plan_mod.ObservedRow("replay_sess", 1, 3, True, None),
            plan_mod.ObservedRow("flush_sess", 2, PARK_ID, True, None),
            plan_mod.ObservedRow(
                # A session-first FINAL chunk (not a regular mid-session
                # one): turn 2 treats chunk_reg as a continuing decode row
                # regardless of whether the burst was empty, and only a
                # FINALIZED session's drained queue is a legal FLUSH
                # (an unfinalized drained queue is ROW_STATUS_SESSION_
                # PROTOCOL — a regular chunk expects another CHUNK next,
                # never a decode/replay/flush step).
                "chunk_reg", 3, PLACEHOLDER_ID, False,
                _header(valid=REG_FINAL_SAMPLES, geometry=GEOM_REG, final=True, seq=0),
            ),
            plan_mod.ObservedRow(
                "chunk_final", 4, PLACEHOLDER_ID, False,
                _header(valid=FINAL_SAMPLES, geometry=GEOM_FINAL, final=True, seq=0),
            ),
        ]
        context1 = _context(registry, rows1, now_ns=100, step=1)
        plan1 = plan_mod.build_row_plan(
            context1, num_decodes=2, num_prefills=2, null_block_id=NULL_BLOCK_ID, num_pool_blocks=num_blocks
        )
        input_ids1 = torch.tensor([3, PARK_ID, PLACEHOLDER_ID, PLACEHOLDER_ID], dtype=torch.long, device=device)
        embeds1 = torch.stack(
            [
                torch.zeros(CARRIER_HIDDEN),
                torch.zeros(CARRIER_HIDDEN),
                _carrier(chunk_reg_samples, final=True, seq=0, geometry=GEOM_REG, prompt=0, hidden=CARRIER_HIDDEN),
                _carrier(chunk_final_samples, final=True, seq=0, geometry=GEOM_FINAL, prompt=0, hidden=CARRIER_HIDDEN),
            ]
        ).to(device)

        rows2 = [
            plan_mod.ObservedRow("chunk_reg", 3, ref_reg, True, None),
            plan_mod.ObservedRow("chunk_final", 4, ref_final, True, None),
        ]
        input_ids2 = torch.tensor([ref_reg, ref_final], dtype=torch.long, device=device)
        embeds2 = torch.zeros(2, CARRIER_HIDDEN, device=device)

        def turn1() -> torch.Tensor:
            return advance.advance_model_rows(
                core, input_ids1, embeds1, plan1,
                adapter=adapter, decode_resolver=_dense_eager_resolver,
                placeholder_id=PLACEHOLDER_ID, park_id=PARK_ID,
                commit_sink=sink, capture=False, graph_covers_decode=False,
                staging=staging, **pools,
            )

        def turn2() -> torch.Tensor:
            # Built at call time: the continuing plan reads the registry
            # AFTER turn1/collect1 have advanced it.
            context2 = _context(registry, rows2, now_ns=200, step=2)
            plan2 = plan_mod.build_row_plan(
                context2, num_decodes=2, num_prefills=0, null_block_id=NULL_BLOCK_ID, num_pool_blocks=num_blocks
            )
            return advance.advance_model_rows(
                core, input_ids2, embeds2, plan2,
                adapter=adapter, decode_resolver=_dense_eager_resolver,
                placeholder_id=PLACEHOLDER_ID, park_id=PARK_ID,
                commit_sink=sink, capture=False, graph_covers_decode=False,
                staging=staging, **pools,
            )

        return SimpleNamespace(sink=sink, turn1=turn1, turn2=turn2)

    trace_dir = out_dir / "traces"
    trace_dir.mkdir(parents=True, exist_ok=True)

    def _run_guarded(fn: Any, label: str) -> tuple[bool, Any]:
        """Run one transaction under the sync-debug tripwire, capturing
        the COMPLETE failure (type, message, traceback) into the cell
        immediately — a fired tripwire, or any real error, must not be
        reduced to a downstream ``collect()`` ValueError. Always drains
        outstanding CUDA work in ``finally`` so a partial transaction
        cannot leak queued work into the next fixture."""
        torch.cuda.set_sync_debug_mode(2)
        try:
            out = fn()
            cell[f"{label}_tripwire"] = "clean"
            return True, out
        except Exception as err:  # tripwire raises RuntimeError; catch real errors too
            cell[f"{label}_tripwire"] = f"FIRED: {type(err).__name__}: {err}"
            cell[f"{label}_traceback"] = traceback.format_exc()
            return False, None
        finally:
            torch.cuda.set_sync_debug_mode(0)
            torch.accelerator.synchronize()

    # ======================================================================
    # Fixture A — TRIPWIRE ONLY (never profiled): the correctness signal
    # (each transaction is sync-free) plus the report's status/lease
    # content. Profiler instrumentation can itself introduce syncs, so a
    # FIRED result here names a real transaction sync, not a measurement
    # artifact.
    # ======================================================================
    fx_tw = build_fixture()

    ok1, turn1_out = _run_guarded(fx_tw.turn1, "turn1")
    check.equal(cell["turn1_tripwire"], "clean", "sync.turn1.tripwire")
    if not ok1:
        # PRIMARY failure fully captured (see turn1_traceback). Do NOT
        # call collect() — its "no staged commit" ValueError would
        # overwrite this evidence with an unrelated symptom.
        cell["arm_outcome"] = "halted: turn1 tripwire fired before staging (see turn1_traceback)"
        return cell
    del turn1_out
    # A clean turn1 MUST have staged; a clean-but-unstaged transaction is
    # a distinct postcondition defect, recorded as itself.
    if not fx_tw.sink.has_staged:
        check.ok(False, "sync.turn1.postcondition_staged")
        cell["arm_outcome"] = "halted: turn1 returned clean but nothing staged"
        return cell

    # collect() is the sole sanctioned host sync (via
    # torch.cuda.Event.synchronize()) — but PyTorch's own docs mark
    # set_sync_debug_mode "experimental... not all synchronizing
    # operations are currently covered", and Event.synchronize()
    # confirmed NOT covered on this build (pod evidence, 2026-07-21:
    # a "dry, expected-to-fire-then-retry" call here silently
    # succeeded and released the ONLY staged commit, so the real
    # retry crashed on an already-empty sink — a design that was
    # never actually exercised until this round, since earlier bugs
    # always crashed first). collect() is therefore called exactly
    # ONCE and its result used directly; the sync itself is verified
    # by the Kineto-only fixture below (collect1_kineto's marker
    # presence check), not by this tripwire.
    reports1, _records1, lease_ok1 = fx_tw.sink.collect()
    check.ok(lease_ok1, "sync.turn1.lease_ok")
    check.equal(
        sorted((r.request_id, r.row_status) for r in reports1),
        sorted(
            [
                ("replay_sess", 0), ("flush_sess", 0),
                ("chunk_reg", 0), ("chunk_final", 0),
            ]
        ),
        "sync.turn1.reports",
    )

    ok2, _turn2_out = _run_guarded(fx_tw.turn2, "turn2")
    check.equal(cell["turn2_tripwire"], "clean", "sync.turn2.tripwire")
    if not ok2:
        cell["arm_outcome"] = "halted: turn2 tripwire fired before staging (see turn2_traceback)"
        return cell
    if not fx_tw.sink.has_staged:
        check.ok(False, "sync.turn2.postcondition_staged")
        cell["arm_outcome"] = "halted: turn2 returned clean but nothing staged"
        return cell
    # See the collect1 comment above: called once, verified via Kineto
    # markers (collect2_kineto) instead of the (unsound for
    # Event.synchronize()) sync-debug tripwire.
    reports2, _records2, lease_ok2 = fx_tw.sink.collect()
    check.ok(lease_ok2, "sync.turn2.lease_ok")
    check.equal(
        sorted((r.request_id, r.row_status) for r in reports2),
        sorted([("chunk_reg", 0), ("chunk_final", 0)]),
        "sync.turn2.reports",
    )

    # ======================================================================
    # Fixture B — KINETO ONLY (never sync-debugged): the trace exports and
    # sync-marker counts, on second, independent, identically-seeded
    # state. Kept fully separate so profiler behavior can never
    # contaminate fixture A's tripwire verdicts above. Guarded so a
    # failure HERE (the secondary, trace-only fixture) degrades to a
    # recorded note rather than discarding fixture A's authoritative
    # correctness evidence already in ``cell``.
    # ======================================================================
    try:
        fx_kn = build_fixture()
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof1:
            fx_kn.turn1()
        cell["turn1_kineto"] = _sync_counts(prof1, TURN_BODY_SYNC_MARKERS)
        check.equal(cell["turn1_kineto"], {}, "sync.turn1.kineto_sync_markers")
        prof1.export_chrome_trace(str(trace_dir / "turn1.json"))
        # collect() is expected to sync — profile the real call to
        # confirm markers ARE present (its sanctioned host sync), and
        # advance the fixture for turn 2.
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as profc1:
            fx_kn.sink.collect()
        cell["collect1_kineto"] = _sync_counts(profc1)
        check.ok(bool(cell["collect1_kineto"]), "sync.collect1.kineto_sync_markers_present")
        profc1.export_chrome_trace(str(trace_dir / "collect1.json"))
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof2:
            fx_kn.turn2()
        cell["turn2_kineto"] = _sync_counts(prof2, TURN_BODY_SYNC_MARKERS)
        check.equal(cell["turn2_kineto"], {}, "sync.turn2.kineto_sync_markers")
        prof2.export_chrome_trace(str(trace_dir / "turn2.json"))
        fx_kn.sink.collect()  # drain the staged step
        cell["arm_outcome"] = "complete"
    except Exception as err:
        check.ok(False, "sync.kineto_fixture_uncaught")
        cell["kineto_fixture_error"] = f"{type(err).__name__}: {err}"
        cell["kineto_fixture_traceback"] = traceback.format_exc()
        cell["arm_outcome"] = "tripwire-complete; kineto fixture failed (see kineto_fixture_traceback)"
    return cell


def _sync_counts(
    prof: Any, markers: tuple[str, ...] = SYNC_MARKERS
) -> dict[str, int]:
    counts: dict[str, int] = {}
    for evt in prof.events():
        for marker in markers:
            if marker in evt.name:
                counts[marker] = counts.get(marker, 0) + 1
    return counts


# ---- arm (b): allocation-free commit window --------------------------------


def run_allocation_arm(check: Check, device: str) -> dict[str, Any]:
    cell: dict[str, Any] = {}
    core = _tiny_core(device, seed=3)
    num_blocks = 4
    pools = _fresh_pools(
        n_layers=N_LAYERS, window=WINDOW, d_model=D_MODEL, kernel=KERNEL,
        pred_layers=2, pred_hidden=PRED_HIDDEN, cap=CAP, raw_tail=RAW_TAIL,
        n_mels=FEAT, num_blocks=num_blocks, device=device,
    )
    advance.warmup_advance_model_rows_scatter(**pools)
    registry = plan_mod.SessionRegistry()
    sink = commit_sink_mod.BoundedCommitSink(registry, max_rows=4, device=torch.device(device))
    staging = advance.HostStaging(4)
    adapter = advance.make_mrv1_adapter(hidden_size=CARRIER_HIDDEN, park_id=PARK_ID, blank_id=core.blank_id)

    torch.manual_seed(31)
    samples = torch.randn(REG_SAMPLES) * 0.01
    rows = [
        plan_mod.ObservedRow(
            "a", 1, PLACEHOLDER_ID, False,
            _header(valid=REG_SAMPLES, geometry=GEOM_REG, final=False, seq=0),
        )
    ]
    context = _context(registry, rows, now_ns=1, step=1)
    plan = plan_mod.build_row_plan(
        context, num_decodes=0, num_prefills=1, null_block_id=NULL_BLOCK_ID, num_pool_blocks=num_blocks
    )
    input_ids = torch.tensor([PLACEHOLDER_ID], dtype=torch.long, device=device)
    embeds = _carrier(samples, final=False, seq=0, geometry=GEOM_REG, prompt=0, hidden=CARRIER_HIDDEN).unsqueeze(0).to(device)

    real_exec = advance._execute_masked_page_scatter_
    real_stage = commit_sink_mod.CommitTicket.stage
    snap: dict[str, int] = {}
    calls = {"n": 0}

    def counting_exec(pool: torch.Tensor, scratch: torch.Tensor, blocks: torch.Tensor, row_status: torch.Tensor) -> None:
        calls["n"] += 1
        if calls["n"] == 1:
            torch.accelerator.synchronize()
            snap["before_scatter"] = torch.cuda.memory_stats()["allocation.all.allocated"]
        return real_exec(pool, scratch, blocks, row_status)

    def counting_stage(self: Any, row_status: torch.Tensor) -> None:
        result = real_stage(self, row_status)
        torch.accelerator.synchronize()
        snap["after_stage"] = torch.cuda.memory_stats()["allocation.all.allocated"]
        return result

    advance._execute_masked_page_scatter_ = counting_exec  # type: ignore[assignment]
    commit_sink_mod.CommitTicket.stage = counting_stage  # type: ignore[method-assign]
    torch.accelerator.synchronize()
    whole_turn_before = torch.cuda.memory_stats()["allocation.all.allocated"]
    try:
        advance.advance_model_rows(
            core, input_ids, embeds, plan,
            adapter=adapter, decode_resolver=_dense_eager_resolver,
            placeholder_id=PLACEHOLDER_ID, park_id=PARK_ID,
            commit_sink=sink, capture=False, graph_covers_decode=False,
            staging=staging, **pools,
        )
    finally:
        advance._execute_masked_page_scatter_ = real_exec
        commit_sink_mod.CommitTicket.stage = real_stage  # type: ignore[method-assign]
    torch.accelerator.synchronize()
    whole_turn_after = torch.cuda.memory_stats()["allocation.all.allocated"]

    cell["commit_window_scatter_calls"] = calls["n"]
    cell["commit_window_alloc_before"] = snap.get("before_scatter")
    cell["commit_window_alloc_after"] = snap.get("after_stage")
    commit_delta = snap.get("after_stage", -1) - snap.get("before_scatter", -1)
    cell["commit_window_alloc_delta"] = commit_delta
    check.equal(commit_delta, 0, "allocation.commit_window_delta")
    cell["whole_turn_alloc_before"] = whole_turn_before
    cell["whole_turn_alloc_after"] = whole_turn_after
    cell["whole_turn_alloc_delta_informational"] = whole_turn_after - whole_turn_before
    sink.collect()
    return cell


# ---- arm (c): warmup / no-late-JIT ------------------------------------------


def _triton_cache_dir() -> Path:
    configured = os.environ.get("TRITON_CACHE_DIR")
    return Path(configured) if configured else Path.home() / ".triton" / "cache"


def _cache_file_count(directory: Path) -> int:
    if not directory.exists():
        return 0
    return sum(1 for p in directory.rglob("*") if p.is_file())


def run_warmup_arm(check: Check, device: str) -> dict[str, Any]:
    cell: dict[str, Any] = {}
    core = _tiny_core(device, seed=5)
    num_blocks = 4
    pools = _fresh_pools(
        n_layers=N_LAYERS, window=WINDOW, d_model=D_MODEL, kernel=KERNEL,
        pred_layers=2, pred_hidden=PRED_HIDDEN, cap=CAP, raw_tail=RAW_TAIL,
        n_mels=FEAT, num_blocks=num_blocks, device=device,
    )
    advance.warmup_advance_model_rows_scatter(**pools)
    cache_dir = _triton_cache_dir()
    files_after_warmup = _cache_file_count(cache_dir)
    cell["triton_cache_dir"] = str(cache_dir)
    cell["files_after_warmup"] = files_after_warmup

    registry = plan_mod.SessionRegistry()
    sink = commit_sink_mod.BoundedCommitSink(registry, max_rows=4, device=torch.device(device))
    staging = advance.HostStaging(4)
    adapter = advance.make_mrv1_adapter(hidden_size=CARRIER_HIDDEN, park_id=PARK_ID, blank_id=core.blank_id)

    def one_turn(req_id: str, block: int, seed: int, now_ns: int) -> float:
        torch.manual_seed(seed)
        samples = torch.randn(REG_SAMPLES) * 0.01
        rows = [
            plan_mod.ObservedRow(
                req_id, block, PLACEHOLDER_ID, False,
                _header(valid=REG_SAMPLES, geometry=GEOM_REG, final=False, seq=0),
            )
        ]
        context = _context(registry, rows, now_ns=now_ns, step=now_ns)
        plan = plan_mod.build_row_plan(
            context, num_decodes=0, num_prefills=1, null_block_id=NULL_BLOCK_ID, num_pool_blocks=num_blocks
        )
        input_ids = torch.tensor([PLACEHOLDER_ID], dtype=torch.long, device=device)
        embeds = (
            _carrier(samples, final=False, seq=0, geometry=GEOM_REG, prompt=0, hidden=CARRIER_HIDDEN)
            .unsqueeze(0)
            .to(device)
        )
        real_exec = advance._execute_masked_page_scatter_
        real_stage = commit_sink_mod.CommitTicket.stage
        window: dict[str, float] = {}

        def timed_exec(pool: torch.Tensor, scratch: torch.Tensor, blocks: torch.Tensor, row_status: torch.Tensor) -> None:
            if "start" not in window:
                torch.accelerator.synchronize()
                window["start"] = time.perf_counter()
            return real_exec(pool, scratch, blocks, row_status)

        def timed_stage(self: Any, row_status: torch.Tensor) -> None:
            result = real_stage(self, row_status)
            torch.accelerator.synchronize()
            window["end"] = time.perf_counter()
            return result

        advance._execute_masked_page_scatter_ = timed_exec  # type: ignore[assignment]
        commit_sink_mod.CommitTicket.stage = timed_stage  # type: ignore[method-assign]
        try:
            advance.advance_model_rows(
                core, input_ids, embeds, plan,
                adapter=adapter, decode_resolver=_dense_eager_resolver,
                placeholder_id=PLACEHOLDER_ID, park_id=PARK_ID,
                commit_sink=sink, capture=False, graph_covers_decode=False,
                staging=staging, **pools,
            )
        finally:
            advance._execute_masked_page_scatter_ = real_exec
            commit_sink_mod.CommitTicket.stage = real_stage  # type: ignore[method-assign]
        sink.collect()
        return window["end"] - window["start"]

    t1 = one_turn("warm_a", 1, 41, 1)
    files_after_first = _cache_file_count(cache_dir)
    t2 = one_turn("warm_b", 2, 42, 2)
    files_after_second = _cache_file_count(cache_dir)

    cell["commit_window_s_first"] = t1
    cell["commit_window_s_second"] = t2
    cell["files_after_first"] = files_after_first
    cell["files_after_second"] = files_after_second
    check.equal(files_after_first, files_after_warmup, "warmup.cache_files_unchanged_after_first")
    check.equal(files_after_second, files_after_warmup, "warmup.cache_files_unchanged_after_second")
    check.ok(
        t2 <= 3 * max(t1, 1e-4),
        f"warmup.no_late_jit_spike: second={t2:.4f}s first={t1:.4f}s",
    )

    # Negative control: a layout warmup never covered (extra block
    # column changes page_numel/stride) must fail closed.
    unwarmed = _fresh_pools(
        n_layers=N_LAYERS, window=WINDOW + 1, d_model=D_MODEL, kernel=KERNEL,
        pred_layers=2, pred_hidden=PRED_HIDDEN, cap=CAP, raw_tail=RAW_TAIL,
        n_mels=FEAT, num_blocks=num_blocks, device=device,
    )
    scratch = torch.zeros(1, WINDOW + 1, D_MODEL, device=device)
    block_ids = torch.zeros(1, dtype=torch.int64, device=device)
    row_status = torch.zeros(1, dtype=torch.int32, device=device)
    try:
        state_scatter.validate_masked_page_scatter(
            unwarmed["channel_pools"][0], scratch, block_ids, row_status
        )
        cell["unwarmed_layout"] = "validated (UNEXPECTED)"
    except ValueError as err:
        cell["unwarmed_layout"] = f"raised-as-expected: {err}"
    check.ok(
        cell["unwarmed_layout"].startswith("raised-as-expected"),
        "warmup.unwarmed_layout_raises",
    )
    return cell


# ---- arm (d): multi-bucket resolver override --------------------------------


def _two_bucket_turn(
    registry: plan_mod.SessionRegistry, now_ns: int, num_blocks: int, device: str
) -> tuple[Any, torch.Tensor, torch.Tensor]:
    torch.manual_seed(51)
    reg_samples = torch.randn(REG_SAMPLES) * 0.01
    torch.manual_seed(52)
    final_samples = torch.randn(FINAL_SAMPLES) * 0.01
    rows = [
        plan_mod.ObservedRow(
            "resolver_reg", 1, PLACEHOLDER_ID, False,
            _header(valid=REG_SAMPLES, geometry=GEOM_REG, final=False, seq=0),
        ),
        plan_mod.ObservedRow(
            "resolver_final", 2, PLACEHOLDER_ID, False,
            _header(valid=FINAL_SAMPLES, geometry=GEOM_FINAL, final=True, seq=0),
        ),
    ]
    context = _context(registry, rows, now_ns=now_ns, step=now_ns)
    plan = plan_mod.build_row_plan(
        context, num_decodes=0, num_prefills=2, null_block_id=NULL_BLOCK_ID, num_pool_blocks=num_blocks
    )
    input_ids = torch.tensor([PLACEHOLDER_ID, PLACEHOLDER_ID], dtype=torch.long, device=device)
    embeds = torch.stack(
        [
            _carrier(reg_samples, final=False, seq=0, geometry=GEOM_REG, prompt=0, hidden=CARRIER_HIDDEN),
            _carrier(final_samples, final=True, seq=0, geometry=GEOM_FINAL, prompt=0, hidden=CARRIER_HIDDEN),
        ]
    ).to(device)
    return plan, input_ids, embeds


def run_resolver_arm(check: Check, device: str) -> dict[str, Any]:
    cell: dict[str, Any] = {}
    core = _tiny_core(device, seed=7)
    num_blocks = 4
    pools = _fresh_pools(
        n_layers=N_LAYERS, window=WINDOW, d_model=D_MODEL, kernel=KERNEL,
        pred_layers=2, pred_hidden=PRED_HIDDEN, cap=CAP, raw_tail=RAW_TAIL,
        n_mels=FEAT, num_blocks=num_blocks, device=device,
    )
    advance.warmup_advance_model_rows_scatter(**pools)
    adapter = advance.make_mrv1_adapter(hidden_size=CARRIER_HIDDEN, park_id=PARK_ID, blank_id=core.blank_id)

    # (1) recording wrapper around the dense-eager fixed resolver.
    registry1 = plan_mod.SessionRegistry()
    sink1 = commit_sink_mod.BoundedCommitSink(registry1, max_rows=4, device=torch.device(device))
    staging1 = advance.HostStaging(4)
    plan1, ids1, embeds1 = _two_bucket_turn(registry1, 1, num_blocks, device)
    recording = _RecordingResolver(_dense_eager_resolver)
    advance.advance_model_rows(
        core, ids1, embeds1, plan1,
        adapter=adapter, decode_resolver=recording,
        placeholder_id=PLACEHOLDER_ID, park_id=PARK_ID,
        commit_sink=sink1, capture=False, graph_covers_decode=False,
        staging=staging1, **pools,
    )
    sink1.collect()
    cell["dense_fixed_requests"] = len(recording.requests)
    cell["dense_fixed_ready_buckets"] = [r.ready_decode_buckets for r in recording.requests]
    check.equal(len(recording.requests), 2, "resolver.dense_fixed.request_count")
    check.ok(
        all(r.ready_decode_buckets == 2 for r in recording.requests),
        f"resolver.dense_fixed.ready_decode_buckets == 2: {cell['dense_fixed_ready_buckets']}",
    )

    # (2) a compact-eager DECLARED arm, forced to dense-eager by the
    # multi-bucket override (nemotron_asr.build_decode_resolver).
    pools2 = _fresh_pools(
        n_layers=N_LAYERS, window=WINDOW, d_model=D_MODEL, kernel=KERNEL,
        pred_layers=2, pred_hidden=PRED_HIDDEN, cap=CAP, raw_tail=RAW_TAIL,
        n_mels=FEAT, num_blocks=num_blocks, device=device,
    )
    advance.warmup_advance_model_rows_scatter(**pools2)
    registry2 = plan_mod.SessionRegistry()
    sink2 = commit_sink_mod.BoundedCommitSink(registry2, max_rows=4, device=torch.device(device))
    staging2 = advance.HostStaging(4)
    plan2, ids2, embeds2 = _two_bucket_turn(registry2, 2, num_blocks, device)
    declared = build_decode_resolver(
        SimpleNamespace(decode_dispatch_arm="compact-eager", decode_dispatch_table=None)
    )
    recording_declared = _RecordingResolver(declared)
    advance.advance_model_rows(
        core, ids2, embeds2, plan2,
        adapter=adapter, decode_resolver=recording_declared,
        placeholder_id=PLACEHOLDER_ID, park_id=PARK_ID,
        commit_sink=sink2, capture=False, graph_covers_decode=False,
        staging=staging2, **pools2,
    )
    sink2.collect()
    cell["declared_compact_arms"] = [r.arm for r in recording_declared.resolutions]
    cell["declared_compact_override_reasons"] = [r.override_reason for r in recording_declared.resolutions]
    check.ok(
        all(r.arm == "dense-eager" for r in recording_declared.resolutions),
        f"resolver.declared_compact.forced_dense_eager: {cell['declared_compact_arms']}",
    )
    check.ok(
        all(r.override_reason == "multi-bucket-serialization-guard" for r in recording_declared.resolutions),
        f"resolver.declared_compact.override_reason: {cell['declared_compact_override_reasons']}",
    )
    return cell


# ---- arm (e): production-shape smoke ----------------------------------------


def run_production_arm(check: Check, device: str) -> dict[str, Any]:
    cell: dict[str, Any] = {}
    torch.manual_seed(91)
    core = NemotronASRCore(
        vocab_size=PROD_VOCAB,
        n_layers=PROD_N_LAYERS,
        filterbank=torch.zeros(128, 257),
        window=torch.zeros(400),
    ).to(device)
    core.eval()
    num_blocks = PROD_BATCH + 1
    pools = _fresh_pools(
        n_layers=PROD_N_LAYERS, window=PROD_WINDOW, d_model=1024, kernel=PROD_KERNEL,
        pred_layers=2, pred_hidden=PROD_PRED_HIDDEN, cap=PROD_QUEUE_CAPACITY,
        raw_tail=RAW_TAIL, n_mels=128, num_blocks=num_blocks, device=device,
    )
    advance.warmup_advance_model_rows_scatter(**pools)
    registry = plan_mod.SessionRegistry()
    sink = commit_sink_mod.BoundedCommitSink(registry, max_rows=PROD_BATCH, device=torch.device(device))
    staging = advance.HostStaging(PROD_BATCH)
    adapter = advance.make_mrv1_adapter(hidden_size=PROD_HIDDEN, park_id=PROD_PARK_ID, blank_id=core.blank_id)

    torch.manual_seed(92)
    # A REGULAR (non-final) full-width chunk: PROD_SAMPLES ==
    # cadence*hop exactly for geometry 4, which the envelope's
    # non-final width check requires; a FINAL row at this width would
    # instead trip ROW_STATUS_FINAL_OVERSIZE (valid_samples >=
    # cadence*hop, the "strictly under one cadence" final-tail rule).
    signals = [torch.randn(PROD_SAMPLES) * 0.01 for _ in range(PROD_BATCH)]
    rows = [
        plan_mod.ObservedRow(
            f"prod_{i}", i + 1, PROD_PLACEHOLDER_ID, False,
            _header(valid=PROD_SAMPLES, geometry=PROD_GEOM, final=False, seq=0),
        )
        for i in range(PROD_BATCH)
    ]
    context = plan_mod.prepare_plan_context(
        registry, rows,
        resident_request_ids=[r.request_id for r in rows],
        placeholder_id=PROD_PLACEHOLDER_ID, num_prompts=128,
        num_geometries=NUM_GEOMETRIES, now_ns=1, step=1,
    )
    plan = plan_mod.build_row_plan(
        context, num_decodes=0, num_prefills=PROD_BATCH,
        null_block_id=NULL_BLOCK_ID, num_pool_blocks=num_blocks,
    )
    input_ids = torch.full((PROD_BATCH,), PROD_PLACEHOLDER_ID, dtype=torch.long, device=device)
    embeds = torch.stack(
        [
            _carrier(s, final=False, seq=0, geometry=PROD_GEOM, prompt=0, hidden=PROD_HIDDEN)
            for s in signals
        ]
    ).to(device)

    torch.accelerator.synchronize()
    torch.accelerator.reset_peak_memory_stats(device)
    t0 = time.perf_counter()
    advance.advance_model_rows(
        core, input_ids, embeds, plan,
        adapter=adapter, decode_resolver=_dense_eager_resolver,
        placeholder_id=PROD_PLACEHOLDER_ID, park_id=PROD_PARK_ID,
        commit_sink=sink, capture=False, graph_covers_decode=False,
        staging=staging, **pools,
    )
    torch.accelerator.synchronize()
    wall_s = time.perf_counter() - t0
    peak_bytes = torch.accelerator.max_memory_allocated(device)
    reports, _records, lease_ok = sink.collect()

    cell["wall_s"] = wall_s
    cell["peak_memory_bytes"] = peak_bytes
    cell["lease_ok"] = lease_ok
    cell["row_statuses"] = {r.request_id: r.row_status for r in reports}
    check.ok(lease_ok, "production.lease_ok")
    check.ok(
        all(status == 0 for status in cell["row_statuses"].values()),
        f"production.clean_row_statuses: {cell['row_statuses']}",
    )
    return cell


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--device", default="cuda")
    ap.add_argument(
        "--arm", default="all",
        choices=["all", "sync", "allocation", "warmup", "resolver", "production"],
    )
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("FATAL: --device cuda but no CUDA device", flush=True)
        sys.exit(2)

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False

    out_dir = args.out.parent if args.out else Path("p6c-full-turn-out")
    out_dir.mkdir(parents=True, exist_ok=True)
    check = Check()
    report: dict[str, Any] = {
        "probe": "p6c_full_turn",
        "fingerprint": {
            "device": device,
            "device_name": (
                torch.cuda.get_device_name(0) if device == "cuda" else platform.processor()
            ),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
            "tf32_cudnn": torch.backends.cudnn.allow_tf32,
        },
        "arms": {},
    }
    if device != "cuda":
        print("FATAL: this probe is CUDA-only (sync/allocation/warmup arms need a real device)", flush=True)
        sys.exit(2)

    def _checkpoint() -> None:
        # Persist the running report after every arm so a later arm's
        # exception can never destroy earlier arms' evidence.
        report["checks"] = check.count
        report["failures"] = check.failures
        report["pass"] = not check.failures
        if args.out:
            args.out.write_text(json.dumps(report, indent=2, default=str))

    arms = (
        ("sync", lambda: run_sync_arm(check, device, out_dir)),
        ("allocation", lambda: run_allocation_arm(check, device)),
        ("warmup", lambda: run_warmup_arm(check, device)),
        ("resolver", lambda: run_resolver_arm(check, device)),
        ("production", lambda: run_production_arm(check, device)),
    )
    for name, fn in arms:
        if args.arm not in ("all", name):
            continue
        try:
            report["arms"][name] = fn()
        except Exception as err:  # a crash must not erase earlier arms
            check.ok(False, f"{name}.arm_uncaught_exception")
            report["arms"][name] = {
                "arm_outcome": f"UNCAUGHT: {type(err).__name__}: {err}",
                "traceback": traceback.format_exc(),
            }
        _checkpoint()

    _checkpoint()
    text = json.dumps(report, indent=2, default=str)
    print(text)
    print(
        f"{'PASS' if report['pass'] else 'FAIL'}: {check.count} checks, {len(check.failures)} failures",
        flush=True,
    )
    sys.exit(0 if report["pass"] else 1)


if __name__ == "__main__":
    main()
