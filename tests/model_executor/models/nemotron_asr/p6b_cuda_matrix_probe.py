# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CUDA correctness matrix for the length-aware sync-free transition.

Phase 6b step 2 (ledger §8): on-device confirmation of what the local
4b milestone proved on CPU. Three arms, all evidence into one JSON
report:

``matrix``  — repeated differential correctness runs of the canonical
  ``advance_session`` across all five geometries: mixed-phase buckets
  (session-first / continuing / committable-final / max-legal-final /
  zero-sample-final / dropped-residual-final) equal to their single-row
  runs; multi-chunk lockstep drives (20 units at 80 ms, 4 elsewhere)
  with mixed prompts and mixed-length final batches sweeping the
  reachable residual range (drop side, commit side, max legal C+6);
  protocol rows (wrong sequence / wrong geometry / oversize final /
  post-finalization audio) resolving to exact ``ROW_STATUS_*`` bits
  with zero state mutation. Both unselected decode candidates run
  every differential and must emit identical token streams
  (PORT-DEC-008 keeps them unselected until the profile).

``sync``    — ``torch.cuda.set_sync_debug_mode(2)`` as a TRIPWIRE
  around the dense-decode valid path (PyTorch labels the mode
  experimental and non-exhaustive, so it is never the proof), paired
  with a Kineto host trace counting synchronizing calls
  (``cudaStreamSynchronize`` / ``cudaDeviceSynchronize`` / sync
  memcpy / ``aten::item`` / ``_local_scalar_dense`` / ``nonzero``).
  The compact arm runs as the positive control: its sanctioned
  ``nonzero`` syncs must fire the tripwire and appear in the trace.

``graph``   — CUDA-graph capture of the sync-free path
  (``decode_dense_masked``) after side-stream warmup, replayed from a
  state snapshot and compared against the eager run: token surfaces
  exact, state at the provisional 1e-6 bound. A compact-arm capture
  attempt is the negative control (its shape sync must refuse
  capture).

The transition's dependency closure is torch-only, loaded by file
path under a stubbed package chain (the ``test_advance_session_local``
loader), so ``--device cpu`` runs the matrix arm anywhere as a smoke
gate; ``sync`` and ``graph`` require CUDA.

Run (on the pod):
    /opt/venv-port/bin/python tests/model_executor/models/nemotron_asr/\
      p6b_cuda_matrix_probe.py --device cuda --repeats 3 \
        --out /workspace/evidence/p6b-cuda-matrix/report.json
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import platform
import re
import sys
import time
import traceback
import types
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch

_PKG = (
    Path(__file__).resolve().parents[4]
    / "vllm_omni/model_executor/models/nemotron_asr"
)
_BASE = "vllm_omni.model_executor.models.nemotron_asr"

FEAT = 16
D_MODEL = 32
N_LAYERS = 2
KERNEL = 5
WINDOW = 8
VOCAB = 12
PRED_HIDDEN = 16
RAW_TAIL = 1_953
HOP = 160
#: CUDA-side synchronization evidence — the dense-path hard gate.
#: Pageable HtoD/DtoH memcpys synchronize; a CUDA-tensor ``item()``
#: shows up here as its stream sync + DtoH pair.
SYNC_MARKERS = (
    "cudaStreamSynchronize",
    "cudaDeviceSynchronize",
    "aten::nonzero",
    "Memcpy DtoH",
    "Memcpy HtoD",
)
#: Host-tensor readbacks (``int()`` on a CPU tensor) are legal — no
#: device is involved — but reported so shape-math regressions stay
#: visible.
HOST_READ_MARKERS = (
    "aten::_local_scalar_dense",
    "aten::item",
)


def _load_chain() -> dict[str, Any]:
    for name in (
        "vllm_omni",
        "vllm_omni.model_executor",
        "vllm_omni.model_executor.models",
        _BASE,
    ):
        if name not in sys.modules:
            sys.modules[name] = types.ModuleType(name)
    loaded: dict[str, Any] = {}
    for mod in (
        "precision", "masks", "featurizer", "encoder", "lid",
        "manifests", "frontend", "rnnt_cell", "rnnt", "advance",
    ):
        spec = importlib.util.spec_from_file_location(
            f"{_BASE}.{mod}", _PKG / f"{mod}.py"
        )
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[f"{_BASE}.{mod}"] = module
        spec.loader.exec_module(module)
        loaded[mod] = module
    return loaded


MODS = _load_chain()
advance = MODS["advance"]
frontend = MODS["frontend"]
rnnt = MODS["rnnt"]
manifests = MODS["manifests"]

#: (geometry_id, label, lookahead, cadence, chunk_samples) — geometry
#: id indexes CADENCES order, matching advance_session's lookup.
GEOMETRIES = [
    (
        gid,
        label,
        right,
        8 * (right + 1),
        manifests.RAW_SAMPLES_PER_CHUNK[label],
    )
    for gid, (label, (_, right)) in enumerate(
        manifests.CADENCES.items()
    )
]


class Check:
    """Failure collector: the probe records every miss and exits
    nonzero at the end, so one bad cell never hides the rest of the
    matrix."""

    def __init__(self) -> None:
        self.failures: list[str] = []
        self.count = 0
        self.max_diffs: dict[str, float] = {}

    def ok(self, cond: bool, ctx: str) -> None:
        self.count += 1
        if not cond:
            self.failures.append(ctx)

    def equal(self, a: Any, b: Any, ctx: str) -> None:
        self.ok(a == b, f"{ctx}: {a!r} != {b!r}")

    def close(
        self,
        a: torch.Tensor,
        b: torch.Tensor,
        atol: float,
        ctx: str,
    ) -> None:
        diff = float((a - b).abs().max()) if a.numel() else 0.0
        key = re.sub(r"\[[^\]]*\]", "", ctx)
        self.max_diffs[key] = max(self.max_diffs.get(key, 0.0), diff)
        self.ok(diff <= atol, f"{ctx}: max|diff|={diff:.3e} > {atol}")


def _core(lookahead: int, device: str, seed: int) -> Any:
    torch.manual_seed(seed)
    encoder = MODS["encoder"].FastConformerEncoder(
        feat_in=FEAT, d_model=D_MODEL, d_ff=64, n_layers=N_LAYERS,
        n_heads=4, conv_kernel=KERNEL, subsampling_channels=16,
        att_context=(WINDOW, lookahead),
    )
    core = SimpleNamespace(
        encoder=encoder,
        lid=MODS["lid"].PromptConditioner(
            enc_hidden=D_MODEL, num_prompts=4
        ),
        predictor=rnnt.Predictor(
            vocab_size=VOCAB, pred_hidden=PRED_HIDDEN, pred_rnn_layers=2
        ),
        joint=rnnt.Joint(
            enc_hidden=D_MODEL, pred_hidden=PRED_HIDDEN,
            joint_hidden=16, vocab_size=VOCAB,
        ),
        featurizer=MODS["featurizer"].MelFeaturizer(
            filterbank=torch.rand(FEAT, 257) * 0.01,
            window=torch.hann_window(400),
        ),
        blank_id=VOCAB,
    )
    for m in (
        core.encoder, core.lid, core.predictor, core.joint,
        core.featurizer,
    ):
        m.to(device)
        m.eval()
    return core


def _fresh_state(batch: int, device: str) -> Any:
    return advance.SessionStateBatch(
        raw_tail=torch.zeros(batch, RAW_TAIL, device=device),
        mel_tail=torch.zeros(
            batch, FEAT, frontend.MEL_TAIL_FRAMES, device=device
        ),
        frontend_counters=torch.zeros(
            batch, 8, dtype=torch.int64, device=device
        ),
        channel=[
            torch.zeros(batch, WINDOW, D_MODEL, device=device)
            for _ in range(N_LAYERS)
        ],
        window_valid=[
            torch.zeros(batch, 1, dtype=torch.int32, device=device)
            for _ in range(N_LAYERS)
        ],
        time=[
            torch.zeros(batch, D_MODEL, KERNEL - 1, device=device)
            for _ in range(N_LAYERS)
        ],
        h=torch.zeros(batch, 2, PRED_HIDDEN, device=device),
        c=torch.zeros(batch, 2, PRED_HIDDEN, device=device),
        last_label=torch.full(
            (batch,), VOCAB, dtype=torch.long, device=device
        ),
    )


def _slice_state(state: Any, row: int) -> Any:
    r = slice(row, row + 1)
    return advance.SessionStateBatch(
        raw_tail=state.raw_tail[r].clone(),
        mel_tail=state.mel_tail[r].clone(),
        frontend_counters=state.frontend_counters[r].clone(),
        channel=[t[r].clone() for t in state.channel],
        window_valid=[t[r].clone() for t in state.window_valid],
        time=[t[r].clone() for t in state.time],
        h=state.h[r].clone(),
        c=state.c[r].clone(),
        last_label=state.last_label[r].clone(),
    )


def _state_snapshot(state: Any) -> dict[str, Any]:
    return {
        "raw_tail": state.raw_tail.clone(),
        "mel_tail": state.mel_tail.clone(),
        "frontend_counters": state.frontend_counters.clone(),
        "channel": [t.clone() for t in state.channel],
        "window_valid": [t.clone() for t in state.window_valid],
        "time": [t.clone() for t in state.time],
        "h": state.h.clone(),
        "c": state.c.clone(),
        "last_label": state.last_label.clone(),
    }


def _state_restore(state: Any, snap: dict[str, Any]) -> None:
    state.raw_tail.copy_(snap["raw_tail"])
    state.mel_tail.copy_(snap["mel_tail"])
    state.frontend_counters.copy_(snap["frontend_counters"])
    for dst, src in zip(state.channel, snap["channel"]):
        dst.copy_(src)
    for dst, src in zip(state.window_valid, snap["window_valid"]):
        dst.copy_(src)
    for dst, src in zip(state.time, snap["time"]):
        dst.copy_(src)
    state.h.copy_(snap["h"])
    state.c.copy_(snap["c"])
    state.last_label.copy_(snap["last_label"])


def _chunk(
    samples: torch.Tensor,
    valid: torch.Tensor,
    *,
    gid: int,
    finals: torch.Tensor,
    prompts: torch.Tensor,
    seqs: torch.Tensor,
) -> Any:
    return advance.ChunkBatch(
        samples=samples,
        valid_samples=valid,
        geometry_id=torch.full_like(seqs, gid),
        final_tail=finals,
        prompt_index=prompts,
        chunk_sequence=seqs,
    )


def _regular_chunk(
    samples: torch.Tensor, *, gid: int, seq: int, prompt: int = 0
) -> Any:
    batch, width = samples.shape
    device = samples.device
    return _chunk(
        samples,
        torch.full((batch,), width, dtype=torch.long, device=device),
        gid=gid,
        finals=torch.zeros(batch, dtype=torch.bool, device=device),
        prompts=torch.full(
            (batch,), prompt, dtype=torch.long, device=device
        ),
        seqs=torch.full(
            (batch,), seq, dtype=torch.long, device=device
        ),
    )


def _compare_states(
    check: Check,
    state_n: Any,
    row: int,
    state1: Any,
    ctx: str,
) -> None:
    r = slice(row, row + 1)
    check.close(
        state_n.frontend_counters[r].float(),
        state1.frontend_counters.float(),
        0.0,
        f"{ctx}.counters[{row}]",
    )
    for layer in range(N_LAYERS):
        check.close(
            state_n.channel[layer][r],
            state1.channel[layer],
            1e-5,
            f"{ctx}.channel{layer}[{row}]",
        )
        check.close(
            state_n.time[layer][r],
            state1.time[layer],
            1e-5,
            f"{ctx}.time{layer}[{row}]",
        )
        check.close(
            state_n.window_valid[layer][r].float(),
            state1.window_valid[layer].float(),
            0.0,
            f"{ctx}.window_valid{layer}[{row}]",
        )
    check.close(state_n.h[r], state1.h, 1e-6, f"{ctx}.h[{row}]")
    check.close(state_n.c[r], state1.c, 1e-6, f"{ctx}.c[{row}]")
    check.close(
        state_n.last_label[r].float(),
        state1.last_label.float(),
        0.0,
        f"{ctx}.last_label[{row}]",
    )
    check.close(
        state_n.raw_tail[r], state1.raw_tail, 0.0,
        f"{ctx}.raw_tail[{row}]",
    )
    check.close(
        state_n.mel_tail[r], state1.mel_tail, 2e-6,
        f"{ctx}.mel_tail[{row}]",
    )


def _tokens(result: Any, row: int) -> list[int]:
    n = int(result.token_lengths[row])
    return [int(t) for t in result.token_ids[row, :n]]


def _mixed_phase_rows(
    cadence: int, chunk: int
) -> tuple[list[int], list[bool], list[int], list[int]]:
    """Six phases: widths, final flags, sequences, prompts.

    Residual frames past B_k for a final of w samples = w//160 + 7:
    committable mid-range, max legal C+6 (one sample short of a full
    unit), dropped zero-sample (residual 7), dropped off-hop-grid.
    """
    widths = [
        chunk,                       # 0: session-first regular
        chunk,                       # 1: continuing regular
        max(HOP, (cadence // 2) * HOP),  # 2: final, committable
        cadence * HOP - 1,           # 3: final, max legal residual
        0,                           # 4: zero-sample final (drop 7)
        HOP // 2,                    # 5: final, off-grid (drop 7)
    ]
    finals = [False, False, True, True, True, True]
    seqs = [0, 1, 1, 1, 1, 1]
    prompts = [0, 2, 1, 3, 0, 1]
    return widths, finals, seqs, prompts


def _advance(
    core: Any, batch: Any, state: Any, gid: int, decode_fn: Any,
    capture: bool = False,
) -> Any:
    return advance.advance_session(
        core, batch, state, geometry=gid, decode_fn=decode_fn,
        capture=capture,
    )


def run_mixed_phase(
    check: Check, gid: int, lookahead: int, cadence: int, chunk: int,
    device: str, seed: int,
) -> None:
    """The mixed-phase bucket differential, both decoders, on device."""
    core = _core(lookahead, device, seed)
    widths, finals, seqs, prompts = _mixed_phase_rows(cadence, chunk)
    n = len(widths)
    torch.manual_seed(seed + 1)
    signals = [
        (torch.randn(2 * chunk) * a).to(device)
        for a in (0.1, 0.2, 0.15, 0.25, 0.12, 0.18)
    ]
    for decode_name in ("decode_dense_masked", "decode_compact_active"):
        decode_fn = getattr(rnnt, decode_name)
        ctx = f"mixed[{gid}][{decode_name}]"
        singles = []
        for row in range(n):
            s1 = _fresh_state(1, device)
            if seqs[row] != 0:
                _advance(
                    core,
                    _regular_chunk(
                        signals[row][:chunk].unsqueeze(0),
                        gid=gid, seq=0, prompt=prompts[row],
                    ),
                    s1, gid, decode_fn,
                )
            singles.append(s1)
        stateN = advance.SessionStateBatch(
            raw_tail=torch.cat([s.raw_tail for s in singles]),
            mel_tail=torch.cat([s.mel_tail for s in singles]),
            frontend_counters=torch.cat(
                [s.frontend_counters for s in singles]
            ),
            channel=[
                torch.cat([s.channel[i] for s in singles])
                for i in range(N_LAYERS)
            ],
            window_valid=[
                torch.cat([s.window_valid[i] for s in singles])
                for i in range(N_LAYERS)
            ],
            time=[
                torch.cat([s.time[i] for s in singles])
                for i in range(N_LAYERS)
            ],
            h=torch.cat([s.h for s in singles]),
            c=torch.cat([s.c for s in singles]),
            last_label=torch.cat([s.last_label for s in singles]),
        )
        samples = torch.zeros(n, chunk, device=device)
        valid = torch.zeros(n, dtype=torch.long, device=device)
        for row in range(n):
            w = widths[row]
            samples[row, :w] = signals[row][chunk : chunk + w]
            valid[row] = w
        batchN = _chunk(
            samples, valid, gid=gid,
            finals=torch.tensor(finals, device=device),
            prompts=torch.tensor(
                prompts, dtype=torch.long, device=device
            ),
            seqs=torch.tensor(seqs, dtype=torch.long, device=device),
        )
        resultN = _advance(
            core, batchN, stateN, gid, decode_fn, capture=True
        )
        check.equal(
            resultN.row_status.tolist(), [0] * n, f"{ctx}.row_status"
        )
        for row in range(n):
            s1 = singles[row]
            batch1 = _chunk(
                samples[row : row + 1], valid[row : row + 1], gid=gid,
                finals=batchN.final_tail[row : row + 1],
                prompts=batchN.prompt_index[row : row + 1],
                seqs=batchN.chunk_sequence[row : row + 1],
            )
            r1 = _advance(core, batch1, s1, gid, decode_fn, capture=True)
            check.equal(
                _tokens(resultN, row), _tokens(r1, 0),
                f"{ctx}.tokens[{row}]",
            )
            _compare_states(check, stateN, row, s1, ctx)
            capsN, caps1 = resultN.captures, r1.captures
            check.ok(
                capsN is not None and caps1 is not None,
                f"{ctx}.captures[{row}] missing",
            )
            check.equal(
                int(capsN.mel_lengths[row]),
                int(caps1.mel_lengths[0]),
                f"{ctx}.mel_len[{row}]",
            )
            ml = int(capsN.mel_lengths[row])
            el = int(capsN.encoder_lengths[row])
            check.equal(
                el, int(caps1.encoder_lengths[0]),
                f"{ctx}.enc_len[{row}]",
            )
            check.close(
                capsN.frontend_mel[row, :, :ml],
                caps1.frontend_mel[0, :, :ml],
                2e-6,
                f"{ctx}.cap_mel[{row}]",
            )
            check.close(
                capsN.encoder_conditioned[row, :el],
                caps1.encoder_conditioned[0, :el],
                1e-5,
                f"{ctx}.cap_cond[{row}]",
            )
        # Zero-sample final: finalized, zero burst, dropped residual.
        check.equal(
            int(
                stateN.frontend_counters[4, frontend.CTR_FINALIZED]
            ),
            1,
            f"{ctx}.finalized[4]",
        )
        check.equal(
            int(resultN.token_lengths[4]), 0, f"{ctx}.zero_burst[4]"
        )


def run_lockstep_drive(
    check: Check, gid: int, lookahead: int, cadence: int, chunk: int,
    device: str, seed: int, units: int,
) -> None:
    """Multi-chunk lockstep drive with a mixed-length final bucket.

    Eight sessions advance through ``units`` regular cadence units in
    lockstep buckets (mixed prompts), then one final bucket carries
    per-row residual widths sweeping the reachable range. Each session
    must equal its single-row drive; dense and compact token streams
    must be identical.
    """
    n = 8
    final_widths = [
        0,                     # residual 7 → dropped
        HOP // 2,              # residual 7 off-grid → dropped
        HOP,                   # residual 8 → committed
        min(chunk - 1, 4 * HOP),
        (cadence // 2) * HOP,
        cadence * HOP - 1,     # max legal residual C+6
        HOP,
        0,
    ]
    prompts = [0, 1, 2, 3, 0, 1, 2, 3]
    torch.manual_seed(seed + 2)
    total = units * chunk + max(final_widths)
    signals = [
        (torch.randn(total) * (0.08 + 0.02 * row)).to(device)
        for row in range(n)
    ]
    core = _core(lookahead, device, seed)
    streams: dict[str, list[list[int]]] = {}
    for decode_name in ("decode_dense_masked", "decode_compact_active"):
        decode_fn = getattr(rnnt, decode_name)
        ctx = f"drive[{gid}][{decode_name}]"
        stateN = _fresh_state(n, device)
        prompts_t = torch.tensor(
            prompts, dtype=torch.long, device=device
        )
        emitted: list[list[int]] = [[] for _ in range(n)]
        for u in range(units):
            samples = torch.stack(
                [s[u * chunk : (u + 1) * chunk] for s in signals]
            )
            batch = _chunk(
                samples,
                torch.full(
                    (n,), chunk, dtype=torch.long, device=device
                ),
                gid=gid,
                finals=torch.zeros(
                    n, dtype=torch.bool, device=device
                ),
                prompts=prompts_t,
                seqs=torch.full(
                    (n,), u, dtype=torch.long, device=device
                ),
            )
            res = _advance(core, batch, stateN, gid, decode_fn)
            check.equal(
                res.row_status.tolist(), [0] * n,
                f"{ctx}.unit{u}.row_status",
            )
            for row in range(n):
                emitted[row].extend(_tokens(res, row))
        fsamples = torch.zeros(n, max(final_widths), device=device)
        fvalid = torch.zeros(n, dtype=torch.long, device=device)
        for row in range(n):
            w = final_widths[row]
            fsamples[row, :w] = signals[row][
                units * chunk : units * chunk + w
            ]
            fvalid[row] = w
        fbatch = _chunk(
            fsamples, fvalid, gid=gid,
            finals=torch.ones(n, dtype=torch.bool, device=device),
            prompts=prompts_t,
            seqs=torch.full(
                (n,), units, dtype=torch.long, device=device
            ),
        )
        fres = _advance(core, fbatch, stateN, gid, decode_fn)
        check.equal(
            fres.row_status.tolist(), [0] * n,
            f"{ctx}.final.row_status",
        )
        check.equal(
            stateN.frontend_counters[
                :, frontend.CTR_FINALIZED
            ].tolist(),
            [1] * n,
            f"{ctx}.final.finalized",
        )
        for row in range(n):
            emitted[row].extend(_tokens(fres, row))
        streams[decode_name] = emitted

        # Single-row drives must match the lockstep bucket exactly.
        for row in range(n):
            s1 = _fresh_state(1, device)
            solo: list[int] = []
            for u in range(units):
                b1 = _chunk(
                    signals[row][
                        u * chunk : (u + 1) * chunk
                    ].unsqueeze(0),
                    torch.tensor(
                        [chunk], dtype=torch.long, device=device
                    ),
                    gid=gid,
                    finals=torch.zeros(
                        1, dtype=torch.bool, device=device
                    ),
                    prompts=prompts_t[row : row + 1],
                    seqs=torch.tensor(
                        [u], dtype=torch.long, device=device
                    ),
                )
                r1 = _advance(core, b1, s1, gid, decode_fn)
                solo.extend(_tokens(r1, 0))
            w = final_widths[row]
            fb1 = _chunk(
                fsamples[row : row + 1, :], fvalid[row : row + 1],
                gid=gid,
                finals=torch.ones(
                    1, dtype=torch.bool, device=device
                ),
                prompts=prompts_t[row : row + 1],
                seqs=torch.tensor(
                    [units], dtype=torch.long, device=device
                ),
            )
            fr1 = _advance(core, fb1, s1, gid, decode_fn)
            solo.extend(_tokens(fr1, 0))
            check.equal(
                emitted[row], solo, f"{ctx}.stream[{row}]"
            )
            _compare_states(check, stateN, row, s1, f"{ctx}.final")
    check.equal(
        streams["decode_dense_masked"],
        streams["decode_compact_active"],
        f"drive[{gid}].dense-vs-compact token streams",
    )


def run_protocol_rows(
    check: Check, gid: int, lookahead: int, cadence: int, chunk: int,
    device: str, seed: int,
) -> None:
    """Protocol violations resolve to exact status bits on device."""
    core = _core(lookahead, device, seed)
    decode_fn = rnnt.decode_dense_masked
    torch.manual_seed(seed + 3)
    signals = (torch.randn(4, 2 * chunk) * 0.1).to(device)
    ctx = f"protocol[{gid}]"
    singles = [_fresh_state(1, device) for _ in range(4)]
    for row in range(4):
        _advance(
            core,
            _regular_chunk(
                signals[row : row + 1, :chunk], gid=gid, seq=0
            ),
            singles[row], gid, decode_fn,
        )
    stateN = advance.SessionStateBatch(
        raw_tail=torch.cat([s.raw_tail for s in singles]),
        mel_tail=torch.cat([s.mel_tail for s in singles]),
        frontend_counters=torch.cat(
            [s.frontend_counters for s in singles]
        ),
        channel=[
            torch.cat([s.channel[i] for s in singles])
            for i in range(N_LAYERS)
        ],
        window_valid=[
            torch.cat([s.window_valid[i] for s in singles])
            for i in range(N_LAYERS)
        ],
        time=[
            torch.cat([s.time[i] for s in singles])
            for i in range(N_LAYERS)
        ],
        h=torch.cat([s.h for s in singles]),
        c=torch.cat([s.c for s in singles]),
        last_label=torch.cat([s.last_label for s in singles]),
    )
    before = _state_snapshot(stateN)
    batch = _chunk(
        signals[:, chunk:].clone(),
        torch.full((4,), chunk, dtype=torch.long, device=device),
        gid=gid,
        finals=torch.tensor(
            [False, False, True, False], device=device
        ),
        prompts=torch.zeros(4, dtype=torch.long, device=device),
        seqs=torch.tensor(
            [5, 1, 1, 1], dtype=torch.long, device=device
        ),
    )
    batch.geometry_id[1] = (gid + 1) % len(GEOMETRIES)
    result = _advance(core, batch, stateN, gid, decode_fn)
    check.equal(
        result.row_status.tolist(),
        [
            frontend.ROW_STATUS_SEQUENCE,
            frontend.ROW_STATUS_GEOMETRY,
            frontend.ROW_STATUS_FINAL_OVERSIZE,
            0,
        ],
        f"{ctx}.row_status",
    )
    for row in range(3):
        check.equal(
            int(result.token_lengths[row]), 0,
            f"{ctx}.masked_burst[{row}]",
        )
        check.close(
            stateN.frontend_counters[row].float(),
            before["frontend_counters"][row].float(),
            0.0,
            f"{ctx}.counters_unchanged[{row}]",
        )
        check.close(
            stateN.h[row], before["h"][row], 0.0,
            f"{ctx}.h_unchanged[{row}]",
        )
        check.close(
            stateN.channel[0][row], before["channel"][0][row], 0.0,
            f"{ctx}.channel_unchanged[{row}]",
        )
    # Audio after finalization: FINALIZED bit, no mutation.
    s = singles[3]
    fin = _chunk(
        torch.zeros(1, HOP, device=device),
        torch.tensor([0], dtype=torch.long, device=device),
        gid=gid,
        finals=torch.ones(1, dtype=torch.bool, device=device),
        prompts=torch.zeros(1, dtype=torch.long, device=device),
        seqs=torch.tensor([1], dtype=torch.long, device=device),
    )
    _advance(core, fin, s, gid, decode_fn)
    after_final = _state_snapshot(s)
    late = _regular_chunk(
        signals[3:4, chunk:].clone(), gid=gid, seq=2
    )
    r = _advance(core, late, s, gid, decode_fn)
    check.equal(
        int(r.row_status[0]) & frontend.ROW_STATUS_FINALIZED,
        frontend.ROW_STATUS_FINALIZED,
        f"{ctx}.finalized_bit",
    )
    check.close(
        s.frontend_counters.float(),
        after_final["frontend_counters"].float(),
        0.0,
        f"{ctx}.finalized_counters_unchanged",
    )


def _sync_bucket(
    gid: int, lookahead: int, cadence: int, chunk: int, device: str,
    seed: int,
) -> tuple[Any, Any, Any]:
    """A warm mixed-phase bucket for the sync and graph arms."""
    core = _core(lookahead, device, seed)
    widths, finals, seqs, prompts = _mixed_phase_rows(cadence, chunk)
    n = len(widths)
    torch.manual_seed(seed + 4)
    signals = [
        (torch.randn(2 * chunk) * 0.1).to(device) for _ in range(n)
    ]
    state = _fresh_state(n, device)
    warm = torch.stack([s[:chunk] for s in signals])
    _advance(
        core,
        _regular_chunk(warm, gid=gid, seq=0),
        state, gid, rnnt.decode_dense_masked,
    )
    samples = torch.zeros(n, chunk, device=device)
    valid = torch.zeros(n, dtype=torch.long, device=device)
    for row in range(n):
        w = widths[row] if seqs[row] else chunk
        samples[row, :w] = signals[row][chunk : chunk + w]
        valid[row] = w
    batch = _chunk(
        samples, valid, gid=gid,
        finals=torch.tensor(finals, device=device),
        prompts=torch.tensor(prompts, dtype=torch.long, device=device),
        seqs=torch.ones(n, dtype=torch.long, device=device),
    )
    return core, batch, state


def run_sync_arm(
    check: Check, device: str, seed: int, out_dir: Path
) -> dict[str, Any]:
    """Tripwire + Kineto host-trace sync accounting (CUDA only)."""
    from torch.profiler import ProfilerActivity, profile

    report: dict[str, Any] = {}
    for gid, label, lookahead, cadence, chunk in GEOMETRIES:
        if label not in ("80ms", "1120ms"):
            continue
        core, batch, state = _sync_bucket(
            gid, lookahead, cadence, chunk, device, seed
        )
        snap = _state_snapshot(state)
        # Warm both arms outside any measurement.
        for fn in (rnnt.decode_dense_masked, rnnt.decode_compact_active):
            _state_restore(state, snap)
            _advance(core, batch, state, gid, fn)
        torch.accelerator.synchronize()

        cell: dict[str, Any] = {}
        # TRIPWIRE: the dense valid path must survive error mode; the
        # compact arm is the positive control (its nonzero must fire).
        _state_restore(state, snap)
        torch.cuda.set_sync_debug_mode(2)
        try:
            _advance(
                core, batch, state, gid, rnnt.decode_dense_masked
            )
            cell["tripwire_dense"] = "clean"
        except RuntimeError as err:
            cell["tripwire_dense"] = f"FIRED: {err}"
            # The raise happens AT the synchronizing op — this
            # traceback is the diagnosis, keep all of it.
            cell["tripwire_dense_traceback"] = traceback.format_exc()
        finally:
            torch.cuda.set_sync_debug_mode(0)
        torch.accelerator.synchronize()
        check.equal(
            cell["tripwire_dense"], "clean",
            f"sync[{label}].tripwire_dense",
        )
        _state_restore(state, snap)
        torch.cuda.set_sync_debug_mode(2)
        try:
            _advance(
                core, batch, state, gid, rnnt.decode_compact_active
            )
            cell["tripwire_compact"] = "no-fire"
        except RuntimeError:
            cell["tripwire_compact"] = "fired-as-expected"
        finally:
            torch.cuda.set_sync_debug_mode(0)
        torch.accelerator.synchronize()
        check.equal(
            cell["tripwire_compact"], "fired-as-expected",
            f"sync[{label}].tripwire_compact_control",
        )

        # Kineto host trace: count synchronizing calls per arm.
        for name, fn in (
            ("dense", rnnt.decode_dense_masked),
            ("compact", rnnt.decode_compact_active),
        ):
            _state_restore(state, snap)
            torch.accelerator.synchronize()
            with profile(
                activities=[
                    ProfilerActivity.CPU, ProfilerActivity.CUDA
                ]
            ) as prof:
                _advance(core, batch, state, gid, fn)
            torch.accelerator.synchronize()
            counts: dict[str, int] = {}
            host_reads: dict[str, int] = {}
            for evt in prof.events():
                for marker in SYNC_MARKERS:
                    if marker in evt.name:
                        counts[marker] = counts.get(marker, 0) + 1
                for marker in HOST_READ_MARKERS:
                    if marker in evt.name:
                        host_reads[marker] = (
                            host_reads.get(marker, 0) + 1
                        )
            trace = out_dir / f"trace-{label}-{name}.json"
            prof.export_chrome_trace(str(trace))
            cell[f"profiler_{name}"] = counts
            cell[f"host_reads_{name}"] = host_reads
            cell[f"trace_{name}"] = str(trace)
        dense_counts = cell["profiler_dense"]
        check.equal(
            sum(dense_counts.values()), 0,
            f"sync[{label}].dense device-sync count {dense_counts}",
        )
        compact_counts = cell["profiler_compact"]
        check.ok(
            sum(compact_counts.values()) > 0,
            f"sync[{label}].compact control saw no syncs "
            f"{compact_counts}",
        )
        report[label] = cell
    return report


def run_graph_arm(
    check: Check, device: str, seed: int
) -> dict[str, Any]:
    """CUDA-graph capture of the sync-free path after warmup."""
    report: dict[str, Any] = {}
    gid, label, lookahead, cadence, chunk = GEOMETRIES[4]
    core, batch, state = _sync_bucket(
        gid, lookahead, cadence, chunk, device, seed
    )
    snap = _state_snapshot(state)

    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(3):
            _state_restore(state, snap)
            _advance(
                core, batch, state, gid, rnnt.decode_dense_masked
            )
    torch.cuda.current_stream().wait_stream(side)
    torch.accelerator.synchronize()

    _state_restore(state, snap)
    graph = torch.cuda.CUDAGraph()
    try:
        with torch.cuda.graph(graph):
            captured = _advance(
                core, batch, state, gid, rnnt.decode_dense_masked
            )
        report["dense_capture"] = "ok"
    except RuntimeError as err:
        report["dense_capture"] = f"FAILED: {err}"
        report["dense_capture_traceback"] = traceback.format_exc()
        check.ok(False, f"graph[{label}].dense capture: {err}")
        return report
    check.ok(True, "")

    # Replay from the snapshot and compare against the eager run.
    _state_restore(state, snap)
    graph.replay()
    torch.accelerator.synchronize()
    replay_tokens = [
        _tokens(captured, row)
        for row in range(batch.samples.shape[0])
    ]
    replay_status = captured.row_status.tolist()
    replay_state = _state_snapshot(state)

    eager_state = _fresh_state(batch.samples.shape[0], device)
    _state_restore(eager_state, snap)
    eager = _advance(
        core, batch, eager_state, gid, rnnt.decode_dense_masked
    )
    torch.accelerator.synchronize()
    check.equal(
        replay_status, eager.row_status.tolist(),
        f"graph[{label}].replay row_status",
    )
    for row in range(batch.samples.shape[0]):
        check.equal(
            replay_tokens[row], _tokens(eager, row),
            f"graph[{label}].replay tokens[{row}]",
        )
    for key in ("h", "c"):
        check.close(
            replay_state[key], getattr(eager_state, key), 1e-6,
            f"graph[{label}].replay {key}",
        )
    check.close(
        replay_state["frontend_counters"].float(),
        eager_state.frontend_counters.float(),
        0.0,
        f"graph[{label}].replay counters",
    )
    report["replay_matches_eager"] = not check.failures

    # New input values through the SAME static tensors and graph.
    torch.manual_seed(seed + 9)
    batch.samples.copy_(torch.randn_like(batch.samples) * 0.1)
    _state_restore(state, snap)
    graph.replay()
    torch.accelerator.synchronize()
    replay2 = [
        _tokens(captured, row)
        for row in range(batch.samples.shape[0])
    ]
    _state_restore(eager_state, snap)
    eager2 = _advance(
        core, batch, eager_state, gid, rnnt.decode_dense_masked
    )
    torch.accelerator.synchronize()
    for row in range(batch.samples.shape[0]):
        check.equal(
            replay2[row], _tokens(eager2, row),
            f"graph[{label}].replay2 tokens[{row}]",
        )
    report["replay_tracks_new_inputs"] = True

    # Negative control: the compact arm's shape sync must refuse
    # capture.
    _state_restore(state, snap)
    bad = torch.cuda.CUDAGraph()
    try:
        with torch.cuda.graph(bad):
            _advance(
                core, batch, state, gid, rnnt.decode_compact_active
            )
        report["compact_capture"] = "captured (UNEXPECTED)"
        check.ok(
            False, f"graph[{label}].compact capture unexpectedly ok"
        )
    except RuntimeError:
        report["compact_capture"] = "refused-as-expected"
    return report


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--seed-base", type=int, default=1_000)
    ap.add_argument(
        "--arm", default="all",
        choices=["all", "matrix", "sync", "graph"],
    )
    ap.add_argument("--units-80ms", type=int, default=20)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("FATAL: --device cuda but no CUDA device", flush=True)
        sys.exit(2)

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False

    out_dir = (
        args.out.parent if args.out else Path("p6b-cuda-matrix-out")
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    check = Check()
    report: dict[str, Any] = {
        "probe": "p6b_cuda_matrix",
        "fingerprint": {
            "device": device,
            "device_name": (
                torch.cuda.get_device_name(0)
                if device == "cuda" else platform.processor()
            ),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
            "tf32_cudnn": torch.backends.cudnn.allow_tf32,
            "repeats": args.repeats,
            "seed_base": args.seed_base,
        },
        "arms": {},
    }

    if args.arm in ("all", "matrix"):
        t0 = time.time()
        for rep in range(args.repeats):
            seed = args.seed_base + 97 * rep
            for gid, label, lookahead, cadence, chunk in GEOMETRIES:
                units = (
                    args.units_80ms if label == "80ms" else 4
                )
                run_mixed_phase(
                    check, gid, lookahead, cadence, chunk, device,
                    seed + gid,
                )
                run_lockstep_drive(
                    check, gid, lookahead, cadence, chunk, device,
                    seed + gid, units,
                )
                run_protocol_rows(
                    check, gid, lookahead, cadence, chunk, device,
                    seed + gid,
                )
                print(
                    f"matrix rep {rep} {label}: "
                    f"{len(check.failures)} failures so far",
                    flush=True,
                )
        report["arms"]["matrix"] = {
            "repeats": args.repeats,
            "checks": check.count,
            "elapsed_s": round(time.time() - t0, 1),
            "max_state_diffs": {
                k: f"{v:.3e}"
                for k, v in sorted(check.max_diffs.items())
                if v > 0.0
            },
        }

    if args.arm in ("all", "sync"):
        if device != "cuda":
            print("sync arm skipped (needs CUDA)", flush=True)
        else:
            report["arms"]["sync"] = run_sync_arm(
                check, device, args.seed_base, out_dir
            )

    if args.arm in ("all", "graph"):
        if device != "cuda":
            print("graph arm skipped (needs CUDA)", flush=True)
        else:
            report["arms"]["graph"] = run_graph_arm(
                check, device, args.seed_base
            )

    report["checks"] = check.count
    report["failures"] = check.failures
    report["pass"] = not check.failures
    text = json.dumps(report, indent=2)
    if args.out:
        args.out.write_text(text)
    print(text)
    print(
        f"{'PASS' if report['pass'] else 'FAIL'}: "
        f"{check.count} checks, {len(check.failures)} failures",
        flush=True,
    )
    sys.exit(0 if report["pass"] else 1)


if __name__ == "__main__":
    main()
