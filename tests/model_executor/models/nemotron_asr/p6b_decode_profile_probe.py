# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Dense-vs-compact decode profile (Phase-6b step 3, PORT-DEC-008).

Measures the two correctness-equal RNN-T decode candidates at
PRODUCTION SHAPES (decode touches only predictor + joint — 640-wide
two-layer LSTM and a 1024/640 -> 640 -> 13088 joint — so real dims
are cheap and timing needs shapes, not weights) over the
dispatch-table axes of decisions/decode-dispatch-regime.md (r2):

  geometry {5} x padded batch tier {1..1024, incl. 128/512 around the
  predicted crossovers} x precision {fp32, bf16-joint: joint in BF16,
  predictor + recurrent state fp32 per PORT-PREC-001/005} x arm
  {dense-eager, dense-graphed, compact-eager} x activity
  {silence, speech ~3 labels/chunk, saturated, mixed 50% zero-length}.

Dense timing is ACTIVITY-INVARIANT (fixed T_pad x S trips, identical
kernel shapes regardless of emission), so the dense arms run once per
(geometry, tier, precision) at the speech bias; compact — the only
distribution-sensitive arm — runs all four activity levels.

Activity is forced by a CALIBRATED blank-logit bias per (geometry,
precision); realized labels-per-chunk is recorded next to every cell.
Timing reports BOTH protocols: per-call latency (terminal sync per
call) and K-calls-one-sync throughput; the host/GPU occupancy ratio
under the throughput protocol is the recorded proxy for compact's
host-thread serialization (the multi-bucket magnitude stays
unmeasured at step 3 — the regime record says so). The graphed arm
times replay-only and records capture time + memory delta as a
startup budget, freeing each graph before the next cell. Iteration
counts adapt to cell cost against a per-bracket time budget. Every
cell asserts dense-vs-compact token equality once, outside timing,
same-lane. SM clock is logged per cell (L4 throttles; the log is the
drift record).

The probe MEASURES; it selects nothing. Selection rules (hysteresis,
crossover-gap sync-free preference) belong to the startup generator,
and the numbers are per-GPU-class evidence (an L4 profile validates
methodology and is never the A100 table — A12).

Run (on the pod):
    /opt/venv-port/bin/python tests/model_executor/models/nemotron_asr/\
      p6b_decode_profile_probe.py --device cuda \
        --out /workspace/evidence/p6b-decode-profile/report.json

Local smoke: --device cpu --tiny (small dims, tiers {1,4}, eager
arms, fp32 only).
"""

from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import platform
import subprocess
import sys
import time
import types
from pathlib import Path
from typing import Any

import torch

_PKG = (
    Path(__file__).resolve().parents[4]
    / "vllm_omni/model_executor/models/nemotron_asr"
)
_BASE = "vllm_omni.model_executor.models.nemotron_asr"


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
rnnt = MODS["rnnt"]
manifests = MODS["manifests"]

#: Production decode dims (nemotron_asr.py NemotronASRCore defaults;
#: vocab = num_asr_labels, blank = index vocab). Asserted against the
#: model config below where that module is importable.
PROD = {
    "enc_hidden": 1024,
    "pred_hidden": 640,
    "joint_hidden": 640,
    "vocab": 13_087,
    "pred_layers": 2,
}
TINY = {
    "enc_hidden": 64,
    "pred_hidden": 32,
    "joint_hidden": 32,
    "vocab": 128,
    "pred_layers": 2,
}

GEOMETRIES = [
    (gid, label, right, 8 * (right + 1))
    for gid, (label, (_, right)) in enumerate(
        manifests.CADENCES.items()
    )
]

#: Labels-per-chunk calibration targets; ``None`` = saturated (strong
#: negative bias, emission at the cap). "mixed" reuses the speech
#: bias with 50% zero-length rows — no separate calibration.
ACTIVITY_TARGETS = {
    "silence": 0.3,
    "speech": 3.0,
    "saturated": None,
}
COMPACT_ACTIVITIES = ("silence", "speech", "saturated", "mixed")


def _assert_config_dims(dims: dict[str, int]) -> bool:
    """Assert probe dims against the model config where importable
    (the tiny-vocab lesson, made structural). Returns whether the
    assert ran."""
    try:
        spec = importlib.util.spec_from_file_location(
            f"{_BASE}.configuration_nemotron_asr",
            _PKG / "configuration_nemotron_asr.py",
        )
        assert spec is not None and spec.loader is not None
        cfg_mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cfg_mod)
    except Exception:
        return False
    cfg = cfg_mod.NemotronASRConfig()
    assert dims["vocab"] == cfg.num_asr_labels, (
        dims["vocab"], cfg.num_asr_labels,
    )
    assert dims["enc_hidden"] == cfg.d_model
    assert dims["pred_hidden"] == cfg.pred_hidden
    assert dims["joint_hidden"] == cfg.joint_hidden
    assert dims["pred_layers"] == cfg.pred_rnn_layers
    return True


def _sm_clock_mhz() -> int | None:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=clocks.sm",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
        return int(out.stdout.strip().splitlines()[0])
    except Exception:
        return None


def _t_pad(cadence: int) -> int:
    """The bucket's padded encoder width, from the same host formula
    advance_session uses; dims don't matter, only kernel/stride/
    stages."""
    enc = MODS["encoder"].FastConformerEncoder(
        feat_in=16, d_model=32, d_ff=32, n_layers=1, n_heads=2,
        conv_kernel=5, subsampling_channels=8, att_context=(4, 0),
    )
    mel_width = 9 + cadence + 6
    return int(
        enc.pre_encode.output_lengths(torch.tensor([mel_width]))[0]
    )


def _stack(
    dims: dict[str, int], device: str, seed: int
) -> tuple[Any, dict[str, Any]]:
    """Predictor (fp32, PORT-PREC-005) + per-precision joints."""
    torch.manual_seed(seed)
    predictor = rnnt.Predictor(
        vocab_size=dims["vocab"],
        pred_hidden=dims["pred_hidden"],
        pred_rnn_layers=dims["pred_layers"],
    )
    joint = rnnt.Joint(
        enc_hidden=dims["enc_hidden"],
        pred_hidden=dims["pred_hidden"],
        joint_hidden=dims["joint_hidden"],
        vocab_size=dims["vocab"],
    )
    predictor.to(device).eval()
    joint.to(device).eval()
    joints: dict[str, Any] = {"fp32": joint}
    if device == "cuda":
        jb = copy.deepcopy(joint).to(torch.bfloat16)
        jb.eval()
        joints["bf16-joint"] = jb
    return predictor, joints


def _fresh_state(
    batch: int, dims: dict[str, int], device: str
) -> Any:
    return rnnt.DecodeState(
        h=torch.zeros(
            dims["pred_layers"], batch, dims["pred_hidden"],
            device=device,
        ),
        c=torch.zeros(
            dims["pred_layers"], batch, dims["pred_hidden"],
            device=device,
        ),
        last_label=torch.full(
            (batch,), dims["vocab"], dtype=torch.long, device=device
        ),
    )


def _labels_per_chunk(
    frames: torch.Tensor,
    lengths: torch.Tensor,
    predictor: Any,
    joint: Any,
    dims: dict[str, int],
    device: str,
) -> float:
    with torch.no_grad():
        _, tl, _ = rnnt.decode_dense_masked(
            frames, lengths, predictor, joint,
            _fresh_state(frames.shape[0], dims, device),
        )
    return float(tl.float().mean())


def calibrate_blank_bias(
    joint: Any,
    frames: torch.Tensor,
    lengths: torch.Tensor,
    predictor: Any,
    dims: dict[str, int],
    device: str,
    target: float | None,
) -> tuple[float, float]:
    """Binary-search a blank-logit bias delta realizing ~target labels
    per chunk; returns (delta, realized). Random-init nets at a wide
    argmax emit near the cap by default, so positive deltas suppress
    emission."""
    out = joint.joint_net[1]
    blank = dims["vocab"]
    base = out.bias.detach().clone()

    def realized(delta: float) -> float:
        with torch.no_grad():
            out.bias.copy_(base)
            out.bias[blank] += delta
        return _labels_per_chunk(
            frames, lengths, predictor, joint, dims, device
        )

    try:
        if target is None:
            delta = -8.0
            return delta, realized(delta)
        lo, hi = -8.0, 24.0
        got = 0.0
        for _ in range(18):
            mid = (lo + hi) / 2
            got = realized(mid)
            if abs(got - target) <= max(0.15 * target, 0.05):
                return mid, got
            if got > target:
                lo = mid
            else:
                hi = mid
        return (lo + hi) / 2, got
    finally:
        with torch.no_grad():
            out.bias.copy_(base)


def _time_arm(
    fn: Any,
    args: tuple,
    device: str,
    iters: int,
    brackets: int,
    budget_us: float,
) -> dict[str, Any]:
    """Both timing protocols. Throughput: N un-synced calls between
    events, one terminal sync (median of brackets; adaptive N against
    the bracket budget). Latency: per-call terminal sync, median."""
    with torch.no_grad():
        for _ in range(3):
            fn(*args)
        if device != "cuda":
            t0 = time.perf_counter()
            fn(*args)
            est = (time.perf_counter() - t0) * 1e6
            n = max(3, min(iters, int(budget_us / max(est, 1.0))))
            walls: list[float] = []
            for _ in range(brackets):
                t0 = time.perf_counter()
                for _ in range(n):
                    fn(*args)
                walls.append((time.perf_counter() - t0) * 1e6 / n)
            walls.sort()
            return {
                "wall_us": round(walls[len(walls) // 2], 1),
                "iters": n,
            }
        torch.accelerator.synchronize()
        t0 = time.perf_counter()
        fn(*args)
        torch.accelerator.synchronize()
        est = (time.perf_counter() - t0) * 1e6
        n = max(3, min(iters, int(budget_us / max(est, 1.0))))
        pairs: list[tuple[float, float]] = []
        for _ in range(brackets):
            torch.accelerator.synchronize()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            t0 = time.perf_counter()
            start.record()
            for _ in range(n):
                fn(*args)
            end.record()
            enqueue_us = (time.perf_counter() - t0) * 1e6 / n
            torch.accelerator.synchronize()
            pairs.append(
                (start.elapsed_time(end) * 1e3 / n, enqueue_us)
            )
        pairs.sort()
        gpu_us, enqueue_us = pairs[len(pairs) // 2]
        lats: list[float] = []
        for _ in range(min(n, 10)):
            torch.accelerator.synchronize()
            t0 = time.perf_counter()
            fn(*args)
            torch.accelerator.synchronize()
            lats.append((time.perf_counter() - t0) * 1e6)
        lats.sort()
    return {
        "gpu_us": round(gpu_us, 1),
        "enqueue_us": round(enqueue_us, 1),
        "host_occupancy": round(
            min(enqueue_us / max(gpu_us, 1e-9), 1.0), 3
        ),
        "latency_us": round(lats[len(lats) // 2], 1),
        "iters": n,
    }


def run_cells_for_tier(
    label: str,
    t_pad: int,
    batch: int,
    precision: str,
    deltas: dict[str, tuple[float, float]],
    dims: dict[str, int],
    predictor: Any,
    joint: Any,
    device: str,
    seed: int,
    iters: int,
    brackets: int,
    budget_us: float,
) -> list[dict[str, Any]]:
    torch.manual_seed(seed)
    frames = (
        torch.randn(batch, t_pad, dims["enc_hidden"], device=device)
        * 0.1
    )
    full = torch.full(
        (batch,), t_pad, dtype=torch.long, device=device
    )
    out = joint.joint_net[1]
    bias_base = out.bias.detach().clone()
    rows: list[dict[str, Any]] = []

    def cell_base(activity: str, lengths: torch.Tensor) -> dict:
        delta, cal = deltas[
            "speech" if activity == "mixed" else activity
        ]
        with torch.no_grad():
            out.bias.copy_(bias_base)
            out.bias[dims["vocab"]] += delta
        realized = _labels_per_chunk(
            frames, lengths, predictor, joint, dims, device
        )
        state = _fresh_state(batch, dims, device)
        with torch.no_grad():
            d_ids, d_len, _ = rnnt.decode_dense_masked(
                frames, lengths, predictor, joint, state
            )
            c_ids, c_len, _ = rnnt.decode_compact_active(
                frames, lengths, predictor, joint, state
            )
        return {
            "geometry": label,
            "t_pad": t_pad,
            "batch": batch,
            "precision": precision,
            "activity": activity,
            "blank_bias": round(delta, 3),
            "labels_per_chunk": round(realized, 3),
            "calibration_labels_per_chunk": round(cal, 3),
            "tokens_match": bool(
                torch.equal(d_len, c_len)
                and torch.equal(d_ids, c_ids)
            ),
            "sm_clock_mhz": _sm_clock_mhz(),
            "_state": state,
            "_lengths": lengths,
        }

    # Dense arms: activity-invariant timing — one run at the speech
    # bias per (geometry, tier, precision).
    base = cell_base("speech", full)
    state, lengths = base.pop("_state"), base.pop("_lengths")
    rows.append({
        **base, "arm": "dense-eager",
        **_time_arm(
            rnnt.decode_dense_masked,
            (frames, lengths, predictor, joint, state),
            device, iters, brackets, budget_us,
        ),
    })
    if device == "cuda":
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side), torch.no_grad():
            for _ in range(3):
                rnnt.decode_dense_masked(
                    frames, lengths, predictor, joint, state
                )
        torch.cuda.current_stream().wait_stream(side)
        torch.accelerator.synchronize()
        # Allocated, not reserved: capture allocates into the graph's
        # private pool and reserved-delta went NEGATIVE in the first
        # L4 round (allocator-release artifact) — a broken metric.
        alloc0 = torch.cuda.memory_allocated()
        graph = torch.cuda.CUDAGraph()
        t0 = time.perf_counter()
        with torch.cuda.graph(graph), torch.no_grad():
            rnnt.decode_dense_masked(
                frames, lengths, predictor, joint, state
            )
        capture_ms = (time.perf_counter() - t0) * 1e3
        capture_mb = (
            torch.cuda.memory_allocated() - alloc0
        ) / 2**20
        rows.append({
            **base, "arm": "dense-graphed",
            "capture_ms": round(capture_ms, 1),
            "capture_mb": round(capture_mb, 1),
            **_time_arm(
                graph.replay, (), device, iters * 3, brackets,
                budget_us,
            ),
        })
        # Free the graph's private pool before the next cell — ~100
        # accumulated capture pools would exhaust the device.
        del graph
        torch.accelerator.synchronize()
        torch.accelerator.empty_cache()

    # Compact: the distribution-sensitive arm — all activity levels,
    # including the heterogeneous mixed bucket.
    for activity in COMPACT_ACTIVITIES:
        lengths = full
        if activity == "mixed":
            lengths = full.clone()
            lengths[: batch // 2] = 0
        base = cell_base(activity, lengths)
        state, lengths = base.pop("_state"), base.pop("_lengths")
        rows.append({
            **base, "arm": "compact-eager",
            **_time_arm(
                rnnt.decode_compact_active,
                (frames, lengths, predictor, joint, state),
                device, iters, brackets, budget_us,
            ),
        })
    with torch.no_grad():
        out.bias.copy_(bias_base)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--tiny", action="store_true")
    ap.add_argument(
        "--tiers", default="1,4,8,16,32,64,128,256,512,1024",
        help="comma-separated padded batch tiers (run ascending)",
    )
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--brackets", type=int, default=5)
    ap.add_argument(
        "--bracket-budget-ms", type=float, default=200.0,
        help="target time per throughput bracket; iteration counts "
        "adapt to it",
    )
    ap.add_argument("--seed", type=int, default=2_000)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("FATAL: --device cuda but no CUDA device", flush=True)
        sys.exit(2)

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    dims = TINY if args.tiny else PROD
    config_asserted = (not args.tiny) and _assert_config_dims(dims)
    tiers = sorted(int(t) for t in args.tiers.split(","))
    if args.tiny:
        tiers = [t for t in tiers if t <= 4] or [1, 4]
    budget_us = args.bracket_budget_ms * 1e3

    predictor, joints = _stack(dims, device, args.seed)
    shape_digest = manifests.manifest_hash({
        "dims": dims,
        "max_symbols": rnnt.MAX_SYMBOLS_PER_STEP,
        "geometries": {
            label: 8 * (right + 1)
            for _, label, right, _ in GEOMETRIES
        },
    })
    cells: list[dict[str, Any]] = []
    mismatches = 0
    bf16_mismatches = 0
    for gid, label, lookahead, cadence in GEOMETRIES:
        t_pad = _t_pad(cadence)
        for precision, joint in joints.items():
            torch.manual_seed(args.seed + gid)
            cal_frames = (
                torch.randn(
                    16, t_pad, dims["enc_hidden"], device=device
                ) * 0.1
            )
            cal_lengths = torch.full(
                (16,), t_pad, dtype=torch.long, device=device
            )
            deltas = {
                activity: calibrate_blank_bias(
                    joint, cal_frames, cal_lengths, predictor,
                    dims, device, target,
                )
                for activity, target in ACTIVITY_TARGETS.items()
            }
            for batch in tiers:
                rows = run_cells_for_tier(
                    label, t_pad, batch, precision, deltas, dims,
                    predictor, joint, device,
                    args.seed + 31 * gid + batch, args.iters,
                    args.brackets, budget_us,
                )
                for row in rows:
                    if not row["tokens_match"]:
                        # BF16 argmax is not batch-shape-invariant:
                        # compact's compacted GEMMs change reduction
                        # order, and the speech calibration parks
                        # blank-vs-label logits near a tie, so bf16
                        # cross-arm flips at large B are a measured
                        # property of the lane, not a decoder defect
                        # (L4 round: 18/300 bf16 cells, 0/300 fp32).
                        # Recorded, gating only the fp32 lane.
                        if row["precision"] == "fp32":
                            mismatches += 1
                        else:
                            bf16_mismatches += 1
                cells.extend(rows)
                probe_us = [
                    f"{r['arm']}[{r['activity']}]="
                    f"{r.get('latency_us', r.get('wall_us', 0)):.0f}us"
                    for r in rows
                ]
                print(
                    f"{label} {precision} B={batch}: "
                    + ", ".join(probe_us),
                    flush=True,
                )

    report = {
        "probe": "p6b_decode_profile",
        "fingerprint": {
            "device": device,
            "device_name": (
                torch.cuda.get_device_name(0)
                if device == "cuda" else platform.processor()
            ),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
            "dims": dims,
            "model_shape_digest": shape_digest,
            "config_asserted": config_asserted,
            "max_symbols": rnnt.MAX_SYMBOLS_PER_STEP,
            "tiers": tiers,
            "iters_cap": args.iters,
            "brackets": args.brackets,
            "bracket_budget_ms": args.bracket_budget_ms,
            "seed": args.seed,
        },
        "cells": cells,
        "token_mismatch_cells": mismatches,
        "bf16_token_mismatch_cells": bf16_mismatches,
        "pass": mismatches == 0,
    }
    text = json.dumps(report, indent=2)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text)
    print(text[-1_500:] if len(text) > 1_500 else text)
    print(
        f"{'PASS' if report['pass'] else 'FAIL'}: {len(cells)} cells, "
        f"{mismatches} token-mismatch cells",
        flush=True,
    )
    sys.exit(0 if report["pass"] else 1)


if __name__ == "__main__":
    main()
