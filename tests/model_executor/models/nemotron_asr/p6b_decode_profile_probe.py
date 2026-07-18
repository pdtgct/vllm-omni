# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Dense-vs-compact decode profile (Phase-6b step 3, PORT-DEC-008).

Measures the two correctness-equal RNN-T decode candidates at
PRODUCTION SHAPES (decode touches only predictor + joint — 640-wide
two-layer LSTM and a 1024/640 -> 640 -> 13088 joint — so real dims are
cheap and timing needs shapes, not weights) over the dispatch-table
axes:

  geometry {80,160,320,560,1120 ms}  x  padded batch tier
  x  arm {dense-eager, dense-graphed, compact-eager}
  x  activity {silence-like, speech-like ~3 labels/chunk, saturated}

Activity is forced by biasing the joint's blank logit; the bias is
CALIBRATED per geometry against a labels-per-chunk target and the
REALIZED emission rate is recorded next to every cell — the forcing
is valid only when the realized distribution brackets the serving
anchor (design decisions/decode-dispatch-regime.md; LLD §MRV1 gives
L≈3 per chunk). Timing: CUDA events bracketing N un-synced iterations
(median-of-cells over repeated brackets), warmups first; the graphed
arm times replay and records one-time capture cost separately. Each
cell cross-checks dense-vs-compact token equality once, outside
timing, so the numbers stay visibly attached to the proven
correctness bar.

The probe MEASURES; it selects nothing. Its JSON feeds the startup
dispatch-table generator, and its numbers are per-GPU-class evidence
(an L4 profile validates methodology and is never the A100 table —
A12 fingerprint discipline).

Run (on the pod):
    /opt/venv-port/bin/python tests/model_executor/models/nemotron_asr/\
      p6b_decode_profile_probe.py --device cuda \
        --out /workspace/evidence/p6b-decode-profile/report.json

Local smoke: --device cpu --tiny (small dims, tiers {1,4}, eager arms
only).
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import platform
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
#: vocab = num_asr_labels, blank = index vocab_size).
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

#: Activity targets in labels per chunk (per row). "saturated" is the
#: emission cap — no calibration target, just a strongly negative
#: blank bias.
ACTIVITY_TARGETS = {
    "silence": 0.3,
    "speech": 3.0,
    "saturated": None,
}


def _t_pad(cadence: int) -> int:
    """The bucket's padded encoder width, from the same host formula
    advance_session uses (mel grid 9 + C + 6 through the pre-encode
    subsampling); dims don't matter, only kernel/stride/stages."""
    enc = MODS["encoder"].FastConformerEncoder(
        feat_in=16, d_model=32, d_ff=32, n_layers=1, n_heads=2,
        conv_kernel=5, subsampling_channels=8, att_context=(4, 0),
    )
    mel_width = 9 + cadence + 6
    return int(
        enc.pre_encode.output_lengths(torch.tensor([mel_width]))[0]
    )


def _stack(dims: dict[str, int], device: str, seed: int) -> Any:
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
    for m in (predictor, joint):
        m.to(device)
        m.eval()
    return predictor, joint


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
    per chunk; returns (delta, realized). ``None`` target = saturated
    (strong negative bias). Random-init nets at a 13k-wide argmax emit
    near the cap by default, so positive deltas suppress emission."""
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

    if target is None:
        delta = -8.0
        got = realized(delta)
        return delta, got
    lo, hi = -8.0, 24.0  # emission decreases as delta rises
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


def _time_eager(
    fn: Any, args: tuple, device: str, warmup: int, iters: int,
    brackets: int,
) -> dict[str, float]:
    """Median-of-brackets timing: each bracket runs ``iters`` calls
    between two CUDA events with NO per-call sync (compact self-syncs
    internally; that is part of its real cost)."""
    with torch.no_grad():
        for _ in range(warmup):
            fn(*args)
        if device != "cuda":
            walls: list[float] = []
            for _ in range(brackets):
                t0 = time.perf_counter()
                for _ in range(iters):
                    fn(*args)
                walls.append(
                    (time.perf_counter() - t0) * 1e6 / iters
                )
            walls.sort()
            return {"wall_us": walls[len(walls) // 2]}
        pairs: list[tuple[float, float]] = []
        for _ in range(brackets):
            torch.accelerator.synchronize()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            t0 = time.perf_counter()
            start.record()
            for _ in range(iters):
                fn(*args)
            end.record()
            torch.accelerator.synchronize()
            host_us = (time.perf_counter() - t0) * 1e6 / iters
            pairs.append((start.elapsed_time(end) * 1e3 / iters,
                          host_us))
    pairs.sort()
    gpu_us, host_med = pairs[len(pairs) // 2]
    return {"gpu_us": gpu_us, "host_us": host_med}


def run_cell(
    label: str,
    t_pad: int,
    batch: int,
    activity: str,
    delta: float,
    dims: dict[str, int],
    predictor: Any,
    joint: Any,
    device: str,
    seed: int,
    arms: list[str],
    iters: int,
    brackets: int,
) -> list[dict[str, Any]]:
    torch.manual_seed(seed)
    frames = (
        torch.randn(batch, t_pad, dims["enc_hidden"], device=device)
        * 0.1
    )
    lengths = torch.full(
        (batch,), t_pad, dtype=torch.long, device=device
    )
    out = joint.joint_net[1]
    with torch.no_grad():
        bias_base = out.bias.detach().clone()
        out.bias[dims["vocab"]] += delta

    realized = _labels_per_chunk(
        frames, lengths, predictor, joint, dims, device
    )
    # Correctness cross-check, once, outside timing: the two arms must
    # agree on tokens exactly (the step-2 bar, kept attached).
    state = _fresh_state(batch, dims, device)
    with torch.no_grad():
        d_ids, d_len, _ = rnnt.decode_dense_masked(
            frames, lengths, predictor, joint, state
        )
        c_ids, c_len, _ = rnnt.decode_compact_active(
            frames, lengths, predictor, joint, state
        )
    match = bool(
        torch.equal(d_len, c_len)
        and torch.equal(d_ids, c_ids)
    )

    base = {
        "geometry": label,
        "t_pad": t_pad,
        "batch": batch,
        "activity": activity,
        "blank_bias": round(delta, 3),
        "labels_per_chunk": round(realized, 3),
        "tokens_match": match,
    }
    rows: list[dict[str, Any]] = []
    if "dense-eager" in arms:
        rows.append({
            **base, "arm": "dense-eager",
            **_time_eager(
                rnnt.decode_dense_masked,
                (frames, lengths, predictor, joint, state),
                device, 3, iters, brackets,
            ),
        })
    if "compact-eager" in arms:
        rows.append({
            **base, "arm": "compact-eager",
            **_time_eager(
                rnnt.decode_compact_active,
                (frames, lengths, predictor, joint, state),
                device, 3, iters, brackets,
            ),
        })
    if "dense-graphed" in arms and device == "cuda":
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side), torch.no_grad():
            for _ in range(3):
                rnnt.decode_dense_masked(
                    frames, lengths, predictor, joint, state
                )
        torch.cuda.current_stream().wait_stream(side)
        torch.accelerator.synchronize()
        graph = torch.cuda.CUDAGraph()
        t0 = time.perf_counter()
        with torch.cuda.graph(graph), torch.no_grad():
            rnnt.decode_dense_masked(
                frames, lengths, predictor, joint, state
            )
        capture_ms = (time.perf_counter() - t0) * 1e3
        rows.append({
            **base, "arm": "dense-graphed",
            "capture_ms": round(capture_ms, 1),
            **_time_eager(
                graph.replay, (), device, 3, iters * 3, brackets
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
        "--tiers", default="1,4,8,16,32,64,256,1024",
        help="comma-separated padded batch tiers",
    )
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--brackets", type=int, default=5)
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
    tiers = [int(t) for t in args.tiers.split(",")]
    if args.tiny:
        tiers = [t for t in tiers if t <= 4] or [1, 4]
    arms = ["dense-eager", "compact-eager"]
    if device == "cuda":
        arms.append("dense-graphed")

    predictor, joint = _stack(dims, device, args.seed)
    cells: list[dict[str, Any]] = []
    mismatches = 0
    for gid, label, lookahead, cadence in GEOMETRIES:
        t_pad = _t_pad(cadence)
        # Calibrate the blank bias once per (geometry, activity) at a
        # mid tier; realized rates are re-recorded per cell.
        torch.manual_seed(args.seed + gid)
        cal_frames = (
            torch.randn(16, t_pad, dims["enc_hidden"], device=device)
            * 0.1
        )
        cal_lengths = torch.full(
            (16,), t_pad, dtype=torch.long, device=device
        )
        deltas = {}
        for activity, target in ACTIVITY_TARGETS.items():
            deltas[activity] = calibrate_blank_bias(
                joint, cal_frames, cal_lengths, predictor, dims,
                device, target,
            )
        for activity, (delta, cal_realized) in deltas.items():
            for batch in tiers:
                rows = run_cell(
                    label, t_pad, batch, activity, delta, dims,
                    predictor, joint, device,
                    args.seed + 31 * gid + batch, arms,
                    args.iters, args.brackets,
                )
                for row in rows:
                    row["calibration_labels_per_chunk"] = round(
                        cal_realized, 3
                    )
                    if not row["tokens_match"]:
                        mismatches += 1
                cells.extend(rows)
                print(
                    f"{label} B={batch} {activity}: "
                    + ", ".join(
                        f"{r['arm']}="
                        f"{r.get('host_us', r.get('wall_us', 0)):.0f}us"
                        for r in rows
                    ),
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
            "tiers": tiers,
            "iters": args.iters,
            "brackets": args.brackets,
            "seed": args.seed,
        },
        "cells": cells,
        "token_mismatch_cells": mismatches,
        "pass": mismatches == 0,
    }
    text = json.dumps(report, indent=2)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text)
    print(text[-2_000:] if len(text) > 2_000 else text)
    print(
        f"{'PASS' if report['pass'] else 'FAIL'}: {len(cells)} cells, "
        f"{mismatches} token-mismatch cells",
        flush=True,
    )
    sys.exit(0 if report["pass"] else 1)


if __name__ == "__main__":
    main()
