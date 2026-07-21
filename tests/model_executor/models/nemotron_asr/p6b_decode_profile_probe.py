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

Activity is REALIZED, not merely declared (PR #94 round 3): the
blank-logit bias is calibrated PER CELL on the cell's exact frames
and lengths (a delta minted on scout frames does not transfer — the
first L4 round realized 0–20 labels/chunk against target 3). Every
dcp cell records target, realized, absolute error, and
`policy_realized` against the declared tolerance window; a cell
outside its window is NONSELECTIVE (the generator must pick the
sync-free arm there — small tiers that cannot realize a fractional
target become dense-only, which is costless since dense wins them
uncontested). Saturated and mixed are explicitly nonselective.
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
        "manifests", "frontend", "rnnt_cell", "rnnt",
        "state_scatter", "advance",
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

#: Artifact schema revision — bumped whenever report structure or
#: semantics change; the generator admits known schemas only.
PROBE_SCHEMA = "p6b-decode-profile-v3"

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

#: The dispatch-calibration policy the artifact must REALIZE, not
#: merely declare (decisions/decode-dispatch-regime.md): selective
#: cells (silence/speech) are calibrated per cell on the cell's own
#: frames and must land inside the declared tolerance window to be
#: `policy_realized`; a cell that misses is NONSELECTIVE — the
#: generator must pick the sync-free arm there and may not select
#: compact from it. Saturated and mixed are explicitly nonselective
#: (stress/heterogeneity bounds outside dcp-v1).
DCP_V1: dict[str, Any] = {
    "version": "dcp-v1",
    "targets": {"silence": 0.3, "speech": 3.0},
    "tolerance": {"silence": (0.0, 1.0), "speech": (1.5, 4.5)},
    "selective_activities": ("silence", "speech"),
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


def _driver_version() -> str | None:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version",
             "--format=csv,noheader"],
            capture_output=True, text=True, timeout=5,
        )
        return out.stdout.strip().splitlines()[0]
    except Exception:
        return None


def _probe_revision() -> dict[str, Any]:
    """The fork revision + dirty flag this probe ran from — evidence
    identity, so an artifact binds to exact code. Dirty means
    TRACKED modifications only (untracked scratch on a pod checkout
    cannot change probe behavior and made round 6's artifact
    generator-ineligible for no reproducibility reason); the paths
    are recorded so a dirty flag is auditable, and untracked files
    are counted separately."""
    root = Path(__file__).resolve().parents[4]
    try:
        rev = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=5,
        ).stdout.strip()
        tracked = subprocess.run(
            ["git", "-C", str(root), "status", "--porcelain",
             "--untracked-files=no"],
            capture_output=True, text=True, timeout=10,
        ).stdout.strip()
        untracked = subprocess.run(
            ["git", "-C", str(root), "ls-files", "--others",
             "--exclude-standard"],
            capture_output=True, text=True, timeout=10,
        ).stdout.strip()
        untracked_paths = (
            untracked.splitlines() if untracked else []
        )
        return {
            "fork_commit": rev,
            "dirty_tree": bool(tracked),
            "dirty_paths": tracked.splitlines()[:20],
            # EVERY untracked path, uncapped, so the artifact itself
            # is auditable (a count alone cannot prove all of them
            # were benign, and a truncated list would let file N+1
            # escape auditing — admission cross-checks the count
            # against the list length); the generator allowlists the
            # narrow lock-file class and fails on anything else.
            "untracked_files": len(untracked_paths),
            "untracked_paths": untracked_paths,
        }
    except Exception:
        return {"fork_commit": None, "dirty_tree": None}


def _t_pad(cadence: int) -> int:
    """The bucket's padded encoder width, from the same host formula
    advance_session uses; dims don't matter, only kernel/stride/
    stages."""
    enc = MODS["encoder"].FastConformerEncoder(
        feat_in=16, d_model=32, d_ff=32, n_layers=1, n_heads=2,
        conv_kernel=5, subsampling_channels=8, att_context=(4, 0),
    )
    mel_width = 9 + cadence
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
    warm_delta: float | None = None,
) -> tuple[float, float]:
    """Binary-search a blank-logit bias delta realizing ~target labels
    per chunk ON THESE EXACT frames/lengths; returns (delta,
    realized). The first L4 round proved a delta minted on scout
    frames does NOT transfer (realized 0–20 against target 3), so
    calibration runs per cell — ``warm_delta`` only narrows the
    initial bracket. Random-init nets at a wide argmax emit near the
    cap by default, so positive deltas suppress emission."""
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
        if warm_delta is not None:
            wlo, whi = warm_delta - 3.0, warm_delta + 3.0
            # Only narrow if the tight bracket still straddles the
            # target (emission decreases as delta rises).
            if realized(wlo) >= target >= realized(whi):
                lo, hi = wlo, whi
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
    warm: dict[str, float],
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
        # Per-cell dcp realization: silence/speech calibrate on THIS
        # cell's frames/lengths (warm-started from the previous tier);
        # mixed reuses this tier's speech delta (nonselective);
        # saturated is the fixed strong-negative bias (nonselective).
        cal_activity = (
            "speech" if activity == "mixed" else activity
        )
        target = DCP_V1["targets"].get(cal_activity)
        # Restore the pristine bias FIRST: calibration captures the
        # current bias as its base, and the previous activity's delta
        # is still applied at this point (base-drift bug caught in
        # the r6 smoke: silence cells realized at the cap).
        with torch.no_grad():
            out.bias.copy_(bias_base)
        if activity == "saturated":
            delta, _ = calibrate_blank_bias(
                joint, frames, lengths, predictor, dims, device,
                None,
            )
        elif activity == "mixed":
            delta = warm.get("speech", 0.0)
        else:
            delta, _ = calibrate_blank_bias(
                joint, frames, lengths, predictor, dims, device,
                target, warm_delta=warm.get(activity),
            )
            warm[activity] = delta
        with torch.no_grad():
            out.bias.copy_(bias_base)
            out.bias[dims["vocab"]] += delta
        realized = _labels_per_chunk(
            frames, lengths, predictor, joint, dims, device
        )
        selective_intent = (
            activity in DCP_V1["selective_activities"]
        )
        if selective_intent:
            lo_t, hi_t = DCP_V1["tolerance"][activity]
            policy_realized = lo_t <= realized <= hi_t
            abs_error = abs(realized - float(target))
        else:
            policy_realized = False
            abs_error = None
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
            "target_labels_per_chunk": target,
            "labels_per_chunk": round(realized, 3),
            "abs_error": (
                round(abs_error, 3) if abs_error is not None else None
            ),
            "policy_realized": policy_realized,
            # A cell the generator may select compact FROM: dcp
            # activity realized within tolerance. Everything else is
            # sync-free-forced (small tiers that cannot realize a
            # fractional target become dense-only — costless, dense
            # wins them uncontested anyway).
            "selective": selective_intent and policy_realized,
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
            # Warm-start bounds carried across ascending tiers within
            # one (geometry, lane) — calibration itself is PER CELL,
            # on the cell's own frames (dcp realization, PR #94 r3).
            warm: dict[str, float] = {}
            for batch in tiers:
                rows = run_cells_for_tier(
                    label, t_pad, batch, precision, warm, dims,
                    predictor, joint, device,
                    args.seed + 31 * gid + batch, args.iters,
                    args.brackets, budget_us,
                )
                for row in rows:
                    if not row["tokens_match"]:
                        # BF16 argmax is not batch-shape-invariant:
                        # compact's compacted GEMMs change reduction
                        # order, and the speech calibration parks
                        # blank-vs-label logits near a tie (L4 round:
                        # 18/300 bf16 cells, 0/300 fp32). The bf16
                        # lane is therefore EXPLORATORY — timed but
                        # correctness-unqualified, never generator
                        # input (PORT-DEC-008: candidates match
                        # before they are dispatchable; bf16 dispatch
                        # waits on real-checkpoint precision
                        # qualification). Counted per lane; never
                        # silently ignored.
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

    lane_definitions = {
        "fp32": "all decode modules fp32",
        "bf16-joint": (
            "joint BF16; predictor + recurrent state fp32 "
            "(PORT-PREC-001/005 seam)"
        ),
    }
    report: dict[str, Any] = {
        "probe": "p6b_decode_profile",
        "fingerprint": {
            "probe_schema": PROBE_SCHEMA,
            "decode_algo_revision": rnnt.DECODE_ALGO_REVISION,
            "lane_definitions_digest": manifests.manifest_hash(
                {"lanes": lane_definitions}
            ),
            "device": device,
            "device_name": (
                torch.cuda.get_device_name(0)
                if device == "cuda" else platform.processor()
            ),
            "driver": _driver_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
            **_probe_revision(),
            "weights": f"synthetic-seed-{args.seed}",
            "config_source": (
                "configuration_nemotron_asr defaults"
                if config_asserted else "unasserted (shape-only)"
            ),
            "dims": dims,
            "model_shape_digest": shape_digest,
            "config_asserted": config_asserted,
            "max_symbols": rnnt.MAX_SYMBOLS_PER_STEP,
            "lanes": {
                "fp32": {
                    "definition": "all decode modules fp32",
                    "status": "performance-qualified",
                },
                "bf16-joint": {
                    "definition": (
                        "joint BF16; predictor + recurrent state "
                        "fp32 (PORT-PREC-001/005 seam)"
                    ),
                    "status": (
                        "EXPLORATORY: performance-measured, "
                        "correctness-unqualified (BF16 argmax not "
                        "batch-shape-invariant); dispatch pending "
                        "real-checkpoint precision qualification"
                    ),
                },
            },
            "activity_forcing": (
                "synthetic blank-bias calibration; table generation "
                "must bind its own versioned activity-envelope id"
            ),
            "tiers": tiers,
            "iters_cap": args.iters,
            "brackets": args.brackets,
            "bracket_budget_ms": args.bracket_budget_ms,
            "seed": args.seed,
        },
        "cells": cells,
    }
    # Qualification is structurally PER LANE — there is no global
    # "pass" a generator could mistake for whole-run qualification.
    # execution_ok means the sweep completed; dispatch eligibility is
    # a lane property.
    def _lane_policy(lane: str) -> dict[str, int]:
        intended = [
            c for c in cells
            if c["precision"] == lane
            and c["activity"] in DCP_V1["selective_activities"]
        ]
        realized_n = sum(1 for c in intended if c["policy_realized"])
        return {
            "dcp_intended_cells": len(intended),
            "dcp_realized_cells": realized_n,
            "dcp_nonselective_fallback_cells": (
                len(intended) - realized_n
            ),
        }

    lane_results: dict[str, dict[str, Any]] = {
        lane: {
            "mismatch_cells": (
                mismatches if lane == "fp32" else bf16_mismatches
            ),
            **_lane_policy(lane),
            "performance_qualified": (
                lane == "fp32" and mismatches == 0
            ),
            # Eligibility = correctness-clean; every dcp cell is
            # either policy_realized (selective) or explicitly
            # nonselective (sync-free-forced) — the generator applies
            # the fallback, never a drifted measurement.
            "dispatch_eligible": (
                lane == "fp32" and mismatches == 0
            ),
        }
        for lane in joints
    }
    eligible = [
        lane
        for lane, res in lane_results.items()
        if res["dispatch_eligible"]
    ]
    report["execution_ok"] = True
    report["dcp"] = {
        **{
            k: v for k, v in DCP_V1.items()
        },
        "fallback_rule": (
            "a dcp cell outside tolerance is nonselective: the "
            "generator must select the sync-free arm there"
        ),
    }
    report["lane_results"] = lane_results
    report["generator_eligible_lanes"] = eligible
    text = json.dumps(report, indent=2)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text)
    print(text[-1_500:] if len(text) > 1_500 else text)
    fp32_ok = mismatches == 0
    summary = [
        f"FP32 {'PASS' if fp32_ok else 'FAIL'} "
        f"({mismatches} mismatch cells)"
    ]
    if "bf16-joint" in joints:
        summary.append(
            f"BF16 UNQUALIFIED ({bf16_mismatches} mismatch cells, "
            "exploratory)"
        )
    summary.append(
        "generator-eligible lanes: " + (", ".join(eligible) or "NONE")
    )
    print(f"{len(cells)} cells; " + "; ".join(summary), flush=True)
    sys.exit(0 if fp32_ok else 1)


if __name__ == "__main__":
    main()
