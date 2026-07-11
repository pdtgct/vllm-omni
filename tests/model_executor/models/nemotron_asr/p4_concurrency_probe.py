# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""P4 concurrency probe: batched chunk-steps (venv-port, dev pod).

The port's throughput lever is batching chunk-steps across sessions
(port-design §Performance model) — continuous batching makes chunk
steps scheduler steps, so the model-compute concurrency curve is the
capacity backbone. This probe sweeps batched sessions through the
steady-state per-chunk work (window encoder forward + LID + batched
greedy decode steps) at fp32 on one GPU, reporting per-stream
chunk latency percentiles and the EVAL-PERF-003 knee (first N where
p95 exceeds 2x the single-stream p95). Labeled port-internal,
compute-tier: engine scheduler overhead and transport ride on top at
the served tier.

Usage:
    /opt/venv-port/bin/python p4_concurrency_probe.py \
        --dump /workspace/weights/nemo-dump [--chunk-ms 560]
"""

import argparse
import json
import time
from pathlib import Path

import torch
from safetensors.torch import load_file

from vllm_omni.model_executor.models.nemotron_asr.convert import (
    convert_state_dict,
)
from vllm_omni.model_executor.models.nemotron_asr.encoder import (
    FastConformerEncoder,
)
from vllm_omni.model_executor.models.nemotron_asr.lid import (
    PromptConditioner,
)
from vllm_omni.model_executor.models.nemotron_asr.rnnt import (
    Joint,
    Predictor,
)
from vllm_omni.model_executor.models.nemotron_asr.rules import NEMO_RULES

CHUNK_FRAMES = {80: 1, 160: 2, 320: 4, 560: 7, 1120: 14}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dump", type=Path, required=True)
    parser.add_argument("--chunk-ms", type=int, default=560)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    device = torch.device(args.device)
    frames = CHUNK_FRAMES[args.chunk_ms]

    state = load_file(str(args.dump / "nemo_state.safetensors"))
    meta = json.loads((args.dump / "meta.json").read_text())
    converted, _ = convert_state_dict(state, NEMO_RULES)
    vocab = int(meta["vocab_size"])

    encoder = FastConformerEncoder(att_context=(56, args.chunk_ms // 80 - 1))
    lid = PromptConditioner(enc_hidden=1024, num_prompts=128)
    predictor = Predictor(
        vocab_size=vocab, pred_hidden=640, pred_rnn_layers=2
    )
    joint = Joint(
        enc_hidden=1024, pred_hidden=640, joint_hidden=640, vocab_size=vocab
    )
    for module, prefix in (
        (encoder, "encoder."),
        (lid, "lid."),
        (predictor, "predictor."),
        (joint, "joint."),
    ):
        sub = {
            k[len(prefix) :]: v
            for k, v in converted.items()
            if k.startswith(prefix)
        }
        module.load_state_dict(sub, strict=False)
        module.to(device).eval()

    # Steady-state window: 56 left-context frames + one chunk, as mel.
    window_mel = (56 + frames) * 8 + 9

    def batched_chunk_step(batch: int) -> None:
        mel = torch.randn(batch, 128, window_mel, device=device)
        lengths = torch.full((batch,), window_mel, device=device)
        with torch.inference_mode():
            out, _ = encoder(mel, lengths)
            new = out[:, -frames:]
            cond = lid(new, prompt_index=0)
            # Batched decode: frames x (predictor + joint) steps with ~2
            # extra emission steps — the per-chunk decode critical path
            # batched across sessions (label content is irrelevant to
            # cost; greedy loop length is frames + emissions).
            h = torch.zeros(2, batch, 640, device=device)
            c = torch.zeros(2, batch, 640, device=device)
            labels = torch.full(
                (batch,), predictor.blank_id, dtype=torch.long, device=device
            )
            pred_out, (h, c) = predictor.step(labels, (h, c))
            for t in range(frames + 2):
                logits = joint.logits(cond[:, min(t, frames - 1)], pred_out)
                labels = logits.argmax(dim=-1)
                pred_out, (h, c) = predictor.step(labels, (h, c))
        torch.cuda.synchronize()

    chunk_s = args.chunk_ms / 1000.0
    results = []
    single_p95 = None
    for batch in (1, 2, 4, 8, 16, 32, 64, 128, 256):
        batched_chunk_step(batch)  # warmup
        times = []
        for _ in range(12):
            t0 = time.perf_counter()
            batched_chunk_step(batch)
            times.append((time.perf_counter() - t0) * 1000.0)
        times.sort()
        p50, p95 = times[6], times[11]
        mem = torch.cuda.max_memory_allocated() / (1 << 30)
        rtf_capacity = batch * chunk_s * 1000.0 / p50
        row = {
            "streams": batch,
            "p50_ms": round(p50, 1),
            "p95_ms": round(p95, 1),
            "gpu_gib": round(mem, 2),
            "realtime_capacity_x": round(rtf_capacity / 1000.0 * 1000, 0),
        }
        results.append(row)
        if batch == 1:
            single_p95 = p95
        knee = single_p95 is not None and p95 > 2.0 * single_p95
        print(
            f"streams={batch:4d} p50={p50:7.1f}ms p95={p95:7.1f}ms "
            f"mem={mem:5.2f}GiB deadline={'OK' if p95 < args.chunk_ms else 'MISS'}"
            f"{'  <-- KNEE (p95 > 2x single)' if knee else ''}"
        )
        if p95 > args.chunk_ms:  # past real-time deadline; stop sweep
            break
    print(json.dumps({"chunk_ms": args.chunk_ms, "sweep": results}))


if __name__ == "__main__":
    main()
