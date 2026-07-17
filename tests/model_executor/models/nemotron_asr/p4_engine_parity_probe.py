# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Engine parity probe: one real chunk through the booted engine (BU-c2 m3).

The first end-to-end check that the *engine* (not the core directly)
transcribes: feed chunk 1's audio as a realtime TokensPrompt through
``LLM.generate`` and compare the emitted, SentencePiece-decoded labels
against the golden per-chunk partial. Chunk 1 needs no cross-chunk state
(fresh session), so a single offline ``generate`` is a valid single-pool
forward; rigorous multi-chunk streaming parity is the streaming probe.

Run (on the pod):
    /opt/venv-port/bin/python -m \
      vllm_omni.model_executor.models.nemotron_asr... (this is a tests/ probe;
      run by path)
    /opt/venv-port/bin/python tests/model_executor/models/nemotron_asr/\
      p4_engine_parity_probe.py \
        --served-dir /workspace/weights/served-nemotron-asr \
        --clip /workspace/datasets/clips/en-US_sample.wav \
        --golden-set /workspace/goldens/<matrix>/en-US_sample/560ms
"""

from __future__ import annotations

import argparse
import json
import wave
from pathlib import Path

import numpy as np

#: encoder frame = 8 mel frames * 160-sample hop.
_SAMPLES_PER_ENC_FRAME = 8 * 160


def _read_wav_16k_mono(path: Path) -> np.ndarray:
    with wave.open(str(path)) as handle:
        assert handle.getframerate() == 16000, "clip must be 16 kHz"
        assert handle.getnchannels() == 1, "clip must be mono"
        raw = handle.readframes(handle.getnframes())
    return np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--served-dir", type=Path, required=True)
    ap.add_argument("--clip", type=Path, required=True)
    ap.add_argument("--golden-set", type=Path, required=True)
    ap.add_argument("--chunk-index", type=int, default=0)
    ap.add_argument("--gpu-mem", type=float, default=0.5)
    args = ap.parse_args()

    manifest = json.loads((args.golden_set / "manifest.json").read_text())
    # Five-cadence golden schema: cell workload identity is nested
    # (harness EVAL-GOLD-004); pre-P5-0 six-cell matrices are not
    # valid evidence and are deliberately unreadable here.
    att_context = tuple(manifest["workload_fingerprint"]["att_context_size"])
    lookahead = int(att_context[1])
    frames_per_chunk = lookahead + 1
    chunk_samples = frames_per_chunk * _SAMPLES_PER_ENC_FRAME
    golden_partials = manifest["partial_transcripts"]

    wav = _read_wav_16k_mono(args.clip)
    lo = args.chunk_index * chunk_samples
    hi = lo + chunk_samples
    chunk = wav[lo:hi]
    print(
        f"chunk {args.chunk_index}: samples[{lo}:{hi}] "
        f"= {chunk.shape[0]} ({frames_per_chunk} enc frames), "
        f"att_context={att_context}"
    )

    # Served config: minted specials to strip from the transcript.
    cfg = json.loads((args.served_dir / "config.json").read_text())
    vocab_size = int(cfg["vocab_size"])
    blank_id = int(cfg["num_asr_labels"])  # = V
    park_id = int(cfg["eos_token_id"])
    placeholder_id = int(cfg["audio_chunk_token_id"])
    specials = {blank_id, park_id, placeholder_id}

    import sentencepiece as spm

    sp = spm.SentencePieceProcessor()
    sp.load(str(args.served_dir / "tokenizer.model"))

    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt

    llm = LLM(
        model=str(args.served_dir),
        trust_remote_code=False,
        enforce_eager=True,
        # fp32: the FP32_BRINGUP policy the goldens were made under, and
        # the decision carrier rides the hidden dtype (bf16 cannot hold
        # ids > 256 integer-exact). Do not let vLLM downcast.
        dtype="float32",
        gpu_memory_utilization=args.gpu_mem,
        max_model_len=512,
    )
    prompt = TokensPrompt(
        prompt_token_ids=[placeholder_id],
        multi_modal_data={"audio": chunk},
    )
    out = llm.generate(
        prompt,
        SamplingParams(temperature=0.0, max_tokens=141, detokenize=False),
    )
    emitted = list(out[0].outputs[0].token_ids)
    labels = [t for t in emitted if t not in specials and t < vocab_size]
    ours = sp.decode(labels)
    golden = golden_partials[args.chunk_index] if golden_partials else ""

    print(f"emitted token ids ({len(emitted)}): {emitted}")
    print(f"label ids (specials stripped): {labels}")
    print(f"ours  : {ours!r}")
    print(f"golden: {golden!r}")
    print("MATCH" if ours == golden else "DIFFERS (inspect prefix)")


if __name__ == "__main__":
    main()
