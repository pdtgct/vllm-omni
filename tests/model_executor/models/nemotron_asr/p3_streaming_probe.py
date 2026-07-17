# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""P3 streaming-semantics probe (venv-port, dev pod).

Validates the streaming regime against a golden chunk cell BEFORE the
paged-cache engineering: because the encoder is strictly block-causal
(proven exactly 0.0 in test_encoder), running the prefix through the
encoder under the streaming mask and slicing each chunk's new frames is
mathematically identical to cache-carried computation. Decode state
threads across chunks exactly as in NeMo (PORT-DEC-001); partials are
the per-chunk cumulative hypotheses (EVAL-PAR-001/002 cadence).

Usage:
    /opt/venv-port/bin/python p3_streaming_probe.py \
        --dump /workspace/weights/nemo-dump \
        --clip /workspace/datasets/clips/en-US_sample.wav \
        --golden-set /workspace/goldens/<matrix>/en-US_sample/560ms \
        --target-lang en-US [--device cuda]
"""

import argparse
import json
import wave
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file

from vllm_omni.model_executor.models.nemotron_asr.convert import (
    convert_state_dict,
)
from vllm_omni.model_executor.models.nemotron_asr.encoder import (
    FastConformerEncoder,
)
from vllm_omni.model_executor.models.nemotron_asr.featurizer import (
    MelFeaturizer,
)
from vllm_omni.model_executor.models.nemotron_asr.lid import (
    PromptConditioner,
    resolve_prompt_index,
)
from vllm_omni.model_executor.models.nemotron_asr.rnnt import (
    DecodeState,
    Joint,
    Predictor,
    greedy_decode_chunk,
)
from vllm_omni.model_executor.models.nemotron_asr.rules import NEMO_RULES


def _read_wav(path: Path) -> torch.Tensor:
    with wave.open(str(path), "rb") as handle:
        assert handle.getframerate() == 16000
        pcm = np.frombuffer(
            handle.readframes(handle.getnframes()), dtype=np.int16
        )
    return torch.from_numpy(pcm.astype(np.float32) / 32768.0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dump", type=Path, required=True)
    parser.add_argument("--clip", type=Path, required=True)
    parser.add_argument("--golden-set", type=Path, required=True)
    parser.add_argument("--target-lang", default="en-US")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    device = torch.device(args.device)

    manifest = json.loads((args.golden_set / "manifest.json").read_text())
    att_context = tuple(manifest["workload_fingerprint"]["att_context_size"])
    lookahead = att_context[1]
    frames_per_chunk = lookahead + 1
    # New-schema goldens append one synthetic zero-work final-tail entry
    # (repeats the last hypothesis); zip below ignores it until the
    # Phase-5 final-tail transition produces the matching port entry.
    golden_partials = manifest["partial_transcripts"]
    n_chunks = len(golden_partials)

    state = load_file(str(args.dump / "nemo_state.safetensors"))
    meta = json.loads((args.dump / "meta.json").read_text())
    converted, _ = convert_state_dict(state, NEMO_RULES)
    vocab = int(meta["vocab_size"])

    featurizer = MelFeaturizer(
        filterbank=converted["featurizer.fb"][0],
        window=converted["featurizer.window"],
    )
    encoder = FastConformerEncoder(att_context=att_context)
    lid = PromptConditioner(enc_hidden=1024, num_prompts=128)
    predictor = Predictor(
        vocab_size=vocab, pred_hidden=640, pred_rnn_layers=2
    )
    joint = Joint(
        enc_hidden=1024, pred_hidden=640, joint_hidden=640, vocab_size=vocab
    )

    def load_into(module: torch.nn.Module, prefix: str) -> None:
        sub = {
            k[len(prefix) :]: v
            for k, v in converted.items()
            if k.startswith(prefix)
        }
        missing, unexpected = module.load_state_dict(sub, strict=False)
        real = [m for m in missing if not m.endswith(("fb", "window", "pe"))]
        if real or unexpected:
            raise SystemExit(f"{prefix}: missing={real} bad={unexpected}")

    load_into(encoder, "encoder.")
    load_into(lid, "lid.")
    load_into(predictor, "predictor.")
    load_into(joint, "joint.")
    for module in (featurizer, encoder, lid, predictor, joint):
        module.to(device).eval()

    import sentencepiece as spm

    sp = spm.SentencePieceProcessor(str(args.dump / "tokenizer.model"))
    prompt_index = resolve_prompt_index(
        meta["prompt_dictionary"], args.target_lang
    )

    waveform = _read_wav(args.clip).unsqueeze(0).to(device)
    with torch.inference_mode():
        mel, mel_len = featurizer(
            waveform, torch.tensor([waveform.shape[1]], device=device)
        )
    total_mel = int(mel_len[0])
    mel_per_chunk = frames_per_chunk * 8
    # NeMo chunking (setup_streaming_params + buffer __iter__ tail rule
    # @ de242add): first chunk is 8L+1 mel, shifts are 8L+8, and a tail
    # shorter than one subsampled frame (8 mel) is DROPPED, never
    # processed. Cumulative processed-mel boundaries per chunk:
    boundaries = [8 * lookahead + 1]
    while boundaries[-1] < total_mel:
        boundaries.append(boundaries[-1] + mel_per_chunk)
    boundaries = [min(b, total_mel) for b in boundaries]
    if len(boundaries) >= 2 and total_mel - boundaries[-2] < 8:
        boundaries.pop()  # dropped tail (< 8 mel)

    decode_state = DecodeState(
        h=torch.zeros(2, 1, 640, device=device),
        c=torch.zeros(2, 1, 640, device=device),
        last_label=torch.tensor([predictor.blank_id], device=device),
    )
    all_labels: list[int] = []
    partials: list[str] = []
    emitted_frames = 0
    with torch.inference_mode():
        for chunk in range(n_chunks):
            mel_end = boundaries[min(chunk, len(boundaries) - 1)]
            prefix = mel[:, :, :mel_end]
            enc_out, enc_len = encoder(
                prefix, torch.tensor([mel_end], device=device)
            )
            # With NeMo-exact boundaries (first chunk 8L+1 mel), the
            # cumulative valid frame count is enc_len(boundary) directly
            # — the valid_out_len / drop_extra_pre_encoded interplay is
            # embedded in the boundary arithmetic.
            valid = int(enc_len[0])
            new = enc_out[:, emitted_frames:valid]
            if new.shape[1] == 0 and chunk < n_chunks - 1:
                partials.append(partials[-1] if partials else "")
                continue
            cond = lid(new, prompt_index=prompt_index)
            labels, decode_state = greedy_decode_chunk(
                cond[0], predictor, joint, decode_state
            )
            all_labels.extend(labels)
            partials.append(sp.decode(all_labels))
            emitted_frames = valid

    print(f"chunks: ours {len(partials)} vs golden {n_chunks}")
    mismatches = 0
    for idx, (ours, golden) in enumerate(zip(partials, golden_partials)):
        flag = "==" if ours == golden else "!="
        if ours != golden:
            mismatches += 1
        print(f"  [{idx}] {flag} ours={ours!r} golden={golden!r}")
    final_match = partials[-1] == manifest["final_transcript"]
    print(f"final: {'MATCH' if final_match else 'MISMATCH'}")
    print(
        f"partial cadence: {n_chunks - mismatches}/{n_chunks} identical"
    )
    raise SystemExit(0 if final_match and mismatches == 0 else 1)


if __name__ == "__main__":
    main()
