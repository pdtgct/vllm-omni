# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""P2 full-context parity probe (venv-port, dev pod).

Assembles the port pipeline — featurizer -> subsampling/encoder (full
context [56,13]) -> LID -> greedy label-looping decode — from the
dumped checkpoint weights and compares against the pinned golden
matrix's sixth cell (PORT-REGIME-002/003): final transcript identity
(blocking) and named-checkpoint tensor diffs (advisory off the golden's
exact fingerprint, EVAL-PAR-003 — torch builds differ between
venv-oracle and venv-port).

Usage:
    /opt/venv-port/bin/python p2_parity_probe.py \
        --dump /workspace/weights/nemo-dump \
        --clip /workspace/datasets/clips/en-US_sample.wav \
        --golden-set /workspace/goldens/<matrix>/en-US_sample/full-context \
        --target-lang en-US [--device cuda]
"""

import argparse
import json
import wave
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open
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


def _read_wav(path: Path) -> tuple[torch.Tensor, int]:
    with wave.open(str(path), "rb") as handle:
        assert handle.getsampwidth() == 2 and handle.getnchannels() == 1
        rate = handle.getframerate()
        pcm = np.frombuffer(
            handle.readframes(handle.getnframes()), dtype=np.int16
        )
    return torch.from_numpy(pcm.astype(np.float32) / 32768.0), rate


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dump", type=Path, required=True)
    parser.add_argument("--clip", type=Path, required=True)
    parser.add_argument("--golden-set", type=Path, required=True)
    parser.add_argument("--target-lang", default="en-US")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    device = torch.device(args.device)

    state = load_file(str(args.dump / "nemo_state.safetensors"))
    meta = json.loads((args.dump / "meta.json").read_text())
    converted, report = convert_state_dict(state, NEMO_RULES)
    print(f"converted {len(report.consumed)} tensors")

    vocab = int(meta["vocab_size"])  # excludes blank; blank = vocab
    featurizer = MelFeaturizer(
        filterbank=converted["featurizer.fb"][0],
        window=converted["featurizer.window"],
    )
    encoder = FastConformerEncoder(att_context=(56, 13))
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
        real_missing = [
            m for m in missing if not m.endswith(("fb", "window", "pe"))
        ]
        if real_missing or unexpected:
            raise SystemExit(
                f"{prefix}: missing={real_missing} unexpected={unexpected}"
            )

    load_into(encoder, "encoder.")
    load_into(lid, "lid.")
    load_into(predictor, "predictor.")
    load_into(joint, "joint.")
    for module in (featurizer, encoder, lid, predictor, joint):
        module.to(device).eval()

    waveform, rate = _read_wav(args.clip)
    assert rate == 16000
    waveform = waveform.unsqueeze(0).to(device)
    prompt_index = resolve_prompt_index(
        meta["prompt_dictionary"], args.target_lang
    )

    with torch.inference_mode():
        mel, mel_len = featurizer(
            waveform, torch.tensor([waveform.shape[1]], device=device)
        )
        enc_raw, enc_len = encoder(mel, mel_len.to(device))
        n = int(enc_len[0])
        enc_raw = enc_raw[:, :n]
        enc_cond = lid(enc_raw, prompt_index=prompt_index)
        state0 = DecodeState(
            h=torch.zeros(2, 1, 640, device=device),
            c=torch.zeros(2, 1, 640, device=device),
            last_label=torch.tensor([predictor.blank_id], device=device),
        )
        labels, _ = greedy_decode_chunk(
            enc_cond[0], predictor, joint, state0
        )

    import sentencepiece as spm

    sp = spm.SentencePieceProcessor(str(args.dump / "tokenizer.model"))
    text = sp.decode(labels)
    print(f"PORT transcript: {text!r}")

    manifest = json.loads((args.golden_set / "manifest.json").read_text())
    golden_final = manifest["final_transcript"]
    print(f"GOLDEN final  : {golden_final!r}")

    with safe_open(
        str(args.golden_set / "tensors.safetensors"), framework="pt"
    ) as handle:
        golden_raw = handle.get_tensor("encoder_raw/00000")
        golden_cond = handle.get_tensor("encoder_conditioned/00000")

    # Golden tensors are (D, T) from NeMo's (B, D, T); ours are (T, D).
    for name, ours, golden in (
        ("encoder_raw", enc_raw[0].float().cpu(), golden_raw.T),
        ("encoder_conditioned", enc_cond[0].float().cpu(), golden_cond.T),
    ):
        t = min(ours.shape[0], golden.shape[0])
        diff = (ours[:t] - golden[:t]).abs()
        print(
            f"{name}: shapes ours={tuple(ours.shape)} "
            f"golden={tuple(golden.shape)} max_abs={diff.max():.3e} "
            f"mean_abs={diff.mean():.3e}"
        )

    verdict = "MATCH" if text == golden_final else "MISMATCH"
    print(f"P2 transcript parity: {verdict}")
    raise SystemExit(0 if verdict == "MATCH" else 1)


if __name__ == "__main__":
    main()
