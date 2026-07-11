# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Dump the .nemo checkpoint to safetensors (venv-oracle ONLY).

Bridges the two venvs through files: the restored model's full fp32
state dict (including the featurizer's persisted ``fb``/``window``
buffers) goes to safetensors, the SentencePiece tokenizer model and the
prompt dictionary to plain files, so the port side (venv-port, no NeMo)
can load weights and detokenize. Offline conversion artifact — never
part of the runtime path (PORT-WGT-001).

Usage (dev pod):
    /opt/venv-oracle/bin/python dump_nemo_weights.py \
        --model /workspace/weights/nemotron-3.5-asr-streaming-0.6b.nemo \
        --out /workspace/weights/nemo-dump
"""

import argparse
import json
import shutil
from pathlib import Path

import torch
from nemo.collections.asr.models import ASRModel
from safetensors.torch import save_file


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    model = ASRModel.restore_from(
        restore_path=str(args.model), map_location=torch.device("cpu")
    )
    model.eval()

    state = {
        k: v.to(torch.float32).contiguous()
        for k, v in model.state_dict().items()
    }
    save_file(state, str(args.out / "nemo_state.safetensors"))

    tok = model.tokenizer
    spm_path = getattr(tok, "model_path", None) or getattr(
        tok.tokenizer, "model_path", None
    )
    if spm_path:
        shutil.copy(spm_path, args.out / "tokenizer.model")

    meta = {
        "prompt_dictionary": dict(
            model.cfg.model_defaults.get("prompt_dictionary", {})
        ),
        "vocab_size": int(model.tokenizer.vocab_size),
        "blank_id": int(model.tokenizer.vocab_size),
        "num_prompts": int(model.cfg.get("num_prompts", 128)),
    }
    (args.out / "meta.json").write_text(json.dumps(meta, indent=1))
    print(f"dumped {len(state)} tensors to {args.out}")


if __name__ == "__main__":
    main()
