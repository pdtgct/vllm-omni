# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Publish the served checkpoint from the .nemo dump (BU-c2, PORT-WGT-004).

The offline conversion: raw NeMo state dict → the port's tree
(``convert_state_dict`` under ``NEMO_RULES``) → derived V and authored
``config.json`` (``author_config``) → a served directory
(``config.json`` + ``model.safetensors`` + tokenizer) the engine boots
from the standard HF path (PORT-WGT-001). Run on the pod against the
real ``.nemo`` dump; the enforcement is the same consume-exactly-once
hard-fail as the conversion tests pin.

Usage (on the pod):
    python -m vllm_omni.model_executor.models.nemotron_asr.publish \
        --nemo-state /workspace/weights/nemo-dump/nemo_state.safetensors \
        --tokenizer-dir /workspace/weights/nemo-dump/tokenizer \
        --out /workspace/weights/served-nemotron-asr
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from safetensors.torch import load_file, save_file

from vllm_omni.model_executor.models.nemotron_asr.configuration_nemotron_asr import (  # noqa: E501
    NemotronASRConfig,
)
from vllm_omni.model_executor.models.nemotron_asr.convert import (
    author_config,
    convert_state_dict,
)
from vllm_omni.model_executor.models.nemotron_asr.rules import NEMO_RULES

#: Minted engine specials, strictly past blank (= V): park = V+1,
#: placeholder = V+2 (author_config enforces > V and distinctness).
_PARK_OFFSET = 1
_PLACEHOLDER_OFFSET = 2
#: The measured carrier width (BU-c1: 128 × (113 stft cols + 9 overlap)
#: + 1 frame-count slot).
_HIDDEN_SIZE = 15617


def publish(
    nemo_state: Path,
    out_dir: Path,
    *,
    tokenizer_dir: Path | None,
    reference_vocab_size: int | None,
) -> dict:
    """Convert + author + write the served checkpoint. Returns the config."""
    raw = load_file(str(nemo_state))
    converted, report = convert_state_dict(raw, NEMO_RULES)

    from vllm_omni.model_executor.models.nemotron_asr.convert import (
        derive_vocab_size,
    )

    v = derive_vocab_size(converted)
    cfg_dict = author_config(
        converted,
        eos_token_id=v + _PARK_OFFSET,
        audio_chunk_token_id=v + _PLACEHOLDER_OFFSET,
        hidden_size=_HIDDEN_SIZE,
        reference_vocab_size=reference_vocab_size,
    )

    config = NemotronASRConfig(
        vocab_size=cfg_dict["vocab_size"],
        num_asr_labels=cfg_dict["num_asr_labels"],
        hidden_size=cfg_dict["hidden_size"],
        eos_token_id=cfg_dict["eos_token_id"],
        audio_chunk_token_id=cfg_dict["audio_chunk_token_id"],
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    save_file(
        {k: t.contiguous() for k, t in converted.items()},
        str(out_dir / "model.safetensors"),
    )
    config.save_pretrained(str(out_dir))
    if tokenizer_dir is not None and tokenizer_dir.exists():
        for f in tokenizer_dir.iterdir():
            if f.is_file():
                shutil.copy2(f, out_dir / f.name)

    summary = {
        "derived_V": v,
        "vocab_size": cfg_dict["vocab_size"],
        "eos_token_id": cfg_dict["eos_token_id"],
        "audio_chunk_token_id": cfg_dict["audio_chunk_token_id"],
        "hidden_size": cfg_dict["hidden_size"],
        "architectures": cfg_dict["architectures"],
        "tensors_consumed": len(report.consumed),
        "out_dir": str(out_dir),
    }
    (out_dir / "publish_summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--nemo-state", type=Path, required=True)
    ap.add_argument("--tokenizer-dir", type=Path, default=None)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument(
        "--reference-vocab-size",
        type=int,
        default=None,
        help="cross-check against .nemo meta.json / model card; "
        "hard-fails on disagreement with the derived V.",
    )
    args = ap.parse_args()
    summary = publish(
        args.nemo_state,
        args.out,
        tokenizer_dir=args.tokenizer_dir,
        reference_vocab_size=args.reference_vocab_size,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
