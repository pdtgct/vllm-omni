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
from vllm_omni.model_executor.models.nemotron_asr.manifests import (
    ENVELOPE_HEADER_FIELDS,
    RAW_SAMPLES_PER_CHUNK,
    author_checkpoint_profile,
    author_emission_manifest,
    author_geometry_manifest,
    author_state_manifest,
    author_transition_manifest,
    canonical_json,
    manifest_hash,
)
from vllm_omni.model_executor.models.nemotron_asr.rules import NEMO_RULES
from vllm_omni.model_executor.models.nemotron_asr.staging import (
    atomic_publish_dir,
)

#: (attribute name, output filename) for the four checkpoint-derived
#: manifests (PORT-WGT-004) written alongside config.json.
_MANIFEST_FILES = (
    ("state", "state-manifest.json"),
    ("geometry", "geometry-manifest.json"),
    ("transition", "transition-manifest.json"),
    ("emission", "emission-manifest.json"),
)

#: Minted engine specials, strictly past blank (= V): park = V+1,
#: placeholder = V+2 (author_config enforces > V and distinctness).
_PARK_OFFSET = 1
_PLACEHOLDER_OFFSET = 2
#: The chunk-envelope carrier width: the header slots + the largest
#: admitted raw cadence (1120 ms at 16 kHz = 17,920 samples). DERIVED
#: from the manifests, never a copied constant — author_geometry_
#: manifest re-validates it (supersedes the BU-c1 mel-carrier width
#: 15,617; PORT-REGIME-001 raw envelope).
_HIDDEN_SIZE = len(ENVELOPE_HEADER_FIELDS) + max(RAW_SAMPLES_PER_CHUNK.values())
#: Tokenizer artifacts to copy into the served dir. A strict whitelist,
#: NOT "every file in tokenizer_dir": the .nemo dump mixes the tokenizer
#: with the raw-name ``nemo_state.safetensors`` + ``meta.json``, and
#: copying those would leave a second *.safetensors that the engine's
#: HF loader globs and feeds to ``load_weights`` under raw NeMo names.
_TOKENIZER_FILES = frozenset({
    "tokenizer.model",
    "spiece.model",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "vocab.json",
    "merges.txt",
    "added_tokens.json",
})


def publish(
    nemo_state: Path,
    out_dir: Path,
    *,
    tokenizer_dir: Path | None,
    reference_vocab_size: int | None,
    decode_dispatch_arm: str,
) -> dict:
    """Convert + author + write the served checkpoint. Returns the config."""
    if decode_dispatch_arm not in {"dense-eager", "compact-eager"}:
        raise ValueError(
            "decode_dispatch_arm must name a currently executable eager "
            "arm; dense-graphed is not a graph binding"
        )
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
        n_layers=cfg_dict["n_layers"],
        decode_dispatch_arm=decode_dispatch_arm,
        performance_gated=False,
    )

    # Author and validate EVERYTHING before touching the filesystem
    # (PORT-WGT-004 / PORT-INT-005): a failing author must not damage
    # an existing artifact or leave a plausible-looking partial one.
    manifest_authors = {
        "state": author_state_manifest,
        "geometry": author_geometry_manifest,
        "transition": author_transition_manifest,
        "emission": author_emission_manifest,
    }
    authored_manifests: dict[str, dict] = {}
    manifest_hashes: dict[str, str] = {}
    for key, _filename in _MANIFEST_FILES:
        authored = manifest_authors[key](config)
        authored_manifests[key] = authored
        manifest_hashes[key] = manifest_hash(authored)
    profile = author_checkpoint_profile(manifest_hashes)

    def _build(staging_dir: Path) -> None:
        save_file(
            {k: t.contiguous() for k, t in converted.items()},
            str(staging_dir / "model.safetensors"),
        )
        config.save_pretrained(str(staging_dir))
        if tokenizer_dir is not None and tokenizer_dir.exists():
            for f in tokenizer_dir.iterdir():
                if f.is_file() and f.name in _TOKENIZER_FILES:
                    shutil.copy2(f, staging_dir / f.name)
        for key, filename in _MANIFEST_FILES:
            (staging_dir / filename).write_text(
                canonical_json(authored_manifests[key]) + "\n"
            )
        (staging_dir / "checkpoint-profile.json").write_text(
            canonical_json(profile) + "\n"
        )

    summary = {
        "derived_V": v,
        "vocab_size": cfg_dict["vocab_size"],
        "eos_token_id": cfg_dict["eos_token_id"],
        "audio_chunk_token_id": cfg_dict["audio_chunk_token_id"],
        "hidden_size": cfg_dict["hidden_size"],
        "n_layers": cfg_dict["n_layers"],
        "architectures": cfg_dict["architectures"],
        "decode_dispatch_arm": decode_dispatch_arm,
        "tensors_consumed": len(report.consumed),
        "out_dir": str(out_dir),
        "checkpoint_profile_id": profile["id"],
        "state_manifest_hash": manifest_hashes["state"],
        "geometry_manifest_hash": manifest_hashes["geometry"],
        "transition_manifest_hash": manifest_hashes["transition"],
        "emission_manifest_hash": manifest_hashes["emission"],
        "checkpoint_profile_content_hash": profile["content_hash"],
    }

    def _build_with_summary(staging_dir: Path) -> None:
        _build(staging_dir)
        (staging_dir / "publish_summary.json").write_text(
            json.dumps(summary, indent=2)
        )

    atomic_publish_dir(out_dir, _build_with_summary)
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
    ap.add_argument(
        "--decode-dispatch-arm",
        choices=("dense-eager", "compact-eager"),
        required=True,
        help="explicit executable bring-up arm authored into config.json",
    )
    args = ap.parse_args()
    summary = publish(
        args.nemo_state,
        args.out,
        tokenizer_dir=args.tokenizer_dir,
        reference_vocab_size=args.reference_vocab_size,
        decode_dispatch_arm=args.decode_dispatch_arm,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
