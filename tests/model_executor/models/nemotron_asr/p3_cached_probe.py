# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Cached-regime streaming probe: stream_step vs golden cells.

Validates ``encoder.stream_step`` (the cached streaming regime — NeMo
``update_cache`` semantics) against golden chunk cells: per chunk, feed
``[9-mel pre-encode context | chunk mel]`` with ``drop_extra=2`` (first
chunk: no context, drop 0), thread StreamingCaches + DecodeState, and
assert the per-chunk cumulative partials and final against the golden.
Success means the cached path == the prefix path == the oracle.

Usage:
    /opt/venv-port/bin/python p3_cached_probe.py \
        --dump /workspace/weights/nemo-dump \
        --clip <wav> --golden-set <matrix>/<clip>/<cell>
"""

import argparse
import json
import wave
from pathlib import Path

import numpy as np
import torch

from vllm_omni.model_executor.models.nemotron_asr.encoder import (
    StreamingCaches,
    stream_step,
)
from vllm_omni.model_executor.models.nemotron_asr.nemotron_asr import (
    load_core_from_dump,
    resolve_prompt_index,
)

PRE_ENCODE_CACHE_MEL = 9
DROP_EXTRA = 2


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
    att_context = tuple(manifest["att_context_size"])
    lookahead = att_context[1]
    golden_partials = manifest["partial_transcripts"]
    n_chunks = len(golden_partials)

    core, meta = load_core_from_dump(
        args.dump, device=device, att_context=att_context
    )
    import sentencepiece as spm

    sp = spm.SentencePieceProcessor(str(args.dump / "tokenizer.model"))
    prompt_index = resolve_prompt_index(
        meta["prompt_dictionary"], args.target_lang
    )

    with wave.open(str(args.clip)) as handle:
        pcm = np.frombuffer(
            handle.readframes(handle.getnframes()), dtype=np.int16
        )
    waveform = (
        torch.from_numpy(pcm.astype(np.float32) / 32768.0)
        .unsqueeze(0)
        .to(device)
    )
    with torch.inference_mode():
        mel, mel_len = core.featurizer(
            waveform, torch.tensor([waveform.shape[1]], device=device)
        )
    total_mel = int(mel_len[0])

    # NeMo boundaries: first chunk 8L+1 mel, shifts 8(L+1); drop sub-8 tail.
    boundaries = [8 * lookahead + 1]
    while boundaries[-1] < total_mel:
        boundaries.append(boundaries[-1] + 8 * (lookahead + 1))
    boundaries = [min(b, total_mel) for b in boundaries]
    if len(boundaries) >= 2 and total_mel - boundaries[-2] < 8:
        boundaries.pop()

    caches = StreamingCaches(
        n_layers=len(core.encoder.layers),
        batch=1,
        d_model=1024,
        left_context=att_context[0],
        conv_kernel=9,
        device=device,
    )
    state = core.fresh_decode_state(device)
    labels: list[int] = []
    partials: list[str] = []
    prev_end = 0
    with torch.inference_mode():
        for chunk in range(n_chunks):
            mel_end = boundaries[min(chunk, len(boundaries) - 1)]
            if chunk == 0:
                chunk_mel = mel[:, :, :mel_end]
                drop = 0
            else:
                start = prev_end - PRE_ENCODE_CACHE_MEL
                if start < 0:
                    # NeMo zero-pads a short pre-encode history
                    # (streaming_utils.py:1640 @ de242add).
                    pad = torch.zeros(
                        1, mel.shape[1], -start, device=device
                    )
                    chunk_mel = torch.cat(
                        [pad, mel[:, :, :mel_end]], dim=2
                    )
                else:
                    chunk_mel = mel[:, :, start:mel_end]
                drop = DROP_EXTRA
            frames = stream_step(
                core.encoder, chunk_mel, caches, drop_extra=drop
            )
            if frames.shape[1] > 0:
                cond = core.lid(frames, prompt_index=prompt_index)
                emitted, state = core.transcribe_chunk(cond[0], state)
                labels.extend(emitted)
            partials.append(sp.decode(labels))
            prev_end = mel_end

    mismatches = sum(
        ours != golden for ours, golden in zip(partials, golden_partials)
    )
    for idx, (ours, golden) in enumerate(zip(partials, golden_partials)):
        if ours != golden:
            print(f"  [{idx}] != ours={ours!r} golden={golden!r}")
    final_match = partials[-1] == manifest["final_transcript"]
    print(
        f"cached-path: final {'MATCH' if final_match else 'MISMATCH'}; "
        f"cadence {n_chunks - mismatches}/{n_chunks}"
    )
    raise SystemExit(0 if final_match and mismatches == 0 else 1)


if __name__ == "__main__":
    main()
