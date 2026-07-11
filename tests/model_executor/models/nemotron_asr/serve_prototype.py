# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Served-tier validation prototype (venv-port, dev pod).

Option-B-shaped serving used exactly as the design sanctions it — the
validation vehicle (port-design §Decisions: 'fallback/validation
only'). Wraps ``NemotronASRCore`` behind a WebSocket endpoint with the
NeMo-exact streaming semantics proven by the P3 probes (full-prefix
recompute ≡ cache-carried computation, by strict block-causality) and
a micro-batcher standing in for the engine's continuous batching:
pending chunk-steps are collected for a few milliseconds, pad-batched
through one encoder forward, and decoded per-session.

Protocol (one WS connection per session):
  client -> {"target_lang": "en-US", "chunk_ms": 560}     (JSON, once)
  client -> binary PCM16 frames (any size; server rechunks)
  client -> {"commit": true}                              (JSON, end)
  server -> {"delta": str, "cumulative": str, "chunk": int} per chunk
  server -> {"final": str}

This measures REAL served chunk-to-partial latency (transport + queue
+ batch + compute) for requirement (b)/(c) at the served tier while
Option-A engine integration (the contribution) follows the RFC.
"""

import argparse
import asyncio
import json
import time
from pathlib import Path

import numpy as np
import torch

from vllm_omni.model_executor.models.nemotron_asr.nemotron_asr import (
    load_core_from_dump,
    resolve_prompt_index,
)

MEL_HOP = 160
SUBSAMPLE = 8


class Session:
    """One streaming session: raw-audio accumulation + decode state."""

    def __init__(self, core, meta, target_lang: str, chunk_ms: int) -> None:
        self.core = core
        self.lookahead = chunk_ms // 80 - 1
        self.frames_per_chunk = self.lookahead + 1
        self.prompt_index = resolve_prompt_index(
            meta["prompt_dictionary"], target_lang
        )
        self.pcm = np.zeros(0, dtype=np.float32)
        self.emitted_mel = 0  # processed-mel boundary (NeMo-exact)
        self.emitted_frames = 0
        self.state = core.fresh_decode_state(
            next(core.parameters()).device
        )
        self.labels: list[int] = []
        self.chunk_index = 0

    def next_boundary(self) -> int:
        """NeMo boundary: first chunk 8L+1 mel, then +8(L+1) each."""
        if self.emitted_mel == 0:
            return 8 * self.lookahead + 1
        return self.emitted_mel + 8 * self.frames_per_chunk

    def available_mel(self) -> int:
        # center=True STFT: frames = samples // hop (NeMo get_seq_len).
        return len(self.pcm) // MEL_HOP


class Batcher:
    """Micro-batcher: pad-batch pending chunk-steps every few ms."""

    def __init__(self, core, window_ms: float = 5.0) -> None:
        self.core = core
        self.window_s = window_ms / 1000.0
        self.queue: asyncio.Queue = asyncio.Queue()
        self.device = next(core.parameters()).device

    async def submit(self, session: Session, mel_end: int) -> None:
        future: asyncio.Future = asyncio.get_event_loop().create_future()
        await self.queue.put((session, mel_end, future))
        await future

    async def run(self) -> None:
        while True:
            batch = [await self.queue.get()]
            deadline = time.monotonic() + self.window_s
            while True:
                timeout = deadline - time.monotonic()
                if timeout <= 0:
                    break
                try:
                    batch.append(
                        await asyncio.wait_for(self.queue.get(), timeout)
                    )
                except asyncio.TimeoutError:
                    break
            await asyncio.get_event_loop().run_in_executor(
                None, self._process, batch
            )
            for _, _, future in batch:
                if not future.done():
                    future.set_result(None)

    def _process(self, batch) -> None:
        core = self.core
        with torch.inference_mode():
            mels = []
            for session, mel_end, _ in batch:
                samples = torch.from_numpy(
                    session.pcm[: mel_end * MEL_HOP + 240]
                ).unsqueeze(0).to(self.device)
                mel, _ = core.featurizer(
                    samples,
                    torch.tensor([samples.shape[1]], device=self.device),
                )
                mels.append(mel[0, :, :mel_end])
            max_len = max(m.shape[1] for m in mels)
            padded = torch.zeros(
                len(mels), 128, max_len, device=self.device
            )
            lengths = torch.tensor(
                [m.shape[1] for m in mels], device=self.device
            )
            for i, m in enumerate(mels):
                padded[i, :, : m.shape[1]] = m
            enc, enc_len = core.encoder(padded, lengths)
            for i, (session, mel_end, _) in enumerate(batch):
                valid = int(enc_len[i])
                new = enc[i : i + 1, session.emitted_frames : valid]
                if new.shape[1] > 0:
                    cond = core.lid(
                        new, prompt_index=session.prompt_index
                    )
                    labels, session.state = core.transcribe_chunk(
                        cond[0], session.state
                    )
                    session.labels.extend(labels)
                session.emitted_frames = valid
                session.emitted_mel = mel_end
                session.chunk_index += 1


async def handle(ws, core, meta, batcher, sp) -> None:
    setup = json.loads(await ws.recv())
    session = Session(
        core, meta, setup.get("target_lang", "en-US"),
        int(setup.get("chunk_ms", 560)),
    )
    committed = False
    while True:
        if not committed:
            message = await ws.recv()
            if isinstance(message, bytes):
                pcm = np.frombuffer(message, dtype=np.int16)
                session.pcm = np.concatenate(
                    [session.pcm, pcm.astype(np.float32) / 32768.0]
                )
            else:
                committed = json.loads(message).get("commit", False)
        boundary = session.next_boundary()
        available = session.available_mel()
        ready = available >= boundary or (
            committed and available - session.emitted_mel >= SUBSAMPLE
        )
        if ready:
            mel_end = min(boundary, available)
            before = len(session.labels)
            await batcher.submit(session, mel_end)
            if len(session.labels) > before:
                delta_ids = session.labels[before:]
                await ws.send(json.dumps({
                    "delta": sp.decode(delta_ids),
                    "cumulative": sp.decode(session.labels),
                    "chunk": session.chunk_index - 1,
                }))
            continue
        if committed:
            await ws.send(json.dumps({"final": sp.decode(session.labels)}))
            return


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dump", type=Path, required=True)
    parser.add_argument("--port", type=int, default=8790)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    core, meta = load_core_from_dump(
        args.dump, device=torch.device(args.device), att_context=(56, 6)
    )
    import sentencepiece as spm
    import websockets

    sp = spm.SentencePieceProcessor(str(args.dump / "tokenizer.model"))
    batcher = Batcher(core)
    asyncio.get_event_loop().create_task(batcher.run())

    async def _handler(ws):
        try:
            await handle(ws, core, meta, batcher, sp)
        except websockets.ConnectionClosed:
            pass

    async with websockets.serve(_handler, "0.0.0.0", args.port):
        print(f"serving on :{args.port}")
        await asyncio.Future()


if __name__ == "__main__":
    asyncio.run(main())
