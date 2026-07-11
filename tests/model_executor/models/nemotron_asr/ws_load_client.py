# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Real-time-paced WS load client for the served prototype.

Drives N concurrent sessions of the sample clip against
``serve_prototype.py``, pacing PCM at real time, and reports served
chunk-to-partial latency percentiles (arrival minus the wall time the
chunk's last sample was sent) plus final-transcript checks — the
served-tier counterpart of the NIM gRPC sweep.

Usage:
    /opt/venv-port/bin/python ws_load_client.py \
        --server ws://127.0.0.1:8790 --clip <wav> --golden-final "<text>"
"""

import argparse
import asyncio
import json
import time
import wave
from pathlib import Path

import numpy as np
import websockets


async def one_session(server, pcm, chunk_ms, results) -> str:
    latencies: list[float] = []
    final = ""
    samples_per_frame = int(16000 * chunk_ms / 1000.0)
    async with websockets.connect(server, max_size=None) as ws:
        await ws.send(json.dumps(
            {"target_lang": "en-US", "chunk_ms": chunk_ms}
        ))
        last_send = time.monotonic()

        async def receiver():
            nonlocal final
            async for message in ws:
                arrival = time.monotonic()
                payload = json.loads(message)
                if "final" in payload:
                    final = payload["final"]
                    return
                latencies.append((arrival - last_send) * 1000.0)

        recv_task = asyncio.create_task(receiver())
        for start in range(0, len(pcm), samples_per_frame):
            frame = pcm[start : start + samples_per_frame]
            await ws.send(frame.tobytes())
            last_send = time.monotonic()
            await asyncio.sleep(len(frame) / 16000.0)
        await ws.send(json.dumps({"commit": True}))
        await asyncio.wait_for(recv_task, timeout=30)
    results.extend(latencies)
    return final


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server", required=True)
    parser.add_argument("--clip", type=Path, required=True)
    parser.add_argument("--chunk-ms", type=int, default=560)
    parser.add_argument("--golden-final", default=None)
    args = parser.parse_args()

    with wave.open(str(args.clip)) as handle:
        pcm = np.frombuffer(
            handle.readframes(handle.getnframes()), dtype=np.int16
        )

    for n in (1, 2, 4, 8, 16, 32):
        latencies: list[float] = []
        finals = await asyncio.gather(*[
            one_session(args.server, pcm, args.chunk_ms, latencies)
            for _ in range(n)
        ])
        latencies.sort()
        p50 = latencies[len(latencies) // 2]
        p95 = latencies[int(len(latencies) * 0.95)]
        match = (
            "n/a" if args.golden_final is None
            else f"{sum(f == args.golden_final for f in finals)}/{n}"
        )
        deadline = "OK" if p95 < args.chunk_ms else "MISS"
        print(
            f"streams={n:3d} p50={p50:7.1f}ms p95={p95:7.1f}ms "
            f"finals_match={match} deadline={deadline}"
        )


if __name__ == "__main__":
    asyncio.run(main())
