# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""NIM sidecar concurrency sweep (Riva client venv ONLY).

Drives N parallel real-time-paced streaming sessions against the NIM
sidecar over gRPC (public nvidia-riva/python-clients API) and reports
per-stream chunk-to-partial latency percentiles per concurrency level —
the like-for-like counterpart of the port's batched chunk-step sweep
(same GPU class, same clip, same pacing).

Usage:
    /workspace/riva-venv/bin/python nim_concurrency.py \
        --server <ip:port> --clip /workspace/datasets/clips/en-US_sample.wav
"""

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import riva.client
from riva.client import (
    ASRService,
    AudioChunkFileIterator,
    AudioEncoding,
    Auth,
    RecognitionConfig,
    StreamingRecognitionConfig,
    get_wav_file_parameters,
    sleep_audio_length,
)


def stream_once(server: str, clip: Path, chunk_ms: int) -> list[float]:
    params = get_wav_file_parameters(clip)
    framerate = float(params["framerate"])
    config = RecognitionConfig(
        encoding=AudioEncoding.LINEAR_PCM,
        sample_rate_hertz=int(framerate),
        language_code="en-US",
        max_alternatives=1,
        enable_automatic_punctuation=True,
        audio_channel_count=int(params["nchannels"]),
    )
    streaming_config = StreamingRecognitionConfig(
        config=config, interim_results=True
    )
    frames = max(1, int(framerate * chunk_ms / 1000.0))
    chunks = AudioChunkFileIterator(
        clip, frames, delay_callback=sleep_audio_length
    )
    send_times: list[float] = []

    def instrumented():
        for chunk in chunks:
            send_times.append(time.monotonic())
            yield chunk

    service = ASRService(Auth(uri=server))
    latencies: list[float] = []
    start = time.monotonic()
    for response in service.streaming_response_generator(
        instrumented(), streaming_config
    ):
        arrival = time.monotonic()
        last = send_times[-1] if send_times else start
        for _result in response.results:
            latencies.append((arrival - last) * 1000.0)
    return latencies


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server", required=True)
    parser.add_argument("--clip", type=Path, required=True)
    parser.add_argument("--chunk-ms", type=int, default=560)
    args = parser.parse_args()

    results = []
    for n in (1, 2, 4, 8, 16, 32):
        with ThreadPoolExecutor(max_workers=n) as pool:
            futures = [
                pool.submit(stream_once, args.server, args.clip, args.chunk_ms)
                for _ in range(n)
            ]
            all_lat = sorted(
                lat for fut in futures for lat in fut.result()
            )
        if not all_lat:
            print(f"streams={n}: no responses")
            continue
        p50 = all_lat[len(all_lat) // 2]
        p95 = all_lat[int(len(all_lat) * 0.95)]
        row = {
            "streams": n,
            "p50_ms": round(p50, 1),
            "p95_ms": round(p95, 1),
            "n_responses": len(all_lat),
        }
        results.append(row)
        deadline = "OK" if p95 < args.chunk_ms else "MISS"
        print(
            f"streams={n:3d} p50={p50:7.1f}ms p95={p95:7.1f}ms "
            f"responses={len(all_lat):5d} deadline={deadline}"
        )
    print(json.dumps({"chunk_ms": args.chunk_ms, "sweep": results}))


if __name__ == "__main__":
    main()
