# Realtime Streaming ASR

vLLM-Omni serves NVIDIA's cache-aware streaming speech-recognition model
[`nvidia/nemotron-3.5-asr-streaming-0.6b`](https://huggingface.co/nvidia/nemotron-3.5-asr-streaming-0.6b)
natively over the `/v1/realtime` WebSocket route. Each connection owns one
streaming session with persistent per-session encoder state on the GPU: audio
advances the model chunk by chunk, interim transcripts stream back as they
form, and a final transcript is produced when the client ends the stream.

Each server instance runs a single model (specified at startup via
`vllm-omni serve <model> --omni`).

## Quick Start

### Download the model

Serve directly from a plain checkout of the HuggingFace repository — no
conversion step and no `--trust-remote-code`:

```bash
hf download nvidia/nemotron-3.5-asr-streaming-0.6b \
    --local-dir nemotron-3.5-asr-streaming-0.6b
```

The served directory must contain `config.json`, `model.safetensors`,
`processor_config.json` (the locale table), and the tokenizer files — all part
of the published repository.

### Start the server

```bash
vllm-omni serve ./nemotron-3.5-asr-streaming-0.6b \
    --served-model-name nemotron \
    --omni \
    --port 8000
```

The deploy profile resolves from the model type: the streaming scheduler,
FP32 weights, eager execution, and prefix caching disabled are selected
automatically. `--served-model-name` sets the public model id clients must
use; the checkpoint path is never a client-facing identity.

### Stream audio

The route speaks the vLLM realtime dialect. The client sequence is:

1. `session.update` with the served model name — validates the model and
   admits the session;
2. `input_audio_buffer.commit` with `final: false` — starts generation;
3. repeated `input_audio_buffer.append` — base64 PCM16 mono at 16 kHz;
4. `input_audio_buffer.commit` with `final: true` — closes audio;
5. receive `transcription.delta` events while streaming and one
   `transcription.done` carrying the final transcript.

```python
import asyncio, base64, json, wave
import websockets


async def transcribe(path: str) -> str:
    with wave.open(path, "rb") as w:
        assert w.getframerate() == 16000 and w.getnchannels() == 1
        pcm = w.readframes(w.getnframes())

    async with websockets.connect("ws://localhost:8000/v1/realtime") as ws:
        await ws.send(json.dumps({"type": "session.update", "model": "nemotron"}))
        await ws.send(json.dumps({"type": "input_audio_buffer.commit", "final": False}))
        chunk = 2 * 16000 // 10  # 100 ms of PCM16
        for i in range(0, len(pcm), chunk):
            await ws.send(json.dumps({
                "type": "input_audio_buffer.append",
                "audio": base64.b64encode(pcm[i : i + chunk]).decode(),
            }))
            await asyncio.sleep(0.1)  # real-time pacing
        await ws.send(json.dumps({"type": "input_audio_buffer.commit", "final": True}))

        async for message in ws:
            event = json.loads(message)
            if event.get("type") == "transcription.delta":
                print(event.get("text", ""), end="", flush=True)
            elif event.get("type") == "transcription.done":
                return event.get("text", "")
            elif event.get("type") == "error":
                raise RuntimeError(event)


print(asyncio.run(transcribe("sample.wav")))
```

Omitting the non-final commit in step 2 is the most common client mistake:
the session then accepts audio but never starts generation and produces no
transcript.

## Session options

`session.update` accepts, alongside `model`:

| Field | Values | Default | Meaning |
|---|---|---|---|
| `cadence` | `80ms`, `320ms`, `560ms`, `1120ms` | `560ms` | Streaming chunk size: smaller is lower latency, larger is higher throughput. Admission validates the cadence against the checkpoint's declared supported lookahead arms; `160ms` is rejected for this checkpoint because its implied lookahead is not in the card's declared set. |
| `locale` | a locale from the checkpoint's `processor_config.json` (e.g. `en-US`), or `auto` | `auto` | Language conditioning. `auto` lets the model detect the language and emit a language tag. |
| `endpointing` | `{"mode": "greedy_blank" \| "disabled", "stop_history_ms": int, "residue_frames": int}` | `greedy_blank`, 800 ms, 2 | Server-side end-of-segment detection over the decode stream. |

The locale can also be changed mid-session with a later `session.update`; it
applies from the next audio chunk.

## Runtime limits

Serving needs no extra configuration: every runtime limit resolves to a
qualified default. The defaults can be overridden through
`--additional-config` with a JSON object; commonly adjusted keys:

| Key | Default | Meaning |
|---|---|---|
| `max_resident_sessions` | 8 | Concurrent streaming sessions holding GPU state. |
| `streaming_accepted_audio_capacity_samples` | 480000 (30 s) | Per-session accepted-audio backlog before backpressure. |
| `streaming_session_idle_timeout_s` | 60 | Idle session reclamation. |
| `streaming_session_finalization_timeout_s` | max(40, safe drain floor) | Bound on end-of-stream drain; values below the published safe floor are rejected. |
| `streaming_max_session_duration_s` | unbounded | Optional per-session duration policy. |

Derived limits follow overridden inputs (queue and tombstone bounds track the
session cap; the finalization floor tracks the audio budget), and an explicit
invalid value fails startup rather than being replaced by a default.

## Notes

- Audio must be mono 16 kHz PCM16. Send pieces at any size; the server owns
  chunking (cadence) and never drops accepted audio to admit more — a full
  per-session buffer rejects the new piece instead.
- Sessions end by client finalization, disconnect, idle timeout, or an
  explicit error; state is reclaimed idempotently on every path.
- The model runs in FP32; a single L4-class GPU serves the qualified
  eight-session profile in real time.
- Compatibility frontends (e.g. Riva/NIM dialects) are separate downstream
  packages layered on this server's public session API; this route is the
  native surface.
