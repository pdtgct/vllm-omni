# nemotron_asr_ingress — Riva/NIM-family compatibility frontend

A **sans-IO** ingress package that lets existing NVIDIA Riva/NIM-family ASR
clients drive a vLLM-Omni streaming-ASR session core unchanged. It is the
reference prototype for a proposed *Riva frontend compatibility* RFC — a
separate proposal from the engine-managed streaming-state RFC (the
[`nemotron-asr-port`](https://github.com/pdtgct/vllm-omni/tree/nemotron-asr-port)
branch). This branch is `unmodified vLLM-Omni v0.24.0 + this package`, so it
reads as a clean, self-contained deliverable.

## What it is

One transport-agnostic session core with dialect adapters over it (a dialect
codec translates wire messages ⇄ session-core events; exactly one core owns
behavior). The adapters cover four surfaces:

| Surface | Module | Speaks |
|---|---|---|
| vLLM `/v1/realtime` | `vllm_dialect.py` | vLLM-Omni's own realtime WebSocket dialect |
| Riva gRPC | `riva_grpc.py` | `RivaSpeechRecognition` — `StreamingRecognize`, `Recognize`, `GetRivaSpeechRecognitionConfig` |
| NIM realtime WS | `nim_ws.py` | the public NIM Realtime transcription-intent event dialect |
| NIM HTTP shim | `nim_http.py` | NIM auxiliary endpoint paths + error-body shapes |

Plus the shared pieces: the session core (`core.py`), the sans-IO client
(`client.py`), the event set (`events.py`), the error catalog (`errors.py`),
the admission provider (`provider.py`), and the audio front-end
(`frontend.py` — the accept-matrix, ITU G.711 μ-law/A-law tables, and a
stateful `resample_poly(x, 2, 1)` reimplementation proven ≤2e-7 against SciPy).

## Sans-IO by design

The package imports **no** vLLM, vLLM-Omni, or model/engine code — only
`numpy`, `scipy`, `grpc`, and the public `nvidia-riva-client` protos. It is a
pure protocol layer: sockets, serialization, and the engine live outside it.
The wiring that stands a real engine behind the core (weight loading, the
batched kernel, the serve entrypoints) is deliberately *not* in this package;
it composes the same session core an engine already exposes. That separation
is what keeps every dialect adapter independently testable with no GPU.

## Provenance (public sources only)

Every NVIDIA-facing shape is built clean-room from public sources: the public
[`nvidia-riva/python-clients`](https://github.com/nvidia-riva/python-clients)
tooling and its generated protos (the `nvidia-riva-client` PyPI wheel), the
public NIM Realtime API reference (docs.nvidia.com/nim/speech), and the HF
model card. Behavioral differences from NIM's servers (e.g. locale-tag
handling, punctuation defaults, delta-vs-cumulative event semantics) are
documented deviations observed against those public references — each labeled
in the module that owns it. No proprietary source is referenced.

## Evidence

- **275 GPU-free contract tests** (`tests/nemotron_asr_ingress/`) — the
  session-core event set, admission, the error catalog, all four dialects,
  the front-end/telephony path, and the honest-subset disposition matrix
  (which walks the live `nvidia-riva-client` proto descriptor, so a wheel bump
  that adds a config field breaks loudly).
- The canonical `nvidia-riva/python-clients` CLIs (`transcribe_file.py`,
  `realtime_asr_client.py`) drive this core **unmodified, with default flags**,
  in the project's serving harness, and it has been compared like-for-like
  against a self-hosted reference NIM sidecar on the same GPU.

## Status / caveat

This package predates the engine's Phase-6c transaction rework; it was
validated against the earlier serving path. Re-verification against the
current batched `advance_model_rows` engine is owed once that path's parity
gate closes — the sans-IO boundary means the package itself is unaffected, but
the end-to-end serving wiring must be re-run.
