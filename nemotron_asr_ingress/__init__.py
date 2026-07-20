"""Ingress: session core, providers, dialect adapters (ING segment).

The session core is a sans-IO protocol — a typed event interface with
no sockets, no serialization, no framework imports (ingress-design.md
§The session-core seam). Adapters translate dialects; exactly one
session core owns behavior. ING-1 scope: the session-core event set,
the error catalog, admission (watermark + bounded queue + admission-
wait timeout), the vLLM ``/v1/realtime`` dialect binding for the β
server, and the shared WebSocket session client's event core. ING-2
adds the downstream audio front-end (accept-matrix, G.711, the pinned
resampler) and the Riva gRPC servicer over the generated
``nvidia-riva-client`` stubs. ING-3 adds the NIM-realtime WebSocket
adapter (the public NIM Realtime dialect over the same core) and the
NIM-shaped HTTP shim (aux endpoints with our provenance, delegated
inference).
"""
