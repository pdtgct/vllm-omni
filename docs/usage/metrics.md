# Production Metrics

vLLM-Omni exposes Prometheus metrics via the `/metrics` endpoint on the OpenAI-compatible API server. This page covers the text and audio surface; diffusion / image / video metrics are tracked in a follow-up PR.

```bash
vllm-omni serve Qwen/Qwen3-Omni-30B-A3B-Instruct --port 8000
curl http://localhost:8000/metrics
```

**Statistics collection is on by default for `vllm-omni serve`; disable it with `--disable-log-stats`.** With statistics disabled the endpoint still returns `200 OK` and every family stays registered, but no sample data is written — the runtime cost is essentially zero for deployments that don't need monitoring.

## Metric Namespaces

| Prefix | Source | Present when |
|--------|--------|--------------|
| `vllm_omni:` | vLLM-Omni orchestrator / audio modality / cross-stage transfer | Pipeline-dependent |
| `vllm:` | Upstream vLLM engine, wrapped by `OmniPrometheusStatLogger` to expose `{stage, replica}` | Pipeline includes an LLM (AR) stage |
| `http_` / `process_` | Uvicorn / Python runtime | Always |

## Pipeline-Level Metrics (`vllm_omni:`)

Defined in `vllm_omni/metrics/prometheus.py`. Track request lifecycle across the full multi-stage pipeline.

### Request counts

| Metric | Type | Labels | Description |
|--------|------|--------|-------------|
| `vllm_omni:num_requests_running` | Gauge | `model_name` | Pipeline-global in-flight requests (dispatched to engine, not yet finalized) |
| `vllm_omni:num_requests_waiting` | Gauge | `model_name` | Requests waiting in the Orchestrator queue |
| `vllm_omni:requests_success_total` | Counter | `model_name`, `finished_reason` | Total requests by completion reason. `finished_reason` ∈ {`stop`, `length`, `abort`, ...} mirroring upstream `vllm:request_success_total`; aborts cover client disconnect / cancellation paths in addition to upstream `FinishReason.ABORT` |

### Latency

| Metric | Type | Labels | Description |
|--------|------|--------|-------------|
| `vllm_omni:e2e_request_latency_s` | Histogram | `model_name` | Pipeline-global end-to-end request latency in seconds |

## Audio Modality Metrics (`vllm_omni:`)

Emitted at request finalize, except for `audio_ttfp_s` (streaming-hook at the first audio packet) and `audio_underrun_s` / `audio_continuity_ok_total` (streaming finalize, after the chunk stream is exhausted). All carry `{model_name, stage, replica}` plus the listed extra label.

| Metric | Type | Extra label | Description |
|--------|------|-------------|-------------|
| `vllm_omni:audio_ttfp_s` | Histogram | — | Time from request arrival to first audio packet/frame |
| `vllm_omni:audio_duration_s` | Histogram | — | Audio content duration (`audio_frames / sample_rate`) |
| `vllm_omni:audio_rtf` | Histogram | — | Real-time factor (`stage_gen_time_s / audio_duration_s`); streaming TTS SLO red line `< 1`; uses `RTF_BUCKETS` |
| `vllm_omni:audio_frames_total` | Counter | — | Cumulative audio frame count; throughput via `rate()` |
| `vllm_omni:audio_underrun_s` | Histogram | — | Per-request worst-case player deficit; `> 0` indicates listener heard silent gaps |
| `vllm_omni:audio_continuity_ok_total` | Counter | `threshold_ms` | Incremented when the request's worst underrun stayed below `threshold_ms` |
| `vllm_omni:audio_skipped_requests_total` | Counter | `reason` | Silent-loss counter — code2wav rejected malformed codec input and returned `200 OK` with empty audio |

The continuity math comes from `vllm_omni/benchmarks/audio_continuity.py::compute_continuity_stats` so the server-side observation aligns with the bench-side definition.

## Cross-Stage Transfer Metrics (`vllm_omni:`)

Per-physical-transfer histograms tracking the data hop between adjacent stages. Labels `{model_name, from_stage, from_replica, to_stage, to_replica}` let dashboards attribute latency to specific replica edges. `from_replica` / `to_replica` are resolved from the orchestrator's sticky-routing binding (`stage_pool.get_bound_replica_id(request_id)`), so no extra plumbing through `TransferEdgeStats` is needed.

| Metric | Type | Description |
|--------|------|-------------|
| `vllm_omni:transfer_size_bytes` | Histogram | Per-transfer payload size in bytes |
| `vllm_omni:transfer_tx_s` | Histogram | Sender-side time (serialize + submit to connector) |
| `vllm_omni:transfer_rx_s` | Histogram | Receiver-side time (recv + deserialize) |
| `vllm_omni:transfer_in_flight_s` | Histogram | Network in-flight time (TX done → RX recv start) |

## Streaming Session Metrics (`vllm_omni:streaming_*`)

Defined in `vllm_omni/metrics/streaming.py`. Ten families covering realtime streaming-ASR sessions on the `/v1/realtime` path (and any transport frontend built on the transport-neutral session factory). They are designed as **autoscaler inputs** (HPA/KEDA) and SLO signals: backlog, deadline misses, active sessions, and effective batch size are the series to scale and alert on.

Every family labels `model_name` with the **served alias** (the `--served-model-name` identity), not the checkpoint path. Note this deliberately differs from the older `vllm_omni:` request-tracking families above, which label the canonical model path — an inherited inconsistency that was not copied into the new families.

`cadence_ms` is the session's admitted streaming cadence, a bounded enumeration `{80, 160, 320, 560, 1120}`. `chunk_type` distinguishes `regular` cadence units from the single `final_tail` unit minted at finalization. All label values are bounded enumerations — an out-of-vocabulary value is dropped, never a new series.

### Session lifecycle

| Metric | Type | Labels | Description |
|--------|------|--------|-------------|
| `vllm_omni:streaming_sessions_active` | Gauge | `cadence_ms` | Sessions currently open. Increments at model-session construction (after validation, before the engine request exists — never at WebSocket accept); decrements exactly once in the session's terminal section, so opens − finished == active |
| `vllm_omni:streaming_sessions_finished_total` | Counter | `cadence_ms`, `reason` | Terminal outcomes, `reason ∈ {completed, aborted, error}`. `completed` iff model finalization succeeded with no work left outstanding — transport delivery is not part of completion. `aborted` covers client disconnect and cancellation; `error` covers engine/protocol failure and lifecycle divergence |
| `vllm_omni:streaming_session_open_rejections_total` | Counter | `reason` | Denied session opens, `reason ∈ {model, cadence, locale, config}`. Diagnostic-only by design: rejections precede admission, carry no cadence label, and should not drive autoscaling |

### Chunk service and SLO

| Metric | Type | Labels | Description |
|--------|------|--------|-------------|
| `vllm_omni:streaming_chunk_latency_s` | Histogram | `cadence_ms`, `chunk_type` | Per-chunk service time from readiness (cadence completion; finalize acceptance for the final tail) to the committed park completing the chunk. Queueing is included, so alert on this p95/p99. The bucket ladder (`0.01 … 10.0` s) has an edge at every admitted cadence period, so "within cadence budget" reads directly off the histogram |
| `vllm_omni:streaming_chunks_total` | Counter | `cadence_ms`, `chunk_type`, `outcome` | Every ready unit's single terminal disposition, `outcome ∈ {parked, aborted, error}`. `parked` is the success denominator |
| `vllm_omni:streaming_deadline_misses_total` | Counter | `cadence_ms`, `chunk_type` | Parked chunks whose service latency strictly exceeded their cadence period — the realtime SLO violation counter |

### Load and backpressure (the autoscaling inputs)

| Metric | Type | Labels | Description |
|--------|------|--------|-------------|
| `vllm_omni:streaming_backlog_chunks` | Gauge | `cadence_ms` | Units past readiness without a terminal disposition (waiting-ready ∪ in-flight), summed over sessions. The primary backpressure signal: sustained growth means the deployment is under-provisioned |
| `vllm_omni:streaming_backlog_overflows_total` | Counter | `kind` | Bound trips, `kind ∈ {input_queue, carrier, receipt}`: the per-session accepted-audio occupancy budget on any path, and the transport-side ledger bounds on the leased path. A nonzero rate means clients are outrunning the deployment |
| `vllm_omni:streaming_input_audio_seconds_total` | Counter | `cadence_ms` | Accepted input audio duration, accrued at acceptance (rejected pieces contribute nothing). Divided by wall time this is ingest throughput in audio-seconds per second |
| `vllm_omni:streaming_chunk_batch_size` | Histogram | `stage`, `replica`, `cadence_ms` | Rows in each executed nonempty streaming batch — the **effective batch size** on the GPU, for utilization and capacity planning. The one engine-side series: recorded in the model runner and carried to the serving process on the omni-owned engine outputs (no vLLM core change) |

### Operational notes

- Gated by the same host statistics switch as everything else: with `--disable-log-stats` all ten families stay registered but sample-free, and every observation path returns before recording.
- The streaming gauges are process-local; serving asserts the single-API-server invariant (`--api-server-count 1`) at startup rather than assuming it.
- Observation is non-authoritative and non-fatal: an exporter failure can never fail audio acceptance, parking, finalization, or scheduling.

## vLLM Engine Metrics (`vllm:`)

When the pipeline includes an LLM stage, the upstream vLLM engine exposes its full set of ~37 metric families under the `vllm:` prefix.

vLLM-Omni wraps the upstream `vllm.v1.metrics.loggers.PrometheusStatLogger` with `OmniPrometheusStatLogger` so that the original `engine` single label is reshaped into `stage` + `replica`. Every `vllm:*` family — TTFT, ITL, TPOT, e2e latency, KV cache usage, scheduler running/waiting, request success counts, etc. — therefore gains per-`(stage, replica)` visibility automatically. No omni-side duplicate is needed for the text path.

```text
# Before wrap:
vllm:num_requests_running{model_name="...", engine="1"}              3.0

# After wrap:
vllm:num_requests_running{model_name="...", stage="1", replica="0"}  2.0
vllm:num_requests_running{model_name="...", stage="1", replica="1"}  1.0
```

For the full list of upstream metrics, see [the vLLM docs](https://github.com/vllm-project/vllm/blob/main/docs/usage/metrics.md).

## Metric Availability by Pipeline Type

| Metric group | Present when |
|---|---|
| `vllm_omni:` request tracking + latency | Statistics enabled (the default; off with `--disable-log-stats`) |
| `vllm_omni:` audio modality | Statistics enabled, if pipeline has a talker stage |
| `vllm_omni:` transfer | Statistics enabled, if pipeline has ≥ 2 stages |
| `vllm_omni:streaming_*` session metrics | Statistics enabled, realtime streaming path served |
| `vllm:` engine metrics (per `(stage, replica)`) | Statistics enabled |
| `vllm:` MFU metrics | Statistics enabled, plus `--enable-mfu-metrics` |

## Naming Convention

- All time-bearing metrics use the `_s` suffix (values in seconds). Buckets are `SECONDS_BUCKETS` for e2e / generation-style values and `SECONDS_FAST_BUCKETS` (1 ms → 60 s) for the fine-grained transfer and audio-underrun values.
- Counters use the `_total` suffix (auto-appended by `prometheus_client`).
- Sizes use the `_bytes` suffix.
- All omni-specific families are prefixed `vllm_omni:`.
- Text and audio first-output use distinct families (`vllm:time_to_first_token_seconds` reused from upstream for text; `vllm_omni:audio_ttfp_s` for audio) rather than a single metric with a `modality` label.
