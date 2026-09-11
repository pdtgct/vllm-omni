# Model Runner v2 RFC review against v0.26.0

**Status:** maintainer review; changes requested  
**Review baseline:** `rfc2-host-v025`, `rfc1-5212-main`, vLLM-Omni
[v0.26.0](https://github.com/vllm-project/vllm-omni/releases/tag/v0.26.0), and
vLLM [v0.26.0](https://github.com/vllm-project/vllm/releases/tag/v0.26.0)

## Decision

Keep the RFCs' product boundary and persistent-state design, but do **not** merge
them as a v0.25 backport and then perform a mechanical dependency bump. Rebase
both RFCs onto vLLM-Omni v0.26 first, make Model Runner v2 (MRv2) the sole
qualification target, and re-run the lifecycle and capacity evidence on that
base.

The manager-owned, purpose-named persistent-state group remains a sound design.
It is preferable to disguising session state as an attention KV group, and the
thin MRv2 projection remains preferable to copying the core runner. The release
does not provide production-grade persistent session leases, recovery, or
admission; vLLM-Omni explicitly describes its new full-duplex runtime as an
experimental preview without those guarantees. RFC1 therefore still fills a
real gap.

The current patch series is not merge-ready on v0.26. It backports and overrides
several moving upstream seams. vLLM 0.26 modularizes the MRv2 GPU runner, adds
native zero-cache-group handling, changes warmup/profiling behavior, adds
persisted startup profiling, and introduces per-cache-group attention backend
selection. vLLM-Omni 0.26 also adds a full-duplex session runtime and vLLM adds
an endpoint-plugin framework. Those changes require reconciliation, not a
parallel implementation with subtly different ownership.

## Required RFC changes

### 1. Change the declared baseline and compatibility contract

Both RFCs must name `vllm==0.26.*` and `vllm-omni==0.26.*` as their implementation
baseline. Compatibility must be expressed in terms of public or explicitly
accepted extension points, not file paths or private method signatures.

Add a CI lane that installs the released v0.26 pair and performs import and
contract tests. A source-tree-only test run is insufficient: it will not catch
packaging omissions or import drift. Keep the existing Python 3.10 coverage.

### 2. Make the modular MRv2 runner the only implementation target

The current `GPUARModelRunnerV2` direction is correct: it subclasses
`vllm.v1.worker.gpu.model_runner.GPUModelRunner` and brackets core request
management instead of replacing it. Update the RFC to make that modular class
an explicit dependency and remove MRv1 as a behavioral oracle required for
production.

MRv1 may remain temporarily as a test oracle behind an opt-in flag, but it must
not be an automatic fallback. Fail startup clearly when MRv2 capability
validation fails. Silent fallback would conceal correctness bugs in lifecycle,
preemption, or scheduler block-group ordering.

### 3. Remove the custom empty-block-table compatibility layer if core v0.26
passes the persistent-only tests

vLLM 0.26 handles an empty `kv_cache_groups` topology in MRv2. The RFC currently
carries `_EmptyOrdinaryBlockTables` to emulate that behavior. First run the
persistent-only suite directly against the native v0.26 block tables. If it
passes, delete the adapter. If it does not, reduce the adapter to the failing
operation and propose that missing zero-group contract upstream.

The RFC must not claim native zero-group support while retaining a broad proxy
that can mask new core operations. Add a test that fails whenever an unexpected
block-table method is reached.

### 4. Rework warmup as a narrow core hook, not a copied worker postamble

The persistent-only warmup currently duplicates core orchestration: LoRA
removal, kernel warmup, seed reset, JIT monitor activation, GC freezing, GPU
sync checks, and compilation timing. This is fragile and already intersects
v0.26's changed warmup and CUDA-graph memory accounting.

Revise the RFC to request or introduce a narrow core hook for the synthetic
request portion of warmup. Omni should supply a persistent-state dummy/profile
batch through that hook; core must continue to own the postamble. Until that
hook exists, pin an exact vLLM patch version and add a parity test comparing the
core and Omni postamble call sequence.

### 5. Integrate persisted memory profiling deliberately

vLLM 0.26 can persist and reuse memory-profiling results across boots. A cached
profile must not be accepted solely because model weights and ordinary engine
arguments match. Persistent-state capacity depends on the state specification,
qualified cadence/arms, dtype, state-slot count, endpoint-control headroom, and
execution mode.

Extend the startup-profile fingerprint with those inputs, or explicitly disable
profile reuse for persistent-state models. The hard-cap, profile-free startup
policy remains useful, but the RFC must distinguish:

* core activation/CUDA-graph memory evidence;
* persistent resident-state allocation evidence; and
* service/admission geometry evidence.

No evidence class may substitute for another.

### 6. Specify cache-group ownership under per-group backend selection

Per-KV-cache-group attention backend selection is new in vLLM 0.26. The
persistent-state group is storage, not attention, and must never participate in
attention-backend selection, sliding-window capability checks, connector
initialization, prefix-cache accounting, or KV offload/tiering.

Add this as a normative invariant and test it with at least two ordinary cache
groups using different backends plus one persistent group. Keep the current
restrictions on prefix caching and DCP/PCP until this mixed topology is
qualified; do not infer support from core's new heterogeneous-group support.

### 7. Reconcile RFC2 with both plugin frameworks

Do not replace the v0.26 endpoint-plugin framework with a second endpoint
registration mechanism. RFC2's application plugin may remain a higher-level,
lifecycle-aware supervisor, but its relationship to core endpoint plugins must
be explicit:

* endpoint plugins own route contribution;
* the application plugin owns session resources, readiness, drain, failure
  propagation, and shutdown;
* reserved `/live`, `/health`, and `/metrics` ownership is declared once;
* route collision detection runs after all core and Omni routers are composed;
* multi-worker and launcher semantics are tested on the v0.26 launcher.

Prefer an adapter from the RFC2 lifecycle object to the upstream endpoint
plugin contract. Do not fork route discovery or CLI configuration unless an
upstream limitation is documented.

### 8. Reconcile RFC1 with v0.26 full-duplex session semantics

The new MiniCPM-o runtime establishes names and behavior for commit, cancel,
barge-in, overlap policy, playback acknowledgement, disconnect, and connector
tail flushing. RFC1 should reuse that external vocabulary where semantics
match, while retaining the stronger persistent-lease guarantees.

Add a mapping table to RFC1 for native duplex and Realtime-compatible events.
In particular, specify which component owns session identity, when a commit
becomes durable, how cancel differs from terminal finish, how preemption is
reported, and whether disconnect drains, parks, or releases resident state.
RFC1's forced end-of-utterance park must not be confused with vLLM's terminal
finish status.

### 9. Add explicit v0.26 non-goals and qualification gates

The first v0.26 merge should remain single-stage, single API-server worker,
eager execution, no CUDA graphs, no prefix caching, and DCP=PCP=1 for
persistent-state models. Core support for a feature is not evidence that the
persistent-state composition supports it.

Before relaxing any gate, require tests for allocation, profile/warmup,
preemption/recovery, cancellation, shutdown, and metrics cardinality. Add a
long-duration soak with repeated commit/cancel/barge-in cycles and process-kill
recovery; the v0.26 release notes specifically do not claim those production
guarantees.

## Branch-level maintainer findings

### `rfc1-5212-main`

**Keep:** purpose-named model state, manager ownership, explicit transaction
status, bounded leases/tombstones, recovery, admission receipts, and the
separation of service geometry from allocation.

**Change before merge:** rebase the scheduler/model-state projection onto the
modular MRv2 APIs; remove assumptions based on MRv1 ordering; prove native
zero-group behavior; fingerprint or disable persisted profiling; and map the
session state machine onto v0.26 duplex events.

### `rfc2-host-v025`

**Keep:** supervised lifecycle, readiness, ordered drain/exit, failure
propagation, and strict operations-route ownership.

**Change before merge:** rebuild on the v0.26 launcher; adapt route registration
to the endpoint-plugin framework; eliminate duplicated router discovery; and
re-run shutdown race tests against v0.26's full-duplex disconnect and tail-flush
behavior.

## Merge plan

1. Rebase the host RFC onto v0.26 and land only the generic lifecycle adapter
   plus launcher parity tests.
2. Rebase the persistent-state RFC onto that result, MRv2-only.
3. Land storage/specification and scheduler plumbing without a public model.
4. Land the Nemotron model projection and persistent-only native zero-group
   execution.
5. Land recovery/admission and the duplex event mapping.
6. Enable the public deployment only after GPU integration, soak, packaging,
   and shutdown-fault tests pass on the released v0.26 dependency pair.

This ordering keeps the generic host reviewable, avoids coupling route plumbing
to one model, and makes every persistent-state slice testable on the actual
compatibility target.
