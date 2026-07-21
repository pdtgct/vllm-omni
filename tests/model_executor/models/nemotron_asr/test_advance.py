# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests-first contract for the advance seam split (ledger P5-1/6c).

Pins the ``advance_session`` / ``advance_model_rows`` contract
(``advance.py``) that replaced the legacy ``run_forward_step``
(``forward_step.py``, deleted at Task 7). POD-TIER: importing
``vllm_omni.model_executor.models.nemotron_asr.*`` pulls the
``vllm_omni`` package, which pulls ``vllm`` — this file cannot be
collected on macOS and must be run on the pod venv (contrast
``test_advance_model_rows_local.py``, the stubbed-chain local bar).

Reconciled 2026-07-19 to the Phase-6c settled seams (design §Phase-6c
transaction seams; ledger Phase-6c block): the resolver/sink/status
parameters, the host role/geometry authority in ``RowPlan``, the
pinned engine null id 0 (``NULL_BLOCK_ID``, utils.py:46 @ ee0da84 —
this file previously modeled −1), int32 queue/book pools per the
state manifest, legal-cadence envelopes (a 3,840-sample final tail is
OVERSIZE at 80 ms under the C+6 contract and moves to the 320 ms
geometry), and the row-tier echo contract (PORT-DEC-007 as amended —
the prior whole-call echo-abort expectation encoded the superseded
boundary and contradicted PORT-STATE-008's row tier).

The Phase-6c implementation (``make_mrv1_adapter``,
``advance_model_rows``) has since landed and ``forward_step.py`` was
deleted at Task 7. Both source-text pins (naming lock + forward
wiring) have flipped from ``xfail(strict=True)`` to real passing
assertions — the forward-wiring pin at Task 5, the naming lock at
Task 7 once the five-cadence parity gate passed.
"""

from pathlib import Path
from typing import Any

import pytest
import torch

from vllm_omni.model_executor.models.nemotron_asr import advance as advance_mod
from vllm_omni.model_executor.models.nemotron_asr.advance import (
    ENV_CHUNK_SEQUENCE,
    ENV_FINAL_TAIL,
    ENV_GEOMETRY_ID,
    ENV_PROMPT_INDEX,
    ENV_VALID_SAMPLES,
    ENV_VERSION,
    ENVELOPE_HEADER_SLOTS,
    ENVELOPE_VERSION,
    ROW_STATUS_BOOK_IDENTITY,
    ROW_STATUS_ECHO_MISMATCH,
    ROW_STATUS_PROMPT_MISMATCH,
    ROW_STATUS_QUEUE_NOT_DRAINED,
    AdvanceResult,
    ChunkBatch,
    DecodeRequest,
    EmissionAdapter,
    PreparedCaptures,
    PreparedRowBinding,
    ResolvedDecode,
    RowPlan,
    SessionStateBatch,
    advance_model_rows,
    advance_session,
    make_mrv1_adapter,
)
from vllm_omni.model_executor.models.nemotron_asr.manifests import (
    BOOK_FIELDS,
    FRONTEND_COUNTER_FIELDS,
)
from vllm_omni.model_executor.models.nemotron_asr.precision import (
    FP32_BRINGUP,
)
from vllm_omni.model_executor.models.nemotron_asr.rnnt import (
    MAX_SYMBOLS_PER_STEP,
    decode_dense_masked,
    read_decision_carrier,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

FEAT = 16
D_MODEL = 32
N_LAYERS = 2
KERNEL = 5
WINDOW = 8
PARK_ID = 9000
PLACEHOLDER_ID = 9001
VOCAB = 12
CAP = 48  # holds a cap-saturated burst (max_symbols × enc_frames)
#: Header (7 slots, design §Ingress-deadline plumbing) + the largest
#: cadence THESE fixtures admit (320 ms = 5,120 raw samples) — the
#: tiny-core analogue of the production 17,927.
CARRIER_HIDDEN = 5_127
RAW_TAIL = 1_953  # pre_encode_cache(9) * hop(160) + n_fft(512) + 1
#: The reserved null block id at the pin (``NULL_BLOCK_ID``,
#: vllm/v1/attention/backends/utils.py:46 @ ee0da84): block 0 is the
#: null block; live session pages start at 1. Graph-padding rows carry
#: it and no real row may (PORT-STATE-007).
NULL_INDEX = 0

#: Geometry ids follow manifests.CADENCES order: 0=80ms(C=8),
#: 2=320ms(C=32). A regular CHUNK carries exactly one cadence unit;
#: a legal final residual is STRICTLY under one unit (C+6 contract,
#: PORT-SESS-001/003) — so final-tail fixtures use the 320 ms
#: geometry, whose 5,120-sample unit legally covers a 3,840-sample
#: session-first final (24 mel frames >= the 8-new-frame commit rule).
GEOM_REG = 0
REG_SAMPLES = 1_280  # one 80 ms cadence unit at 16 kHz
GEOM_FINAL = 2
FINAL_SAMPLES = 3_840  # 24 mel frames at hop 160, < one 320 ms unit

#: Book / counter slot indices, derived from the manifest's canonical
#: order — the single source (manifests.py).
_BOOK = {name: i for i, (name, _) in enumerate(BOOK_FIELDS)}
_CTR = {name: i for i, name in enumerate(FRONTEND_COUNTER_FIELDS)}
BOOK_WIDTH = len(BOOK_FIELDS)
CTR_WIDTH = len(FRONTEND_COUNTER_FIELDS)

#: A pool bundle: per-layer tensor lists (channel/time/len) plus the
#: whole-pool tensors (h/c/queue/book/frontend) — the shapes the engine
#: binds. Left as ``dict[str, Any]`` — call sites index the per-layer
#: lists AND the whole-pool tensors interchangeably by key.
Pools = dict[str, Any]


def _tiny_core(seed: int = 7) -> Any:
    # Seed 7 is the shared fixture default; the burst-arc test builds
    # a seed-40 core — through the REAL featurizer path the seed-7
    # tiny joint is single-label on every probed signal family (the
    # log-guard-dominated mel is near-constant), while seed 40 yields
    # a deterministic cap-saturated burst with three distinct labels
    # across two transitions, the drain-order discriminator. NOTE ON
    # FRAGILITY (found 2026-07-20, pod round): this class of fixture —
    # untrained random weights feeding a recurrent greedy-argmax decode
    # over 40 steps — is inherently sensitive to environment-level
    # floating-point differences (torch build, BLAS backend, thread
    # count); a prior seed choice (1) that reportedly qualified when
    # authored no longer produces a multi-distinct-label burst in
    # EITHER of two independently checked environments (an Apple
    # Silicon local run and an A100 SXM pod, both landing on the same
    # degenerate single-label collapse). Seed 40 was swept and
    # confirmed stable across repeated local runs, but no seed choice
    # here is guaranteed permanent — if this test's own precondition
    # ("need a multi-label, multi-distinct burst") starts failing
    # again after a torch/dependency bump, re-sweep rather than assume
    # a code regression.
    # Returns a duck-typed SimpleNamespace, not a real NemotronASRCore
    # (Any — the minimal-core fixture idiom used across these tests).
    from types import SimpleNamespace

    from vllm_omni.model_executor.models.nemotron_asr.encoder import (
        FastConformerEncoder,
    )
    from vllm_omni.model_executor.models.nemotron_asr.featurizer import (
        MelFeaturizer,
    )
    from vllm_omni.model_executor.models.nemotron_asr.lid import (
        PromptConditioner,
    )
    from vllm_omni.model_executor.models.nemotron_asr.rnnt import (
        Joint,
        Predictor,
    )

    torch.manual_seed(seed)
    encoder = FastConformerEncoder(
        feat_in=FEAT,
        d_model=D_MODEL,
        d_ff=64,
        n_layers=N_LAYERS,
        n_heads=4,
        conv_kernel=KERNEL,
        subsampling_channels=16,
        att_context=(WINDOW, 1),
    )
    encoder.eval()
    predictor = Predictor(vocab_size=VOCAB, pred_hidden=16, pred_rnn_layers=2)
    joint = Joint(enc_hidden=D_MODEL, pred_hidden=16, joint_hidden=16, vocab_size=VOCAB)
    lid = PromptConditioner(enc_hidden=D_MODEL, num_prompts=4)
    featurizer = MelFeaturizer(
        filterbank=torch.rand(FEAT, 257) * 0.01,
        window=torch.hann_window(400),
    )
    return SimpleNamespace(
        encoder=encoder,
        lid=lid,
        predictor=predictor,
        joint=joint,
        featurizer=featurizer,
        blank_id=VOCAB,
    )


def _whole_signal_mel(core: Any, samples: torch.Tensor) -> torch.Tensor:
    """Featurize the whole signal — the final-tail reference semantics
    (a final-tail chunk applies ordinary centered-STFT boundary rules,
    identical to the whole-utterance featurizer)."""
    with torch.no_grad():
        mel, mel_len = core.featurizer(samples.unsqueeze(0), torch.tensor([samples.shape[0]]))
    trimmed: torch.Tensor = mel[:, :, : int(mel_len[0])]
    return trimmed


def _reference_burst(core: Any, samples: torch.Tensor, prompt_index: int) -> list[int]:
    """The labels a session-first final-tail chunk should emit, via the
    golden whole-signal path."""
    from vllm_omni.model_executor.models.nemotron_asr.encoder import (
        StreamingCaches,
        stream_step,
    )
    from vllm_omni.model_executor.models.nemotron_asr.rnnt import (
        DecodeState,
        greedy_decode_batch,
    )

    mel = _whole_signal_mel(core, samples)
    caches = StreamingCaches(
        n_layers=N_LAYERS,
        batch=1,
        d_model=D_MODEL,
        left_context=WINDOW,
        conv_kernel=KERNEL,
        device=torch.device("cpu"),
    )
    with torch.no_grad():
        enc = stream_step(core.encoder, mel, caches, drop_extra=0)
        conditioned = core.lid(enc, prompt_index=prompt_index)
        state = DecodeState(
            h=torch.zeros(2, 1, 16),
            c=torch.zeros(2, 1, 16),
            last_label=torch.full((1,), core.blank_id),
        )
        labels, _ = greedy_decode_batch(conditioned, core.predictor, core.joint, state)
    return labels[0]


def _reference_encoder_frames(core: Any, samples: torch.Tensor) -> int:
    """The valid encoder-frame count for ``samples``, measured on the
    golden path — the PORT-INT-002 burst-bound input."""
    from vllm_omni.model_executor.models.nemotron_asr.encoder import (
        StreamingCaches,
        stream_step,
    )

    mel = _whole_signal_mel(core, samples)
    caches = StreamingCaches(
        n_layers=N_LAYERS,
        batch=1,
        d_model=D_MODEL,
        left_context=WINDOW,
        conv_kernel=KERNEL,
        device=torch.device("cpu"),
    )
    with torch.no_grad():
        enc = stream_step(core.encoder, mel, caches, drop_extra=0)
    return enc.shape[1]


def _fresh_pools(num_blocks: int = 3) -> Pools:
    # Block 0 is the reserved null block: allocated like every pool row
    # but never a live session page.
    return {
        "channel_pools": [torch.zeros(num_blocks, WINDOW, D_MODEL) for _ in range(N_LAYERS)],
        "time_pools": [torch.zeros(num_blocks, D_MODEL, KERNEL - 1) for _ in range(N_LAYERS)],
        "len_pools": [torch.zeros(num_blocks, 1, dtype=torch.int32) for _ in range(N_LAYERS)],
        "h_pool": torch.zeros(num_blocks, 2, 16),
        "c_pool": torch.zeros(num_blocks, 2, 16),
        "queue_pool": torch.zeros(num_blocks, CAP, dtype=torch.int32),
        "book_pool": torch.zeros(num_blocks, BOOK_WIDTH, dtype=torch.int32),
        "frontend_raw_pool": torch.zeros(num_blocks, RAW_TAIL),
        "frontend_mel_pool": torch.zeros(num_blocks, FEAT, 9),
        "frontend_counter_pool": torch.zeros(num_blocks, CTR_WIDTH, dtype=torch.int64),
    }


def _sentinel_pools(num_blocks: int = 3, value: float = 12345.0) -> Pools:
    """Pools filled with a distinctive value — never a legal state, so
    any accidental read-before-preflight is detectable, and equality
    against a clone after a rejected call proves no write happened.
    """
    pools = _fresh_pools(num_blocks)
    for val in pools.values():
        if isinstance(val, list):
            for t in val:
                t.fill_(value)
        else:
            val.fill_(int(value) if not val.dtype.is_floating_point else value)
    return pools


def _clone_pools(pools: Pools) -> Pools:
    return {k: [t.clone() for t in v] if isinstance(v, list) else v.clone() for k, v in pools.items()}


def _assert_pools_equal(a: Pools, b: Pools) -> None:
    for key in a:
        av, bv = a[key], b[key]
        if isinstance(av, list):
            for at, bt in zip(av, bv, strict=True):
                torch.testing.assert_close(at, bt, rtol=0, atol=0)
        else:
            torch.testing.assert_close(av, bv, rtol=0, atol=0)


def _plan(
    decodes: list[int] | None = None,
    prefills: list[int] | None = None,
    *,
    num_pool_blocks: int = 3,
    pad_decodes_to: int | None = None,
    padding_value: int = NULL_INDEX,
    decode_columns: int = 1,
    has_initial: list[bool] | None = None,
    live: list[int] | None = None,
    prompts: list[int] | None = None,
    chunk: list[bool] | None = None,
    geometries: list[int] | None = None,
    generations: list[int] | None = None,
    deadlines: list[int] | None = None,
    prior_prompts: list[int] | None = None,
    allow_prompt_transitions: list[bool] | None = None,
    execution_tier: int = 0,
) -> RowPlan:
    """Build a RowPlan the way the plan provider would: decode indices
    as a ``(rows, K)`` tensor (real rows first, graph padding after),
    prefill indices flat, freshness/liveness/roles/geometry/prompt/
    identity from HOST authority (scheduler metadata + the session
    registry) — CPU tensors throughout (design §Phase-6c transaction
    seams)."""
    decodes = decodes or []
    prefills = prefills or []
    num_decodes = len(decodes)
    padded = decodes + [padding_value] * ((pad_decodes_to or num_decodes) - num_decodes)
    d = torch.tensor([[i] * decode_columns for i in padded], dtype=torch.long).reshape(len(padded), decode_columns)
    p = torch.tensor(prefills, dtype=torch.long)
    if has_initial is None:
        has_initial = [False] * len(prefills)
    if live is None:
        live = list(range(1, num_pool_blocks))
    n_real = num_decodes + len(prefills)
    if prompts is None:
        prompts = [0] * n_real  # ADMITTED prompt authority, every row
    if chunk is None:
        # Default host roles: decode rows replay, prefill rows are
        # session-first CHUNKs.
        chunk = [False] * num_decodes + [True] * len(prefills)
    if geometries is None:
        geometries = [GEOM_REG] * n_real
    if generations is None:
        generations = [0] * n_real
    if deadlines is None:
        deadlines = [i + 1 if chunk[i] else 0 for i in range(n_real)]
    request_ids = tuple(f"req-{i}" for i in range(n_real))
    if prior_prompts is None:
        prior_prompts = list(prompts)
    if allow_prompt_transitions is None:
        allow_prompt_transitions = [False] * n_real
    blocks = [*decodes, *prefills]
    bindings = tuple(
        PreparedRowBinding(
            request_id=request_ids[row],
            block_id=blocks[row],
            admission_generation=generations[row],
            geometry_id=geometries[row],
            prompt_index=prompts[row],
            prior_prompt_index=prior_prompts[row],
            allow_prompt_transition=allow_prompt_transitions[row],
            is_chunk=chunk[row],
            ready_deadline_ns=deadlines[row],
        )
        for row in range(n_real)
    )
    return RowPlan(
        state_indices_d=d,
        num_decodes=num_decodes,
        state_indices_p=p,
        num_prefills=len(prefills),
        has_initial_states_p=torch.tensor(has_initial, dtype=torch.bool),
        null_block_id=NULL_INDEX,
        num_pool_blocks=num_pool_blocks,
        live_block_ids=torch.tensor(live, dtype=torch.long),
        geometry_id=torch.tensor(geometries, dtype=torch.long),
        prompt_index=torch.tensor(prompts, dtype=torch.long),
        is_chunk=torch.tensor(chunk, dtype=torch.bool),
        admission_generation=torch.tensor(generations, dtype=torch.long),
        ready_deadline_ns=torch.tensor(deadlines, dtype=torch.long),
        request_ids=request_ids,
        execution_tier=execution_tier,
        bindings=bindings,
    )


def _envelope(
    samples: torch.Tensor,
    *,
    final: bool,
    seq: int,
    geometry: int = GEOM_REG,
    prompt: int = 0,
    hidden: int = CARRIER_HIDDEN,
) -> torch.Tensor:
    """Pack one chunk-envelope carrier row (design §Chunk envelope):
    versioned header, then the raw samples; zeroes elsewhere."""
    n = samples.shape[0]
    assert ENVELOPE_HEADER_SLOTS + n <= hidden
    row = torch.zeros(hidden)
    row[ENV_VERSION] = ENVELOPE_VERSION
    row[ENV_VALID_SAMPLES] = n
    row[ENV_GEOMETRY_ID] = geometry
    row[ENV_FINAL_TAIL] = 1.0 if final else 0.0
    row[ENV_PROMPT_INDEX] = prompt
    row[ENV_CHUNK_SEQUENCE] = seq
    row[ENVELOPE_HEADER_SLOTS : ENVELOPE_HEADER_SLOTS + n] = samples
    return row


def _refuse_adapter(*_args: Any) -> Any:
    raise AssertionError("adapter must not be reached in this test")


def _empty_result(n: int) -> AdvanceResult:
    return AdvanceResult(
        token_ids=torch.zeros(n, 0, dtype=torch.int32),
        token_lengths=torch.zeros(n, dtype=torch.int32),
        row_status=torch.zeros(n, dtype=torch.int32),
    )


def _fixed_resolver(decode_fn: Any = None) -> Any:
    """A deterministic trivial resolver (tests inject; PORT-DEC-008's
    no-hardcoded-default rule applies to production wiring only)."""
    fn = decode_fn or decode_dense_masked

    def resolve(request: DecodeRequest) -> ResolvedDecode:
        return ResolvedDecode(arm="dense-eager", decode_fn=fn)

    return resolve


class _RecordingResolver:
    """Records every DecodeRequest (the exactly-once-per-bucket pin)."""

    def __init__(self, decode_fn: Any = None) -> None:
        self.requests: list[DecodeRequest] = []
        self._fn = decode_fn or decode_dense_masked

    def __call__(self, request: DecodeRequest) -> ResolvedDecode:
        self.requests.append(request)
        return ResolvedDecode(arm="dense-eager", decode_fn=self._fn)


class _CommitTicket:
    def __init__(self, sink: "_CommitRecorder") -> None:
        self._sink = sink

    def stage(self, row_status: torch.Tensor) -> None:
        self._sink.staged.append(row_status.clone())
        candidates = list(self._sink.plans[-1].records)
        self._sink.published.append([r for r in candidates if int(row_status[r.row]) == 0])

    def cancel(self) -> None:
        self._sink.cancels += 1


class _CommitRecorder:
    """Composite status/capture reservation test double."""

    def __init__(self, *, fail_reserve: bool = False) -> None:
        self.plans: list[Any] = []
        self.staged: list[torch.Tensor] = []
        self.published: list[list[Any]] = []
        self.cancels = 0
        self._fail = fail_reserve

    def reserve(self, plan: Any) -> _CommitTicket:
        self.plans.append(plan)
        if self._fail:
            raise RuntimeError("commit sink at capacity")
        return _CommitTicket(self)


def _call(
    core: Any,
    pools: Pools,
    input_ids: torch.Tensor,
    inputs_embeds: torch.Tensor,
    plan: RowPlan,
    adapter: EmissionAdapter | None = None,
    *,
    resolver: Any = None,
    commit_sink: Any = None,
    capture: bool = False,
    graph_covers_decode: bool = False,
) -> torch.Tensor:
    if adapter is None:
        adapter = make_mrv1_adapter(
            hidden_size=CARRIER_HIDDEN,
            park_id=PARK_ID,
            blank_id=core.blank_id,
        )
    return advance_model_rows(
        core,
        input_ids,
        inputs_embeds,
        plan,
        adapter=adapter,
        decode_resolver=resolver if resolver is not None else _fixed_resolver(),
        placeholder_id=PLACEHOLDER_ID,
        park_id=PARK_ID,
        commit_sink=commit_sink,
        capture=capture,
        graph_covers_decode=graph_covers_decode,
        **pools,
    )


def _set_replay_book(
    pools: Pools,
    block: int,
    *,
    queue: list[int],
    head: int,
    expected: int,
) -> None:
    """A VALID mid-replay book under the adapter convention: the burst
    is in the queue, ``head`` labels already emitted (the CHUNK step
    emits the first label and sets head=1), pending-echo armed with
    ``expected`` — the most recently emitted label, awaiting its echo.
    """
    book = pools["book_pool"]
    qp = pools["queue_pool"]
    for i, label in enumerate(queue):
        qp[block, i] = label
    book[block, _BOOK["queue_head"]] = head
    book[block, _BOOK["queue_length"]] = len(queue)
    book[block, _BOOK["last_label"]] = queue[-1]
    book[block, _BOOK["pending_echo"]] = 1
    book[block, _BOOK["expected_label"]] = expected


def _set_drained_book(pools: Pools, block: int, *, blank: int) -> None:
    """A legally parked session: queue drained, echo cleared."""
    book = pools["book_pool"]
    book[block, _BOOK["queue_head"]] = 0
    book[block, _BOOK["queue_length"]] = 0
    book[block, _BOOK["last_label"]] = blank
    book[block, _BOOK["pending_echo"]] = 0
    book[block, _BOOK["expected_label"]] = 0


# ---- PORT-STATE-007: whole-call structural preflight ---------------------
# Structural-reject tests use the refuse-adapter: rejection happens
# before classification, so the adapter must never run. Preflight is
# HOST-side over the plan's CPU authority tensors (design §Phase-6c
# transaction seams) — it needs no device synchronization to raise.


def test_advance_model_rows_rejects_wrong_row_count() -> None:
    # @spec PORT-STATE-007
    core = _tiny_core()
    pools = _sentinel_pools()
    before = _clone_pools(pools)
    plan = _plan(decodes=[1, 2])
    input_ids = torch.tensor([PARK_ID, PARK_ID], dtype=torch.long)
    inputs_embeds = torch.zeros(3, CARRIER_HIDDEN)  # 3 rows vs plan's 2
    with pytest.raises(ValueError):
        _call(core, pools, input_ids, inputs_embeds, plan, _refuse_adapter)
    _assert_pools_equal(pools, before)


def test_advance_model_rows_rejects_null_index_among_real_rows() -> None:
    # @spec PORT-STATE-007
    core = _tiny_core()
    pools = _sentinel_pools()
    before = _clone_pools(pools)
    plan = _plan(decodes=[1, NULL_INDEX])  # null id claimed as real
    input_ids = torch.tensor([PARK_ID, PARK_ID], dtype=torch.long)
    inputs_embeds = torch.zeros(2, CARRIER_HIDDEN)
    with pytest.raises(ValueError):
        _call(core, pools, input_ids, inputs_embeds, plan, _refuse_adapter)
    _assert_pools_equal(pools, before)


def test_advance_model_rows_rejects_out_of_range_index() -> None:
    # @spec PORT-STATE-007
    core = _tiny_core()
    pools = _sentinel_pools(num_blocks=3)
    before = _clone_pools(pools)
    plan = _plan(decodes=[1, 99], num_pool_blocks=3, live=[1, 2, 99])
    input_ids = torch.tensor([PARK_ID, PARK_ID], dtype=torch.long)
    inputs_embeds = torch.zeros(2, CARRIER_HIDDEN)
    with pytest.raises(ValueError):
        _call(core, pools, input_ids, inputs_embeds, plan, _refuse_adapter)
    _assert_pools_equal(pools, before)


def test_advance_model_rows_rejects_valid_but_non_live_index() -> None:
    # @spec PORT-STATE-007
    # Block 2 exists in the pool but is not allocated to any resident
    # session — liveness comes from plan.live_block_ids (the session
    # registry's allocated-block set), never from page contents.
    core = _tiny_core()
    pools = _sentinel_pools(num_blocks=5)
    before = _clone_pools(pools)
    plan = _plan(decodes=[2], num_pool_blocks=5, live=[1, 3])
    input_ids = torch.tensor([PARK_ID], dtype=torch.long)
    inputs_embeds = torch.zeros(1, CARRIER_HIDDEN)
    with pytest.raises(ValueError):
        _call(core, pools, input_ids, inputs_embeds, plan, _refuse_adapter)
    _assert_pools_equal(pools, before)


def test_advance_model_rows_rejects_duplicate_across_composition() -> None:
    # @spec PORT-STATE-007
    # The same block appearing once as a decode row and once as a
    # prefill row — only visible AFTER decode-then-prefill composition,
    # which is exactly why the transaction owns it.
    core = _tiny_core()
    pools = _sentinel_pools()
    before = _clone_pools(pools)
    plan = _plan(decodes=[1], prefills=[1])
    input_ids = torch.tensor([PARK_ID, PLACEHOLDER_ID], dtype=torch.long)
    inputs_embeds = torch.zeros(2, CARRIER_HIDDEN)
    with pytest.raises(ValueError):
        _call(core, pools, input_ids, inputs_embeds, plan, _refuse_adapter)
    _assert_pools_equal(pools, before)


def test_advance_model_rows_rejects_extra_speculative_column() -> None:
    # @spec PORT-STATE-007
    # A (rows, 2) decode index tensor: extra columns are a
    # speculative-decode configuration error, never values to flatten.
    core = _tiny_core()
    pools = _sentinel_pools()
    before = _clone_pools(pools)
    plan = _plan(decodes=[1, 2], decode_columns=2)
    input_ids = torch.tensor([PARK_ID, PARK_ID], dtype=torch.long)
    inputs_embeds = torch.zeros(2, CARRIER_HIDDEN)
    with pytest.raises(ValueError):
        _call(core, pools, input_ids, inputs_embeds, plan, _refuse_adapter)
    _assert_pools_equal(pools, before)


def test_advance_model_rows_rejects_live_index_in_padding_position() -> None:
    # @spec PORT-STATE-007
    # A graph-padding row (beyond num_decodes) carrying a live block id
    # is a real/padding mismatch — distinct from the null-among-real
    # defect above (there a REAL row carries null; here a PADDING row
    # carries a live id).
    core = _tiny_core()
    pools = _sentinel_pools()
    before = _clone_pools(pools)
    plan = _plan(decodes=[1], pad_decodes_to=2, padding_value=2)
    input_ids = torch.tensor([PARK_ID], dtype=torch.long)
    inputs_embeds = torch.zeros(1, CARRIER_HIDDEN)
    with pytest.raises(ValueError):
        _call(core, pools, input_ids, inputs_embeds, plan, _refuse_adapter)
    _assert_pools_equal(pools, before)


def test_advance_model_rows_requires_a_resolver_before_state_reads() -> None:
    # @spec PORT-DEC-008
    # No hardcoded decode default: an explicit None resolver fails the
    # call before any resident read (pools sentinel-proven untouched).
    core = _tiny_core()
    pools = _sentinel_pools()
    before = _clone_pools(pools)
    plan = _plan(decodes=[1])
    with pytest.raises(ValueError):
        _call(
            core,
            pools,
            torch.tensor([PARK_ID], dtype=torch.long),
            torch.zeros(1, CARRIER_HIDDEN),
            plan,
            _refuse_adapter,
            resolver=False,  # explicit non-callable sentinel
        )
    _assert_pools_equal(pools, before)


def test_advance_model_rows_accepts_legal_graph_padding() -> None:
    # @spec PORT-STATE-007
    # The same call SHAPE as the mismatch cases, but structurally legal:
    # padding rows carry the null id and num_decodes excludes them.
    # This is the accept half that makes the reject tests non-vacuous.
    core = _tiny_core()
    pools = _fresh_pools()
    _set_replay_book(pools, 1, queue=[3, 5], head=1, expected=3)
    plan = _plan(decodes=[1], pad_decodes_to=2, padding_value=NULL_INDEX)
    input_ids = torch.tensor([3], dtype=torch.long)  # the correct echo
    out = _call(core, pools, input_ids, torch.zeros(1, CARRIER_HIDDEN), plan)
    assert out.shape == (1, CARRIER_HIDDEN)
    assert int(read_decision_carrier(out)[0]) == 5  # next queued label


def test_graph_covered_outer_call_rejects_before_resolution() -> None:
    # @spec PORT-DEC-008 — legal padding metadata does not imply that
    # the outer transaction has exact execution padding; graph-covered
    # invocation remains fail-closed until that separate slice lands.
    core = _tiny_core()
    pools = _sentinel_pools()
    before = _clone_pools(pools)
    resolver = _RecordingResolver()
    with pytest.raises(ValueError, match="graph-covered"):
        _call(
            core,
            pools,
            torch.tensor([PLACEHOLDER_ID]),
            torch.zeros(1, CARRIER_HIDDEN),
            _plan(prefills=[1], execution_tier=1),
            _refuse_adapter,
            resolver=resolver,
            graph_covers_decode=True,
        )
    assert resolver.requests == []
    _assert_pools_equal(pools, before)


# ---- PORT-ADV-003: decode-then-prefill composition ------------------------


def test_composition_orders_decode_rows_before_prefill_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # @spec PORT-ADV-003
    # A resumed decode session (block 2, poisoned-distinct h) and a
    # fresh prefill (block 1) in one batch: the gathered state handed
    # to advance_session must be decode-then-prefill — row 0 carries
    # block 2's resumed h, row 1 the fresh zero h.
    seen: dict[str, torch.Tensor] = {}

    def _recorder(
        _core: Any,
        batch: ChunkBatch,
        state: SessionStateBatch,
        **_kw: Any,
    ) -> AdvanceResult:
        seen["h"] = state.h.clone()
        seen["seq"] = batch.chunk_sequence.clone()
        return _empty_result(batch.samples.shape[0])

    monkeypatch.setattr(advance_mod, "advance_session", _recorder)
    core = _tiny_core()
    pools = _fresh_pools()
    pools["h_pool"][2].fill_(0.5)  # the resumed session's signature
    pools["book_pool"][2, _BOOK["last_label"]] = core.blank_id
    pools["frontend_counter_pool"][2, _CTR["expected_chunk_sequence"]] = 3
    plan = _plan(decodes=[2], prefills=[1], chunk=[True, True])
    torch.manual_seed(5)
    samples = torch.randn(REG_SAMPLES) * 0.01
    embeds = torch.stack(
        [
            _envelope(samples, final=False, seq=3),
            _envelope(samples, final=False, seq=0),
        ]
    )
    input_ids = torch.tensor([PLACEHOLDER_ID, PLACEHOLDER_ID], dtype=torch.long)
    _call(core, pools, input_ids, embeds, plan)
    assert torch.all(seen["h"][0] == 0.5)  # decode row first
    assert torch.count_nonzero(seen["h"][1]) == 0  # fresh prefill after
    assert seen["seq"].tolist() == [3, 0]


def test_mixed_geometries_bucket_and_resolve_per_geometry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # @spec PORT-PERF-001 / PORT-DEC-008
    # Two CHUNK rows at different admitted geometries: one
    # advance_session call per geometry bucket, each with its own
    # geometry kwarg and membership; the resolver is invoked exactly
    # once per bucket with ready_decode_buckets == 2. Deadlines oppose
    # manifest order, proving final geometry resolves/executes first.
    calls: list[dict[str, Any]] = []

    def _recorder(
        _core: Any,
        batch: ChunkBatch,
        _state: SessionStateBatch,
        **kw: Any,
    ) -> AdvanceResult:
        calls.append(
            {
                "rows": batch.samples.shape[0],
                "geometry": kw.get("geometry"),
            }
        )
        return _empty_result(batch.samples.shape[0])

    monkeypatch.setattr(advance_mod, "advance_session", _recorder)
    core = _tiny_core()
    pools = _fresh_pools(num_blocks=4)
    resolver = _RecordingResolver()
    plan = _plan(
        prefills=[1, 2],
        num_pool_blocks=4,
        geometries=[GEOM_REG, GEOM_FINAL],
        deadlines=[200, 100],
    )
    torch.manual_seed(5)
    reg = _envelope(
        torch.randn(REG_SAMPLES) * 0.01,
        final=False,
        seq=0,
        geometry=GEOM_REG,
    )
    fin = _envelope(
        torch.randn(FINAL_SAMPLES) * 0.01,
        final=True,
        seq=0,
        geometry=GEOM_FINAL,
    )
    input_ids = torch.tensor([PLACEHOLDER_ID, PLACEHOLDER_ID], dtype=torch.long)
    _call(
        core,
        pools,
        input_ids,
        torch.stack([reg, fin]),
        plan,
        resolver=resolver,
    )
    assert [c["geometry"] for c in calls] == [GEOM_FINAL, GEOM_REG]
    assert all(c["rows"] == 1 for c in calls)
    assert len(resolver.requests) == 2  # exactly once per bucket
    assert [r.geometry for r in resolver.requests] == [GEOM_FINAL, GEOM_REG]
    assert all(r.ready_decode_buckets == 2 for r in resolver.requests)
    assert all(r.execution_batch_size == 1 for r in resolver.requests)


# ---- PORT-ADV-001: advance_session is CHUNK-only -------------------------


def test_replay_only_batch_never_calls_advance_session(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # @spec PORT-ADV-001
    def _recorder(*_args: Any, **_kwargs: Any) -> None:
        raise AssertionError("advance_session must not run for REPLAY rows")

    monkeypatch.setattr(advance_mod, "advance_session", _recorder)
    core = _tiny_core()
    pools = _fresh_pools()
    # A VALID mid-replay book: burst [3, 5], label 3 emitted (head=1),
    # echo of label 3 armed.
    _set_replay_book(pools, 1, queue=[3, 5], head=1, expected=3)
    plan = _plan(decodes=[1])
    _call(
        core,
        pools,
        torch.tensor([3], dtype=torch.long),  # the correct echo
        torch.zeros(1, CARRIER_HIDDEN),
        plan,
    )


def test_flush_only_batch_never_calls_advance_session(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # @spec PORT-ADV-001 / PORT-DEC-009
    def _recorder(*_args: Any, **_kwargs: Any) -> None:
        raise AssertionError("advance_session must not run for FLUSH rows")

    monkeypatch.setattr(advance_mod, "advance_session", _recorder)
    core = _tiny_core()
    pools = _fresh_pools()
    # A VALID flush row: queue drained, no pending echo, the session's
    # frontend finalized, and the engine's final sentinel fed back
    # (the step after the drain emitted park) — never an arbitrary
    # token.
    _set_drained_book(pools, 1, blank=core.blank_id)
    pools["frontend_counter_pool"][1, _CTR["finalized"]] = 1
    pools["frontend_counter_pool"][1, _CTR["expected_chunk_sequence"]] = 1
    plan = _plan(decodes=[1])
    out = _call(
        core,
        pools,
        torch.tensor([PARK_ID], dtype=torch.long),
        torch.zeros(1, CARRIER_HIDDEN),
        plan,
    )
    assert int(read_decision_carrier(out)[0]) == PARK_ID


def test_mixed_batch_calls_advance_session_with_only_chunk_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # @spec PORT-ADV-001
    seen: dict[str, int] = {}

    def _recorder(
        _core: Any,
        batch: ChunkBatch,
        _state: SessionStateBatch,
        **_kw: Any,
    ) -> AdvanceResult:
        seen["n_chunk_rows"] = batch.samples.shape[0]
        return _empty_result(batch.samples.shape[0])

    monkeypatch.setattr(advance_mod, "advance_session", _recorder)
    core = _tiny_core()
    pools = _fresh_pools()
    _set_replay_book(pools, 2, queue=[7], head=1, expected=7)  # REPLAY
    torch.manual_seed(5)
    samples = torch.randn(REG_SAMPLES) * 0.01
    embeds = torch.stack(
        [
            torch.zeros(CARRIER_HIDDEN),
            _envelope(samples, final=False, seq=0),
        ]
    )
    input_ids = torch.tensor([7, PLACEHOLDER_ID], dtype=torch.long)
    plan = _plan(decodes=[2], prefills=[1])
    _call(core, pools, input_ids, embeds, plan)
    assert seen["n_chunk_rows"] == 1  # only the one CHUNK row gathered


def test_advance_session_result_contract() -> None:
    # @spec PORT-ADV-001 / PORT-ADV-004 / PORT-INT-002
    # The direct transition contract at the CURRENT seam
    # (geometry= / decode_fn= / capture=True). A session-first
    # FINAL-TAIL chunk at the 320 ms geometry: final semantics equal
    # the whole-signal featurizer, so the reference is exact.
    core = _tiny_core()
    torch.manual_seed(5)
    samples = torch.randn(FINAL_SAMPLES) * 0.01
    enc_frames = _reference_encoder_frames(core, samples)
    batch = ChunkBatch(
        samples=samples.unsqueeze(0),
        valid_samples=torch.tensor([FINAL_SAMPLES], dtype=torch.long),
        geometry_id=torch.full((1,), GEOM_FINAL, dtype=torch.long),
        final_tail=torch.tensor([True]),
        prompt_index=torch.zeros(1, dtype=torch.long),
        chunk_sequence=torch.zeros(1, dtype=torch.long),
    )
    state = SessionStateBatch(
        raw_tail=torch.zeros(1, RAW_TAIL),
        mel_tail=torch.zeros(1, FEAT, 9),
        frontend_counters=torch.zeros(1, CTR_WIDTH, dtype=torch.int64),
        channel=[torch.zeros(1, WINDOW, D_MODEL) for _ in range(N_LAYERS)],
        window_valid=[torch.zeros(1, 1, dtype=torch.int32) for _ in range(N_LAYERS)],
        time=[torch.zeros(1, D_MODEL, KERNEL - 1) for _ in range(N_LAYERS)],
        h=torch.zeros(1, 2, 16),
        c=torch.zeros(1, 2, 16),
        last_label=torch.full((1,), core.blank_id, dtype=torch.long),
    )
    result = advance_session(
        core,
        batch,
        state,
        geometry=GEOM_FINAL,
        decode_fn=decode_dense_masked,
        capture=True,
    )
    assert isinstance(result, AdvanceResult)
    # GPU-resident result: padded token_ids + per-row lengths + the
    # per-row status, never Python lists (PORT-PERF-001/ADV-004).
    assert result.row_status is not None
    assert int(result.row_status[0]) == 0
    n_tok = int(result.token_lengths[0])
    burst = result.token_ids[0, :n_tok]
    assert result.token_ids.dtype == torch.int32
    assert not bool((burst == core.blank_id).any())
    # PORT-INT-002: bounded by valid encoder frames × max symbols per
    # step — NOT by the attention window.
    assert n_tok <= enc_frames * MAX_SYMBOLS_PER_STEP
    caps = result.captures
    assert isinstance(caps, PreparedCaptures)
    # Captures are PADDED to the bucket's host-derived widths with
    # exact logical lengths (PORT-HOOK-001).
    assert caps.frontend_mel.shape[0] == 1
    assert int(caps.mel_lengths[0]) == FINAL_SAMPLES // 160
    assert int(caps.mel_lengths[0]) <= caps.frontend_mel.shape[2]
    assert caps.encoder_raw.shape == caps.encoder_conditioned.shape
    assert int(caps.encoder_lengths[0]) == enc_frames
    assert int(caps.encoder_lengths[0]) <= caps.encoder_raw.shape[1]
    # The transition advanced the frontend state it was handed.
    ctr = state.frontend_counters[0]
    assert int(ctr[_CTR["total_valid_samples"]]) == FINAL_SAMPLES
    assert int(ctr[_CTR["committed_mel_frames"]]) > 0
    assert int(ctr[_CTR["finalized"]]) == 1


# ---- PORT-ADV-003: fresh-session init + the echo row tier -----------------


def test_fresh_session_rows_ignore_a_recycled_blocks_poison() -> None:
    # @spec PORT-ADV-003 / PORT-STATE-003
    # A block carrying a prior session's garbage in EVERY pool, but
    # marked fresh by scheduler metadata (~has_initial_states_p), must
    # decode exactly as a clean block — freshness never reads page
    # contents (so there is no zero-length-slot sentinel to trip on).
    core = _tiny_core()
    torch.manual_seed(5)
    samples = torch.randn(FINAL_SAMPLES) * 0.01
    carrier = _envelope(samples, final=True, seq=0, geometry=GEOM_FINAL).unsqueeze(0)
    input_ids = torch.tensor([PLACEHOLDER_ID], dtype=torch.long)
    plan = _plan(prefills=[1], has_initial=[False], geometries=[GEOM_FINAL])

    clean = _fresh_pools()
    out_clean = _call(core, clean, input_ids, carrier, plan)

    dirty = _sentinel_pools(value=3.14)  # poison EVERYTHING, book incl.
    out_dirty = _call(core, dirty, input_ids, carrier, plan)
    torch.testing.assert_close(read_decision_carrier(out_dirty), read_decision_carrier(out_clean))


def test_corrupted_echo_masks_only_that_row_and_peers_advance() -> None:
    # @spec PORT-DEC-007 (amended 2026-07-19) / PORT-STATE-008
    # The row tier: after trusted structural preflight, an echo
    # mismatch is evidence about ONE session's resident state. Row 0
    # replays with the CORRECT echo and must advance normally; row 1's
    # corrupted echo masks that row — book bit-identical, park-only
    # emission (never client text) — and surfaces ECHO_MISMATCH
    # through the single staged status handoff.
    core = _tiny_core()
    pools = _fresh_pools()
    _set_replay_book(pools, 1, queue=[3, 5], head=1, expected=3)
    _set_replay_book(pools, 2, queue=[4, 6], head=1, expected=4)
    before = _clone_pools(pools)
    plan = _plan(decodes=[1, 2])
    status = _CommitRecorder()
    out = _call(
        core,
        pools,
        torch.tensor([3, 8], dtype=torch.long),  # row 1: wrong echo
        torch.zeros(2, CARRIER_HIDDEN),
        plan,
        commit_sink=status,
    )
    # Row 0 advanced: next queued label emitted, head past it.
    assert int(read_decision_carrier(out)[0]) == 5
    assert int(pools["book_pool"][1, _BOOK["queue_head"]]) == 2
    assert int(pools["book_pool"][1, _BOOK["expected_label"]]) == 5
    # Row 1 masked: park only, book/queue bit-identical.
    assert int(read_decision_carrier(out)[1]) == PARK_ID
    torch.testing.assert_close(pools["book_pool"][2], before["book_pool"][2], rtol=0, atol=0)
    torch.testing.assert_close(pools["queue_pool"][2], before["queue_pool"][2], rtol=0, atol=0)
    # Exactly one staged status: row 0 clean, row 1 flagged.
    assert len(status.staged) == 1
    staged = status.staged[0]
    assert int(staged[0]) == 0
    assert int(staged[1]) & ROW_STATUS_ECHO_MISMATCH


def test_absurd_echo_id_cannot_reach_unsafe_indexing() -> None:
    # @spec PORT-DEC-007 (amended) — echoed values are never used for
    # addressing: an out-of-range echoed id must produce the same
    # masked park outcome as any mismatch, not an index fault.
    core = _tiny_core()
    pools = _fresh_pools()
    _set_replay_book(pools, 1, queue=[3, 5], head=1, expected=3)
    before = _clone_pools(pools)
    plan = _plan(decodes=[1])
    out = _call(
        core,
        pools,
        torch.tensor([10**6], dtype=torch.long),
        torch.zeros(1, CARRIER_HIDDEN),
        plan,
    )
    assert int(read_decision_carrier(out)[0]) == PARK_ID
    _assert_pools_equal(pools, before)


def test_chunk_on_undrained_queue_is_masked_and_reported() -> None:
    # @spec PORT-SESS-001 (one in-flight CHUNK) — a new CHUNK arriving
    # while the replay queue still holds labels is a protocol defect:
    # the row masks (park, no state mutation) and reports
    # QUEUE_NOT_DRAINED through the status handoff.
    core = _tiny_core()
    pools = _fresh_pools()
    _set_replay_book(pools, 1, queue=[3, 5], head=1, expected=3)
    before = _clone_pools(pools)
    torch.manual_seed(5)
    samples = torch.randn(REG_SAMPLES) * 0.01
    carrier = _envelope(samples, final=False, seq=0).unsqueeze(0)
    plan = _plan(decodes=[1], chunk=[True])
    status = _CommitRecorder()
    out = _call(
        core,
        pools,
        torch.tensor([PLACEHOLDER_ID], dtype=torch.long),
        carrier,
        plan,
        commit_sink=status,
    )
    assert int(read_decision_carrier(out)[0]) == PARK_ID
    _assert_pools_equal(pools, before)
    assert len(status.staged) == 1
    assert int(status.staged[0][0]) & ROW_STATUS_QUEUE_NOT_DRAINED


# ---- PORT-STATE-008: no partial scatter on compute failure ----------------


def test_resident_pools_unchanged_when_bucket_compute_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # @spec PORT-STATE-008
    # An UNEXPECTED exception in bucket compute is a whole-call
    # failure before any scatter (scratch-first): every resident pool
    # is untouched and the exception propagates (a CUDA context
    # failure is worker-fatal by propagation, never row suppression).
    def _boom(*_args: Any, **_kwargs: Any) -> AdvanceResult:
        raise RuntimeError("simulated mid-bucket compute failure")

    monkeypatch.setattr(advance_mod, "advance_session", _boom)
    core = _tiny_core()
    pools = _fresh_pools()
    before = _clone_pools(pools)
    torch.manual_seed(5)
    samples = torch.randn(REG_SAMPLES) * 0.01
    carrier = _envelope(samples, final=False, seq=0).unsqueeze(0)
    plan = _plan(prefills=[1])
    with pytest.raises(RuntimeError):
        _call(
            core,
            pools,
            torch.tensor([PLACEHOLDER_ID], dtype=torch.long),
            carrier,
            plan,
        )
    _assert_pools_equal(pools, before)


# ---- PORT-HOOK-001: capture reservation ownership -------------------------


def test_capture_reservation_failure_is_pre_commit_fatal() -> None:
    # @spec PORT-HOOK-001
    # Reservation happens BEFORE any resident commit: a sink that
    # cannot reserve fails the call with every pool untouched.
    core = _tiny_core()
    pools = _fresh_pools()
    before = _clone_pools(pools)
    torch.manual_seed(5)
    samples = torch.randn(FINAL_SAMPLES) * 0.01
    carrier = _envelope(samples, final=True, seq=0, geometry=GEOM_FINAL).unsqueeze(0)
    plan = _plan(prefills=[1], geometries=[GEOM_FINAL])
    sink = _CommitRecorder(fail_reserve=True)
    with pytest.raises(RuntimeError):
        _call(
            core,
            pools,
            torch.tensor([PLACEHOLDER_ID], dtype=torch.long),
            carrier,
            plan,
            commit_sink=sink,
            capture=True,
        )
    _assert_pools_equal(pools, before)
    assert sink.published == []


def test_capture_publishes_prepared_records_after_commit() -> None:
    # @spec PORT-HOOK-001
    # With capture enabled the transaction reserves the exact bounded
    # footprint, commits, then publishes the prepared records into the
    # reserved slots. Records ride with device identity + status; a
    # zero-frame/masked row's record survives with zero valid length
    # and its status view, and the consumer filters at ITS readback.
    core = _tiny_core()
    pools = _fresh_pools()
    torch.manual_seed(5)
    samples = torch.randn(FINAL_SAMPLES) * 0.01
    carrier = _envelope(samples, final=True, seq=0, geometry=GEOM_FINAL).unsqueeze(0)
    plan = _plan(prefills=[1], geometries=[GEOM_FINAL])
    sink = _CommitRecorder()
    _call(
        core,
        pools,
        torch.tensor([PLACEHOLDER_ID], dtype=torch.long),
        carrier,
        plan,
        commit_sink=sink,
        capture=True,
    )
    assert len(sink.plans) == 1
    assert sink.plans[0].capture.rows == 1
    assert sink.plans[0].capture.payload_bytes > 0
    assert len(sink.published) == 1 and len(sink.published[0]) == 1
    record = sink.published[0][0]
    assert record.request_id == "req-0"
    assert record.block_id == 1
    assert int(record.mel_length) == FINAL_SAMPLES // 160
    assert record.frontend_mel.shape[0] == FEAT
    assert sink.cancels == 0


def test_capture_off_means_no_capture_work() -> None:
    # @spec PORT-HOOK-001 — capture=False disables capture
    # creation entirely; the sink recorder proves no reserve/publish
    # by never existing, and the call still succeeds.
    core = _tiny_core()
    pools = _fresh_pools()
    torch.manual_seed(5)
    samples = torch.randn(FINAL_SAMPLES) * 0.01
    carrier = _envelope(samples, final=True, seq=0, geometry=GEOM_FINAL).unsqueeze(0)
    plan = _plan(prefills=[1], geometries=[GEOM_FINAL])
    out = _call(
        core,
        pools,
        torch.tensor([PLACEHOLDER_ID], dtype=torch.long),
        carrier,
        plan,
        commit_sink=None,
    )
    assert out.shape == (1, CARRIER_HIDDEN)


# ---- correction-pass hardening pins (mirrored from the local twin) --------


def test_second_bucket_failure_leaves_pools_untouched(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # @spec PORT-STATE-008 / PORT-ADV-003 (as amended)
    # A failure in bucket 2 AFTER bucket 1 computed is a whole-call
    # failure before any scatter (scratch-first).
    real = advance_mod.advance_session
    calls = {"n": 0}

    def flaky(*args: Any, **kwargs: Any) -> Any:
        calls["n"] += 1
        if calls["n"] == 2:
            raise RuntimeError("second bucket exploded")
        return real(*args, **kwargs)

    monkeypatch.setattr(advance_mod, "advance_session", flaky)
    core = _tiny_core()
    pools = _fresh_pools(num_blocks=4)
    before = _clone_pools(pools)
    torch.manual_seed(5)
    reg = _envelope(
        torch.randn(REG_SAMPLES) * 0.01,
        final=False,
        seq=0,
        geometry=GEOM_REG,
    )
    fin = _envelope(
        torch.randn(FINAL_SAMPLES) * 0.01,
        final=True,
        seq=0,
        geometry=GEOM_FINAL,
    )
    plan = _plan(
        prefills=[1, 2],
        num_pool_blocks=4,
        geometries=[GEOM_REG, GEOM_FINAL],
    )
    with pytest.raises(RuntimeError):
        _call(
            core,
            pools,
            torch.tensor([PLACEHOLDER_ID, PLACEHOLDER_ID], dtype=torch.long),
            torch.stack([reg, fin]),
            plan,
        )
    assert calls["n"] == 2
    _assert_pools_equal(pools, before)


def test_composite_reserve_failure_is_precommit_fatal() -> None:
    # @spec PORT-ADV-003 / PORT-HOOK-001: atomic reserve failure
    # creates no ticket and leaves all resident state untouched.
    core = _tiny_core()
    pools = _fresh_pools()
    before = _clone_pools(pools)
    torch.manual_seed(5)
    carrier = _envelope(
        torch.randn(FINAL_SAMPLES) * 0.01,
        final=True,
        seq=0,
        geometry=GEOM_FINAL,
    ).unsqueeze(0)
    plan = _plan(prefills=[1], geometries=[GEOM_FINAL])
    cap_sink = _CommitRecorder(fail_reserve=True)
    with pytest.raises(RuntimeError):
        _call(
            core,
            pools,
            torch.tensor([PLACEHOLDER_ID], dtype=torch.long),
            carrier,
            plan,
            commit_sink=cap_sink,
            capture=True,
        )
    _assert_pools_equal(pools, before)
    assert cap_sink.cancels == 0
    assert cap_sink.published == []


def test_wrong_but_valid_prompt_masks_and_admission_conditions() -> None:
    # @spec PORT-LID-003 (as amended): the envelope prompt is only a
    # device cross-check; a valid-but-different stamped prompt masks
    # the row instead of silently changing model output.
    core = _tiny_core()  # num_prompts == 4
    pools = _fresh_pools(num_blocks=4)
    torch.manual_seed(5)
    samples = torch.randn(FINAL_SAMPLES) * 0.01
    ok_row = _envelope(samples, final=True, seq=0, geometry=GEOM_FINAL, prompt=2)
    bad_row = _envelope(samples, final=True, seq=0, geometry=GEOM_FINAL, prompt=2)
    plan = _plan(
        prefills=[1, 2],
        num_pool_blocks=4,
        geometries=[GEOM_FINAL, GEOM_FINAL],
        prompts=[2, 1],  # admitted authority per row
    )
    status = _CommitRecorder()
    out = _call(
        core,
        pools,
        torch.tensor([PLACEHOLDER_ID, PLACEHOLDER_ID], dtype=torch.long),
        torch.stack([ok_row, bad_row]),
        plan,
        commit_sink=status,
    )
    decisions = read_decision_carrier(out).tolist()
    assert decisions[0] != PARK_ID
    assert decisions[1] == PARK_ID
    assert int(status.staged[0][0]) == 0
    assert int(status.staged[0][1]) & ROW_STATUS_PROMPT_MISMATCH


def test_persisted_prompt_mismatch_masks_only_that_row() -> None:
    # @spec PORT-LID-003 / PORT-STATE-008: the session book must agree
    # with the CPU registry authority before conditioning or commit.
    core = _tiny_core()
    pools = _fresh_pools(num_blocks=4)
    for block, prompt in ((1, 2), (2, 1)):
        pools["book_pool"][block, _BOOK["last_label"]] = core.blank_id
        pools["book_pool"][block, _BOOK["geometry"]] = GEOM_REG
        pools["book_pool"][block, _BOOK["prompt"]] = prompt
    before = _clone_pools(pools)
    torch.manual_seed(5)
    samples = torch.randn(REG_SAMPLES) * 0.01
    rows = torch.stack(
        [
            _envelope(samples, final=False, seq=0, prompt=1),
            _envelope(samples, final=False, seq=0, prompt=1),
        ]
    )
    sink = _CommitRecorder()
    out = _call(
        core,
        pools,
        torch.tensor([PLACEHOLDER_ID, PLACEHOLDER_ID]),
        rows,
        _plan(
            decodes=[1, 2],
            chunk=[True, True],
            prompts=[1, 1],
            num_pool_blocks=4,
        ),
        commit_sink=sink,
    )
    decisions = read_decision_carrier(out).tolist()
    assert decisions[0] == PARK_ID
    assert decisions[1] != PARK_ID
    assert int(sink.staged[0][0]) & ROW_STATUS_BOOK_IDENTITY
    for value, old in zip(pools.values(), before.values(), strict=True):
        values = value if isinstance(value, list) else [value]
        olds = old if isinstance(old, list) else [old]
        for tensor, original in zip(values, olds, strict=True):
            torch.testing.assert_close(tensor[1], original[1], rtol=0, atol=0)


# ---- PORT-ADV-001 / MRV1 emission: burst-then-park ------------------------


def test_advance_model_rows_emits_the_burst_then_parks() -> None:
    # @spec PORT-ADV-001 / PORT-DEC-002/003
    # The full MRV1 session arc on a session-first FINAL-TAIL chunk:
    # burst queued with echo state armed (head=1 past the emitted
    # first label), drained one label per step against the golden
    # reference, park once drained + finalized.
    core = _tiny_core(seed=40)  # three-distinct-label burst by design
    torch.manual_seed(5)
    samples = torch.randn(FINAL_SAMPLES) * 0.01
    expected = _reference_burst(core, samples, prompt_index=0)
    assert len(expected) >= 2 and len(set(expected)) >= 2, (
        "need a multi-label, multi-distinct burst to test drain order"
    )

    pools = _fresh_pools()
    plan = _plan(prefills=[1], geometries=[GEOM_FINAL])
    carrier = _envelope(samples, final=True, seq=0, geometry=GEOM_FINAL).unsqueeze(0)
    out = _call(
        core,
        pools,
        torch.tensor([PLACEHOLDER_ID], dtype=torch.long),
        carrier,
        plan,
    )
    emitted = [int(read_decision_carrier(out)[0])]
    book = pools["book_pool"]
    assert int(book[1, _BOOK["queue_length"]]) == len(expected)
    assert int(book[1, _BOOK["queue_head"]]) == 1
    assert int(book[1, _BOOK["pending_echo"]]) == 1
    assert int(book[1, _BOOK["expected_label"]]) == emitted[0]
    assert int(pools["frontend_counter_pool"][1, _CTR["finalized"]]) == 1

    # geometries must match what the CHUNK phase actually committed to
    # the book (GEOM_FINAL) — _plan's default (GEOM_REG) would trip
    # ROW_STATUS_BOOK_IDENTITY on every iteration and mask every
    # decision to park_id, which is what the pre-reseed fixture never
    # surfaced (its len(set(expected))>=2 precondition always failed
    # first, so this decode_plan was never actually exercised before).
    decode_plan = _plan(decodes=[1], geometries=[GEOM_FINAL])
    for _ in range(len(expected) + 1):
        if emitted[-1] == PARK_ID:
            break
        out = _call(
            core,
            pools,
            torch.tensor([emitted[-1]], dtype=torch.long),
            torch.zeros(1, CARRIER_HIDDEN),
            decode_plan,
        )
        emitted.append(int(read_decision_carrier(out)[0]))

    assert emitted[:-1] == expected
    assert emitted[-1] == PARK_ID
    assert int(book[1, _BOOK["pending_echo"]]) == 0


# ---- PORT-STATE-001: page lifecycle by kind -------------------------------


def test_persist_across_session_park_by_kind() -> None:
    # @spec PORT-STATE-001
    # The reconciled design: EVERY state page persists across a legal
    # park — the replay/session-book page too (it also carries
    # last-label, prompt, geometry, and echo state; port-design.md
    # §Session State Pages), and the frontend-continuity pair (fp32
    # buffers + int64 counters, physically split for typed-view
    # alignment) exists as first-class state layers.
    from vllm_omni.model_executor.models.nemotron_asr.state_layers import (
        ConvCachePage,
        FrontendBufferPage,
        FrontendCounterPage,
        LSTMStatePage,
        ReplayQueuePage,
        WindowCachePage,
    )

    window = WindowCachePage(
        prefix="encoder.layers.0.window",
        window=WINDOW,
        d_model=D_MODEL,
        policy=FP32_BRINGUP,
    )
    conv = ConvCachePage(
        prefix="encoder.layers.0.conv",
        d_model=D_MODEL,
        kernel=KERNEL,
        policy=FP32_BRINGUP,
    )
    lstm = LSTMStatePage(
        prefix="predictor.layers.0.lstm_state",
        pred_rnn_layers=2,
        pred_hidden=16,
        policy=FP32_BRINGUP,
    )
    replay = ReplayQueuePage(
        prefix="decode.layers.0.replay",
        max_symbols_per_step=10,
        max_frames_per_chunk=4,
        policy=FP32_BRINGUP,
    )
    frontend = FrontendBufferPage(
        prefix="frontend.layers.0.buffers",
        raw_tail=RAW_TAIL,
        n_mels=FEAT,
        policy=FP32_BRINGUP,
    )
    counters = FrontendCounterPage(
        prefix="frontend.layers.0.counters",
        policy=FP32_BRINGUP,
    )
    assert window.persist_across_session_park is True
    assert conv.persist_across_session_park is True
    assert lstm.persist_across_session_park is True
    assert replay.persist_across_session_park is True
    assert frontend.persist_across_session_park is True
    assert counters.persist_across_session_park is True


def test_carrier_width_covers_header_plus_largest_raw_cadence() -> None:
    # @spec PORT-INT-003 / PORT-REGIME-001
    # The raw-audio/control envelope replaced the packed-mel carrier:
    # the configuration-derived width is the header plus the largest
    # admitted raw cadence (1120 ms at 16 kHz), never a mel formula.
    from vllm_omni.model_executor.models.nemotron_asr.configuration_nemotron_asr import (  # noqa: E501
        NemotronASRConfig,
    )
    from vllm_omni.model_executor.models.nemotron_asr.manifests import (
        RAW_SAMPLES_PER_CHUNK,
    )

    needed = ENVELOPE_HEADER_SLOTS + max(RAW_SAMPLES_PER_CHUNK.values())
    assert needed == 17_927
    assert NemotronASRConfig().hidden_size >= needed, (
        f"hidden_size must cover the raw chunk envelope: need {needed}, config has {NemotronASRConfig().hidden_size}"
    )


def test_embed_input_ids_canonical_merge_places_carriers_and_zeros() -> None:
    # @spec PORT-INT-003
    from vllm_omni.model_executor.models.nemotron_asr.nemotron_asr import (
        NemotronASRForRNNT,
    )

    model = object.__new__(NemotronASRForRNNT)  # seam only; no __init__
    model.num_logits = 13090
    hidden = 8
    model.config = type("C", (), {"hidden_size": hidden})()
    input_ids = torch.tensor([500, 7, 501, 0], dtype=torch.long)
    is_mm = torch.tensor([True, False, True, False])
    mm = torch.stack([torch.full((hidden,), 1.0), torch.full((hidden,), 2.0)])
    out = model.embed_input_ids(input_ids, multimodal_embeddings=mm, is_multimodal=is_mm)
    assert out.shape == (4, hidden)
    assert torch.all(out[0] == 1.0) and torch.all(out[2] == 2.0)
    assert torch.count_nonzero(out[1]) == 0
    assert torch.count_nonzero(out[3]) == 0


# ---- PORT-REGIME-001 / PORT-INT-003: naming-lock source pins --------------

_NEMOTRON_ASR_DIR = Path(__file__).resolve().parents[4] / ("vllm_omni/model_executor/models/nemotron_asr")


def test_run_forward_step_is_removed_and_advance_model_rows_is_wired() -> None:
    # @spec PORT-REGIME-001
    # A source-level pin (Path.read_text — no imports needed): the P5-1
    # split has landed and the Task-7 parity gate passed, so the package
    # no longer defines run_forward_step (forward_step.py is deleted) and
    # nemotron_asr.py calls advance_model_rows. The xfail(strict=True)
    # mark was removed in that same change (Task 7); the read handles an
    # absent forward_step.py so this must never FileNotFoundError.
    forward_step_path = _NEMOTRON_ASR_DIR / "forward_step.py"
    forward_step_src = forward_step_path.read_text() if forward_step_path.exists() else ""
    model_src = (_NEMOTRON_ASR_DIR / "nemotron_asr.py").read_text()
    assert "def run_forward_step" not in forward_step_src
    assert "advance_model_rows" in model_src


def test_forward_routes_through_advance_model_rows() -> None:
    # @spec PORT-INT-003
    # Flipped from xfail at Task 5: forward now routes through the
    # shared transaction. forward_step.py served as the regression
    # oracle until the Task-7 parity gate passed, then was deleted
    # (the companion naming-lock test above flipped in that change).
    model_src = (_NEMOTRON_ASR_DIR / "nemotron_asr.py").read_text()
    assert "advance_model_rows(" in model_src
    assert "run_forward_step(" not in model_src
