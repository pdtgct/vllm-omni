# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Phase-5 tests-first: the advance seam split (ledger P5-1).

Pins the future ``advance_session`` / ``advance_model_rows`` contract
(``advance.py``) that replaces ``forward_step.py``'s
``run_forward_step`` in Phase 6. POD-TIER: importing
``vllm_omni.model_executor.models.nemotron_asr.*`` pulls the
``vllm_omni`` package, which pulls ``vllm`` — this file cannot be
collected on macOS and must be run on the pod venv (contrast
``test_manifests.py``, which is torch-free and loaded by file path).

Most tests here exercise a still-``NotImplementedError``-raising stub
and are EXPECTED TO FAIL until Phase 6 lands the real implementation
alongside ``forward_step.py``'s deletion — that failure is the
recorded tests-first evidence, not a bug in this file. The two
source-text pins (naming lock + forward wiring) are ``xfail(strict=
True)`` so they flip to a hard error if the split lands without
removing the mark, or if someone removes the mark without doing the
split.

``test_forward_step.py`` stays in place, UNCHANGED, this round — it
dies together with ``forward_step.py`` in the Phase-6 change that
lands this module's real bodies (ledger P5-1: a semantic split, never
a second legacy forward path).
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
    AdvanceResult,
    ChunkBatch,
    EmissionAdapter,
    PreparedCaptures,
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
SAMPLES = 3_840  # 24 mel frames at hop 160
CARRIER_HIDDEN = 4_096  # >= ENVELOPE_HEADER_SLOTS + SAMPLES
RAW_TAIL = 1_953  # pre_encode_cache(9) * hop(160) + n_fft(512) + 1
#: The reserved null block id (PORT-STATE-007) — never a live page.
NULL_INDEX = -1

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


def _tiny_core() -> Any:
    # Seed 7 in component order yields a deterministic cap-saturated
    # burst with two distinct labels in order — good drain-order test
    # data (test_forward_step.py's rationale, unchanged by the split).
    # Returns a duck-typed SimpleNamespace, not a real NemotronASRCore
    # (Any, matching test_forward_step.py's fixture idiom).
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

    torch.manual_seed(7)
    encoder = FastConformerEncoder(
        feat_in=FEAT, d_model=D_MODEL, d_ff=64, n_layers=N_LAYERS,
        n_heads=4, conv_kernel=KERNEL, subsampling_channels=16,
        att_context=(WINDOW, 1),
    )
    encoder.eval()
    predictor = Predictor(vocab_size=VOCAB, pred_hidden=16, pred_rnn_layers=2)
    joint = Joint(
        enc_hidden=D_MODEL, pred_hidden=16, joint_hidden=16, vocab_size=VOCAB
    )
    lid = PromptConditioner(enc_hidden=D_MODEL, num_prompts=4)
    featurizer = MelFeaturizer(
        filterbank=torch.rand(FEAT, 257) * 0.01,
        window=torch.hann_window(400),
    )
    return SimpleNamespace(
        encoder=encoder, lid=lid, predictor=predictor, joint=joint,
        featurizer=featurizer, blank_id=VOCAB,
    )


def _whole_signal_mel(core: Any, samples: torch.Tensor) -> torch.Tensor:
    """Featurize the whole signal — the final-tail reference semantics
    (a final-tail chunk applies ordinary centered-STFT boundary rules,
    identical to the whole-utterance featurizer)."""
    with torch.no_grad():
        mel, mel_len = core.featurizer(
            samples.unsqueeze(0), torch.tensor([samples.shape[0]])
        )
    trimmed: torch.Tensor = mel[:, :, : int(mel_len[0])]
    return trimmed


def _reference_burst(
    core: Any, samples: torch.Tensor, prompt_index: int
) -> list[int]:
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
        n_layers=N_LAYERS, batch=1, d_model=D_MODEL, left_context=WINDOW,
        conv_kernel=KERNEL, device=torch.device("cpu"),
    )
    with torch.no_grad():
        enc = stream_step(core.encoder, mel, caches, drop_extra=0)
        conditioned = core.lid(enc, prompt_index=prompt_index)
        state = DecodeState(
            h=torch.zeros(2, 1, 16),
            c=torch.zeros(2, 1, 16),
            last_label=torch.full((1,), core.blank_id),
        )
        labels, _ = greedy_decode_batch(
            conditioned, core.predictor, core.joint, state
        )
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
        n_layers=N_LAYERS, batch=1, d_model=D_MODEL, left_context=WINDOW,
        conv_kernel=KERNEL, device=torch.device("cpu"),
    )
    with torch.no_grad():
        enc = stream_step(core.encoder, mel, caches, drop_extra=0)
    return enc.shape[1]


def _fresh_pools(num_blocks: int = 2) -> Pools:
    return {
        "channel_pools": [
            torch.zeros(num_blocks, WINDOW, D_MODEL) for _ in range(N_LAYERS)
        ],
        "time_pools": [
            torch.zeros(num_blocks, D_MODEL, KERNEL - 1)
            for _ in range(N_LAYERS)
        ],
        "len_pools": [
            torch.zeros(num_blocks, 1) for _ in range(N_LAYERS)
        ],
        "h_pool": torch.zeros(num_blocks, 2, 16),
        "c_pool": torch.zeros(num_blocks, 2, 16),
        "queue_pool": torch.zeros(num_blocks, CAP),
        "book_pool": torch.zeros(num_blocks, BOOK_WIDTH),
        "frontend_raw_pool": torch.zeros(num_blocks, RAW_TAIL),
        "frontend_mel_pool": torch.zeros(num_blocks, FEAT, 9),
        "frontend_counter_pool": torch.zeros(
            num_blocks, CTR_WIDTH, dtype=torch.int64
        ),
    }


def _sentinel_pools(num_blocks: int = 2, value: float = 12345.0) -> Pools:
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
            val.fill_(int(value) if val.dtype == torch.int64 else value)
    return pools


def _clone_pools(pools: Pools) -> Pools:
    return {
        k: [t.clone() for t in v] if isinstance(v, list) else v.clone()
        for k, v in pools.items()
    }


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
    num_pool_blocks: int = 2,
    pad_decodes_to: int | None = None,
    padding_value: int = NULL_INDEX,
    decode_columns: int = 1,
    has_initial: list[bool] | None = None,
    live: list[int] | None = None,
    prompts: list[int] | None = None,
) -> RowPlan:
    """Build a RowPlan the way the scheduler would: decode indices as a
    ``(rows, K)`` tensor (real rows first, graph padding after),
    prefill indices flat, freshness/liveness from metadata."""
    decodes = decodes or []
    prefills = prefills or []
    num_decodes = len(decodes)
    padded = decodes + [padding_value] * (
        (pad_decodes_to or num_decodes) - num_decodes
    )
    d = torch.tensor(
        [[i] * decode_columns for i in padded], dtype=torch.long
    ).reshape(len(padded), decode_columns)
    p = torch.tensor(prefills, dtype=torch.long)
    if has_initial is None:
        has_initial = [False] * len(prefills)
    if live is None:
        live = list(range(num_pool_blocks))
    if prompts is None:
        prompts = [0] * len(prefills)
    n_real = num_decodes + len(prefills)
    return RowPlan(
        state_indices_d=d,
        num_decodes=num_decodes,
        state_indices_p=p,
        num_prefills=len(prefills),
        has_initial_states_p=torch.tensor(has_initial, dtype=torch.bool),
        null_block_id=NULL_INDEX,
        num_pool_blocks=num_pool_blocks,
        live_block_ids=torch.tensor(live, dtype=torch.long),
        geometry_id=torch.zeros(n_real, dtype=torch.long),
        prompt_index=torch.tensor(prompts, dtype=torch.long),
    )


def _envelope(
    samples: torch.Tensor,
    *,
    final: bool,
    seq: int,
    geometry: int = 0,
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
    )


def _call(
    core: Any,
    pools: Pools,
    input_ids: torch.Tensor,
    inputs_embeds: torch.Tensor,
    plan: RowPlan,
    adapter: EmissionAdapter | None = None,
) -> torch.Tensor:
    if adapter is None:
        adapter = make_mrv1_adapter(
            hidden_size=CARRIER_HIDDEN, park_id=PARK_ID,
            blank_id=core.blank_id,
        )
    return advance_model_rows(
        core, input_ids, inputs_embeds, plan,
        adapter=adapter, placeholder_id=PLACEHOLDER_ID, park_id=PARK_ID,
        feat=FEAT, **pools,
    )


def _set_replay_book(
    pools: Pools, block: int, *, pending: list[int], expected: int
) -> None:
    """A VALID mid-replay book: labels queued, head past the emitted
    prefix, pending-echo armed with the label awaiting verification."""
    book = pools["book_pool"]
    queue = pools["queue_pool"]
    for i, label in enumerate(pending):
        queue[block, i] = float(label)
    book[block, _BOOK["queue_head"]] = 0.0
    book[block, _BOOK["queue_length"]] = float(len(pending))
    book[block, _BOOK["last_label"]] = float(expected)
    book[block, _BOOK["pending_echo"]] = 1.0
    book[block, _BOOK["expected_label"]] = float(expected)


# ---- PORT-STATE-007: whole-call structural preflight ---------------------
# Structural-reject tests use the refuse-adapter: rejection happens
# before classification, so the adapter must never run.


def test_advance_model_rows_rejects_wrong_row_count() -> None:
    # @spec PORT-STATE-007
    core = _tiny_core()
    pools = _sentinel_pools()
    before = _clone_pools(pools)
    plan = _plan(decodes=[0, 1])
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
    plan = _plan(decodes=[0, NULL_INDEX])  # null id claimed as real
    input_ids = torch.tensor([PARK_ID, PARK_ID], dtype=torch.long)
    inputs_embeds = torch.zeros(2, CARRIER_HIDDEN)
    with pytest.raises(ValueError):
        _call(core, pools, input_ids, inputs_embeds, plan, _refuse_adapter)
    _assert_pools_equal(pools, before)


def test_advance_model_rows_rejects_out_of_range_index() -> None:
    # @spec PORT-STATE-007
    core = _tiny_core()
    pools = _sentinel_pools(num_blocks=2)
    before = _clone_pools(pools)
    plan = _plan(decodes=[0, 99], num_pool_blocks=2, live=[0, 1, 99])
    input_ids = torch.tensor([PARK_ID, PARK_ID], dtype=torch.long)
    inputs_embeds = torch.zeros(2, CARRIER_HIDDEN)
    with pytest.raises(ValueError):
        _call(core, pools, input_ids, inputs_embeds, plan, _refuse_adapter)
    _assert_pools_equal(pools, before)


def test_advance_model_rows_rejects_valid_but_non_live_index() -> None:
    # @spec PORT-STATE-007
    # Block 1 exists in the pool but is not allocated to any resident
    # session — liveness comes from plan.live_block_ids (scheduler
    # metadata), never from page contents.
    core = _tiny_core()
    pools = _sentinel_pools(num_blocks=4)
    before = _clone_pools(pools)
    plan = _plan(decodes=[1], num_pool_blocks=4, live=[0, 2])
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
    plan = _plan(decodes=[0], prefills=[0])
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
    plan = _plan(decodes=[0, 1], decode_columns=2)
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
    plan = _plan(decodes=[0], pad_decodes_to=2, padding_value=1)
    input_ids = torch.tensor([PARK_ID], dtype=torch.long)
    inputs_embeds = torch.zeros(1, CARRIER_HIDDEN)
    with pytest.raises(ValueError):
        _call(core, pools, input_ids, inputs_embeds, plan, _refuse_adapter)
    _assert_pools_equal(pools, before)


def test_advance_model_rows_accepts_legal_graph_padding() -> None:
    # @spec PORT-STATE-007
    # The same call SHAPE as the mismatch cases, but structurally legal:
    # padding rows carry the null id and num_decodes excludes them.
    # This is the accept half that makes the reject tests non-vacuous.
    core = _tiny_core()
    pools = _fresh_pools()
    plan = _plan(decodes=[0], pad_decodes_to=2, padding_value=NULL_INDEX)
    torch.manual_seed(5)
    samples = torch.randn(SAMPLES) * 0.01
    carrier = _envelope(samples, final=False, seq=0).unsqueeze(0)
    input_ids = torch.tensor([PLACEHOLDER_ID], dtype=torch.long)
    out = _call(core, pools, input_ids, carrier, plan)
    assert out.shape == (1, CARRIER_HIDDEN)


# ---- PORT-ADV-003: decode-then-prefill composition ------------------------


def test_composition_orders_decode_rows_before_prefill_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # @spec PORT-ADV-003
    # A resumed decode session (block 1, poisoned-distinct h) and a
    # fresh prefill (block 0) in one batch: the gathered state handed
    # to advance_session must be decode-then-prefill — row 0 carries
    # block 1's resumed h, row 1 the fresh zero h.
    seen: dict[str, torch.Tensor] = {}

    def _recorder(
        _core: Any, batch: ChunkBatch, state: SessionStateBatch
    ) -> AdvanceResult:
        seen["h"] = state.h.clone()
        seen["seq"] = batch.chunk_sequence.clone()
        return _empty_result(batch.samples.shape[0])

    monkeypatch.setattr(advance_mod, "advance_session", _recorder)
    core = _tiny_core()
    pools = _fresh_pools()
    pools["h_pool"][1].fill_(0.5)  # the resumed session's signature
    pools["book_pool"][1, _BOOK["last_label"]] = float(core.blank_id)
    pools["frontend_counter_pool"][
        1, _CTR["expected_chunk_sequence"]
    ] = 3
    plan = _plan(decodes=[1], prefills=[0])
    torch.manual_seed(5)
    samples = torch.randn(SAMPLES) * 0.01
    embeds = torch.stack([
        _envelope(samples, final=False, seq=3),
        _envelope(samples, final=False, seq=0),
    ])
    input_ids = torch.tensor(
        [PLACEHOLDER_ID, PLACEHOLDER_ID], dtype=torch.long
    )
    _call(core, pools, input_ids, embeds, plan)
    assert torch.all(seen["h"][:, 0] == 0.5)  # decode row first
    assert torch.count_nonzero(seen["h"][:, 1]) == 0  # fresh prefill after
    assert seen["seq"].tolist() == [3, 0]


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
    # A VALID mid-replay book: labels 3,5 queued, echo of label 3 armed.
    _set_replay_book(pools, 0, pending=[3, 5], expected=3)
    plan = _plan(decodes=[0])
    _call(
        core, pools,
        torch.tensor([3], dtype=torch.long),  # the correct echo
        torch.zeros(1, CARRIER_HIDDEN), plan,
    )


def test_flush_only_batch_never_calls_advance_session(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # @spec PORT-ADV-001
    def _recorder(*_args: Any, **_kwargs: Any) -> None:
        raise AssertionError("advance_session must not run for FLUSH rows")

    monkeypatch.setattr(advance_mod, "advance_session", _recorder)
    core = _tiny_core()
    pools = _fresh_pools()
    # A VALID flush row: queue drained, no pending echo, the session's
    # frontend finalized, and the engine's final sentinel echoed back
    # (the step after the drain emitted park) — never an arbitrary
    # token.
    pools["book_pool"][0, _BOOK["last_label"]] = float(core.blank_id)
    pools["frontend_counter_pool"][0, _CTR["finalized"]] = 1
    plan = _plan(decodes=[0])
    out = _call(
        core, pools,
        torch.tensor([PARK_ID], dtype=torch.long),
        torch.zeros(1, CARRIER_HIDDEN), plan,
    )
    assert int(read_decision_carrier(out)[0]) == PARK_ID


def test_mixed_batch_calls_advance_session_with_only_chunk_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # @spec PORT-ADV-001
    seen: dict[str, int] = {}

    def _recorder(
        _core: Any, batch: ChunkBatch, _state: SessionStateBatch
    ) -> AdvanceResult:
        seen["n_chunk_rows"] = batch.samples.shape[0]
        return _empty_result(batch.samples.shape[0])

    monkeypatch.setattr(advance_mod, "advance_session", _recorder)
    core = _tiny_core()
    pools = _fresh_pools()
    _set_replay_book(pools, 1, pending=[7], expected=7)  # row 1: REPLAY
    torch.manual_seed(5)
    samples = torch.randn(SAMPLES) * 0.01
    embeds = torch.stack([
        torch.zeros(CARRIER_HIDDEN),
        _envelope(samples, final=False, seq=0),
    ])
    input_ids = torch.tensor([7, PLACEHOLDER_ID], dtype=torch.long)
    plan = _plan(decodes=[1], prefills=[0])
    _call(core, pools, input_ids, embeds, plan)
    assert seen["n_chunk_rows"] == 1  # only the one CHUNK row gathered


def test_advance_session_result_contract() -> None:
    # @spec PORT-ADV-001 / PORT-INT-002
    # Written as a normal test against the real contract — fails
    # NotImplementedError on the pod until Phase 6 (expected-fail
    # evidence). A session-first FINAL-TAIL chunk: final semantics
    # equal the whole-signal featurizer, so the reference is exact.
    core = _tiny_core()
    torch.manual_seed(5)
    samples = torch.randn(SAMPLES) * 0.01
    enc_frames = _reference_encoder_frames(core, samples)
    batch = ChunkBatch(
        samples=samples.unsqueeze(0),
        valid_samples=torch.tensor([SAMPLES], dtype=torch.long),
        geometry_id=torch.zeros(1, dtype=torch.long),
        final_tail=torch.tensor([True]),
        prompt_index=torch.zeros(1, dtype=torch.long),
        chunk_sequence=torch.zeros(1, dtype=torch.long),
    )
    state = SessionStateBatch(
        raw_tail=torch.zeros(1, RAW_TAIL),
        mel_tail=torch.zeros(1, FEAT, 9),
        frontend_counters=torch.zeros(1, CTR_WIDTH, dtype=torch.int64),
        channel=[torch.zeros(1, WINDOW, D_MODEL) for _ in range(N_LAYERS)],
        window_valid=[
            torch.zeros(1, 1, dtype=torch.int32) for _ in range(N_LAYERS)
        ],
        time=[torch.zeros(1, D_MODEL, KERNEL - 1) for _ in range(N_LAYERS)],
        h=torch.zeros(1, 2, 16),
        c=torch.zeros(1, 2, 16),
        last_label=torch.full((1,), core.blank_id, dtype=torch.long),
    )
    result = advance_session(core, batch, state)
    assert isinstance(result, AdvanceResult)
    # GPU-resident result: padded token_ids + per-row lengths, never
    # Python lists (PORT-PERF-001).
    n_tok = int(result.token_lengths[0])
    burst = result.token_ids[0, :n_tok]
    assert result.token_ids.dtype == torch.int32
    assert not bool((burst == core.blank_id).any())
    # PORT-INT-002: bounded by valid encoder frames × max symbols per
    # step — NOT by the attention window.
    assert n_tok <= enc_frames * MAX_SYMBOLS_PER_STEP
    caps = result.captures
    assert isinstance(caps, PreparedCaptures)
    assert caps.frontend_mel.shape[0] == 1
    assert int(caps.mel_lengths[0]) == caps.frontend_mel.shape[2]
    assert caps.encoder_raw.shape == caps.encoder_conditioned.shape
    assert int(caps.encoder_lengths[0]) == caps.encoder_raw.shape[1]
    # The transition advanced the frontend state it was handed.
    ctr = state.frontend_counters[0]
    assert int(ctr[_CTR["total_valid_samples"]]) == SAMPLES
    assert int(ctr[_CTR["committed_mel_frames"]]) > 0
    assert int(ctr[_CTR["finalized"]]) == 1


# ---- PORT-ADV-003: fresh-session init + echo guard ------------------------


def test_fresh_session_rows_ignore_a_recycled_blocks_poison() -> None:
    # @spec PORT-ADV-003
    # A block carrying a prior session's garbage in EVERY pool, but
    # marked fresh by scheduler metadata (~has_initial_states_p), must
    # decode exactly as a clean block — freshness never reads page
    # contents (so there is no zero-length-slot sentinel to trip on).
    core = _tiny_core()
    torch.manual_seed(5)
    samples = torch.randn(SAMPLES) * 0.01
    carrier = _envelope(samples, final=False, seq=0).unsqueeze(0)
    input_ids = torch.tensor([PLACEHOLDER_ID], dtype=torch.long)
    plan = _plan(prefills=[0], has_initial=[False])

    clean = _fresh_pools()
    out_clean = _call(core, clean, input_ids, carrier, plan)

    dirty = _sentinel_pools(value=3.14)  # poison EVERYTHING, book incl.
    out_dirty = _call(core, dirty, input_ids, carrier, plan)
    torch.testing.assert_close(
        read_decision_carrier(out_dirty), read_decision_carrier(out_clean)
    )


def test_corrupted_echo_aborts_whole_call_without_advancing_peers() -> None:
    # @spec PORT-ADV-003
    # Echo atomicity across a MULTI-row batch: row 0 replays with the
    # CORRECT echo, row 1 with a corrupted one. The whole call aborts
    # and NO pool changes — the valid peer must not have advanced.
    core = _tiny_core()
    pools = _fresh_pools()
    _set_replay_book(pools, 0, pending=[3, 5], expected=3)
    _set_replay_book(pools, 1, pending=[4, 6], expected=4)
    before = _clone_pools(pools)
    plan = _plan(decodes=[0, 1])
    input_ids = torch.tensor([3, 8], dtype=torch.long)  # row 1: wrong echo
    with pytest.raises(ValueError):
        _call(
            core, pools, input_ids, torch.zeros(2, CARRIER_HIDDEN), plan,
        )
    _assert_pools_equal(pools, before)


# ---- PORT-STATE-008: no partial scatter on compute failure ----------------


def test_resident_pools_unchanged_when_bucket_compute_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # @spec PORT-STATE-008
    # The bucket seam IS advance_session (the transaction's only
    # compute call for CHUNK rows) — a failure there must leave every
    # resident pool untouched (compute runs on scratch; scatter only
    # commits returnable results).
    def _boom(*_args: Any, **_kwargs: Any) -> AdvanceResult:
        raise RuntimeError("simulated mid-bucket compute failure")

    monkeypatch.setattr(advance_mod, "advance_session", _boom)
    core = _tiny_core()
    pools = _fresh_pools()
    before = _clone_pools(pools)
    torch.manual_seed(5)
    samples = torch.randn(SAMPLES) * 0.01
    carrier = _envelope(samples, final=False, seq=0).unsqueeze(0)
    plan = _plan(prefills=[0])
    with pytest.raises(RuntimeError):
        _call(
            core, pools,
            torch.tensor([PLACEHOLDER_ID], dtype=torch.long), carrier, plan,
        )
    _assert_pools_equal(pools, before)


# ---- PORT-ADV-001 / MRV1 emission: burst-then-park ------------------------


def test_advance_model_rows_emits_the_burst_then_parks() -> None:
    # @spec PORT-ADV-001
    # The full MRV1 session arc on a session-first FINAL-TAIL chunk:
    # burst queued with echo state armed, drained one label per step
    # against the golden reference, park once drained + finalized.
    core = _tiny_core()
    torch.manual_seed(5)
    samples = torch.randn(SAMPLES) * 0.01
    expected = _reference_burst(core, samples, prompt_index=0)
    assert len(expected) >= 2 and len(set(expected)) >= 2, (
        "need a multi-label, multi-distinct burst to test drain order"
    )

    pools = _fresh_pools()
    plan = _plan(prefills=[0])
    carrier = _envelope(samples, final=True, seq=0).unsqueeze(0)
    out = _call(
        core, pools,
        torch.tensor([PLACEHOLDER_ID], dtype=torch.long), carrier, plan,
    )
    emitted = [int(read_decision_carrier(out)[0])]
    book = pools["book_pool"]
    assert int(book[0, _BOOK["queue_length"]]) == len(expected)
    assert int(book[0, _BOOK["queue_head"]]) == 1
    assert int(book[0, _BOOK["pending_echo"]]) == 1
    assert int(book[0, _BOOK["expected_label"]]) == emitted[0]
    assert int(pools["frontend_counter_pool"][0, _CTR["finalized"]]) == 1

    decode_plan = _plan(decodes=[0])
    for _ in range(len(expected) + 1):
        if emitted[-1] == PARK_ID:
            break
        out = _call(
            core, pools,
            torch.tensor([emitted[-1]], dtype=torch.long),
            torch.zeros(1, CARRIER_HIDDEN), decode_plan,
        )
        emitted.append(int(read_decision_carrier(out)[0]))

    assert emitted[:-1] == expected
    assert emitted[-1] == PARK_ID
    assert int(book[0, _BOOK["pending_echo"]]) == 0


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
        prefix="encoder.layers.0.window", window=WINDOW, d_model=D_MODEL,
        policy=FP32_BRINGUP,
    )
    conv = ConvCachePage(
        prefix="encoder.layers.0.conv", d_model=D_MODEL, kernel=KERNEL,
        policy=FP32_BRINGUP,
    )
    lstm = LSTMStatePage(
        prefix="predictor.layers.0.lstm_state", pred_rnn_layers=2,
        pred_hidden=16, policy=FP32_BRINGUP,
    )
    replay = ReplayQueuePage(
        prefix="decode.layers.0.replay", max_symbols_per_step=10,
        max_frames_per_chunk=4, policy=FP32_BRINGUP,
    )
    frontend = FrontendBufferPage(
        prefix="frontend.layers.0.buffers", raw_tail=RAW_TAIL,
        n_mels=FEAT, policy=FP32_BRINGUP,
    )
    counters = FrontendCounterPage(
        prefix="frontend.layers.0.counters", policy=FP32_BRINGUP,
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
    assert needed == 17_926
    assert NemotronASRConfig().hidden_size >= needed, (
        f"hidden_size must cover the raw chunk envelope: need {needed}, "
        f"config has {NemotronASRConfig().hidden_size}"
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
    out = model.embed_input_ids(
        input_ids, multimodal_embeddings=mm, is_multimodal=is_mm
    )
    assert out.shape == (4, hidden)
    assert torch.all(out[0] == 1.0) and torch.all(out[2] == 2.0)
    assert torch.count_nonzero(out[1]) == 0
    assert torch.count_nonzero(out[3]) == 0


# ---- PORT-REGIME-001 / PORT-INT-003: naming-lock source pins --------------

_NEMOTRON_ASR_DIR = Path(__file__).resolve().parents[4] / (
    "vllm_omni/model_executor/models/nemotron_asr"
)


@pytest.mark.xfail(
    strict=True, reason="P5-1 split lands in Phase 6"
)
def test_run_forward_step_is_removed_and_advance_model_rows_is_wired() -> None:
    # @spec PORT-REGIME-001
    # A source-level pin (Path.read_text — no imports needed): once the
    # P5-1 split lands, the package no longer defines run_forward_step
    # and nemotron_asr.py calls advance_model_rows. Flips to XPASS
    # (strict → error) if the mark is left behind after the split, and
    # to a hard failure if the mark is removed without doing the split.
    # forward_step.py itself is deleted in that same change (ledger
    # P5-1), so an absent file also satisfies "no longer defines
    # run_forward_step" — this must not FileNotFoundError forever.
    forward_step_path = _NEMOTRON_ASR_DIR / "forward_step.py"
    forward_step_src = (
        forward_step_path.read_text() if forward_step_path.exists() else ""
    )
    model_src = (_NEMOTRON_ASR_DIR / "nemotron_asr.py").read_text()
    assert "def run_forward_step" not in forward_step_src
    assert "advance_model_rows" in model_src


@pytest.mark.xfail(
    strict=True, reason="P5-1 split lands in Phase 6"
)
def test_forward_routes_through_advance_model_rows() -> None:
    # @spec PORT-INT-003
    model_src = (_NEMOTRON_ASR_DIR / "nemotron_asr.py").read_text()
    assert "advance_model_rows(" in model_src
    assert "run_forward_step(" not in model_src
