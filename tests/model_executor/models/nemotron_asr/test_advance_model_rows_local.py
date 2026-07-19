# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Exercised advance_model_rows + MRV1-adapter coverage, locally.

The outer transaction's whole dependency closure is torch-only
(manifests, frontend, encoder, lid, rnnt, decode_dispatch, advance) —
loaded by file path under a stubbed package chain, the CANONICAL
transaction executes on macOS (the ``test_advance_session_local.py``
pattern). ``test_advance.py`` remains the pod-tier contract surface;
this file is the executable differential bar for the Phase-6c seams:
host structural preflight, metadata fresh init, the row-tier echo
contract (PORT-DEC-007 as amended), geometry buckets + the typed
decode resolver, the reservation-owned capture sink, the single
status handoff, and the pool-free MRV1 adapter's emission semantics.
"""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch

_PKG = (
    Path(__file__).resolve().parents[4]
    / "vllm_omni/model_executor/models/nemotron_asr"
)
_BASE = "vllm_omni.model_executor.models.nemotron_asr"


def _load_chain() -> dict[str, Any]:
    for name in (
        "vllm_omni",
        "vllm_omni.model_executor",
        "vllm_omni.model_executor.models",
        _BASE,
    ):
        if name not in sys.modules:
            sys.modules[name] = types.ModuleType(name)
    loaded: dict[str, Any] = {}
    for mod in (
        "precision", "masks", "featurizer", "encoder", "lid",
        "manifests", "frontend", "rnnt_cell", "rnnt",
        "decode_dispatch", "advance",
    ):
        spec = importlib.util.spec_from_file_location(
            f"{_BASE}.{mod}", _PKG / f"{mod}.py"
        )
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[f"{_BASE}.{mod}"] = module
        spec.loader.exec_module(module)
        loaded[mod] = module
    return loaded


mods = _load_chain()
advance = mods["advance"]
frontend = mods["frontend"]
rnnt = mods["rnnt"]
manifests = mods["manifests"]
decode_dispatch = mods["decode_dispatch"]

FEAT = 16
D_MODEL = 32
N_LAYERS = 2
KERNEL = 5
WINDOW = 8
PARK_ID = 9000
PLACEHOLDER_ID = 9001
VOCAB = 12
CAP = 48
#: Header + the largest cadence THESE fixtures admit (320 ms = 5,120
#: raw samples) — the tiny-core analogue of the production 17,926.
CARRIER_HIDDEN = 5_126
RAW_TAIL = 1_953
NULL_INDEX = 0  # NULL_BLOCK_ID at the pin (utils.py:46 @ ee0da84)

GEOM_REG = 0
REG_SAMPLES = 1_280
GEOM_FINAL = 2
FINAL_SAMPLES = 3_840

_BOOK = {name: i for i, (name, _) in enumerate(manifests.BOOK_FIELDS)}
_CTR = {
    name: i for i, name in enumerate(manifests.FRONTEND_COUNTER_FIELDS)
}
BOOK_WIDTH = len(manifests.BOOK_FIELDS)
CTR_WIDTH = len(manifests.FRONTEND_COUNTER_FIELDS)

Pools = dict[str, Any]


def _tiny_core(seed: int = 7) -> Any:
    # Seed 7 is the shared fixture default; the burst-arc test builds
    # a seed-1 core instead — through the REAL featurizer path the
    # seed-7 tiny joint is single-label on every probed signal family
    # (the log-guard-dominated mel is near-constant), while seed 1
    # yields a deterministic cap-saturated burst with two distinct
    # labels, the drain-order discriminator.
    torch.manual_seed(seed)
    encoder = mods["encoder"].FastConformerEncoder(
        feat_in=FEAT, d_model=D_MODEL, d_ff=64, n_layers=N_LAYERS,
        n_heads=4, conv_kernel=KERNEL, subsampling_channels=16,
        att_context=(WINDOW, 1),
    )
    encoder.eval()
    return SimpleNamespace(
        encoder=encoder,
        lid=mods["lid"].PromptConditioner(
            enc_hidden=D_MODEL, num_prompts=4
        ),
        predictor=rnnt.Predictor(
            vocab_size=VOCAB, pred_hidden=16, pred_rnn_layers=2
        ),
        joint=rnnt.Joint(
            enc_hidden=D_MODEL, pred_hidden=16, joint_hidden=16,
            vocab_size=VOCAB,
        ),
        featurizer=mods["featurizer"].MelFeaturizer(
            filterbank=torch.rand(FEAT, 257) * 0.01,
            window=torch.hann_window(400),
        ),
        blank_id=VOCAB,
    )


def _reference_burst(
    core: Any, samples: torch.Tensor, prompt_index: int
) -> list[int]:
    """The labels a session-first final-tail chunk should emit, via
    the golden whole-signal path."""
    with torch.no_grad():
        mel, mel_len = core.featurizer(
            samples.unsqueeze(0), torch.tensor([samples.shape[0]])
        )
        mel = mel[:, :, : int(mel_len[0])]
        caches = mods["encoder"].StreamingCaches(
            n_layers=N_LAYERS, batch=1, d_model=D_MODEL,
            left_context=WINDOW, conv_kernel=KERNEL,
            device=torch.device("cpu"),
        )
        enc = mods["encoder"].stream_step(
            core.encoder, mel, caches, drop_extra=0
        )
        conditioned = core.lid(enc, prompt_index=prompt_index)
        state = rnnt.DecodeState(
            h=torch.zeros(2, 1, 16),
            c=torch.zeros(2, 1, 16),
            last_label=torch.full((1,), core.blank_id),
        )
        labels, _ = rnnt.greedy_decode_batch(
            conditioned, core.predictor, core.joint, state
        )
    return list(labels[0])


def _fresh_pools(num_blocks: int = 3) -> Pools:
    return {
        "channel_pools": [
            torch.zeros(num_blocks, WINDOW, D_MODEL)
            for _ in range(N_LAYERS)
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
        "queue_pool": torch.zeros(num_blocks, CAP, dtype=torch.int32),
        "book_pool": torch.zeros(
            num_blocks, BOOK_WIDTH, dtype=torch.int32
        ),
        "frontend_raw_pool": torch.zeros(num_blocks, RAW_TAIL),
        "frontend_mel_pool": torch.zeros(num_blocks, FEAT, 9),
        "frontend_counter_pool": torch.zeros(
            num_blocks, CTR_WIDTH, dtype=torch.int64
        ),
    }


def _sentinel_pools(num_blocks: int = 3, value: float = 12345.0) -> Pools:
    pools = _fresh_pools(num_blocks)
    for val in pools.values():
        if isinstance(val, list):
            for t in val:
                t.fill_(value)
        else:
            val.fill_(
                int(value) if not val.dtype.is_floating_point else value
            )
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
    execution_tier: int = 0,
) -> Any:
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
        live = list(range(1, num_pool_blocks))
    n_real = num_decodes + len(prefills)
    if prompts is None:
        prompts = [0] * n_real  # ADMITTED prompt authority, every row
    if chunk is None:
        chunk = [False] * num_decodes + [True] * len(prefills)
    if geometries is None:
        geometries = [GEOM_REG] * n_real
    if generations is None:
        generations = [0] * n_real
    return advance.RowPlan(
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
        request_ids=tuple(f"req-{i}" for i in range(n_real)),
        execution_tier=execution_tier,
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
    n = samples.shape[0]
    assert advance.ENVELOPE_HEADER_SLOTS + n <= hidden
    row = torch.zeros(hidden)
    row[advance.ENV_VERSION] = advance.ENVELOPE_VERSION
    row[advance.ENV_VALID_SAMPLES] = n
    row[advance.ENV_GEOMETRY_ID] = geometry
    row[advance.ENV_FINAL_TAIL] = 1.0 if final else 0.0
    row[advance.ENV_PROMPT_INDEX] = prompt
    row[advance.ENV_CHUNK_SEQUENCE] = seq
    row[advance.ENVELOPE_HEADER_SLOTS : advance.ENVELOPE_HEADER_SLOTS + n] = (
        samples
    )
    return row


def _refuse_adapter(*_args: Any) -> Any:
    raise AssertionError("adapter must not be reached in this test")


def _fixed_resolver(decode_fn: Any = None) -> Any:
    fn = decode_fn or rnnt.decode_dense_masked

    def resolve(request: Any) -> Any:
        return advance.ResolvedDecode(arm="dense-eager", decode_fn=fn)

    return resolve


class _RecordingResolver:
    def __init__(self, decode_fn: Any = None) -> None:
        self.requests: list[Any] = []
        self._fn = decode_fn or rnnt.decode_dense_masked

    def __call__(self, request: Any) -> Any:
        self.requests.append(request)
        return advance.ResolvedDecode(
            arm="dense-eager", decode_fn=self._fn
        )


class _Reservation:
    def __init__(self, sink: _RecorderSink) -> None:
        self._sink = sink

    def publish(self, records: Any) -> None:
        self._sink.published.append(list(records))
        self._sink.log.append("publish")
        if self._sink.on_publish is not None:
            self._sink.on_publish()

    def cancel(self) -> None:
        self._sink.cancels += 1
        self._sink.log.append("cancel")


class _RecorderSink:
    def __init__(self, *, fail_reserve: bool = False) -> None:
        self.plans: list[Any] = []
        self.published: list[list[Any]] = []
        self.cancels = 0
        self.log: list[str] = []
        self.on_publish: Any = None
        self._fail = fail_reserve

    def reserve(self, plan: Any) -> _Reservation:
        self.plans.append(plan)
        self.log.append("reserve_capture")
        if self._fail:
            raise RuntimeError("capture sink at capacity")
        return _Reservation(self)


class _StatusTicket:
    def __init__(self, sink: _StatusRecorder) -> None:
        self._sink = sink

    def stage(self, row_status: torch.Tensor) -> None:
        self._sink.staged.append(row_status.clone())
        self._sink.log.append("stage")


class _StatusRecorder:
    """Request-aware ticket sink: reserve pre-commit, stage no-fail."""

    def __init__(self, *, fail_reserve: bool = False) -> None:
        self.reserved: list[tuple[tuple[str, ...], tuple[int, ...]]] = []
        self.staged: list[torch.Tensor] = []
        self.log: list[str] = []
        self._fail = fail_reserve

    def reserve(self, request_ids: Any, generations: Any) -> _StatusTicket:
        self.reserved.append((tuple(request_ids), tuple(generations)))
        self.log.append("reserve_status")
        if self._fail:
            raise RuntimeError("status sink at capacity")
        return _StatusTicket(self)


def _call(
    core: Any,
    pools: Pools,
    input_ids: torch.Tensor,
    inputs_embeds: torch.Tensor,
    plan: Any,
    adapter: Any = None,
    *,
    resolver: Any = None,
    capture_sink: Any = None,
    status_sink: Any = None,
    graph_covers_decode: bool = False,
) -> torch.Tensor:
    if adapter is None:
        adapter = advance.make_mrv1_adapter(
            hidden_size=CARRIER_HIDDEN, park_id=PARK_ID,
            blank_id=core.blank_id,
        )
    out: torch.Tensor = advance.advance_model_rows(
        core, input_ids, inputs_embeds, plan,
        adapter=adapter,
        decode_resolver=(
            resolver if resolver is not None else _fixed_resolver()
        ),
        placeholder_id=PLACEHOLDER_ID, park_id=PARK_ID,
        capture_sink=capture_sink, status_sink=status_sink,
        graph_covers_decode=graph_covers_decode,
        **pools,
    )
    return out


def _decision(out: torch.Tensor) -> list[int]:
    return out[:, 0].long().tolist()


def _set_replay_book(
    pools: Pools,
    block: int,
    *,
    queue: list[int],
    head: int,
    expected: int,
) -> None:
    book = pools["book_pool"]
    qp = pools["queue_pool"]
    for i, label in enumerate(queue):
        qp[block, i] = label
    book[block, _BOOK["queue_head"]] = head
    book[block, _BOOK["queue_length"]] = len(queue)
    book[block, _BOOK["last_label"]] = expected
    book[block, _BOOK["pending_echo"]] = 1
    book[block, _BOOK["expected_label"]] = expected


def _set_drained_book(pools: Pools, block: int, *, blank: int) -> None:
    book = pools["book_pool"]
    book[block, _BOOK["queue_head"]] = 0
    book[block, _BOOK["queue_length"]] = 0
    book[block, _BOOK["last_label"]] = blank
    book[block, _BOOK["pending_echo"]] = 0
    book[block, _BOOK["expected_label"]] = 0


# ---- PORT-STATE-007: host structural preflight ---------------------------


@pytest.mark.parametrize(
    "name",
    [
        "wrong_row_count",
        "null_among_real",
        "out_of_range",
        "non_live",
        "duplicate_composition",
        "speculative_column",
        "live_padding",
    ],
)
def test_structural_defects_fail_whole_call_before_any_read(
    name: str,
) -> None:
    core = _tiny_core()
    pools = _sentinel_pools(num_blocks=5)
    before = _clone_pools(pools)
    if name == "wrong_row_count":
        plan = _plan(decodes=[1, 2], num_pool_blocks=5)
        ids, embeds = [PARK_ID, PARK_ID], torch.zeros(3, CARRIER_HIDDEN)
    elif name == "null_among_real":
        plan = _plan(decodes=[1, NULL_INDEX], num_pool_blocks=5)
        ids, embeds = [PARK_ID, PARK_ID], torch.zeros(2, CARRIER_HIDDEN)
    elif name == "out_of_range":
        plan = _plan(
            decodes=[1, 99], num_pool_blocks=5, live=[1, 2, 99]
        )
        ids, embeds = [PARK_ID, PARK_ID], torch.zeros(2, CARRIER_HIDDEN)
    elif name == "non_live":
        plan = _plan(decodes=[2], num_pool_blocks=5, live=[1, 3])
        ids, embeds = [PARK_ID], torch.zeros(1, CARRIER_HIDDEN)
    elif name == "duplicate_composition":
        plan = _plan(decodes=[1], prefills=[1], num_pool_blocks=5)
        ids = [PARK_ID, PLACEHOLDER_ID]
        embeds = torch.zeros(2, CARRIER_HIDDEN)
    elif name == "speculative_column":
        plan = _plan(decodes=[1, 2], num_pool_blocks=5, decode_columns=2)
        ids, embeds = [PARK_ID, PARK_ID], torch.zeros(2, CARRIER_HIDDEN)
    else:  # live_padding
        plan = _plan(
            decodes=[1], num_pool_blocks=5, pad_decodes_to=2,
            padding_value=2,
        )
        ids, embeds = [PARK_ID], torch.zeros(1, CARRIER_HIDDEN)
    with pytest.raises(ValueError):
        _call(
            core, pools, torch.tensor(ids, dtype=torch.long), embeds,
            plan, _refuse_adapter,
        )
    _assert_pools_equal(pools, before)


def test_missing_resolver_fails_before_state_reads() -> None:
    core = _tiny_core()
    pools = _sentinel_pools()
    before = _clone_pools(pools)
    plan = _plan(decodes=[1])
    with pytest.raises(ValueError):
        _call(
            core, pools,
            torch.tensor([PARK_ID], dtype=torch.long),
            torch.zeros(1, CARRIER_HIDDEN), plan, _refuse_adapter,
            resolver=False,
        )
    _assert_pools_equal(pools, before)


def test_legal_graph_padding_replays_normally() -> None:
    core = _tiny_core()
    pools = _fresh_pools()
    _set_replay_book(pools, 1, queue=[3, 5], head=1, expected=3)
    plan = _plan(decodes=[1], pad_decodes_to=2, padding_value=NULL_INDEX)
    out = _call(
        core, pools,
        torch.tensor([3], dtype=torch.long),
        torch.zeros(1, CARRIER_HIDDEN), plan,
    )
    assert out.shape == (1, CARRIER_HIDDEN)
    assert _decision(out) == [5]
    assert int(pools["book_pool"][1, _BOOK["queue_head"]]) == 2
    assert int(pools["book_pool"][1, _BOOK["expected_label"]]) == 5


# ---- fresh init + the echo row tier --------------------------------------


def test_fresh_rows_ignore_recycled_poison() -> None:
    core = _tiny_core()
    torch.manual_seed(5)
    samples = torch.randn(FINAL_SAMPLES) * 0.01
    carrier = _envelope(
        samples, final=True, seq=0, geometry=GEOM_FINAL
    ).unsqueeze(0)
    ids = torch.tensor([PLACEHOLDER_ID], dtype=torch.long)
    plan = _plan(prefills=[1], geometries=[GEOM_FINAL])
    clean = _fresh_pools()
    out_clean = _call(core, clean, ids, carrier, plan)
    dirty = _sentinel_pools(value=3.14)
    out_dirty = _call(core, dirty, ids, carrier, plan)
    assert _decision(out_dirty) == _decision(out_clean)
    assert _decision(out_clean)[0] != PARK_ID  # a real burst ran


def test_corrupt_echo_is_row_local_and_reported() -> None:
    core = _tiny_core()
    pools = _fresh_pools()
    _set_replay_book(pools, 1, queue=[3, 5], head=1, expected=3)
    _set_replay_book(pools, 2, queue=[4, 6], head=1, expected=4)
    before = _clone_pools(pools)
    plan = _plan(decodes=[1, 2])
    status = _StatusRecorder()
    out = _call(
        core, pools,
        torch.tensor([3, 8], dtype=torch.long),
        torch.zeros(2, CARRIER_HIDDEN), plan,
        status_sink=status,
    )
    assert _decision(out) == [5, PARK_ID]
    # The healthy peer advanced...
    assert int(pools["book_pool"][1, _BOOK["queue_head"]]) == 2
    # ...the corrupted row is bit-identical.
    torch.testing.assert_close(
        pools["book_pool"][2], before["book_pool"][2], rtol=0, atol=0
    )
    torch.testing.assert_close(
        pools["queue_pool"][2], before["queue_pool"][2], rtol=0, atol=0
    )
    assert len(status.staged) == 1
    assert int(status.staged[0][0]) == 0
    assert int(status.staged[0][1]) & advance.ROW_STATUS_ECHO_MISMATCH


def test_absurd_echo_id_is_safe_and_masked() -> None:
    core = _tiny_core()
    pools = _fresh_pools()
    _set_replay_book(pools, 1, queue=[3, 5], head=1, expected=3)
    before = _clone_pools(pools)
    plan = _plan(decodes=[1])
    out = _call(
        core, pools,
        torch.tensor([10**6], dtype=torch.long),
        torch.zeros(1, CARRIER_HIDDEN), plan,
    )
    assert _decision(out) == [PARK_ID]
    _assert_pools_equal(pools, before)


def test_chunk_on_undrained_queue_masks_and_reports() -> None:
    core = _tiny_core()
    pools = _fresh_pools()
    _set_replay_book(pools, 1, queue=[3, 5], head=1, expected=3)
    before = _clone_pools(pools)
    torch.manual_seed(5)
    carrier = _envelope(
        torch.randn(REG_SAMPLES) * 0.01, final=False, seq=0
    ).unsqueeze(0)
    plan = _plan(decodes=[1], chunk=[True])
    status = _StatusRecorder()
    out = _call(
        core, pools,
        torch.tensor([PLACEHOLDER_ID], dtype=torch.long), carrier, plan,
        status_sink=status,
    )
    assert _decision(out) == [PARK_ID]
    _assert_pools_equal(pools, before)
    assert (
        int(status.staged[0][0]) & advance.ROW_STATUS_QUEUE_NOT_DRAINED
    )


def test_flush_parks_only_when_drained_and_finalized() -> None:
    core = _tiny_core()
    pools = _fresh_pools()
    _set_drained_book(pools, 1, blank=core.blank_id)
    pools["frontend_counter_pool"][1, _CTR["finalized"]] = 1
    _set_drained_book(pools, 2, blank=core.blank_id)  # NOT finalized
    plan = _plan(decodes=[1, 2])
    status = _StatusRecorder()
    out = _call(
        core, pools,
        torch.tensor([PARK_ID, PARK_ID], dtype=torch.long),
        torch.zeros(2, CARRIER_HIDDEN), plan,
        status_sink=status,
    )
    assert _decision(out) == [PARK_ID, PARK_ID]
    assert int(status.staged[0][0]) == 0
    assert (
        int(status.staged[0][1]) & advance.ROW_STATUS_SESSION_PROTOCOL
    )


# ---- the full MRV1 arc ----------------------------------------------------


def test_burst_then_drain_then_park_matches_reference() -> None:
    core = _tiny_core(seed=1)  # two-distinct-label burst by design
    torch.manual_seed(5)
    samples = torch.randn(FINAL_SAMPLES) * 0.01
    expected = _reference_burst(core, samples, prompt_index=0)
    assert len(expected) >= 2 and len(set(expected)) >= 2

    pools = _fresh_pools()
    plan = _plan(prefills=[1], geometries=[GEOM_FINAL])
    carrier = _envelope(
        samples, final=True, seq=0, geometry=GEOM_FINAL
    ).unsqueeze(0)
    out = _call(
        core, pools,
        torch.tensor([PLACEHOLDER_ID], dtype=torch.long), carrier, plan,
    )
    emitted = [_decision(out)[0]]
    book = pools["book_pool"]
    assert int(book[1, _BOOK["queue_length"]]) == len(expected)
    assert int(book[1, _BOOK["queue_head"]]) == 1
    assert int(book[1, _BOOK["pending_echo"]]) == 1
    assert int(book[1, _BOOK["expected_label"]]) == emitted[0]
    assert int(pools["frontend_counter_pool"][1, _CTR["finalized"]]) == 1

    decode_plan = _plan(decodes=[1], geometries=[GEOM_FINAL])
    for _ in range(len(expected) + 1):
        if emitted[-1] == PARK_ID:
            break
        out = _call(
            core, pools,
            torch.tensor([emitted[-1]], dtype=torch.long),
            torch.zeros(1, CARRIER_HIDDEN), decode_plan,
        )
        emitted.append(_decision(out)[0])

    assert emitted[:-1] == expected
    assert emitted[-1] == PARK_ID
    assert int(book[1, _BOOK["pending_echo"]]) == 0

    # An INVALID CONTINUING chunk (wrong sequence) on the now-parked
    # session leaves every page of its block bit-identical.
    parked = _clone_pools(pools)
    wrong_seq = _envelope(
        samples, final=True, seq=5, geometry=GEOM_FINAL
    ).unsqueeze(0)
    status = _StatusRecorder()
    out = _call(
        core, pools,
        torch.tensor([PLACEHOLDER_ID], dtype=torch.long), wrong_seq,
        _plan(decodes=[1], chunk=[True], geometries=[GEOM_FINAL]),
        status_sink=status,
    )
    assert _decision(out) == [PARK_ID]
    _assert_pools_equal(pools, parked)
    assert int(status.staged[0][0]) != 0


def test_mixed_geometries_bucket_resolver_and_row_order() -> None:
    # Two CHUNK rows at different geometries plus one replay row in a
    # single call: per-geometry buckets, one resolver query per bucket
    # with ready_decode_buckets == 2, and the returned rows stay
    # row-aligned with the call order.
    core = _tiny_core()
    pools = _fresh_pools(num_blocks=5)
    _set_replay_book(pools, 3, queue=[7, 9], head=1, expected=7)
    resolver = _RecordingResolver()
    plan = _plan(
        decodes=[3],
        prefills=[1, 2],
        num_pool_blocks=5,
        geometries=[GEOM_REG, GEOM_REG, GEOM_FINAL],
    )
    torch.manual_seed(5)
    reg = _envelope(
        torch.randn(REG_SAMPLES) * 0.01, final=False, seq=0,
        geometry=GEOM_REG,
    )
    fin = _envelope(
        torch.randn(FINAL_SAMPLES) * 0.01, final=True, seq=0,
        geometry=GEOM_FINAL,
    )
    input_ids = torch.tensor(
        [7, PLACEHOLDER_ID, PLACEHOLDER_ID], dtype=torch.long
    )
    embeds = torch.stack([torch.zeros(CARRIER_HIDDEN), reg, fin])
    out = _call(core, pools, input_ids, embeds, plan, resolver=resolver)
    assert len(resolver.requests) == 2
    assert {r.geometry for r in resolver.requests} == {
        GEOM_REG, GEOM_FINAL,
    }
    assert all(r.ready_decode_buckets == 2 for r in resolver.requests)
    assert all(r.execution_batch_size == 1 for r in resolver.requests)
    decisions = _decision(out)
    assert decisions[0] == 9  # the replay row advanced its queue
    assert decisions[2] != PARK_ID  # the final chunk burst emitted
    assert int(pools["book_pool"][3, _BOOK["queue_head"]]) == 2


def test_bucket_compute_failure_leaves_pools_untouched(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _boom(*_args: Any, **_kwargs: Any) -> Any:
        raise RuntimeError("simulated mid-bucket compute failure")

    monkeypatch.setattr(advance, "advance_session", _boom)
    core = _tiny_core()
    pools = _fresh_pools()
    before = _clone_pools(pools)
    torch.manual_seed(5)
    carrier = _envelope(
        torch.randn(REG_SAMPLES) * 0.01, final=False, seq=0
    ).unsqueeze(0)
    plan = _plan(prefills=[1])
    with pytest.raises(RuntimeError):
        _call(
            core, pools,
            torch.tensor([PLACEHOLDER_ID], dtype=torch.long),
            carrier, plan,
        )
    _assert_pools_equal(pools, before)


# ---- PORT-HOOK-001: the reservation-owned capture sink -------------------


def test_capture_reservation_failure_is_pre_commit_fatal() -> None:
    core = _tiny_core()
    pools = _fresh_pools()
    before = _clone_pools(pools)
    torch.manual_seed(5)
    carrier = _envelope(
        torch.randn(FINAL_SAMPLES) * 0.01, final=True, seq=0,
        geometry=GEOM_FINAL,
    ).unsqueeze(0)
    plan = _plan(prefills=[1], geometries=[GEOM_FINAL])
    sink = _RecorderSink(fail_reserve=True)
    with pytest.raises(RuntimeError):
        _call(
            core, pools,
            torch.tensor([PLACEHOLDER_ID], dtype=torch.long),
            carrier, plan, capture_sink=sink,
        )
    _assert_pools_equal(pools, before)
    assert sink.published == []


def test_capture_records_publish_after_commit_with_identity() -> None:
    core = _tiny_core()
    pools = _fresh_pools()
    torch.manual_seed(5)
    carrier = _envelope(
        torch.randn(FINAL_SAMPLES) * 0.01, final=True, seq=0,
        geometry=GEOM_FINAL,
    ).unsqueeze(0)
    plan = _plan(
        prefills=[1], geometries=[GEOM_FINAL], generations=[41]
    )
    sink = _RecorderSink()
    _call(
        core, pools,
        torch.tensor([PLACEHOLDER_ID], dtype=torch.long), carrier, plan,
        capture_sink=sink,
    )
    assert len(sink.plans) == 1
    assert sink.plans[0].rows == 1
    assert sink.plans[0].payload_bytes > 0
    [records] = sink.published
    [record] = records
    assert record.row == 0
    assert record.block_id == 1
    assert record.admission_generation == 41
    assert record.geometry == GEOM_FINAL
    assert int(record.row_status) == 0
    assert int(record.mel_length) == FINAL_SAMPLES // 160
    assert record.frontend_mel.shape[0] == FEAT
    assert record.encoder_raw.shape == record.encoder_conditioned.shape
    assert sink.cancels == 0


def test_adapter_failure_precedes_reservation_entirely() -> None:
    # The corrected state machine (PORT-ADV-003 as amended) runs the
    # adapter and every other fallible operation BEFORE reservation:
    # an adapter failure therefore never creates a reservation to
    # leak — nothing reserved, nothing published, pools untouched.
    core = _tiny_core()
    pools = _fresh_pools()
    before = _clone_pools(pools)
    torch.manual_seed(5)
    carrier = _envelope(
        torch.randn(FINAL_SAMPLES) * 0.01, final=True, seq=0,
        geometry=GEOM_FINAL,
    ).unsqueeze(0)
    plan = _plan(prefills=[1], geometries=[GEOM_FINAL])
    sink = _RecorderSink()

    def _bad_adapter(*_args: Any) -> Any:
        raise RuntimeError("adapter blew up")

    with pytest.raises(RuntimeError):
        _call(
            core, pools,
            torch.tensor([PLACEHOLDER_ID], dtype=torch.long),
            carrier, plan, _bad_adapter, capture_sink=sink,
        )
    _assert_pools_equal(pools, before)
    assert sink.plans == []
    assert sink.cancels == 0
    assert sink.published == []


def test_no_capture_sink_stages_nothing() -> None:
    core = _tiny_core()
    pools = _fresh_pools()
    torch.manual_seed(5)
    carrier = _envelope(
        torch.randn(FINAL_SAMPLES) * 0.01, final=True, seq=0,
        geometry=GEOM_FINAL,
    ).unsqueeze(0)
    plan = _plan(prefills=[1], geometries=[GEOM_FINAL])
    out = _call(
        core, pools,
        torch.tensor([PLACEHOLDER_ID], dtype=torch.long), carrier, plan,
    )
    assert out.shape == (1, CARRIER_HIDDEN)


# ---- envelope validation --------------------------------------------------


def test_bad_envelope_on_a_fresh_row_scatters_nothing() -> None:
    # PORT-ADV-004 as amended: a failed row — fresh initialization
    # included — leaves EVERY resident page bit-identical. Sentinel
    # pools prove the fresh init did not leak.
    core = _tiny_core()
    pools = _sentinel_pools()
    before = _clone_pools(pools)
    torch.manual_seed(5)
    carrier = _envelope(
        torch.randn(REG_SAMPLES) * 0.01, final=False, seq=0
    )
    carrier[advance.ENV_VERSION] = 99.0
    plan = _plan(prefills=[1])
    status = _StatusRecorder()
    out = _call(
        core, pools,
        torch.tensor([PLACEHOLDER_ID], dtype=torch.long),
        carrier.unsqueeze(0), plan, status_sink=status,
    )
    assert _decision(out) == [PARK_ID]
    assert int(status.staged[0][0]) & advance.ROW_STATUS_ENVELOPE
    _assert_pools_equal(pools, before)


def test_regular_chunk_with_wrong_cadence_length_masks() -> None:
    core = _tiny_core()
    pools = _fresh_pools()
    torch.manual_seed(5)
    carrier = _envelope(
        torch.randn(REG_SAMPLES - 160) * 0.01, final=False, seq=0
    )
    plan = _plan(prefills=[1])
    status = _StatusRecorder()
    out = _call(
        core, pools,
        torch.tensor([PLACEHOLDER_ID], dtype=torch.long),
        carrier.unsqueeze(0), plan, status_sink=status,
    )
    assert _decision(out) == [PARK_ID]
    assert int(status.staged[0][0]) & advance.ROW_STATUS_ENVELOPE


def test_wrong_but_valid_prompt_masks_and_admission_conditions() -> None:
    # PORT-LID-003 as amended: the envelope prompt is only a device
    # cross-check. Row 0's carrier agrees with its admitted prompt (2)
    # and computes; row 1's carrier stamps a DIFFERENT valid prompt
    # (2) against admitted 1 — masked with PROMPT_MISMATCH, never a
    # silently accepted substitute.
    core = _tiny_core()  # num_prompts == 4
    pools = _fresh_pools(num_blocks=4)
    torch.manual_seed(5)
    samples = torch.randn(FINAL_SAMPLES) * 0.01
    ok_row = _envelope(
        samples, final=True, seq=0, geometry=GEOM_FINAL, prompt=2
    )
    bad_row = _envelope(
        samples, final=True, seq=0, geometry=GEOM_FINAL, prompt=2
    )
    plan = _plan(
        prefills=[1, 2],
        num_pool_blocks=4,
        geometries=[GEOM_FINAL, GEOM_FINAL],
        prompts=[2, 1],  # admitted authority per row
    )
    status = _StatusRecorder()
    out = _call(
        core, pools,
        torch.tensor([PLACEHOLDER_ID, PLACEHOLDER_ID], dtype=torch.long),
        torch.stack([ok_row, bad_row]), plan, status_sink=status,
    )
    decisions = _decision(out)
    assert decisions[0] != PARK_ID  # admitted==stamped: computed
    assert decisions[1] == PARK_ID  # valid-but-different: masked
    assert int(status.staged[0][0]) == 0
    assert (
        int(status.staged[0][1]) & advance.ROW_STATUS_PROMPT_MISMATCH
    )


def test_out_of_dictionary_carrier_prompt_masks() -> None:
    # A carrier prompt outside the dictionary disagrees with the
    # (host-validated) admitted authority by construction.
    core = _tiny_core()
    pools = _fresh_pools()
    torch.manual_seed(5)
    carrier = _envelope(
        torch.randn(REG_SAMPLES) * 0.01, final=False, seq=0, prompt=7
    )
    plan = _plan(prefills=[1])
    status = _StatusRecorder()
    out = _call(
        core, pools,
        torch.tensor([PLACEHOLDER_ID], dtype=torch.long),
        carrier.unsqueeze(0), plan, status_sink=status,
    )
    assert _decision(out) == [PARK_ID]
    assert (
        int(status.staged[0][0]) & advance.ROW_STATUS_PROMPT_MISMATCH
    )


# ---- correction-pass hardening pins ---------------------------------------


def test_second_bucket_failure_leaves_pools_untouched(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A failure in bucket 2 AFTER bucket 1 computed is still a
    # whole-call failure before any scatter: scratch-first means the
    # first bucket's results never reached the pools.
    real = advance.advance_session
    calls = {"n": 0}

    def flaky(*args: Any, **kwargs: Any) -> Any:
        calls["n"] += 1
        if calls["n"] == 2:
            raise RuntimeError("second bucket exploded")
        return real(*args, **kwargs)

    monkeypatch.setattr(advance, "advance_session", flaky)
    core = _tiny_core()
    pools = _fresh_pools(num_blocks=4)
    before = _clone_pools(pools)
    torch.manual_seed(5)
    reg = _envelope(
        torch.randn(REG_SAMPLES) * 0.01, final=False, seq=0,
        geometry=GEOM_REG,
    )
    fin = _envelope(
        torch.randn(FINAL_SAMPLES) * 0.01, final=True, seq=0,
        geometry=GEOM_FINAL,
    )
    plan = _plan(
        prefills=[1, 2], num_pool_blocks=4,
        geometries=[GEOM_REG, GEOM_FINAL],
    )
    with pytest.raises(RuntimeError):
        _call(
            core, pools,
            torch.tensor(
                [PLACEHOLDER_ID, PLACEHOLDER_ID], dtype=torch.long
            ),
            torch.stack([reg, fin]), plan,
        )
    assert calls["n"] == 2
    _assert_pools_equal(pools, before)


def test_status_reserve_failure_cancels_capture_reservation() -> None:
    # The composite reservation: a status-sink reservation failure
    # after a successful capture reservation cancels the capture
    # reservation and fails pre-commit.
    core = _tiny_core()
    pools = _fresh_pools()
    before = _clone_pools(pools)
    torch.manual_seed(5)
    carrier = _envelope(
        torch.randn(FINAL_SAMPLES) * 0.01, final=True, seq=0,
        geometry=GEOM_FINAL,
    ).unsqueeze(0)
    plan = _plan(prefills=[1], geometries=[GEOM_FINAL])
    cap_sink = _RecorderSink()
    bad_status = _StatusRecorder(fail_reserve=True)
    with pytest.raises(RuntimeError):
        _call(
            core, pools,
            torch.tensor([PLACEHOLDER_ID], dtype=torch.long),
            carrier, plan,
            capture_sink=cap_sink, status_sink=bad_status,
        )
    _assert_pools_equal(pools, before)
    assert cap_sink.cancels == 1
    assert cap_sink.published == []


def test_status_ticket_is_request_aware_and_staged_once() -> None:
    core = _tiny_core()
    pools = _fresh_pools()
    _set_replay_book(pools, 1, queue=[3, 5], head=1, expected=3)
    plan = _plan(decodes=[1], generations=[9])
    status = _StatusRecorder()
    _call(
        core, pools,
        torch.tensor([3], dtype=torch.long),
        torch.zeros(1, CARRIER_HIDDEN), plan,
        status_sink=status,
    )
    assert status.reserved == [(("req-0",), (9,))]
    assert len(status.staged) == 1


@pytest.mark.parametrize(
    "corruption", ["negative_head", "oversized_length", "bad_last_label"]
)
def test_corrupt_book_masks_safely(corruption: str) -> None:
    # PORT-ADV-004 as amended: book-invariant violations are sanitized
    # to safe substitutes (no gather fault, no embedding fault), the
    # row masks with BOOK_INVARIANT, and its pages stay bit-identical.
    core = _tiny_core()
    pools = _fresh_pools()
    book = pools["book_pool"]
    if corruption == "negative_head":
        _set_replay_book(pools, 1, queue=[3, 5], head=1, expected=3)
        book[1, _BOOK["queue_head"]] = -3
        ids = [3]
        chunk = [False]
        embeds = torch.zeros(1, CARRIER_HIDDEN)
    elif corruption == "oversized_length":
        _set_replay_book(pools, 1, queue=[3, 5], head=1, expected=3)
        book[1, _BOOK["queue_length"]] = CAP + 10
        ids = [3]
        chunk = [False]
        embeds = torch.zeros(1, CARRIER_HIDDEN)
    else:  # bad_last_label on a CHUNK row: must not reach the embed
        book[1, _BOOK["last_label"]] = 500  # >> vocab
        ids = [PLACEHOLDER_ID]
        chunk = [True]
        torch.manual_seed(5)
        embeds = _envelope(
            torch.randn(REG_SAMPLES) * 0.01, final=False, seq=0
        ).unsqueeze(0)
    before = _clone_pools(pools)
    plan = _plan(decodes=[1], chunk=chunk)
    status = _StatusRecorder()
    out = _call(
        core, pools,
        torch.tensor(ids, dtype=torch.long), embeds, plan,
        status_sink=status,
    )
    assert _decision(out) == [PARK_ID]
    _assert_pools_equal(pools, before)
    assert (
        int(status.staged[0][0]) & advance.ROW_STATUS_BOOK_INVARIANT
    )


def test_burst_overflow_is_rejected_not_truncated() -> None:
    # A decode returning more labels than the queue holds is a port
    # defect: the row masks with BURST_OVERFLOW and nothing scatters —
    # the queue never sees a truncated burst.
    def _overflowing_decode(
        frames: Any, lengths: Any, predictor: Any, joint: Any, state: Any
    ) -> Any:
        b = frames.shape[0]
        ids = torch.ones(b, CAP + 8, dtype=torch.int32)
        lens = torch.full((b,), CAP + 1, dtype=torch.int32)
        return ids, lens, state

    core = _tiny_core()
    pools = _fresh_pools()
    before = _clone_pools(pools)
    torch.manual_seed(5)
    carrier = _envelope(
        torch.randn(FINAL_SAMPLES) * 0.01, final=True, seq=0,
        geometry=GEOM_FINAL,
    ).unsqueeze(0)
    plan = _plan(prefills=[1], geometries=[GEOM_FINAL])
    status = _StatusRecorder()
    out = _call(
        core, pools,
        torch.tensor([PLACEHOLDER_ID], dtype=torch.long), carrier, plan,
        resolver=_fixed_resolver(_overflowing_decode),
        status_sink=status,
    )
    assert _decision(out) == [PARK_ID]
    assert (
        int(status.staged[0][0]) & advance.ROW_STATUS_BURST_OVERFLOW
    )
    torch.testing.assert_close(
        pools["queue_pool"], before["queue_pool"], rtol=0, atol=0
    )
    torch.testing.assert_close(
        pools["book_pool"], before["book_pool"], rtol=0, atol=0
    )


def test_publication_observes_commit_and_stage_first() -> None:
    # Ordering proof: capture reserve → status reserve → (scatters) →
    # stage → publish, and at publish time the resident book already
    # holds the committed value.
    core = _tiny_core()
    pools = _fresh_pools()
    torch.manual_seed(5)
    carrier = _envelope(
        torch.randn(FINAL_SAMPLES) * 0.01, final=True, seq=0,
        geometry=GEOM_FINAL,
    ).unsqueeze(0)
    plan = _plan(prefills=[1], geometries=[GEOM_FINAL])
    shared: list[str] = []
    cap_sink = _RecorderSink()
    status = _StatusRecorder()
    cap_sink.log = shared
    status.log = shared
    seen_at_publish: dict[str, int] = {}

    def snapshot() -> None:
        seen_at_publish["head"] = int(
            pools["book_pool"][1, _BOOK["queue_head"]]
        )

    cap_sink.on_publish = snapshot
    _call(
        core, pools,
        torch.tensor([PLACEHOLDER_ID], dtype=torch.long), carrier, plan,
        capture_sink=cap_sink, status_sink=status,
    )
    assert shared == [
        "reserve_capture", "reserve_status", "stage", "publish",
    ]
    # The CHUNK committed head=1 (first label emitted) BEFORE publish.
    assert seen_at_publish["head"] == 1


# ---- the MRV1 adapter as a pure tensor transform --------------------------


def _context(
    *,
    roles: list[int],
    input_ids: list[int],
    chunk_rows: list[int],
    queue: torch.Tensor,
    book: torch.Tensor,
    row_status: list[int] | None = None,
) -> Any:
    n = len(roles)
    return advance.EmissionContext(
        roles=torch.tensor(roles, dtype=torch.long),
        input_ids=torch.tensor(input_ids, dtype=torch.long),
        chunk_rows=torch.tensor(chunk_rows, dtype=torch.long),
        queue=queue,
        book=book,
        row_status=torch.tensor(
            row_status or [0] * n, dtype=torch.int32
        ),
    )


def _adapter() -> Any:
    return advance.make_mrv1_adapter(
        hidden_size=64, park_id=PARK_ID, blank_id=VOCAB
    )


def test_adapter_chunk_burst_first_label_and_armed_echo() -> None:
    adapter = _adapter()
    result = advance.AdvanceResult(
        token_ids=torch.tensor([[4, 6, 0]], dtype=torch.int32),
        token_lengths=torch.tensor([2], dtype=torch.int32),
    )
    ctx = _context(
        roles=[advance.ROLE_CHUNK], input_ids=[PLACEHOLDER_ID],
        chunk_rows=[0],
        queue=torch.zeros(1, CAP, dtype=torch.int32),
        book=torch.zeros(1, BOOK_WIDTH, dtype=torch.int32),
    )
    proj = adapter(result, ctx)
    assert proj.rows[0, 0] == 4
    assert proj.queue[0, :2].tolist() == [4, 6]
    assert int(proj.book[0, _BOOK["queue_head"]]) == 1
    assert int(proj.book[0, _BOOK["queue_length"]]) == 2
    assert int(proj.book[0, _BOOK["pending_echo"]]) == 1
    assert int(proj.book[0, _BOOK["expected_label"]]) == 4


def test_adapter_zero_burst_parks_without_echo() -> None:
    # PORT-DEC-004: a blank-only chunk parks immediately; park never
    # arms an echo (PORT-DEC-003).
    adapter = _adapter()
    result = advance.AdvanceResult(
        token_ids=torch.zeros(1, 3, dtype=torch.int32),
        token_lengths=torch.zeros(1, dtype=torch.int32),
    )
    ctx = _context(
        roles=[advance.ROLE_CHUNK], input_ids=[PLACEHOLDER_ID],
        chunk_rows=[0],
        queue=torch.zeros(1, CAP, dtype=torch.int32),
        book=torch.zeros(1, BOOK_WIDTH, dtype=torch.int32),
    )
    proj = adapter(result, ctx)
    assert proj.rows[0, 0] == PARK_ID
    assert int(proj.book[0, _BOOK["pending_echo"]]) == 0
    assert int(proj.book[0, _BOOK["queue_length"]]) == 0


def test_adapter_masked_chunk_keeps_scratch_and_parks() -> None:
    adapter = _adapter()
    result = advance.AdvanceResult(
        token_ids=torch.tensor([[4, 6]], dtype=torch.int32),
        token_lengths=torch.tensor([2], dtype=torch.int32),
    )
    book = torch.zeros(1, BOOK_WIDTH, dtype=torch.int32)
    book[0, _BOOK["last_label"]] = 3
    queue = torch.zeros(1, CAP, dtype=torch.int32)
    ctx = _context(
        roles=[advance.ROLE_CHUNK], input_ids=[PLACEHOLDER_ID],
        chunk_rows=[0], queue=queue, book=book,
        row_status=[advance.ROW_STATUS_ENVELOPE],
    )
    proj = adapter(result, ctx)
    assert proj.rows[0, 0] == PARK_ID
    torch.testing.assert_close(proj.book, book, rtol=0, atol=0)
    torch.testing.assert_close(proj.queue, queue, rtol=0, atol=0)


def test_adapter_replay_emits_next_and_advances_head() -> None:
    adapter = _adapter()
    queue = torch.zeros(1, CAP, dtype=torch.int32)
    queue[0, 0], queue[0, 1] = 4, 6
    book = torch.zeros(1, BOOK_WIDTH, dtype=torch.int32)
    book[0, _BOOK["queue_head"]] = 1
    book[0, _BOOK["queue_length"]] = 2
    book[0, _BOOK["pending_echo"]] = 1
    book[0, _BOOK["expected_label"]] = 4
    ctx = _context(
        roles=[advance.ROLE_REPLAY], input_ids=[4], chunk_rows=[],
        queue=queue, book=book,
    )
    proj = adapter(
        advance.AdvanceResult(
            token_ids=torch.zeros(0, 0, dtype=torch.int32),
            token_lengths=torch.zeros(0, dtype=torch.int32),
        ),
        ctx,
    )
    assert proj.rows[0, 0] == 6
    assert int(proj.book[0, _BOOK["queue_head"]]) == 2
    assert int(proj.book[0, _BOOK["expected_label"]]) == 6
    assert int(proj.book[0, _BOOK["pending_echo"]]) == 1


def test_adapter_drained_replay_parks_and_clears_echo() -> None:
    adapter = _adapter()
    queue = torch.zeros(1, CAP, dtype=torch.int32)
    queue[0, 0], queue[0, 1] = 4, 6
    book = torch.zeros(1, BOOK_WIDTH, dtype=torch.int32)
    book[0, _BOOK["queue_head"]] = 2
    book[0, _BOOK["queue_length"]] = 2
    book[0, _BOOK["pending_echo"]] = 1
    book[0, _BOOK["expected_label"]] = 6
    ctx = _context(
        roles=[advance.ROLE_REPLAY], input_ids=[6], chunk_rows=[],
        queue=queue, book=book,
    )
    proj = adapter(
        advance.AdvanceResult(
            token_ids=torch.zeros(0, 0, dtype=torch.int32),
            token_lengths=torch.zeros(0, dtype=torch.int32),
        ),
        ctx,
    )
    assert proj.rows[0, 0] == PARK_ID
    assert int(proj.book[0, _BOOK["pending_echo"]]) == 0
    assert int(proj.book[0, _BOOK["queue_head"]]) == 2


def test_adapter_queue_saturation_full_capacity_burst() -> None:
    adapter = _adapter()
    burst = torch.arange(1, CAP + 1, dtype=torch.int32).unsqueeze(0)
    result = advance.AdvanceResult(
        token_ids=burst,
        token_lengths=torch.tensor([CAP], dtype=torch.int32),
    )
    ctx = _context(
        roles=[advance.ROLE_CHUNK], input_ids=[PLACEHOLDER_ID],
        chunk_rows=[0],
        queue=torch.zeros(1, CAP, dtype=torch.int32),
        book=torch.zeros(1, BOOK_WIDTH, dtype=torch.int32),
    )
    proj = adapter(result, ctx)
    assert proj.queue[0].tolist() == list(range(1, CAP + 1))
    assert int(proj.book[0, _BOOK["queue_length"]]) == CAP
    assert proj.rows[0, 0] == 1


def test_adapter_mixed_roles_keep_row_order() -> None:
    adapter = _adapter()
    n = 4
    queue = torch.zeros(n, CAP, dtype=torch.int32)
    book = torch.zeros(n, BOOK_WIDTH, dtype=torch.int32)
    # row 1: replay mid-queue; row 3: drained replay.
    queue[1, 0], queue[1, 1] = 7, 9
    book[1, _BOOK["queue_head"]] = 1
    book[1, _BOOK["queue_length"]] = 2
    book[1, _BOOK["pending_echo"]] = 1
    book[3, _BOOK["queue_head"]] = 1
    book[3, _BOOK["queue_length"]] = 1
    book[3, _BOOK["pending_echo"]] = 1
    result = advance.AdvanceResult(
        token_ids=torch.tensor([[5, 0]], dtype=torch.int32),
        token_lengths=torch.tensor([1], dtype=torch.int32),
    )
    ctx = _context(
        roles=[
            advance.ROLE_CHUNK, advance.ROLE_REPLAY,
            advance.ROLE_FLUSH, advance.ROLE_REPLAY,
        ],
        input_ids=[PLACEHOLDER_ID, 7, PARK_ID, 11],
        chunk_rows=[0], queue=queue, book=book,
    )
    proj = adapter(result, ctx)
    assert proj.rows[:, 0].long().tolist() == [5, 9, PARK_ID, PARK_ID]
    assert int(proj.book[3, _BOOK["pending_echo"]]) == 0


# ---- the compiled table resolver -----------------------------------------


def _table() -> Any:
    entries = {}
    for g in manifests.CADENCES:
        entries[(g, 8, "fp32")] = "dense-eager"
        entries[(g, 64, "fp32")] = "dense-eager"
        entries[(g, 512, "fp32")] = "compact-eager"
    return decode_dispatch.DispatchTable(
        entries=entries,
        fingerprint={},
        hysteresis_pct=25.0,
        policy_version="dcp-v1",
        lanes=("fp32",),
    )


def _arms() -> dict[str, Any]:
    return {
        "dense-eager": rnnt.decode_dense_masked,
        "dense-graphed": rnnt.decode_dense_masked,
        "compact-eager": rnnt.decode_compact_active,
    }


def test_table_resolver_matches_select_semantics() -> None:
    table = _table()
    resolver = advance.make_table_resolver(
        table, lane="fp32", arms=_arms(), max_batch=1024
    )
    for batch in (1, 8, 33, 64, 100, 512, 700, 1024):
        got = resolver(
            advance.DecodeRequest(
                geometry=0, execution_batch_size=batch,
                graph_covers_decode=False, ready_decode_buckets=1,
            )
        )
        want = table.select(
            "80ms", batch, lane="fp32", graph_covers_decode=False
        )
        assert got.arm == want, f"batch {batch}: {got.arm} != {want}"


def test_table_resolver_bracket_disagreement_takes_sync_free() -> None:
    resolver = advance.make_table_resolver(
        _table(), lane="fp32", arms=_arms(), max_batch=1024
    )
    # 64 (dense) and 512 (compact) bracket 200: sync-free wins.
    got = resolver(
        advance.DecodeRequest(
            geometry=0, execution_batch_size=200,
            graph_covers_decode=False, ready_decode_buckets=1,
        )
    )
    assert got.arm == "dense-eager"


def test_table_resolver_graph_coverage_forces_dense_graphed() -> None:
    resolver = advance.make_table_resolver(
        _table(), lane="fp32", arms=_arms(), max_batch=1024
    )
    got = resolver(
        advance.DecodeRequest(
            geometry=0, execution_batch_size=512,
            graph_covers_decode=True, ready_decode_buckets=1,
        )
    )
    assert got.arm == "dense-graphed"
    assert got.override_reason == "graph-covers-decode"


def test_table_resolver_multi_bucket_forces_sync_free() -> None:
    resolver = advance.make_table_resolver(
        _table(), lane="fp32", arms=_arms(), max_batch=1024
    )
    single = resolver(
        advance.DecodeRequest(
            geometry=0, execution_batch_size=512,
            graph_covers_decode=False, ready_decode_buckets=1,
        )
    )
    assert single.arm == "compact-eager"
    multi = resolver(
        advance.DecodeRequest(
            geometry=0, execution_batch_size=512,
            graph_covers_decode=False, ready_decode_buckets=2,
        )
    )
    assert multi.arm == "dense-eager"
    assert multi.override_reason == "multi-bucket-serialization-guard"


def test_table_resolver_missing_geometry_fails_at_startup() -> None:
    table = _table()
    entries = {
        k: v for k, v in table.entries.items() if k[0] != "320ms"
    }
    broken = decode_dispatch.DispatchTable(
        entries=entries,
        fingerprint={},
        hysteresis_pct=25.0,
        policy_version="dcp-v1",
        lanes=("fp32",),
    )
    with pytest.raises(KeyError):
        advance.make_table_resolver(
            broken, lane="fp32", arms=_arms(), max_batch=64
        )


def test_table_resolver_unbound_arm_fails_at_startup() -> None:
    arms = _arms()
    del arms["compact-eager"]
    with pytest.raises(KeyError):
        advance.make_table_resolver(
            _table(), lane="fp32", arms=arms, max_batch=1024
        )
