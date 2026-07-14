# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""α3 engine-tier decode mechanics, tests-first (D-α3a/b/d).

Specs: PORT-DEC-002 (forced-logits rows + hidden-row decision
carrier), PORT-DEC-003/004 (park = checkpoint EOS; blank-only chunk
parks with no delta), PORT-DEC-007 (the greedy pin is load-bearing —
the exclusion-mask negative), PORT-DEC-008 (fixed-trip tensorized
decode ≡ the math oracle bit-for-bit), PORT-INT-002 (constant
per-burst token budget).

The math-tier ``greedy_decode_batch`` (proven in test_rnnt_loop.py)
is the differential oracle here, never the implementation. CPU tier,
loader-runnable: no vllm imports (the real-sampler binding lives in
test_decode_sampler_binding.py, pod tier).
"""

from types import SimpleNamespace

import pytest
import torch

from vllm_omni.model_executor.models.nemotron_asr.precision import (
    FP32_BRINGUP,
)
from vllm_omni.model_executor.models.nemotron_asr.rnnt import (
    QUEUE_HEAD,
    QUEUE_LAST_LABEL,
    QUEUE_LEN,
    DecodeState,
    Joint,
    Predictor,
    decode_chunk_paged,
    forced_logits_rows,
    greedy_decode_batch,
    park_token_id,
    read_decision_carrier,
    realtime_token_budget,
    replay_step,
    write_decision_carrier,
)
from vllm_omni.model_executor.models.nemotron_asr.state_layers import (
    ReplayQueuePage,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

_VOCAB = 12
_HID = 16
_NUM_LOGITS = 13089  # tokenizer vocab + the park special token


def _nets(seed: int = 5) -> tuple[Predictor, Joint]:
    torch.manual_seed(seed)
    predictor = Predictor(
        vocab_size=_VOCAB, pred_hidden=_HID, pred_rnn_layers=2
    )
    joint = Joint(
        enc_hidden=_HID,
        pred_hidden=_HID,
        joint_hidden=_HID,
        vocab_size=_VOCAB,
    )
    for net in (predictor, joint):
        for param in net.parameters():
            torch.nn.init.uniform_(param, -0.5, 0.5)
    return predictor, joint


# ---- forced emission (D-α3a, PORT-DEC-002) -----------------------------------


def test_forced_logits_rows_contract():
    # 0 at the chosen id, −inf everywhere else, never +inf: the argmax
    # is the emission and the max logit is exactly 0.
    chosen = torch.tensor([0, 7, 13088], dtype=torch.long)
    rows = forced_logits_rows(chosen, num_logits=_NUM_LOGITS)
    assert rows.shape == (3, _NUM_LOGITS)
    assert torch.equal(rows.argmax(dim=-1), chosen)
    assert (rows.amax(dim=-1) == 0).all()
    assert torch.isfinite(rows).sum(dim=-1).eq(1).all()
    assert not torch.isposinf(rows).any()


def test_forced_logits_rows_reject_out_of_range():
    with pytest.raises(ValueError):
        forced_logits_rows(
            torch.tensor([_NUM_LOGITS], dtype=torch.long),
            num_logits=_NUM_LOGITS,
        )


def test_exclusion_mask_defeats_a_forced_row():
    # The PORT-DEC-007 negative, documented in pure tensor math (no
    # stub involved): an allowed_token_ids-style mask that excludes
    # the forced id leaves NO finite logit, and argmax degenerates —
    # forced emission does NOT survive exclusion masks, so the greedy
    # param pin is load-bearing and a per-update duty (the session
    # update path replaces sampling_params wholesale).
    chosen = 7
    row = torch.full((1, _NUM_LOGITS), float("-inf"))
    row[0, chosen] = 0.0
    allowed = torch.zeros(_NUM_LOGITS, dtype=torch.bool)
    allowed[3] = True  # a mask that does not include the forced id
    masked = row.masked_fill(~allowed, float("-inf"))
    assert not torch.isfinite(masked).any()  # the finite logit is gone
    assert int(masked.argmax(dim=-1)) != chosen


# ---- the hidden-row decision carrier (D-α3a, PORT-DEC-002) --------------------


def test_decision_carrier_roundtrip_is_exact():
    hidden = torch.zeros(4, 32, dtype=torch.float32)
    ids = torch.tensor([0, 256, 13088, _NUM_LOGITS - 1], dtype=torch.long)
    write_decision_carrier(hidden, ids)
    assert torch.equal(read_decision_carrier(hidden), ids)


def test_decision_carrier_dtype_guard_is_loud():
    # bf16 has a 7-bit mantissa: ids > 256 collapse. The carrier must
    # refuse the dtype, never corrupt silently.
    hidden = torch.zeros(1, 32, dtype=torch.bfloat16)
    with pytest.raises(ValueError):
        write_decision_carrier(
            hidden, torch.tensor([13088], dtype=torch.long)
        )


# ---- park token and token budget (D-α3b, PORT-DEC-003 / PORT-INT-002) ---------


def test_park_token_is_the_checkpoint_eos():
    # A config VALUE, never engine structure.
    assert park_token_id(SimpleNamespace(eos_token_id=42)) == 42


def test_missing_eos_fails_at_load():
    with pytest.raises(ValueError):
        park_token_id(SimpleNamespace(eos_token_id=None))


def test_token_budget_is_constant_per_burst_all_geometries():
    # bps·max_symbols + 1 for the five published chunk configs.
    for bps in (1, 2, 4, 7, 14):
        assert (
            realtime_token_budget(frames_per_chunk=bps)
            == bps * 10 + 1
        )


# ---- fixed-trip paged decode vs the math oracle (D-α3d, PORT-DEC-008) ---------


def _paged_state(batch: int, capacity: int):
    h_pool = torch.zeros(batch + 2, 2, _HID)
    c_pool = torch.zeros(batch + 2, 2, _HID)
    queue_pool = torch.zeros(batch + 2, capacity)
    book_pool = torch.zeros(batch + 2, 4)
    state_indices = torch.arange(1, batch + 1, dtype=torch.long)
    return h_pool, c_pool, queue_pool, book_pool, state_indices


def _init_books(book_pool, state_indices, blank_id):
    book_pool[state_indices, QUEUE_LAST_LABEL] = float(blank_id)


def test_paged_decode_matches_the_math_oracle_bit_for_bit():
    predictor, joint = _nets()
    batch, frames, max_symbols = 3, 5, 4
    torch.manual_seed(11)
    enc = torch.randn(batch, frames, _HID)

    state = DecodeState(
        h=torch.zeros(2, batch, _HID),
        c=torch.zeros(2, batch, _HID),
        last_label=torch.full((batch,), predictor.blank_id),
    )
    with torch.no_grad():
        want, want_state = greedy_decode_batch(
            enc, predictor, joint, state, max_symbols=max_symbols
        )

    capacity = frames * max_symbols
    h_pool, c_pool, queue_pool, book_pool, idx = _paged_state(
        batch, capacity
    )
    _init_books(book_pool, idx, predictor.blank_id)
    with torch.no_grad():
        decode_chunk_paged(
            enc, predictor, joint,
            h_pool=h_pool, c_pool=c_pool,
            queue_pool=queue_pool, book_pool=book_pool,
            state_indices=idx, max_symbols=max_symbols,
        )
    for b in range(batch):
        row = idx[b]
        n = int(book_pool[row, QUEUE_LEN])
        assert queue_pool[row, :n].long().tolist() == want[b]
        torch.testing.assert_close(
            h_pool[row], want_state.h[:, b], rtol=0.0, atol=0.0
        )
        torch.testing.assert_close(
            c_pool[row], want_state.c[:, b], rtol=0.0, atol=0.0
        )
        assert int(book_pool[row, QUEUE_LAST_LABEL]) == int(
            want_state.last_label[b]
        )


def test_paged_decode_is_fixed_trip():
    # The trip count must not depend on the data: a blank-everything
    # stream and an emission-heavy stream drive the SAME number of
    # predictor steps (time × max_symbols masked trips + the initial
    # step) — no data-dependent host branching (PORT-DEC-008).
    predictor, joint = _nets()
    batch, frames, max_symbols = 2, 4, 3
    capacity = frames * max_symbols
    calls: list[int] = []
    original = predictor.step

    def counting_step(labels, state):
        calls[-1] += 1
        return original(labels, state)

    predictor.step = counting_step  # type: ignore[method-assign]
    for seed, bias in ((3, 100.0), (4, -100.0)):
        torch.manual_seed(seed)
        enc = torch.randn(batch, frames, _HID)
        with torch.no_grad():
            joint.joint_net[1].bias[predictor.blank_id] = bias
        h_pool, c_pool, queue_pool, book_pool, idx = _paged_state(
            batch, capacity
        )
        _init_books(book_pool, idx, predictor.blank_id)
        calls.append(0)
        with torch.no_grad():
            decode_chunk_paged(
                enc, predictor, joint,
                h_pool=h_pool, c_pool=c_pool,
                queue_pool=queue_pool, book_pool=book_pool,
                state_indices=idx, max_symbols=max_symbols,
            )
    assert calls[0] == calls[1]


# ---- replay-and-park cadence (PORT-DEC-002/003/004) ---------------------------


def test_replay_drains_then_parks():
    park = 9000
    capacity = 8
    queue_pool = torch.zeros(3, capacity)
    book_pool = torch.zeros(3, 4)
    idx = torch.tensor([1], dtype=torch.long)
    queue_pool[1, :3] = torch.tensor([5.0, 7.0, 11.0])
    book_pool[1, QUEUE_LEN] = 3.0
    got = [
        int(
            replay_step(
                queue_pool, book_pool, state_indices=idx, park_id=park
            )[0]
        )
        for _ in range(5)
    ]
    assert got == [5, 7, 11, park, park]
    assert int(book_pool[1, QUEUE_HEAD]) == 3


def test_blank_only_chunk_parks_immediately():
    # An empty queue (blank-only chunk) parks on the first step —
    # no label, hence no transcription.delta (PORT-DEC-004).
    park = 9000
    queue_pool = torch.zeros(2, 8)
    book_pool = torch.zeros(2, 4)
    idx = torch.tensor([0], dtype=torch.long)
    out = replay_step(
        queue_pool, book_pool, state_indices=idx, park_id=park
    )
    assert int(out[0]) == park


# ---- contract green: the queue page holds a chunk's worst case ----------------


def test_queue_page_capacity_matches_the_budget():
    # The page's slot tensor is exactly max_symbols × frames — the
    # same worst case realtime_token_budget covers (minus the park).
    page = ReplayQueuePage(
        prefix="decode.replay",
        max_symbols_per_step=10,
        max_frames_per_chunk=7,
        policy=FP32_BRINGUP,
    )
    shapes = tuple(page.get_state_shape())
    assert shapes[0] == (70,)
    assert shapes[1] == (4,)
