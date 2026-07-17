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

Most tests here call a still-``NotImplementedError``-raising stub and
are EXPECTED TO FAIL until Phase 6 lands the real implementation
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
    AdvanceResult,
    ChunkBatch,
    GatheredState,
    advance_model_rows,
    advance_session,
)
from vllm_omni.model_executor.models.nemotron_asr.precision import (
    FP32_BRINGUP,
)
from vllm_omni.model_executor.models.nemotron_asr.rnnt import (
    QUEUE_HEAD,
    QUEUE_LEN,
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
CARRIER_HIDDEN = 640  # > FEAT×32 + 1: the tiny mel-carrier width
MEL_FRAMES = 24  # → 4 encoder frames at 8× subsampling
#: The null/non-live index sentinel (PORT-STATE-007) — never a live block.
NULL_INDEX = -1

#: A pool bundle: per-layer tensor lists (channel/time/len) plus the four
#: whole-pool tensors (h/c/queue/book) — the same shapes the engine binds.
#: Left as ``dict[str, Any]`` (not a stricter union) — call sites index
#: the per-layer lists AND the whole-pool tensors interchangeably by key.
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
    return SimpleNamespace(
        encoder=encoder, lid=lid, predictor=predictor, joint=joint,
        blank_id=VOCAB,
    )


def _reference_burst(
    core: Any, mel: torch.Tensor, prompt_index: int
) -> list[int]:
    """The labels the chunk decode should emit, via the golden path."""
    from vllm_omni.model_executor.models.nemotron_asr.encoder import (
        StreamingCaches,
        stream_step,
    )
    from vllm_omni.model_executor.models.nemotron_asr.rnnt import (
        DecodeState,
        greedy_decode_batch,
    )

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
        "book_pool": torch.zeros(num_blocks, 4),
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
            val.fill_(value)
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


def _call(
    core: Any,
    pools: Pools,
    input_ids: torch.Tensor,
    inputs_embeds: torch.Tensor,
    state_indices: torch.Tensor,
    num_real_rows: int,
) -> torch.Tensor:
    return advance_model_rows(
        core, input_ids, inputs_embeds,
        state_indices=state_indices, num_real_rows=num_real_rows,
        placeholder_id=PLACEHOLDER_ID, park_id=PARK_ID, feat=FEAT,
        drop_extra=0, **pools,
    )


# ---- PORT-STATE-007: whole-call structural preflight ---------------------


def test_advance_model_rows_rejects_wrong_row_count() -> None:
    # @spec PORT-STATE-007
    core = _tiny_core()
    pools = _sentinel_pools()
    before = _clone_pools(pools)
    input_ids = torch.tensor([PARK_ID, PARK_ID], dtype=torch.long)
    inputs_embeds = torch.zeros(3, CARRIER_HIDDEN)  # shape mismatch vs N=2
    idx = torch.tensor([0, 1], dtype=torch.long)
    with pytest.raises(ValueError):
        _call(core, pools, input_ids, inputs_embeds, idx, num_real_rows=2)
    _assert_pools_equal(pools, before)


def test_advance_model_rows_rejects_null_index() -> None:
    # @spec PORT-STATE-007
    core = _tiny_core()
    pools = _sentinel_pools()
    before = _clone_pools(pools)
    input_ids = torch.tensor([PARK_ID, PARK_ID], dtype=torch.long)
    inputs_embeds = torch.zeros(2, CARRIER_HIDDEN)
    idx = torch.tensor([0, NULL_INDEX], dtype=torch.long)
    with pytest.raises(ValueError):
        _call(core, pools, input_ids, inputs_embeds, idx, num_real_rows=2)
    _assert_pools_equal(pools, before)


def test_advance_model_rows_rejects_out_of_range_index() -> None:
    # @spec PORT-STATE-007
    core = _tiny_core()
    pools = _sentinel_pools(num_blocks=2)
    before = _clone_pools(pools)
    input_ids = torch.tensor([PARK_ID, PARK_ID], dtype=torch.long)
    inputs_embeds = torch.zeros(2, CARRIER_HIDDEN)
    idx = torch.tensor([0, 99], dtype=torch.long)  # 99 >= num blocks (2)
    with pytest.raises(ValueError):
        _call(core, pools, input_ids, inputs_embeds, idx, num_real_rows=2)
    _assert_pools_equal(pools, before)


def test_advance_model_rows_rejects_duplicate_index() -> None:
    # @spec PORT-STATE-007
    core = _tiny_core()
    pools = _sentinel_pools()
    before = _clone_pools(pools)
    input_ids = torch.tensor([PARK_ID, PARK_ID], dtype=torch.long)
    inputs_embeds = torch.zeros(2, CARRIER_HIDDEN)
    idx = torch.tensor([0, 0], dtype=torch.long)  # same block, two rows
    with pytest.raises(ValueError):
        _call(core, pools, input_ids, inputs_embeds, idx, num_real_rows=2)
    _assert_pools_equal(pools, before)


def test_advance_model_rows_rejects_extra_speculative_column() -> None:
    # @spec PORT-STATE-007
    # A 2-D input_ids (an extra trailing speculative-decode column)
    # where the whole-call contract requires 1-D.
    core = _tiny_core()
    pools = _sentinel_pools()
    before = _clone_pools(pools)
    input_ids = torch.tensor([[PARK_ID], [PARK_ID]], dtype=torch.long)
    inputs_embeds = torch.zeros(2, CARRIER_HIDDEN)
    idx = torch.tensor([0, 1], dtype=torch.long)
    with pytest.raises(ValueError):
        _call(core, pools, input_ids, inputs_embeds, idx, num_real_rows=2)
    _assert_pools_equal(pools, before)


def test_advance_model_rows_rejects_padding_row_among_real() -> None:
    # @spec PORT-STATE-007
    # num_real_rows claims both rows are real, but row 1 carries the
    # graph-padding null index — a real/padding mismatch, not a bare
    # null-index typo (that defect class is tested independently
    # above): the whole call must still fail before any state read.
    core = _tiny_core()
    pools = _sentinel_pools()
    before = _clone_pools(pools)
    input_ids = torch.tensor([PARK_ID, PARK_ID], dtype=torch.long)
    inputs_embeds = torch.zeros(2, CARRIER_HIDDEN)
    idx = torch.tensor([0, NULL_INDEX], dtype=torch.long)
    with pytest.raises(ValueError):
        _call(core, pools, input_ids, inputs_embeds, idx, num_real_rows=2)
    _assert_pools_equal(pools, before)


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
    pools["book_pool"][0, QUEUE_LEN] = 1  # a pending replay label
    idx = torch.tensor([0], dtype=torch.long)
    _call(
        core, pools,
        torch.tensor([42], dtype=torch.long),  # not the placeholder id
        torch.zeros(1, CARRIER_HIDDEN), idx, num_real_rows=1,
    )


def test_flush_only_batch_never_calls_advance_session(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # @spec PORT-ADV-001
    def _recorder(*_args: Any, **_kwargs: Any) -> None:
        raise AssertionError("advance_session must not run for FLUSH rows")

    monkeypatch.setattr(advance_mod, "advance_session", _recorder)
    core = _tiny_core()
    pools = _fresh_pools()  # QUEUE_LEN == 0: a drained queue
    idx = torch.tensor([0], dtype=torch.long)
    _call(
        core, pools,
        torch.tensor([42], dtype=torch.long),  # not placeholder, drained
        torch.zeros(1, CARRIER_HIDDEN), idx, num_real_rows=1,
    )


def test_mixed_batch_calls_advance_session_with_only_chunk_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # @spec PORT-ADV-001
    from vllm_omni.model_executor.models.nemotron_asr.forward_ops import (
        pack_audio_carrier,
    )

    seen: dict[str, object] = {}

    def _recorder(
        _core: Any, batch: ChunkBatch, _state: GatheredState
    ) -> AdvanceResult:
        seen["n_chunk_rows"] = batch.mel.shape[0]
        return AdvanceResult(bursts=[[] for _ in range(batch.mel.shape[0])])

    monkeypatch.setattr(advance_mod, "advance_session", _recorder)
    core = _tiny_core()
    pools = _fresh_pools()
    pools["book_pool"][1, QUEUE_LEN] = 1  # row 1 is a replay row
    mel = torch.randn(1, FEAT, MEL_FRAMES)
    carrier = pack_audio_carrier(mel[0], hidden_size=CARRIER_HIDDEN)
    input_ids = torch.tensor([PLACEHOLDER_ID, 7], dtype=torch.long)
    inputs_embeds = torch.stack(
        [carrier, torch.zeros(CARRIER_HIDDEN)]
    )
    idx = torch.tensor([0, 1], dtype=torch.long)
    _call(core, pools, input_ids, inputs_embeds, idx, num_real_rows=2)
    assert seen["n_chunk_rows"] == 1  # only the one CHUNK row gathered


def test_advance_session_result_contract() -> None:
    # @spec PORT-ADV-001
    # Written as a normal test against the real contract — fails
    # NotImplementedError on the pod until Phase 6 (expected-fail
    # evidence).
    core = _tiny_core()
    torch.manual_seed(5)
    mel = torch.randn(1, FEAT, MEL_FRAMES)
    batch = ChunkBatch(
        mel=mel,
        prompt_index=torch.zeros(1, dtype=torch.long),
        session_first=torch.tensor([True]),
        drop_extra=0,
    )
    state = GatheredState(
        channel=[torch.zeros(1, WINDOW, D_MODEL) for _ in range(N_LAYERS)],
        time=[torch.zeros(1, D_MODEL, KERNEL - 1) for _ in range(N_LAYERS)],
        valid=[torch.zeros(1) for _ in range(N_LAYERS)],
        h=torch.zeros(2, 1, 16),
        c=torch.zeros(2, 1, 16),
        last_label=torch.full((1,), core.blank_id, dtype=torch.long),
    )
    result = advance_session(core, batch, state)
    assert isinstance(result, AdvanceResult)
    assert len(result.bursts) == 1
    assert core.blank_id not in result.bursts[0]
    assert len(result.bursts[0]) <= WINDOW  # bounded by the geometry
    assert set(result.captures) == {
        "frontend_mel", "encoder_raw", "encoder_conditioned",
    }
    for tensors in result.captures.values():
        assert len(tensors) == 1  # one capture per row


# ---- PORT-ADV-003: fresh-session init + echo guard ------------------------


def test_fresh_session_rows_ignore_a_recycled_blocks_poison() -> None:
    # @spec PORT-ADV-003
    # Migrates test_forward_step_zeroes_a_recycled_block_at_session_first
    # onto advance_model_rows: a block carrying a prior session's
    # garbage, but marked session-first, must decode as if fresh.
    from vllm_omni.model_executor.models.nemotron_asr.forward_ops import (
        pack_audio_carrier,
    )

    core = _tiny_core()
    torch.manual_seed(5)
    mel = torch.randn(1, FEAT, MEL_FRAMES)
    carrier = pack_audio_carrier(mel[0], hidden_size=CARRIER_HIDDEN).unsqueeze(0)
    idx = torch.tensor([0], dtype=torch.long)
    input_ids = torch.tensor([PLACEHOLDER_ID], dtype=torch.long)

    clean = _fresh_pools()
    out_clean = _call(core, clean, input_ids, carrier, idx, num_real_rows=1)

    dirty = _fresh_pools()
    for layer in range(N_LAYERS):
        dirty["channel_pools"][layer][0].fill_(3.14)
        dirty["time_pools"][layer][0].fill_(-2.7)
    # len-slots stay 0 → session-first → must be zeroed before use.
    out_dirty = _call(core, dirty, input_ids, carrier, idx, num_real_rows=1)
    torch.testing.assert_close(
        read_decision_carrier(out_dirty), read_decision_carrier(out_clean)
    )


def test_corrupted_mrv1_echo_aborts_the_whole_call() -> None:
    # @spec PORT-ADV-003
    # Migrates test_forward_step_echo_guard_aborts_on_corruption.
    from vllm_omni.model_executor.models.nemotron_asr.forward_ops import (
        pack_audio_carrier,
    )

    core = _tiny_core()
    torch.manual_seed(5)
    mel = torch.randn(1, FEAT, MEL_FRAMES)
    expected = _reference_burst(core, mel, prompt_index=0)
    assert len(expected) >= 2

    pools = _fresh_pools()
    idx = torch.tensor([0], dtype=torch.long)
    carrier = pack_audio_carrier(mel[0], hidden_size=CARRIER_HIDDEN).unsqueeze(0)
    _call(
        core, pools,
        torch.tensor([PLACEHOLDER_ID], dtype=torch.long), carrier, idx,
        num_real_rows=1,
    )
    wrong = expected[0] + 1 if expected[0] + 1 != PARK_ID else 0
    with pytest.raises(ValueError):
        _call(
            core, pools,
            torch.tensor([wrong], dtype=torch.long),
            torch.zeros(1, CARRIER_HIDDEN), idx, num_real_rows=1,
        )


# ---- PORT-STATE-008: no partial scatter on compute failure ----------------


def test_resident_pools_unchanged_when_bucket_compute_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # @spec PORT-STATE-008
    from vllm_omni.model_executor.models.nemotron_asr.encoder import (
        FastConformerEncoder,
    )
    from vllm_omni.model_executor.models.nemotron_asr.forward_ops import (
        pack_audio_carrier,
    )

    def _boom(*_args: Any, **_kwargs: Any) -> Any:
        raise RuntimeError("simulated mid-bucket compute failure")

    monkeypatch.setattr(FastConformerEncoder, "forward", _boom)
    core = _tiny_core()
    pools = _fresh_pools()
    before = _clone_pools(pools)
    mel = torch.randn(1, FEAT, MEL_FRAMES)
    carrier = pack_audio_carrier(mel[0], hidden_size=CARRIER_HIDDEN).unsqueeze(0)
    idx = torch.tensor([0], dtype=torch.long)
    with pytest.raises(RuntimeError):
        _call(
            core, pools,
            torch.tensor([PLACEHOLDER_ID], dtype=torch.long), carrier, idx,
            num_real_rows=1,
        )
    _assert_pools_equal(pools, before)


# ---- PORT-ADV-001 / MRV1 emission: burst-then-park ------------------------


def test_advance_model_rows_emits_the_burst_then_parks() -> None:
    # @spec PORT-ADV-001
    # Migrates test_forward_step_emits_the_burst_then_parks onto
    # advance_model_rows.
    from vllm_omni.model_executor.models.nemotron_asr.forward_ops import (
        pack_audio_carrier,
    )

    core = _tiny_core()
    hidden = CARRIER_HIDDEN
    torch.manual_seed(5)
    mel = torch.randn(1, FEAT, MEL_FRAMES)
    expected = _reference_burst(core, mel, prompt_index=0)
    assert len(expected) >= 2 and len(set(expected)) >= 2, (
        "need a multi-label, multi-distinct burst to test drain order"
    )

    pools = _fresh_pools()
    idx = torch.tensor([0], dtype=torch.long)

    carrier = pack_audio_carrier(mel[0], hidden_size=hidden).unsqueeze(0)
    out = _call(
        core, pools,
        torch.tensor([PLACEHOLDER_ID], dtype=torch.long),
        carrier, idx, num_real_rows=1,
    )
    emitted = [int(read_decision_carrier(out)[0])]
    assert int(pools["book_pool"][0, QUEUE_LEN]) == len(expected)
    assert int(pools["book_pool"][0, QUEUE_HEAD]) == 1

    for _ in range(len(expected) + 1):
        if emitted[-1] == PARK_ID:
            break
        out = _call(
            core, pools,
            torch.tensor([emitted[-1]], dtype=torch.long),
            torch.zeros(1, hidden), idx, num_real_rows=1,
        )
        emitted.append(int(read_decision_carrier(out)[0]))

    assert emitted[:-1] == expected
    assert emitted[-1] == PARK_ID


# ---- migrated, non-stub-dependent tests (real, green already) ------------


def test_persist_across_session_park_by_kind() -> None:
    # @spec PORT-STATE-001
    from vllm_omni.model_executor.models.nemotron_asr.state_layers import (
        ConvCachePage,
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
    assert window.persist_across_session_park is True
    assert conv.persist_across_session_park is True
    assert lstm.persist_across_session_park is True
    assert replay.persist_across_session_park is False


def test_carrier_width_covers_the_measured_largest_chunk() -> None:
    # @spec PORT-INT-003
    from vllm_omni.model_executor.models.nemotron_asr.configuration_nemotron_asr import (  # noqa: E501
        NemotronASRConfig,
    )
    from vllm_omni.model_executor.models.nemotron_asr.featurizer import (
        MelFeaturizer,
    )

    n_mels, overlap = 128, 9
    feat = MelFeaturizer(
        filterbank=torch.rand(n_mels, 257) * 0.01,
        window=torch.hann_window(400),
    )
    samples = 17920
    mel, mel_len = feat(
        torch.randn(1, samples) * 0.01, torch.tensor([samples])
    )
    measured_cols = mel.shape[2]
    assert measured_cols >= int(mel_len[0])
    needed = n_mels * (measured_cols + overlap) + 1
    assert NemotronASRConfig().hidden_size >= needed, (
        f"hidden_size must cover the measured max chunk: need {needed}, "
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
