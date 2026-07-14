# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""BU-c1: the forward step over page-backed state (loader-runnable).

Specs: PORT-DEC-001/002/003/007/009, PORT-STATE-003, PORT-INT-003.
Consult: D-BUc-1 (the fixed pipeline + four ordering hazards),
D-BUc-3 (carrier width is a measurement), D-BUc-4 (the lifecycle
classifier).

``run_forward_step`` is the pure compute the engine ``forward`` runs
over bound pools; here it is driven with synthetic pools and a tiny
core, its emit/drain/echo behaviour checked against a reference built
from the same golden ``stream_step`` + ``greedy_decode_batch`` the
implementation composes (so the reference is self-validating). The
engine-context extraction in ``forward`` is pod-tier (BU-c2).
"""

import pytest
import torch

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


def _tiny_core():
    # A minimal core namespace (the featurizer is unused by the forward
    # step — chunk rows arrive as packed mel carriers). Seed 7 in
    # component order yields a deterministic cap-saturated burst with two
    # distinct labels in order ([4…, 7…]) — good drain-order test data;
    # the tiny random joint is otherwise bimodal (all-one-label or
    # all-blank), so this is chosen, not incidental.
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


def _reference_burst(core, mel, prompt_index):
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


def _fresh_pools(num_blocks=2):
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


def _step(core, pools, input_ids, inputs_embeds, state_indices):
    from vllm_omni.model_executor.models.nemotron_asr.forward_step import (
        run_forward_step,
    )

    return run_forward_step(
        core, input_ids, inputs_embeds,
        state_indices=state_indices,
        placeholder_id=PLACEHOLDER_ID, park_id=PARK_ID, feat=FEAT,
        drop_extra=0, **pools,
    )


# ---- the emit/drain cycle (PORT-DEC-001/002/003, D-BUc-1) ----------------------


def test_forward_step_emits_the_burst_then_parks():
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

    # Chunk step: one placeholder row carrying the packed mel.
    carrier = pack_audio_carrier(mel[0], hidden_size=hidden).unsqueeze(0)
    out = _step(
        core, pools,
        torch.tensor([PLACEHOLDER_ID], dtype=torch.long),
        carrier, idx,
    )
    emitted = [int(read_decision_carrier(out)[0])]
    # The chunk step filled the queue and drained its first label.
    assert int(pools["book_pool"][0, QUEUE_LEN]) == len(expected)
    assert int(pools["book_pool"][0, QUEUE_HEAD]) == 1

    # Replay steps: the engine feeds back the prior emission as input.
    for _ in range(len(expected) + 1):
        if emitted[-1] == PARK_ID:
            break
        out = _step(
            core, pools,
            torch.tensor([emitted[-1]], dtype=torch.long),
            torch.zeros(1, hidden), idx,
        )
        emitted.append(int(read_decision_carrier(out)[0]))

    assert emitted[:-1] == expected  # every label, in order
    assert emitted[-1] == PARK_ID  # then park on the drained queue


def test_forward_step_echo_guard_aborts_on_corruption():
    # A replay row whose fed-back id does not match the carrier-forced
    # id last step is a corrupted session — abort loudly (PORT-DEC-007).
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
    _step(
        core, pools,
        torch.tensor([PLACEHOLDER_ID], dtype=torch.long), carrier, idx,
    )
    # Feed a WRONG id (not the just-emitted label) on the replay step.
    wrong = expected[0] + 1 if expected[0] + 1 != PARK_ID else 0
    with pytest.raises(ValueError):
        _step(
            core, pools,
            torch.tensor([wrong], dtype=torch.long),
            torch.zeros(1, CARRIER_HIDDEN), idx,
        )


def test_forward_step_zeroes_a_recycled_block_at_session_first():
    # PORT-STATE-003 / ordering hazard 1: a block carrying a prior
    # session's state (len-slot != 0 would mean live; a fresh session
    # has len-slot == 0) must be read-before-write safe — the encoder
    # must see zeroed window/conv for a session-first chunk. Here a
    # block pre-loaded with garbage but len-slot 0 must decode as if
    # fresh (identical to a clean block).
    from vllm_omni.model_executor.models.nemotron_asr.forward_ops import (
        pack_audio_carrier,
    )

    core = _tiny_core()
    torch.manual_seed(5)
    mel = torch.randn(1, FEAT, MEL_FRAMES)
    carrier = pack_audio_carrier(mel[0], hidden_size=CARRIER_HIDDEN).unsqueeze(0)
    idx = torch.tensor([0], dtype=torch.long)

    clean = _fresh_pools()
    out_clean = _step(
        core, clean,
        torch.tensor([PLACEHOLDER_ID], dtype=torch.long), carrier, idx,
    )

    dirty = _fresh_pools()
    for layer in range(N_LAYERS):
        dirty["channel_pools"][layer][0].fill_(3.14)
        dirty["time_pools"][layer][0].fill_(-2.7)
    # len-slots stay 0 → session-first → must be zeroed before use.
    out_dirty = _step(
        core, dirty,
        torch.tensor([PLACEHOLDER_ID], dtype=torch.long), carrier, idx,
    )
    torch.testing.assert_close(
        read_decision_carrier(out_dirty), read_decision_carrier(out_clean)
    )


# ---- the lifecycle classifier (D-BUc-4) — real, green --------------------------


def test_persist_across_session_park_by_kind():
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
    # window / conv / lstm are persistent recurrent state; the replay
    # queue is intra-burst scratch (offload-exempt).
    assert window.persist_across_session_park is True
    assert conv.persist_across_session_park is True
    assert lstm.persist_across_session_park is True
    assert replay.persist_across_session_park is False


# ---- carrier width is a measurement, not the formula (D-BUc-3) -----------------


def test_carrier_width_covers_the_measured_largest_chunk():
    # Run the REAL featurizer over the largest admitted chunk (1120 ms
    # @ 16 kHz = 17920 samples) and size the carrier from the measured
    # column count, not get_seq_len (stft center=True yields one extra
    # column the estimate hides). The config's hidden_size must cover
    # 128 × (measured + pre-encode overlap) + 1.
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
    # stft center=True gives one more column than output_lengths — the
    # slack the formula omits.
    assert measured_cols >= int(mel_len[0])
    needed = n_mels * (measured_cols + overlap) + 1
    assert NemotronASRConfig().hidden_size >= needed, (
        f"hidden_size must cover the measured max chunk: need {needed}, "
        f"config has {NemotronASRConfig().hidden_size}"
    )


# ---- canonical embed_input_ids (D-BUc-2) --------------------------------------


def test_embed_input_ids_canonical_merge_places_carriers_and_zeros():
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
