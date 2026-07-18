# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Exercised advance_session coverage (PORT-ADV-001), locally.

The transition's whole dependency closure is torch-only (featurizer,
masks, encoder, lid, manifests, frontend, rnnt_cell, rnnt, advance) —
loaded by file path under a stubbed package chain, the CANONICAL
transition executes on macOS. This is the seam the review found
uncovered: the local tier was green without ever running
``advance_session`` end-to-end.
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
        "manifests", "frontend", "rnnt_cell", "rnnt", "advance",
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

FEAT = 16
D_MODEL = 32
N_LAYERS = 2
KERNEL = 5
WINDOW = 8
VOCAB = 12
CHUNK = 17_920  # 1120 ms cadence (geometry id 4, lookahead 13)
GEOMETRY_1120 = 4
RAW_TAIL = 1_953


def _core() -> Any:
    torch.manual_seed(7)
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


def _fresh_state(batch: int) -> Any:
    return advance.SessionStateBatch(
        raw_tail=torch.zeros(batch, RAW_TAIL),
        mel_tail=torch.zeros(batch, FEAT, frontend.MEL_TAIL_FRAMES),
        frontend_counters=torch.zeros(batch, 8, dtype=torch.int64),
        channel=[
            torch.zeros(batch, WINDOW, D_MODEL) for _ in range(N_LAYERS)
        ],
        window_valid=[
            torch.zeros(batch, 1, dtype=torch.int32)
            for _ in range(N_LAYERS)
        ],
        time=[
            torch.zeros(batch, D_MODEL, KERNEL - 1)
            for _ in range(N_LAYERS)
        ],
        h=torch.zeros(batch, 2, 16),
        c=torch.zeros(batch, 2, 16),
        last_label=torch.full((batch,), VOCAB, dtype=torch.long),
    )


def _chunk(
    samples: torch.Tensor,
    *,
    seq: int,
    final: bool = False,
    prompts: list[int] | None = None,
) -> Any:
    batch = samples.shape[0]
    return advance.ChunkBatch(
        samples=samples,
        valid_samples=torch.full(
            (batch,), samples.shape[1], dtype=torch.long
        ),
        geometry_id=torch.full(
            (batch,), GEOMETRY_1120, dtype=torch.long
        ),
        final_tail=torch.full((batch,), final, dtype=torch.bool),
        prompt_index=torch.tensor(
            prompts if prompts is not None else [0] * batch,
            dtype=torch.long,
        ),
        chunk_sequence=torch.full((batch,), seq, dtype=torch.long),
    )


def test_session_first_chunk_advances_and_captures() -> None:
    # @spec PORT-ADV-001
    # The canonical transition, exercised: session-first 1120 ms chunk
    # commits B_1 = 105 mel frames, runs encoder/LID/decode, advances
    # every counter, and (capture ON) stages the three named tensors
    # with exact lengths.
    core = _core()
    state = _fresh_state(1)
    torch.manual_seed(21)
    samples = torch.randn(1, CHUNK) * 0.1
    result = advance.advance_session(
        core, _chunk(samples, seq=0), state, capture=True
    )
    ctr = state.frontend_counters[0]
    b1 = frontend.cadence_boundary(1, lookahead=13)
    assert int(ctr[frontend.CTR_COMMITTED_MEL_FRAMES]) == b1 == 105
    assert int(ctr[frontend.CTR_ENCODED_MEL_FRAMES]) == b1
    assert int(ctr[frontend.CTR_EXPECTED_CHUNK_SEQUENCE]) == 1
    caps = result.captures
    assert caps is not None
    assert caps.frontend_mel.shape == (1, FEAT, b1)  # no prefix, first
    assert int(caps.encoder_lengths[0]) == caps.encoder_raw.shape[1]
    assert result.token_ids.shape[0] == 1
    # The encoder window advanced: valid length grew and is mirrored
    # across every layer's slot.
    assert int(state.window_valid[0][0, 0]) > 0
    assert int(state.window_valid[1][0, 0]) == int(
        state.window_valid[0][0, 0]
    )


def test_capture_off_returns_none() -> None:
    # @spec PORT-HOOK-001
    core = _core()
    state = _fresh_state(1)
    torch.manual_seed(22)
    samples = torch.randn(1, CHUNK) * 0.1
    result = advance.advance_session(core, _chunk(samples, seq=0), state)
    assert result.captures is None
    assert result.token_ids.shape[0] == 1


def test_continuing_chunk_prepends_the_prefix_and_drops() -> None:
    # @spec PORT-ADV-001
    # Chunk 2 consumes [9-slot prefix + C frames] with the configured
    # drop; commits land exactly on B_2.
    core = _core()
    state = _fresh_state(1)
    torch.manual_seed(23)
    samples = torch.randn(1, 2 * CHUNK) * 0.1
    advance.advance_session(
        core, _chunk(samples[:, :CHUNK], seq=0), state
    )
    result = advance.advance_session(
        core, _chunk(samples[:, CHUNK:], seq=1), state, capture=True
    )
    b2 = frontend.cadence_boundary(2, lookahead=13)
    assert int(
        state.frontend_counters[0, frontend.CTR_COMMITTED_MEL_FRAMES]
    ) == b2
    caps = result.captures
    assert caps is not None
    # Prefix (9) + the C = 112 new frames.
    assert caps.frontend_mel.shape[2] == frontend.MEL_TAIL_FRAMES + 112


def test_wrong_sequence_is_rejected_before_mutation() -> None:
    # @spec PORT-ADV-001
    core = _core()
    state = _fresh_state(1)
    before = state.frontend_counters.clone()
    torch.manual_seed(24)
    samples = torch.randn(1, CHUNK) * 0.1
    with pytest.raises(ValueError, match="sequence"):
        advance.advance_session(core, _chunk(samples, seq=3), state)
    torch.testing.assert_close(
        state.frontend_counters, before, rtol=0, atol=0
    )


def test_zero_frame_final_skips_model_work_and_stages_captures() -> None:
    # @spec PORT-ADV-001 / PORT-HOOK-001
    # A final tail whose short residual is dropped: no encoder/LID/
    # decode work, state untouched beyond finalization, and (capture
    # ON) all three named tensors present at zero valid length.
    core = _core()
    state = _fresh_state(1)
    torch.manual_seed(25)
    samples = torch.randn(1, CHUNK) * 0.1
    advance.advance_session(core, _chunk(samples, seq=0), state)
    h_before = state.h.clone()
    window_before = state.channel[0].clone()
    result = advance.advance_session(
        core,
        _chunk(torch.zeros(1, 0), seq=1, final=True),
        state,
        capture=True,
    )
    assert int(
        state.frontend_counters[0, frontend.CTR_FINALIZED]
    ) == 1
    torch.testing.assert_close(state.h, h_before, rtol=0, atol=0)
    torch.testing.assert_close(
        state.channel[0], window_before, rtol=0, atol=0
    )
    assert int(result.token_lengths[0]) == 0
    caps = result.captures
    assert caps is not None
    assert caps.frontend_mel.shape[2] == 0
    assert int(caps.mel_lengths[0]) == 0
    assert int(caps.encoder_lengths[0]) == 0


def test_mixed_prompts_condition_row_wise() -> None:
    # @spec PORT-ADV-001
    # Two rows, identical audio, different prompts: conditioned
    # captures must differ (no shared multi-hot prompt), and each row
    # must equal its single-row run exactly.
    core = _core()
    torch.manual_seed(26)
    samples = torch.randn(1, CHUNK) * 0.1
    two = samples.repeat(2, 1)
    state2 = _fresh_state(2)
    result2 = advance.advance_session(
        core, _chunk(two, seq=0, prompts=[0, 2]), state2, capture=True
    )
    caps2 = result2.captures
    assert caps2 is not None
    assert bool(
        (
            caps2.encoder_conditioned[0] != caps2.encoder_conditioned[1]
        ).any()
    )
    for row, prompt in ((0, 0), (1, 2)):
        state1 = _fresh_state(1)
        r1 = advance.advance_session(
            core, _chunk(samples, seq=0, prompts=[prompt]), state1,
            capture=True,
        )
        c1 = r1.captures
        assert c1 is not None
        torch.testing.assert_close(
            caps2.encoder_conditioned[row],
            c1.encoder_conditioned[0],
            rtol=0,
            atol=1e-6,  # provisional cross-batch-shape bound
        )
        assert int(result2.token_lengths[row]) == int(
            r1.token_lengths[0]
        )
