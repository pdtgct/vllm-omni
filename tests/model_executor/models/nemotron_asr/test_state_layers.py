# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""State-page layers: shapes and policy-delegated dtypes.

Specs: PORT-STATE-001/002 (all cross-chunk state via spec pages, one
page per session), PORT-PREC-001/005 (dtypes delegate to the
PrecisionPolicy; recurrent state defaults fp32). The MambaSpec spec
emission itself is engine-side (exercised at bring-up); these GPU-free
tests pin the contract surface the engine reads.
"""

import pytest
import torch

from vllm_omni.model_executor.models.nemotron_asr.precision import (
    FP32_BRINGUP,
)
from vllm_omni.model_executor.models.nemotron_asr.state_layers import (
    ConvCachePage,
    LSTMStatePage,
    ReplayQueuePage,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_conv_cache_page_shape_matches_nemo_cache_last_time():
    page = ConvCachePage(
        prefix="encoder.layers.0.conv",
        d_model=1024,
        kernel=9,
        policy=FP32_BRINGUP,
    )
    assert tuple(page.get_state_shape()) == ((1024, 8),)
    assert page.get_state_dtype() == (torch.float32,)


def test_lstm_page_holds_h_and_c_at_fp32():
    page = LSTMStatePage(
        prefix="predictor.state",
        pred_rnn_layers=2,
        pred_hidden=640,
        policy=FP32_BRINGUP,
    )
    assert tuple(page.get_state_shape()) == ((2, 640), (2, 640))
    assert page.get_state_dtype() == (torch.float32, torch.float32)


def test_replay_queue_capacity_is_symbols_times_frames():
    # 560ms chunk: 10 symbols/step x 7 frames = 70 slots (PORT-INT-002's
    # intrinsic backstop); widest chunk (1120ms) gives 140.
    page = ReplayQueuePage(
        prefix="decode.queue",
        max_symbols_per_step=10,
        max_frames_per_chunk=14,
        policy=FP32_BRINGUP,
    )
    shapes = tuple(page.get_state_shape())
    assert shapes == ((140,), (4,))
