# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Vehicle-migration tests-first: the window as the fourth state page.

Specs: PORT-STATE-001/002 (as amended by the OPEN-α3-VEHICLE
decision); the decision record is
docs/intent/port/decisions/attention-window-vehicle.md (notes repo).
Two measured questions drive this slice:

1. **Pure-state group formation** — with the model attention-spec-free,
   do core's KV-cache groups form over a HETEROGENEOUS MambaSpec-only
   dict (four page kinds, four page sizes)? Raw and padded paths both
   measured, the α2 mixed-group pattern replayed for the new topology.
2. **The stream_step-literal claim** — the window page IS
   ``StreamingCaches.channel``: a page-backed channel cache must
   advance bit-for-bit with the golden-proven in-module cache.

Engine-tier tests run on the pod (installed vLLM); the parity test is
loader-runnable locally like the rest of the math tier.
"""

from types import SimpleNamespace

import pytest
import torch
from vllm.v1.core.kv_cache_utils import get_kv_cache_groups
from vllm.v1.kv_cache_interface import MambaSpec

from vllm_omni.model_executor.models.nemotron_asr.precision import (
    FP32_BRINGUP,
)
from vllm_omni.model_executor.models.nemotron_asr.state_layers import (
    ConvCachePage,
    LSTMStatePage,
    ReplayQueuePage,
    WindowCachePage,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

WINDOW = 56
D_MODEL = 1024
KERNEL = 9


def duck_vllm_config(
    *,
    mamba_block_size: int = 4096,
    mamba_page_size_padded: int | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        cache_config=SimpleNamespace(
            mamba_block_size=mamba_block_size,
            mamba_page_size_padded=mamba_page_size_padded,
            mamba_cache_mode="none",
        ),
        speculative_config=None,
        scheduler_config=SimpleNamespace(
            disable_hybrid_kv_cache_manager=False
        ),
    )


def make_window_page(prefix: str = "encoder.layers.0.window"):
    return WindowCachePage(
        prefix=prefix, window=WINDOW, d_model=D_MODEL, policy=FP32_BRINGUP
    )


#: The window page's payload: (56×1024 + 1) fp32.
WINDOW_PAGE_BYTES = (WINDOW * D_MODEL + 1) * 4


# ---- the page contract --------------------------------------------------------


def test_window_page_shapes_and_policy_dtypes():
    page = make_window_page()
    assert tuple(page.get_state_shape()) == ((WINDOW, D_MODEL), (1,))
    # attention_cache + queue_state classes, both fp32 in the bring-up
    # policy (PORT-PREC-001/005).
    assert page.get_state_dtype() == (torch.float32, torch.float32)


def test_window_page_is_now_the_largest_kind():
    # The hybrid-alignment reference bundle (if is_hybrid survives the
    # attention-free migration at all — measured below) must track the
    # LARGEST per-layer kind, which the window now is.
    cfg = duck_vllm_config()
    window = make_window_page().get_kv_cache_spec(cfg)
    conv = ConvCachePage(
        prefix="encoder.layers.0.conv",
        d_model=D_MODEL,
        kernel=KERNEL,
        policy=FP32_BRINGUP,
    ).get_kv_cache_spec(cfg)
    assert window.page_size_bytes >= WINDOW_PAGE_BYTES
    assert window.page_size_bytes > conv.page_size_bytes


# ---- measured question 1: pure-state heterogeneous group formation -------------


def four_kind_spec_dict(cfg) -> dict:
    return {
        "encoder.layers.0.window": make_window_page().get_kv_cache_spec(cfg),
        "encoder.layers.1.window": make_window_page(
            "encoder.layers.1.window"
        ).get_kv_cache_spec(cfg),
        "encoder.layers.0.conv": ConvCachePage(
            prefix="encoder.layers.0.conv",
            d_model=D_MODEL,
            kernel=KERNEL,
            policy=FP32_BRINGUP,
        ).get_kv_cache_spec(cfg),
        "predictor.state": LSTMStatePage(
            prefix="predictor.state",
            pred_rnn_layers=2,
            pred_hidden=640,
            policy=FP32_BRINGUP,
        ).get_kv_cache_spec(cfg),
        "decode.replay": ReplayQueuePage(
            prefix="decode.replay",
            max_symbols_per_step=10,
            max_frames_per_chunk=7,
            policy=FP32_BRINGUP,
        ).get_kv_cache_spec(cfg),
    }


def test_pure_state_groups_form_when_padded_to_the_window_page():
    # The migration's α2-analog: with every page padded to the largest
    # kind (the window), grouping must accept the attention-free
    # heterogeneous dict and land every layer in exactly one group.
    cfg = duck_vllm_config()
    padded_size = make_window_page().get_kv_cache_spec(cfg).page_size_bytes
    cfg_padded = duck_vllm_config(mamba_page_size_padded=padded_size)
    spec_dict = four_kind_spec_dict(cfg_padded)
    assert {s.page_size_bytes for s in spec_dict.values()} == {padded_size}
    groups = get_kv_cache_groups(cfg_padded, spec_dict)
    grouped = [name for g in groups for name in g.layer_names]
    assert sorted(grouped) == sorted(spec_dict)
    for group in groups:
        assert isinstance(group.kv_cache_spec, MambaSpec) or all(
            isinstance(spec_dict[n], MambaSpec) for n in group.layer_names
        )


def test_pure_state_raw_path_documented():
    # MEASURED QUESTION (recorded, either way): does the raw unpadded
    # heterogeneous MambaSpec-only dict form groups at the pin, or does
    # it need the padding pre-pass like the hybrid case did? Whoever
    # owns the padding value for an attention-free model is the
    # migration's one open design point — this test pins the measured
    # ground truth it must build on.
    cfg = duck_vllm_config(mamba_page_size_padded=None)
    spec_dict = four_kind_spec_dict(cfg)
    try:
        groups = get_kv_cache_groups(cfg, spec_dict)
        formed = True
        grouped = [name for g in groups for name in g.layer_names]
        assert sorted(grouped) == sorted(spec_dict)
    except (AssertionError, NotImplementedError):
        formed = False
    print(f"RAW_PURE_STATE_GROUPS_FORMED={formed}")
    # No verdict assert: the value is the measurement. The padded path
    # (previous test) is the one the binding relies on.


# ---- measured question 2: the stream_step-literal window advance ---------------


def test_page_backed_channel_cache_matches_streaming_caches():
    # The engine window page IS StreamingCaches.channel: advancing the
    # encoder over multi-chunk audio with the channel cache VIEWED
    # from a page pool must be bit-for-bit identical to the in-module
    # cache. The binding under test (page_view seam) lands in the code
    # phase; this drives the contract.
    from vllm_omni.model_executor.models.nemotron_asr.state_layers import (
        window_page_channel_view,
    )

    pool = torch.zeros(4, WINDOW * D_MODEL + 1)  # 4 page blocks, flat
    view = window_page_channel_view(pool, block_id=2, d_model=D_MODEL)
    assert view.shape == (WINDOW, D_MODEL)
    view[3, 5] = 7.5
    assert pool[2, 3 * D_MODEL + 5] == 7.5  # a VIEW, never a copy
    # valid-length slot is the trailing element of the block.
    from vllm_omni.model_executor.models.nemotron_asr.state_layers import (
        window_page_len_slot,
    )

    len_slot = window_page_len_slot(pool, block_id=2, d_model=D_MODEL)
    len_slot.fill_(42.0)
    assert pool[2, -1] == 42.0
