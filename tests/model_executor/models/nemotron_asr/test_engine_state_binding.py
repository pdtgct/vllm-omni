# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""α2 engine binding: state pages against core's real KV-cache stack.

Specs: PORT-STATE-001..004. The math tier (test_window_state_page.py)
proves ``stream_step`` over page-backed pools bit-for-bit; this suite
binds the pages to the engine machinery itself — spec emission through the
inherited ``MambaBase.get_kv_cache_spec`` seam, mixed-group formation
(the consult's first conformance question: three ``MambaSpec`` page
kinds coexisting with a sliding-window attention spec was unverified
at the pin), zero-at-admission under block reuse (recurrent state is
read-before-write, PORT-STATE-003), the ``SHORT_CONV`` backend seam
the non-conv pages borrow, and forward-context registration.

Exercised against the installed vLLM KV stack on CPU (block
bookkeeping only, no GPU tensors) — the ar_diffusion kv-cache suite's
pattern; runs on the pod tiers, not on machines without vllm.
"""

from types import SimpleNamespace

import pytest
import torch
from vllm.v1.core.kv_cache_utils import get_kv_cache_groups
from vllm.v1.kv_cache_interface import MambaSpec, SlidingWindowSpec

from vllm_omni.model_executor.models.nemotron_asr.precision import (
    FP32_BRINGUP,
)
from vllm_omni.model_executor.models.nemotron_asr.state_layers import (
    ConvCachePage,
    HybridStateModelMixin,
    LSTMStatePage,
    ReplayQueuePage,
    register_state_pages,
    zero_state_pages,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

BLOCK = 16
D_MODEL = 1024
KERNEL = 9


#: The attention page size for this suite's geometry (block 16 ×
#: 8 kv-heads × 128 head × fp32 × K+V) — the value core's
#: `_align_hybrid_block_size` hook computes and stamps into
#: ``cache_config.mamba_page_size_padded`` for our numbers (consult
#: D-α2a arithmetic: per-token 8192 B, block size stays 16).
ATTN_PAGE_BYTES = 16 * 8 * 128 * 4 * 2


def duck_vllm_config(
    *,
    mamba_block_size: int = 4096,
    mamba_page_size_padded: int | None = None,
    mamba_cache_mode: str = "none",
) -> SimpleNamespace:
    """The slice of VllmConfig the exercised seams actually read.

    ``MambaBase.get_kv_cache_spec`` touches ``cache_config.mamba_*``
    and ``speculative_config``; ``get_kv_cache_groups`` touches
    ``scheduler_config.disable_hybrid_kv_cache_manager`` on our path.
    Duck-typing keeps the suite off VllmConfig's model-loading ctor
    (``ModelRegistry`` resolution inside the platform hook needs the
    α4 registration — the full two-stage integration test is a
    bring-up rung). ``mamba_block_size`` defaults to a
    max-model-len-like value: core sets it to ``max_model_len`` for
    mode "none" — never a bespoke small block.
    """
    return SimpleNamespace(
        cache_config=SimpleNamespace(
            mamba_block_size=mamba_block_size,
            mamba_page_size_padded=mamba_page_size_padded,
            mamba_cache_mode=mamba_cache_mode,
        ),
        speculative_config=None,
        scheduler_config=SimpleNamespace(
            disable_hybrid_kv_cache_manager=False
        ),
    )


def make_pages():
    conv = ConvCachePage(
        prefix="encoder.layers.0.conv",
        d_model=D_MODEL,
        kernel=KERNEL,
        policy=FP32_BRINGUP,
    )
    lstm = LSTMStatePage(
        prefix="predictor.state",
        pred_rnn_layers=2,
        pred_hidden=640,
        policy=FP32_BRINGUP,
    )
    queue = ReplayQueuePage(
        prefix="decode.replay",
        max_symbols_per_step=10,
        max_frames_per_chunk=7,
        policy=FP32_BRINGUP,
    )
    return conv, lstm, queue


def sliding_window_attn_spec() -> SlidingWindowSpec:
    """The left-context attention spec shape (α1's storage class)."""
    return SlidingWindowSpec(
        block_size=BLOCK,
        num_kv_heads=8,
        head_size=D_MODEL // 8,
        dtype=torch.float32,
        sliding_window=56,
    )


# ---- spec emission through the inherited seam (PORT-STATE-001/002) ----------


def test_pages_emit_mamba_specs_via_the_inherited_seam():
    cfg = duck_vllm_config()
    for page in make_pages():
        spec = page.get_kv_cache_spec(cfg)
        assert isinstance(spec, MambaSpec)
        assert spec.shapes == tuple(page.get_state_shape())
        assert spec.dtypes == page.get_state_dtype()
        assert spec.mamba_type is page.mamba_type
        # One page per session: the engine's cache mode rides through
        # untouched (kv_cache_interface's mamba_cache_mode="none").
        assert spec.mamba_cache_mode == "none"


def test_page_size_math_covers_every_state_tensor():
    # Multi-tensor pages (LSTM h+c, queue slots+bookkeeping) must size
    # their page over ALL constituent tensors.
    cfg = duck_vllm_config()
    for page in make_pages():
        spec = page.get_kv_cache_spec(cfg)
        payload = sum(
            int(torch.Size(shape).numel()) * dtype.itemsize
            for shape, dtype in zip(spec.shapes, spec.dtypes)
        )
        assert spec.page_size_bytes >= payload


# ---- mixed-group formation (pre-equalized, the way core inits run) -----------
#
# Post-migration (OPEN-α3-VEHICLE) this model is attention-spec-free —
# its own grouping topology is the pure-state dict exercised in
# test_window_state_page.py. The mixed attention+state tests below are
# KEPT as documentation of core's hybrid path at the pin (the
# pre-equalization behavior α2 measured), so a core change to that
# path still breaks loudly here.


def preequalized_spec_dict(cfg) -> dict:
    """The spec mix as core's init sequence actually produces it.

    Core hybrids pre-equalize, never unify: the post-load platform
    hook sets ``cache_config.mamba_page_size_padded`` to the attention
    page size before any spec is built, so every ``MambaSpec`` is born
    padded (consult D-α2a/c; ``_align_hybrid_block_size`` @ the pin).
    """
    conv, lstm, queue = make_pages()
    attn = sliding_window_attn_spec()
    return {
        "encoder.layers.0.attn": attn,
        "encoder.layers.1.attn": attn,
        "encoder.layers.0.conv": conv.get_kv_cache_spec(cfg),
        "encoder.layers.1.conv": ConvCachePage(
            prefix="encoder.layers.1.conv",
            d_model=D_MODEL,
            kernel=KERNEL,
            policy=FP32_BRINGUP,
        ).get_kv_cache_spec(cfg),
        "predictor.state": lstm.get_kv_cache_spec(cfg),
        "decode.replay": queue.get_kv_cache_spec(cfg),
    }


def test_preequalized_mixed_groups_form():
    # With the padding the platform hook stamps, grouping accepts the
    # full mix: attention and page state never share a group, and
    # every layer lands in exactly one group.
    cfg = duck_vllm_config(mamba_page_size_padded=ATTN_PAGE_BYTES)
    spec_dict = preequalized_spec_dict(cfg)
    groups = get_kv_cache_groups(cfg, spec_dict)

    grouped_layers = [
        name for group in groups for name in group.layer_names
    ]
    assert sorted(grouped_layers) == sorted(spec_dict)
    for group in groups:
        kinds = {type(spec_dict[name]) for name in group.layer_names}
        assert kinds in ({SlidingWindowSpec}, {MambaSpec})


def test_preequalized_pages_report_one_page_size_and_unify_is_identity():
    # Every born-padded spec reports the attention page size, so
    # unify_kv_cache_spec_page_size early-returns (identity) and the
    # allocator sees exactly one pool page size.
    from vllm.v1.core.kv_cache_utils import unify_kv_cache_spec_page_size

    cfg = duck_vllm_config(mamba_page_size_padded=ATTN_PAGE_BYTES)
    spec_dict = preequalized_spec_dict(cfg)
    assert {
        spec.page_size_bytes for spec in spec_dict.values()
    } == {ATTN_PAGE_BYTES}
    unified = unify_kv_cache_spec_page_size(dict(spec_dict))
    assert unified == spec_dict
    groups = get_kv_cache_groups(cfg, spec_dict)
    page_sizes = {group.kv_cache_spec.page_size_bytes for group in groups}
    assert page_sizes == {ATTN_PAGE_BYTES}


def test_raw_unpadded_mix_still_dies_in_core_unification():
    # Documents UPSTREAM behavior, not ours (consult D-α2c/d): feeding
    # unpadded MambaSpecs into the non-uniform grouping path trips the
    # divisible branch's post-condition (page size does not scale with
    # block_size for constant-state specs) — a bare AssertionError at
    # the pin. If core changes this branch, this test breaks loudly
    # and the RFC observation gets revisited.
    cfg = duck_vllm_config(mamba_page_size_padded=None)
    spec_dict = preequalized_spec_dict(cfg)
    with pytest.raises((AssertionError, NotImplementedError)):
        get_kv_cache_groups(cfg, spec_dict)


# ---- IsHybrid conformance surface (consult D-α2a) -----------------------------


def test_mixin_is_attention_free_and_reports_the_window_bundle():
    # Post-migration (OPEN-α3-VEHICLE): every state kind is a MambaSpec
    # page, so there is no attention spec to align against — is_hybrid
    # is False and the reference bundle is the largest per-layer kind,
    # now the window page.
    assert HybridStateModelMixin.is_hybrid is False
    shapes = HybridStateModelMixin.get_mamba_state_shape_from_config(
        duck_vllm_config()
    )
    assert shapes == ((56, D_MODEL), (1,))
    dtypes = HybridStateModelMixin.get_mamba_state_dtype_from_config(
        duck_vllm_config()
    )
    # One dtype per bundle tensor (channel cache + valid slot), fp32.
    assert dtypes == (torch.float32, torch.float32)


def test_hybrid_mixin_copy_func_is_a_loud_seam():
    # Align-mode prefix caching is off for v1; the hook must fail
    # loudly if something turns it on, never silently no-op.
    with pytest.raises(NotImplementedError):
        HybridStateModelMixin.get_mamba_state_copy_func(duck_vllm_config())


# ---- zero-at-admission under block reuse (PORT-STATE-003) -------------------


def test_reused_block_is_zeroed_at_admission():
    # Recurrent state is read-before-write: a page block freed by one
    # session and reallocated to another MUST be zeroed at admission,
    # or session B decodes from session A's state.
    num_blocks = 4
    state = [
        torch.zeros(num_blocks, D_MODEL, KERNEL - 1),
        torch.zeros(num_blocks, 2, 640),
    ]
    # Session A lived on block 2 and died without cleanup.
    state[0][2] = 7.5
    state[1][2] = -3.25
    assert state[0][2].abs().sum() > 0  # the hazard is real
    zero_state_pages(state, block_ids=[2])
    assert state[0][2].abs().sum() == 0
    assert state[1][2].abs().sum() == 0
    # Other sessions' blocks are untouched.
    state[0][1] = 1.0
    zero_state_pages(state, block_ids=[2])
    assert state[0][1].abs().sum() > 0


# ---- the SHORT_CONV backend seam (borrowed by non-conv pages) ----------------


def test_every_page_resolves_the_short_conv_backend():
    # The LSTM and replay pages BORROW mamba_type=SHORT_CONV (the
    # landed constant-size-state vehicle). The seam guard: the enum
    # member resolves to a real backend class for every page, so a
    # core rename/removal breaks loudly here, not at bring-up.
    from vllm.v1.attention.backends.registry import (
        MambaAttentionBackendEnum,
    )

    for page in make_pages():
        assert page.mamba_type is MambaAttentionBackendEnum.SHORT_CONV
        backend = page.get_attn_backend()
        assert backend is not None
        assert hasattr(backend, "get_builder_cls")


# ---- forward-context registration (PORT-STATE-002) ---------------------------


def test_register_state_pages_lands_each_prefix_once():
    pages = make_pages()
    ctx: dict[str, object] = {}
    cfg = SimpleNamespace(
        compilation_config=SimpleNamespace(static_forward_context=ctx)
    )
    register_state_pages(cfg, pages)
    for page in pages:
        assert ctx[page.prefix] is page


def test_register_state_pages_refuses_duplicate_prefixes():
    conv, _, _ = make_pages()
    duplicate = ConvCachePage(
        prefix=conv.prefix,
        d_model=D_MODEL,
        kernel=KERNEL,
        policy=FP32_BRINGUP,
    )
    ctx: dict[str, object] = {}
    cfg = SimpleNamespace(
        compilation_config=SimpleNamespace(static_forward_context=ctx)
    )
    with pytest.raises(ValueError):
        register_state_pages(cfg, (conv, duplicate))
