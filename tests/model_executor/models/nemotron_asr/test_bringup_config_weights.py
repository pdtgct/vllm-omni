# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Bring-up BU-a: config authoring, weight loading, model __init__.

Specs: PORT-WGT-004 (the conversion publisher authors config.json —
vocab_size DERIVED from checkpoint tensor shapes, hard-fail on
metadata disagreement; hidden_size = the mm-carrier width; AutoConfig
registration), PORT-WGT-001/003 (load_weights by the name ledger,
prompt_kernel-absence fatal), PORT-INT-001 (the shared ARCHITECTURE
constant), PORT-STATE-002 (F4-compliant page prefixes). Consult:
D-BU-5, and OPEN-α4-ARCH (the 13087-vs-13088 vocab off-by-one settles
by derivation, resolved against the real checkpoint at the pod round).

CPU/loader-runnable: transformers + torch only, no vllm. The
registry-driven config load and _configured_vllm_config are pod-tier
(test_omni_serving_binding.py).
"""

from types import SimpleNamespace

import pytest
import torch

from vllm_omni.model_executor.models.nemotron_asr.configuration_nemotron_asr import (
    ARCHITECTURE,
    MODEL_TYPE,
    NemotronASRConfig,
)
from vllm_omni.model_executor.models.nemotron_asr.convert import (
    ConversionError,
    author_config,
    derive_vocab_size,
)
from vllm_omni.model_executor.models.nemotron_asr.state_layers import (
    state_page_prefixes,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

V = 13087  # the label-set size for this checkpoint (derived, not asserted)
JOINT_HIDDEN = 640
PRED_HIDDEN = 640


def _checkpoint_shaped_state_dict(v: int = V) -> dict:
    """A converted state dict with the two vocab-bearing tensors.

    The joint final linear is ``(V+1, joint_hidden)`` (blank last);
    the predictor embedding is ``(V+1, pred_hidden)``.
    """
    return {
        "joint.joint_net.1.weight": torch.zeros(v + 1, JOINT_HIDDEN),
        "joint.joint_net.1.bias": torch.zeros(v + 1),
        "predictor.embed.weight": torch.zeros(v + 1, PRED_HIDDEN),
    }


# ---- the shared architecture constant (PORT-INT-001) --------------------------


def test_architecture_constant_and_model_type():
    assert ARCHITECTURE == "Nemotron3_5AsrForRNNT"
    assert MODEL_TYPE == "nemotron_asr"
    # The config declares the architecture it belongs to.
    assert NemotronASRConfig().architectures == [ARCHITECTURE]


# ---- the config class (PORT-WGT-004) ------------------------------------------


def test_config_carries_the_bringup_fields():
    cfg = NemotronASRConfig(
        vocab_size=13089,
        hidden_size=15488,
        eos_token_id=13087,
        decode_dispatch_arm="dense-eager",
        performance_gated=True,
    )
    assert cfg.model_type == MODEL_TYPE
    assert cfg.vocab_size == 13089
    assert cfg.hidden_size == 15488  # the mm-carrier width, not d_model
    assert cfg.d_model == 1024
    assert cfg.eos_token_id == 13087
    assert cfg.decode_dispatch_arm == "dense-eager"
    assert cfg.decode_dispatch_table is None
    assert cfg.performance_gated is True
    assert cfg.torch_dtype in ("float32", torch.float32)


def test_config_registers_with_autoconfig():
    # Importing the registration module runs AutoConfig.register once;
    # a config.json with our model_type then loads without
    # trust_remote_code.
    from transformers import AutoConfig

    import vllm_omni.transformers_utils.configs.nemotron_asr  # noqa: F401

    built = AutoConfig.for_model(MODEL_TYPE, vocab_size=13090)
    assert isinstance(built, NemotronASRConfig)
    assert built.vocab_size == 13090


def test_config_roundtrips_through_from_dict():
    # Regression (pod BU-a): from_dict re-passes EVERY key, so a config
    # whose dict already carries "architectures" (author_config and
    # to_dict both do) must not collide with an explicit arg. This is
    # exactly the AutoConfig.from_pretrained path for a real
    # checkpoint's config.json.
    d = NemotronASRConfig(
        vocab_size=13090,
        eos_token_id=13088,
        decode_dispatch_table="decode-dispatch.json",
        performance_gated=True,
    ).to_dict()
    assert d["architectures"] == [ARCHITECTURE]
    rebuilt = NemotronASRConfig.from_dict(d)
    assert rebuilt.vocab_size == 13090
    assert rebuilt.architectures == [ARCHITECTURE]
    assert rebuilt.decode_dispatch_arm is None
    assert rebuilt.decode_dispatch_table == "decode-dispatch.json"
    assert rebuilt.performance_gated is True


def test_declared_dense_graphed_arm_is_not_an_eager_binding():
    from vllm_omni.model_executor.models.nemotron_asr.nemotron_asr import (
        build_decode_resolver,
    )

    with pytest.raises(ValueError, match="unknown decode_dispatch_arm"):
        build_decode_resolver(
            NemotronASRConfig(decode_dispatch_arm="dense-graphed")
        )


def test_num_asr_labels_survives_construction():
    # Regression (pod BU-a): the label count must NOT use the reserved
    # ``num_labels`` field — PretrainedConfig resets that to a 2-label
    # default in super().__init__, clobbering it. The renamed field
    # survives.
    cfg = NemotronASRConfig(num_asr_labels=13087)
    assert cfg.num_asr_labels == 13087
    assert cfg.to_dict()["num_asr_labels"] == 13087


# ---- vocab derivation from tensor shapes (PORT-WGT-004, OPEN-α4-ARCH) ----------


def test_derive_vocab_size_from_corroborating_tensors():
    assert derive_vocab_size(_checkpoint_shaped_state_dict(V)) == V


def test_derive_vocab_size_rejects_tensor_disagreement():
    # Joint implies V, predictor implies V+1 — a real corruption, not
    # a metadata footnote; hard-fail, never pick one.
    sd = _checkpoint_shaped_state_dict(V)
    sd["predictor.embed.weight"] = torch.zeros(V + 2, PRED_HIDDEN)
    with pytest.raises(ConversionError):
        derive_vocab_size(sd)


def test_derive_vocab_size_rejects_missing_tensor():
    sd = _checkpoint_shaped_state_dict(V)
    del sd["joint.joint_net.1.weight"]
    with pytest.raises(ConversionError):
        derive_vocab_size(sd)


# ---- config authoring (PORT-WGT-004) ------------------------------------------


def test_author_config_accounts_for_minted_specials():
    # ids 0..V-1 are labels and V is blank, so the minted specials are
    # the two ids past blank: park = V+1, placeholder = V+2.
    cfg = author_config(
        _checkpoint_shaped_state_dict(V),
        eos_token_id=V + 1,
        audio_chunk_token_id=V + 2,
        hidden_size=15488,
    )
    # width covers 0..V+2 = labels + blank + park + placeholder.
    assert cfg["vocab_size"] == V + 3
    assert cfg["num_asr_labels"] == V
    assert cfg["architectures"] == [ARCHITECTURE]
    assert cfg["hidden_size"] == 15488
    assert cfg["eos_token_id"] == V + 1
    assert cfg["audio_chunk_token_id"] == V + 2
    assert cfg["torch_dtype"] == "float32"


def test_author_config_hardfails_on_reference_disagreement():
    # The .nemo meta.json / model card cross-check: if a supplied
    # reference vocab disagrees with the derived V, hard-fail (the
    # 13087-vs-13088 question is settled by the tensors, and a
    # genuine mismatch must never pass silently).
    with pytest.raises(ConversionError):
        author_config(
            _checkpoint_shaped_state_dict(V),
            eos_token_id=V + 1,
            audio_chunk_token_id=V + 2,
            hidden_size=15488,
            reference_vocab_size=V + 1,
        )


def test_author_config_rejects_specials_that_shadow_labels_or_blank():
    # A minted special must be a NEW id past blank (> V) and distinct.
    # An id <= V shadows a real label (< V) or blank (= V) — the pod
    # facts confirmed blank = V = 13087, so eos = V is a collision.
    for bad_eos in (5, V):  # 5 = a real label; V = blank
        with pytest.raises(ConversionError):
            author_config(
                _checkpoint_shaped_state_dict(V),
                eos_token_id=bad_eos,
                audio_chunk_token_id=V + 2,
                hidden_size=15488,
            )
    with pytest.raises(ConversionError):
        author_config(
            _checkpoint_shaped_state_dict(V),
            eos_token_id=V + 1,
            audio_chunk_token_id=V + 1,  # not distinct from park
            hidden_size=15488,
        )


# ---- F4-compliant page prefixes (PORT-STATE-002) ------------------------------


def test_state_page_prefixes_are_extract_layer_index_safe():
    prefixes = state_page_prefixes(24)
    # 24 window + 24 conv + 1 lstm + 1 replay + the frontend pair
    # (fp32 buffers + int64 counters, split for typed-view alignment).
    assert len(prefixes) == 52
    kinds = [k for k, _ in prefixes]
    assert kinds.count("window") == 24 and kinds.count("conv") == 24
    assert kinds.count("lstm") == 1 and kinds.count("replay") == 1
    assert kinds.count("frontend_buffer") == 1
    assert kinds.count("frontend_counter") == 1
    for _, prefix in prefixes:
        ints = [p for p in prefix.split(".") if p.lstrip("-").isdigit()]
        # extract_layer_index asserts exactly one integer component.
        assert len(ints) == 1, prefix
    names = {prefix for _, prefix in prefixes}
    assert "encoder.layers.0.window" in names
    assert "encoder.layers.23.conv" in names
    assert "predictor.layers.0.lstm_state" in names  # synthetic .0.
    assert "decode.layers.0.replay" in names
    assert "frontend.layers.0.buffers" in names
    assert "frontend.layers.0.counters" in names


# ---- model __init__ and load_weights (PORT-WGT-001/003) -----------------------


def _duck_vllm_config(cfg: NemotronASRConfig) -> SimpleNamespace:
    return SimpleNamespace(
        model_config=SimpleNamespace(hf_config=cfg, dtype=torch.float32),
        compilation_config=SimpleNamespace(static_forward_context={}),
        cache_config=SimpleNamespace(mamba_cache_mode="none"),
        scheduler_config=SimpleNamespace(max_num_seqs=16),
    )


def test_init_sets_num_logits_from_config_vocab_size():
    from vllm_omni.model_executor.models.nemotron_asr.nemotron_asr import (
        NemotronASRForRNNT,
    )

    cfg = NemotronASRConfig(
        vocab_size=13089, decode_dispatch_arm="dense-eager"
    )
    model = NemotronASRForRNNT(vllm_config=_duck_vllm_config(cfg))
    assert model.num_logits == 13089


def test_init_registers_all_state_pages():
    from vllm_omni.model_executor.models.nemotron_asr.nemotron_asr import (
        NemotronASRForRNNT,
    )

    cfg = NemotronASRConfig(decode_dispatch_arm="dense-eager")
    ctx: dict = {}
    vc = _duck_vllm_config(cfg)
    vc.compilation_config.static_forward_context = ctx
    NemotronASRForRNNT(vllm_config=vc)
    # Every page landed under an F4-compliant prefix.
    assert "encoder.layers.0.window" in ctx
    assert "predictor.layers.0.lstm_state" in ctx


def test_load_weights_missing_prompt_kernel_is_fatal():
    from vllm_omni.model_executor.models.nemotron_asr.nemotron_asr import (
        NemotronASRForRNNT,
    )

    cfg = NemotronASRConfig(decode_dispatch_arm="dense-eager")
    model = NemotronASRForRNNT(vllm_config=_duck_vllm_config(cfg))
    weights_without_lid = [
        (n, t)
        for n, t in _checkpoint_shaped_state_dict().items()
        if "prompt_kernel" not in n
    ]
    with pytest.raises((ValueError, KeyError)):
        model.load_weights(iter(weights_without_lid))


def test_load_weights_rejects_unexpected_name():
    from vllm_omni.model_executor.models.nemotron_asr.nemotron_asr import (
        NemotronASRForRNNT,
    )

    cfg = NemotronASRConfig(decode_dispatch_arm="dense-eager")
    model = NemotronASRForRNNT(vllm_config=_duck_vllm_config(cfg))
    with pytest.raises((ValueError, KeyError)):
        model.load_weights(iter([("not.a.real.tensor", torch.zeros(4))]))
