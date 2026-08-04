# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Direct serving of the public HuggingFace card (PORT-WGT-004, card path).

The public ``nvidia/nemotron-3.5-asr-streaming-0.6b`` repository is
transformers' ``nemotron3_5_asr`` model: config.json in that schema,
model.safetensors in that module tree, and the locale authority in
``processor_config.json``. These tests pin the three translations that
make ``vllm-omni serve <card checkout>`` work without a conversion
step: config-schema translation with deterministic control minting,
tensor-name remapping into the port tree, and featurizer-buffer
synthesis for the two buffers the card does not persist. The fixtures
are copied from the published card (config.json values; the complete
non-encoder-layer tensor inventory plus one conformer layer from the
model.safetensors header, read 2026-08-03).
"""

from __future__ import annotations

import json

import pytest
import torch

from vllm_omni.model_executor.models.nemotron_asr.configuration_nemotron_asr import (
    CARD_MODEL_TYPE,
    NemotronASRConfig,
    NemotronCardServingConfig,
    ensure_prompt_dictionary,
    translate_card_config,
)
from vllm_omni.model_executor.models.nemotron_asr.featurizer import (
    synthesize_card_featurizer_buffers,
)
from vllm_omni.model_executor.models.nemotron_asr.rules import remap_card_name

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _card_config_dict() -> dict:
    """The published card's config.json, as loaded from disk."""
    return {
        "architectures": ["Nemotron3_5AsrForRNNT"],
        "blank_token_id": 13087,
        "decoder_hidden_size": 640,
        "dtype": "float32",
        "durations": [],
        "encoder_config": {
            "conv_kernel_size": 9,
            "hidden_size": 1024,
            "model_type": "nemotron_asr_streaming_encoder",
            "num_hidden_layers": 24,
            "num_mel_bins": 128,
            "sliding_window": 57,
            "subsampling_conv_channels": 256,
            "subsampling_factor": 8,
            "default_num_lookahead_tokens": 3,
            "supported_num_lookahead_tokens": [3, 0, 6, 13],
        },
        "hidden_act": "relu",
        "is_encoder_decoder": True,
        "max_symbols_per_step": 10,
        "model_type": "nemotron3_5_asr",
        "num_decoder_layers": 2,
        "num_prompts": 128,
        "pad_token_id": 0,
        "prompt_intermediate_size": 2048,
        "vocab_size": 13088,
        "default_prompt_id": 101,
    }


# ---- config translation ------------------------------------------------------


# @spec PORT-WGT-004
def test_card_translation_mints_the_publisher_identity() -> None:
    """Controls are the four ids past blank; vocab covers them."""
    translated = translate_card_config(_card_config_dict())
    assert translated["num_asr_labels"] == 13087
    assert translated["eos_token_id"] == 13088
    assert translated["audio_chunk_token_id"] == 13089
    assert translated["eou_token_id"] == 13090
    assert translated["flush_token_id"] == 13091
    assert translated["vocab_size"] == 13092
    assert translated["d_model"] == 1024
    assert translated["n_layers"] == 24
    assert translated["conv_kernel"] == 9
    assert translated["att_context_left"] == 56
    assert translated["att_context_right"] == 13
    assert translated["n_mels"] == 128
    assert translated["pred_hidden"] == 640
    assert translated["pred_rnn_layers"] == 2
    # PORT-DEC-008: the card publisher declares the qualified eager arm;
    # the runtime never falls back to a hardcoded dispatch default.
    assert translated["decode_dispatch_arm"] == "dense-eager"
    assert "decode_dispatch_table" not in translated


# @spec PORT-WGT-004
def test_card_serving_config_constructs_and_validates() -> None:
    config = NemotronCardServingConfig(**_card_config_dict())
    assert config.model_type == CARD_MODEL_TYPE
    assert config.num_asr_labels == 13087
    assert config.vocab_size == 13092
    assert config.eou_token_id == 13090
    assert config.architectures == ["Nemotron3_5AsrForRNNT"]
    # The served schema round-trips without re-translation.
    again = NemotronCardServingConfig(**{
        k: v for k, v in config.to_dict().items() if k != "encoder_config"
    })
    assert again.vocab_size == 13092
    assert again.num_asr_labels == 13087


def test_card_translation_rejects_inconsistent_identity() -> None:
    broken = _card_config_dict()
    broken["vocab_size"] = 13087  # blank not last
    with pytest.raises(ValueError, match="blank_token_id \\+ 1"):
        translate_card_config(broken)


def test_card_translation_rejects_missing_encoder_fields() -> None:
    broken = _card_config_dict()
    del broken["encoder_config"]["sliding_window"]
    with pytest.raises(ValueError, match="sliding_window"):
        translate_card_config(broken)


def test_card_translation_rejects_unsupported_pinned_lookahead() -> None:
    broken = _card_config_dict()
    broken["encoder_config"]["supported_num_lookahead_tokens"] = [3, 0, 6]
    with pytest.raises(ValueError, match="pinned lookahead"):
        translate_card_config(broken)


# ---- prompt-dictionary sidecar ----------------------------------------------


# @spec PORT-WGT-003, PORT-LID-001
def test_prompt_dictionary_loads_from_the_processor_sidecar(tmp_path) -> None:
    (tmp_path / "processor_config.json").write_text(
        json.dumps({"prompt_dictionary": {"en-US": 0, "auto": 101}})
    )
    config = NemotronCardServingConfig(**_card_config_dict())
    assert not config.prompt_dictionary
    ensure_prompt_dictionary(config, tmp_path)
    assert config.prompt_dictionary == {"en-US": 0, "auto": 101}
    # Idempotent, and never overwrites an authored mapping.
    (tmp_path / "processor_config.json").write_text(
        json.dumps({"prompt_dictionary": {"clobber": 1}})
    )
    ensure_prompt_dictionary(config, tmp_path)
    assert config.prompt_dictionary == {"en-US": 0, "auto": 101}


def test_prompt_dictionary_sidecar_absent_leaves_config_unchanged(tmp_path) -> None:
    config = NemotronCardServingConfig(**_card_config_dict())
    ensure_prompt_dictionary(config, tmp_path)
    assert not config.prompt_dictionary


# ---- tensor-name remapping ---------------------------------------------------

#: (card name, port name) pairs covering every rename class, from the
#: published safetensors header.
_RENAMES = [
    ("encoder.subsampling.conv_in.weight", "encoder.pre_encode.conv.0.weight"),
    ("encoder.subsampling.conv_in.bias", "encoder.pre_encode.conv.0.bias"),
    (
        "encoder.subsampling.layers.0.depthwise_conv.weight",
        "encoder.pre_encode.conv.2.weight",
    ),
    (
        "encoder.subsampling.layers.0.pointwise_conv.bias",
        "encoder.pre_encode.conv.3.bias",
    ),
    (
        "encoder.subsampling.layers.1.depthwise_conv.weight",
        "encoder.pre_encode.conv.5.weight",
    ),
    (
        "encoder.subsampling.layers.1.pointwise_conv.weight",
        "encoder.pre_encode.conv.6.weight",
    ),
    ("encoder.subsampling.linear.weight", "encoder.pre_encode.out.weight"),
    (
        "encoder.layers.0.self_attn.q_proj.weight",
        "encoder.layers.0.self_attn.linear_q.weight",
    ),
    (
        "encoder.layers.23.self_attn.k_proj.weight",
        "encoder.layers.23.self_attn.linear_k.weight",
    ),
    (
        "encoder.layers.5.self_attn.v_proj.weight",
        "encoder.layers.5.self_attn.linear_v.weight",
    ),
    (
        "encoder.layers.5.self_attn.o_proj.weight",
        "encoder.layers.5.self_attn.linear_out.weight",
    ),
    (
        "encoder.layers.7.self_attn.relative_k_proj.weight",
        "encoder.layers.7.self_attn.linear_pos.weight",
    ),
    ("encoder.layers.0.self_attn.bias_u", "encoder.layers.0.self_attn.pos_bias_u"),
    ("encoder.layers.0.self_attn.bias_v", "encoder.layers.0.self_attn.pos_bias_v"),
    ("encoder.layers.3.conv.norm.weight", "encoder.layers.3.conv.batch_norm.weight"),
    ("encoder.layers.3.conv.norm.bias", "encoder.layers.3.conv.batch_norm.bias"),
    ("decoder.embedding.weight", "predictor.embed.weight"),
    ("decoder.lstm.weight_ih_l0", "predictor.rnn.weight_ih_l0"),
    ("decoder.lstm.bias_hh_l1", "predictor.rnn.bias_hh_l1"),
    ("encoder_projector.weight", "joint.enc.weight"),
    ("encoder_projector.bias", "joint.enc.bias"),
    ("decoder.decoder_projector.weight", "joint.pred.weight"),
    ("joint.head.weight", "joint.joint_net.1.weight"),
    ("joint.head.bias", "joint.joint_net.1.bias"),
    ("prompt_projector.linear_1.weight", "lid.prompt_kernel.0.weight"),
    ("prompt_projector.linear_1.bias", "lid.prompt_kernel.0.bias"),
    ("prompt_projector.linear_2.weight", "lid.prompt_kernel.2.weight"),
]

#: Card names the port tree already shares — must pass through untouched.
_SHARED = [
    "encoder.layers.0.feed_forward1.linear1.weight",
    "encoder.layers.0.feed_forward2.linear2.weight",
    "encoder.layers.0.norm_self_att.weight",
    "encoder.layers.0.norm_out.bias",
    "encoder.layers.0.conv.depthwise_conv.weight",
    "encoder.layers.0.conv.pointwise_conv1.weight",
    "encoder.layers.0.conv.pointwise_conv2.weight",
]


# @spec PORT-WGT-001, PORT-WGT-004
def test_card_renames_map_into_the_port_tree() -> None:
    for card, port in _RENAMES:
        assert remap_card_name(card) == port, card


def test_shared_leaves_are_not_remapped() -> None:
    for name in _SHARED:
        assert remap_card_name(name) is None, name
    # Served-artifact names never remap either.
    assert remap_card_name("featurizer.fb") is None
    assert remap_card_name("joint.joint_net.1.weight") is None
    assert remap_card_name("predictor.embed.weight") is None


# ---- featurizer synthesis ----------------------------------------------------


# @spec PORT-FEAT-001, PORT-WGT-004
def test_synthesized_featurizer_buffers_match_the_module_contract() -> None:
    fb, window = synthesize_card_featurizer_buffers(n_mels=128)
    assert fb.shape == (128, 257)
    assert fb.dtype == torch.float32
    assert window.shape == (400,)
    assert window.dtype == torch.float32
    # Slaney filterbank invariants: nonnegative, every filter has
    # support, band edges empty, and slaney normalization keeps each
    # filter's peak well below 1.
    assert torch.all(fb >= 0)
    assert torch.all(fb.sum(dim=1) > 0)
    assert torch.all(fb[:, 0] == 0)
    assert float(fb.max()) < 0.2
    # Symmetric (periodic=False) hann: endpoints zero; the peak of an
    # even-length symmetric window falls between samples, so the max
    # approaches but does not reach 1.
    assert window[0] == 0
    assert window[-1] == pytest.approx(0.0, abs=1e-6)
    assert 0.999 < float(window.max()) <= 1.0


def test_synthesized_filterbank_matches_librosa_when_available() -> None:
    librosa = pytest.importorskip("librosa")
    ours, _ = synthesize_card_featurizer_buffers(n_mels=128)
    reference = torch.from_numpy(
        librosa.filters.mel(
            sr=16000, n_fft=512, n_mels=128, fmin=0.0, fmax=8000.0, norm="slaney"
        )
    ).to(torch.float32)
    assert torch.allclose(ours, reference, atol=1e-6)


# ---- resolution --------------------------------------------------------------


# @spec PORT-INT-001
def test_card_model_type_resolves_to_the_nemotron_pipeline() -> None:
    from vllm_omni.config.pipeline_registry import OMNI_PIPELINES
    from vllm_omni.model_executor.models.nemotron_asr.pipeline import (
        NEMOTRON_ASR_PIPELINE,
    )

    assert OMNI_PIPELINES["nemotron3_5_asr"] is NEMOTRON_ASR_PIPELINE
    assert OMNI_PIPELINES["nemotron_asr"] is NEMOTRON_ASR_PIPELINE
    assert "Nemotron3_5AsrForRNNT" in NEMOTRON_ASR_PIPELINE.hf_architectures


def test_card_model_type_is_registered_with_vllm() -> None:
    from vllm.transformers_utils.config import _CONFIG_REGISTRY

    import vllm_omni.transformers_utils.configs  # noqa: F401

    assert _CONFIG_REGISTRY["nemotron3_5_asr"] is NemotronCardServingConfig


# ---- cadence admission against the declared arms (D1) ------------------------


def _admissible_config() -> NemotronCardServingConfig:
    config = NemotronCardServingConfig(**_card_config_dict())
    config.prompt_dictionary = {"en-US": 0, "auto": 101}
    return config


# @spec PORT-SESS-015
def test_cadence_admission_rejects_an_arm_the_card_does_not_declare() -> None:
    """160 ms implies lookahead 1, outside the card's [3, 0, 6, 13]."""
    from vllm_omni.model_executor.models.nemotron_asr.session import (
        NemotronRealtimeSession,
    )

    config = _admissible_config()
    assert config.supported_num_lookahead_tokens == [3, 0, 6, 13]
    with pytest.raises(ValueError, match="PORT-SESS-015"):
        NemotronRealtimeSession.from_model_config(config, cadence="160ms")


# @spec PORT-SESS-015
def test_cadence_admission_follows_the_declared_set_not_the_code() -> None:
    """Declared arms admit their cadences; no declaration admits all."""
    from vllm_omni.model_executor.models.nemotron_asr.session import (
        NemotronRealtimeSession,
    )

    config = _admissible_config()
    for cadence in ("80ms", "320ms", "560ms", "1120ms"):
        session = NemotronRealtimeSession.from_model_config(
            config, cadence=cadence
        )
        assert session.geometry.cadence == cadence

    undeclared = _admissible_config()
    del undeclared.supported_num_lookahead_tokens
    session = NemotronRealtimeSession.from_model_config(
        undeclared, cadence="160ms"
    )
    assert session.geometry.cadence == "160ms"


# @spec PORT-DEC-005
def test_card_translation_retains_the_decode_cap_declaration() -> None:
    config = NemotronCardServingConfig(**_card_config_dict())
    assert config.max_symbols_per_step == 10


# @spec PORT-DEC-005
def test_partial_generation_config_inherits_the_declared_park_stop() -> None:
    """A shipped generation config without eos must not lose the park.

    The card ships generation_config.json without eos_token_id, which
    blocks vLLM's from-model-config inheritance the authored artifact
    relied on; the stage-0 seam completes a partial file from the model
    config's declaration so every request stops on the park token.
    """
    from types import SimpleNamespace

    from vllm_omni.engine.stage_init_utils import (
        patch_generation_config_if_needed,
    )

    served = NemotronCardServingConfig(**_card_config_dict())

    # Card shape: file present, eos absent -> inherit 13088.
    partial = SimpleNamespace(
        hf_config=served,
        try_get_generation_config=lambda: {"pad_token_id": 0},
    )
    patch_generation_config_if_needed(partial)
    assert partial.try_get_generation_config()["eos_token_id"] == 13088
    assert partial.try_get_generation_config()["pad_token_id"] == 0

    # A file that declares eos is authoritative and untouched.
    explicit = SimpleNamespace(
        hf_config=served,
        try_get_generation_config=lambda: {"eos_token_id": 7},
    )
    patch_generation_config_if_needed(explicit)
    assert explicit.try_get_generation_config() == {"eos_token_id": 7}

    # The existing raise guard still degrades to an empty dict.
    def _boom() -> dict:
        raise ValueError("no model_type")

    broken = SimpleNamespace(hf_config=served, try_get_generation_config=_boom)
    patch_generation_config_if_needed(broken)
    assert broken.try_get_generation_config() == {}


# @spec PORT-WGT-004
def test_card_translation_never_presents_an_encoder_decoder_model() -> None:
    """The card's architectural flag must not select vLLM's enc-dec path.

    vLLM's encoder-decoder renderer requires a decoder start token this
    tokenizer does not define; the port serves the model single-stage
    with the encoder inside its own forward.
    """
    translated = translate_card_config(_card_config_dict())
    assert "is_encoder_decoder" not in translated
    config = NemotronCardServingConfig(**_card_config_dict())
    assert not getattr(config, "is_encoder_decoder", False)


def test_served_schema_still_constructs_directly() -> None:
    """The authored artifact's schema is untouched by the card path."""
    config = NemotronASRConfig(
        vocab_size=13092,
        num_asr_labels=13087,
        eos_token_id=13088,
        audio_chunk_token_id=13089,
        eou_token_id=13090,
        flush_token_id=13091,
    )
    assert config.model_type == "nemotron_asr"
    assert config.vocab_size == 13092
