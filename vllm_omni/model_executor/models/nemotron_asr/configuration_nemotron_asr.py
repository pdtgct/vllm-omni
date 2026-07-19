# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""HF config for the streaming ASR model (PORT-WGT-004).

The served checkpoint has no upstream HF modeling code, so the port
authors its own ``config.json`` at conversion time and registers this
class with ``AutoConfig`` (see ``transformers_utils/configs``) so vLLM
loads it without ``trust_remote_code``. Only the fields vLLM's config
machinery and this model's ``__init__`` actually read live here; the
values are produced by the conversion publisher
(``convert.author_config``), never hand-authored.
"""

from transformers import PretrainedConfig

from vllm_omni.model_executor.models.nemotron_asr.manifests import (
    ENVELOPE_HEADER_FIELDS,
    RAW_SAMPLES_PER_CHUNK,
)

#: The architecture string — the single source of truth shared by the
#: registry, the pipeline, the publisher, and ``config.json``'s
#: ``architectures[0]``. Imported, never re-spelled.
ARCHITECTURE = "Nemotron3_5AsrForRNNT"

MODEL_TYPE = "nemotron_asr"

#: The raw chunk-envelope carrier width: header slots plus the largest
#: admitted raw cadence (1120 ms at 16 kHz) — DERIVED from the
#: manifests, never a copied constant (PORT-INT-004; the retired mel
#: default 15,617 is superseded by this envelope value, 17,926).
_CARRIER_WIDTH = len(ENVELOPE_HEADER_FIELDS) + max(
    RAW_SAMPLES_PER_CHUNK.values()
)


class NemotronASRConfig(PretrainedConfig):
    """Config for the FastConformer RNN-T streaming ASR model.

    ``hidden_size`` is the multimodal-carrier row width (D-BU-2), not
    an LM width — this model has no LM stack; the encoder's true width
    is ``d_model``. ``vocab_size`` covers the checkpoint's label set
    plus the minted specials (park = ``eos_token_id``, the audio-chunk
    placeholder).
    """

    model_type = MODEL_TYPE

    def __init__(
        self,
        *,
        vocab_size: int = 13090,
        num_asr_labels: int = 13087,
        hidden_size: int = _CARRIER_WIDTH,
        eos_token_id: int | None = None,
        audio_chunk_token_id: int | None = None,
        decode_dispatch_arm: str | None = None,
        decode_dispatch_table: str | None = None,
        performance_gated: bool = False,
        d_model: int = 1024,
        conv_kernel: int = 9,
        att_context_left: int = 56,
        att_context_right: int = 13,
        pred_hidden: int = 640,
        pred_rnn_layers: int = 2,
        joint_hidden: int = 640,
        num_prompts: int = 128,
        n_mels: int = 128,
        max_position_embeddings: int = 32768,
        torch_dtype: str = "float32",
        **kwargs: object,
    ) -> None:
        self.vocab_size = vocab_size
        #: The decoded label-set size V (blank = V, joint emits V+1);
        #: distinct from ``vocab_size``, the engine logit width that
        #: also covers the minted park/placeholder specials. NOT named
        #: ``num_labels`` — that is a reserved ``PretrainedConfig``
        #: field (it drives ``id2label`` and is reset to a 2-label
        #: default by ``super().__init__``, silently clobbering ours).
        self.num_asr_labels = num_asr_labels
        self.hidden_size = hidden_size
        self.audio_chunk_token_id = audio_chunk_token_id
        # Startup decode policy is part of the served artifact, not a
        # process-local default. Preserve all three fields through HF
        # serialization so build_decode_resolver sees the declaration
        # authored by the publisher (PORT-WGT-004 / PORT-DEC-008).
        self.decode_dispatch_arm = decode_dispatch_arm
        self.decode_dispatch_table = decode_dispatch_table
        self.performance_gated = performance_gated
        self.d_model = d_model
        self.conv_kernel = conv_kernel
        self.att_context_left = att_context_left
        self.att_context_right = att_context_right
        self.pred_hidden = pred_hidden
        self.pred_rnn_layers = pred_rnn_layers
        self.joint_hidden = joint_hidden
        self.num_prompts = num_prompts
        self.n_mels = n_mels
        self.max_position_embeddings = max_position_embeddings
        # ``architectures`` rides kwargs, never an explicit arg: a
        # config.json (from author_config or to_dict) already carries
        # it, and ``from_dict`` re-passes every key — an explicit
        # ``architectures=`` would then collide with **kwargs and raise
        # on every reload.
        kwargs.setdefault("architectures", [ARCHITECTURE])
        super().__init__(
            eos_token_id=eos_token_id,
            torch_dtype=torch_dtype,
            **kwargs,
        )
