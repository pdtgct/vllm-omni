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

#: The architecture string — the single source of truth shared by the
#: registry, the pipeline, the publisher, and ``config.json``'s
#: ``architectures[0]``. Imported, never re-spelled.
ARCHITECTURE = "Nemotron3_5AsrForRNNT"

MODEL_TYPE = "nemotron_asr"


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
        hidden_size: int = 15617,
        eos_token_id: int | None = None,
        audio_chunk_token_id: int | None = None,
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
