# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Concrete weight-mapping rules for nemotron-3.5-asr-streaming-0.6b.

Written against the restored checkpoint's state dict (657 tensors,
dumped 2026-07-11 on-pod). Encoder-layer and prompt-kernel names map
identity; the predictor and joint re-root; the featurizer's persisted
``fb``/``window`` buffers map to the port featurizer (checkpoint
buffers are inputs, never recomputed). Enforced by
:func:`convert.convert_state_dict`'s consume-exactly-once contract.
"""

from vllm_omni.model_executor.models.nemotron_asr.convert import TensorRule

NEMO_RULES: tuple[TensorRule, ...] = (
    # Featurizer buffers (persisted in the checkpoint). NeMo wraps these
    # in a leading singleton dim; squeeze to the port featurizer's bare
    # (n_mels, n_freq) / (win_length,) and assert the canonical shape.
    TensorRule(
        source=r"preprocessor\.featurizer\.fb",
        target="featurizer.fb",
        transform="squeeze",
        expect_shape=(128, 257),
    ),
    TensorRule(
        source=r"preprocessor\.featurizer\.window",
        target="featurizer.window",
        transform="squeeze",
        expect_shape=(400,),
    ),
    # Subsampling + conformer layers: identity under the encoder root.
    TensorRule(
        source=r"encoder\.pre_encode\.(conv\.\d+\.(?:weight|bias)"
        r"|out\.(?:weight|bias))",
        target=r"encoder.pre_encode.\1",
    ),
    TensorRule(
        source=r"encoder\.layers\.(\d+)\.(.+)",
        target=r"encoder.layers.\1.\2",
    ),
    # Predictor: embed + LSTM re-root (torch LSTM param names shared).
    TensorRule(
        source=r"decoder\.prediction\.embed\.weight",
        target="predictor.embed.weight",
    ),
    TensorRule(
        source=r"decoder\.prediction\.dec_rnn\.lstm\."
        r"((?:weight|bias)_(?:ih|hh)_l\d+)",
        target=r"predictor.rnn.\1",
    ),
    # Joint: side projections identity; the final linear sits at
    # Sequential index 2 in NeMo (activation+dropout precede) and
    # index 1 here (ReLU, Linear).
    TensorRule(
        source=r"joint\.(enc|pred)\.(weight|bias)",
        target=r"joint.\1.\2",
    ),
    TensorRule(
        source=r"joint\.joint_net\.2\.(weight|bias)",
        target=r"joint.joint_net.1.\1",
    ),
    # LID prompt kernel: identity indices (0, 2), with biases.
    TensorRule(
        source=r"prompt_kernel\.(\d+)\.(weight|bias)",
        target=r"lid.prompt_kernel.\1.\2",
    ),
)
