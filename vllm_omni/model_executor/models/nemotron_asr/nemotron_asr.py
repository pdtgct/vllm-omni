# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Nemotron cache-aware streaming RNN-T ASR — model class.

Two context regimes over one class (PORT-REGIME-001/003): the
full-context single-shot regime backs ``/v1/audio/transcriptions``
(``SupportsTranscription``) and is the serving bring-up baseline; the
streaming regime (``SupportsRealtime`` + spec pages) rides on top with
per-chunk state.

RNN-T emission is D-b (PORT-DEC-001/002/003): the forward that ingests
audio runs featurizer -> encoder -> LID -> the complete greedy
label-looping decode in-model, parks the emitted labels in a per-request
replay queue, and every generation step emits the next queued label by
±inf logits masking; the reserved park token (blank id reused — never
detokenized) ends the run. Greedy sampling is pinned by the model, not
trusted from the entrypoint (PORT-DEC-005).
"""

import torch
from torch import nn

from vllm_omni.model_executor.models.nemotron_asr.encoder import (
    FastConformerEncoder,
)
from vllm_omni.model_executor.models.nemotron_asr.featurizer import (
    MelFeaturizer,
)
from vllm_omni.model_executor.models.nemotron_asr.lid import (
    PromptConditioner,
    resolve_prompt_index,
)
from vllm_omni.model_executor.models.nemotron_asr.precision import (
    FP32_BRINGUP,
    PrecisionPolicy,
)
from vllm_omni.model_executor.models.nemotron_asr.rnnt import (
    DecodeState,
    Joint,
    Predictor,
    greedy_decode_chunk,
)


class NemotronASRCore(nn.Module):
    """The assembled pipeline shared by both regimes.

    Engine-facing classes (offline transcription / realtime) wrap this
    core; probes drive it directly. Weight names follow the converted
    tree (rules.py): ``featurizer.* / encoder.* / lid.* / predictor.* /
    joint.*``.
    """

    def __init__(
        self,
        *,
        vocab_size: int,
        att_context: tuple[int, int] = (56, 13),
        enc_hidden: int = 1024,
        pred_hidden: int = 640,
        pred_rnn_layers: int = 2,
        joint_hidden: int = 640,
        num_prompts: int = 128,
        filterbank: torch.Tensor,
        window: torch.Tensor,
        policy: PrecisionPolicy = FP32_BRINGUP,
    ) -> None:
        super().__init__()
        self.policy = policy
        self.vocab_size = vocab_size
        self.blank_id = vocab_size
        self.featurizer = MelFeaturizer(filterbank=filterbank, window=window)
        self.encoder = FastConformerEncoder(att_context=att_context)
        self.lid = PromptConditioner(
            enc_hidden=enc_hidden, num_prompts=num_prompts
        )
        self.predictor = Predictor(
            vocab_size=vocab_size,
            pred_hidden=pred_hidden,
            pred_rnn_layers=pred_rnn_layers,
        )
        self.joint = Joint(
            enc_hidden=enc_hidden,
            pred_hidden=pred_hidden,
            joint_hidden=joint_hidden,
            vocab_size=vocab_size,
        )
        self._pred_layers = pred_rnn_layers
        self._pred_hidden = pred_hidden

    def fresh_decode_state(self, device: torch.device) -> DecodeState:
        """Zeroed decode state; last label = blank (SOS, PORT-STATE-003)."""
        return DecodeState(
            h=torch.zeros(
                self._pred_layers,
                1,
                self._pred_hidden,
                device=device,
                dtype=self.policy.dtype_for("lstm_state"),
            ),
            c=torch.zeros(
                self._pred_layers,
                1,
                self._pred_hidden,
                device=device,
                dtype=self.policy.dtype_for("lstm_state"),
            ),
            last_label=torch.tensor([self.blank_id], device=device),
        )

    @torch.inference_mode()
    def transcribe_full_context(
        self,
        waveform: torch.Tensor,
        *,
        prompt_index: int,
    ) -> list[int]:
        """Full-context single-shot regime (PORT-REGIME-002).

        One window over the whole utterance, cross-chunk state dormant;
        the complete label-looping decode runs once. Returns emitted
        label ids (the replay-queue content).
        """
        device = waveform.device
        lengths = torch.tensor([waveform.shape[1]], device=device)
        mel, mel_len = self.featurizer(waveform, lengths)
        enc, enc_len = self.encoder(mel, mel_len.to(device))
        valid = int(enc_len[0])
        conditioned = self.lid(
            enc[:, :valid], prompt_index=prompt_index
        )
        labels, _ = greedy_decode_chunk(
            conditioned[0],
            self.predictor,
            self.joint,
            self.fresh_decode_state(device),
        )
        return labels

    @torch.inference_mode()
    def transcribe_chunk(
        self,
        chunk_conditioned: torch.Tensor,
        state: DecodeState,
    ) -> tuple[list[int], DecodeState]:
        """One streaming chunk-step's decode (D-b queue fill)."""
        return greedy_decode_chunk(
            chunk_conditioned, self.predictor, self.joint, state
        )


def load_core_from_dump(
    dump_dir, *, device: torch.device, att_context=(56, 13)
) -> tuple[NemotronASRCore, dict]:
    """Assemble the core from the offline conversion dump.

    The engine path loads the same converted tree through the standard
    HF loader once the checkpoint is published in HF layout
    (PORT-WGT-001); this loader serves probes and the serving prototype
    from the referee dump meanwhile.
    """
    import json
    from pathlib import Path

    from safetensors.torch import load_file

    from vllm_omni.model_executor.models.nemotron_asr.convert import (
        convert_state_dict,
    )
    from vllm_omni.model_executor.models.nemotron_asr.rules import NEMO_RULES

    dump = Path(dump_dir)
    state = load_file(str(dump / "nemo_state.safetensors"))
    meta = json.loads((dump / "meta.json").read_text())
    converted, _ = convert_state_dict(state, NEMO_RULES)
    core = NemotronASRCore(
        vocab_size=int(meta["vocab_size"]),
        att_context=att_context,
        filterbank=converted["featurizer.fb"][0],
        window=converted["featurizer.window"],
    )
    # fb/window enter via the constructor (checkpoint-buffer rule);
    # exclude them here — NeMo's fb carries a leading batch dim.
    loadable = {
        k: v
        for k, v in converted.items()
        if k not in ("featurizer.fb", "featurizer.window")
    }
    missing, unexpected = core.load_state_dict(loadable, strict=False)
    real_missing = [
        m
        for m in missing
        if not m.endswith(("fb", "window", "pe"))
    ]
    if real_missing or unexpected:
        raise ValueError(
            f"core load mismatch: missing={real_missing} "
            f"unexpected={unexpected}"
        )
    core.to(device).eval()
    return core, meta


__all__ = [
    "NemotronASRCore",
    "load_core_from_dump",
    "resolve_prompt_index",
]
