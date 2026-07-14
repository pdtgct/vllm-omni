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
replay queue, and every generation step emits the next queued label via
a forced-logits row from ``compute_logits`` (0 at the chosen id, -inf
elsewhere); the park token — the checkpoint's ``eos_token_id`` — ends
the burst through the resumable stop path. Greedy sampling is pinned
declaratively per update by ``pipeline.py sampling_constraints`` and
actively by the replay-echo guard (PORT-DEC-005/007).
"""

from typing import Any

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


def apply_policy_dtypes(core: NemotronASRCore) -> NemotronASRCore:
    """Cast compute modules to the policy's weight dtype (PORT-PREC-001).

    Encoder, LID, predictor, and joint move to ``dtype_for("weights")``;
    activations follow at the module entry seams, so this realization
    requires the two classes to agree. The mel front-end stays fp32 —
    it sits upstream of the activations seam and is the golden input
    boundary. Recurrent/cache state dtypes are independent axes read at
    their own construction sites (PORT-PREC-005), and the manual LSTM
    cell up-casts its weights to the state dtype per step, keeping
    ``(h, c)`` accumulation at fp32 under sub-fp32 weights.
    """
    weights = core.policy.dtype_for("weights")
    activations = core.policy.dtype_for("activations")
    if weights != activations:
        raise ValueError(
            "this realization derives activations from the weight "
            f"dtype at the entry seams; policy {core.policy.identifier} "
            f"declares weights={weights} activations={activations}"
        )
    for module in (core.encoder, core.lid, core.predictor, core.joint):
        module.to(weights)
    return core


def load_core_from_dump(
    dump_dir,
    *,
    device: torch.device,
    att_context=(56, 13),
    policy: PrecisionPolicy = FP32_BRINGUP,
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
        policy=policy,
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
    apply_policy_dtypes(core)
    core.to(device).eval()
    return core, meta


__all__ = [
    "NemotronASRCore",
    "apply_policy_dtypes",
    "load_core_from_dump",
    "resolve_prompt_index",
]


class NemotronASRForRNNT(nn.Module):
    """The engine-facing model class (α4 tests-first skeleton).

    Single-stage LLM_AR omni model implementing ``SupportsRealtime``
    (structural protocol) over ``NemotronASRCore``; registered as
    ``Nemotron3_5AsrForRNNT`` (PORT-INT-001/002). The serving methods
    below are the α4 seams; each raises until its code phase.
    """

    supports_realtime = True
    #: Secondary framework guard only — the omni realtime route reads
    #: the pipeline's explicit ``max_tokens`` (see pipeline.py); this
    #: classvar is the core-route value (PORT-INT-002), worst case
    #: 14 frames × 10 symbols + park.
    realtime_max_tokens = 141

    def __init__(self, *, vllm_config: Any = None, prefix: str = "") -> None:
        super().__init__()
        raise NotImplementedError("α4 code phase")

    @classmethod
    async def buffer_realtime_audio(
        cls,
        audio_stream: Any,
        input_stream: Any,
        model_config: Any,
    ) -> Any:
        """Chunk client audio: one yield = one StreamingUpdate.

        Fixed segmenter built once at generator start from the
        session's admitted chunk config (PORT-SESS-002); holds chunk
        N+1 until the park token id for chunk N appears on
        ``input_stream`` (buffer-until-drained, PORT-SESS-001 —
        defense in depth over core's park-time queue consumption);
        applies the NeMo tail rules on finalize (PORT-SESS-003:
        partial tails as-is, sub-8-mel-frame remainders dropped,
        never zero-padded).
        """
        raise NotImplementedError("α4 code phase")
        yield  # pragma: no cover — makes the stub an async GENERATOR

    def forward(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        """Chunk-ingest or replay step; hidden rows carry decisions."""
        raise NotImplementedError("α4 code phase")

    def compute_logits(
        self, hidden_states: torch.Tensor, sampling_metadata: Any = None
    ) -> torch.Tensor:
        """Forced-logits rows from the hidden-row decision carrier.

        The engine's ``logits_indices`` gather hands this exactly the
        last-scheduled-token rows, row-aligned per request in batch
        order (PORT-DEC-002) — ids are read by ROW POSITION, never by
        request id.
        """
        raise NotImplementedError("α4 code phase")
