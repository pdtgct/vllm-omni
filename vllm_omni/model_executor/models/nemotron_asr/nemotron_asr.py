# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Nemotron cache-aware streaming RNN-T ASR — model class.

This reconstruction slice restores the storage-neutral model core and
checkpoint-loading path first. Cross-chunk state is supplied explicitly by
ordinary tensors to the canonical transition until the generic persistent-
state substrate and direct MRv2 ``ModelState`` land in later slices. The
full-context path survives only as an EVAL-tier parity probe.

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

from pathlib import Path
from typing import Any

import torch
from torch import nn

from vllm_omni.model_executor.models.nemotron_asr.advance import (
    DecodeRequest,
    DecodeResolver,
    ResolvedDecode,
)
from vllm_omni.model_executor.models.nemotron_asr.decode_dispatch import (
    SYNC_FREE_ARMS,
)
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

#: The largest published chunk (1120 ms) emits 14 encoder frames — the
#: replay queue page's per-chunk worst case.
_MAX_FRAMES_PER_CHUNK = 14
#: The checkpoint's mel-bin count; the featurizer's filterbank is built
#: as a placeholder at this width and overwritten by load_weights.
_N_MELS = 128
_STFT_FREQ_BINS = 512 // 2 + 1
_WIN_LENGTH = 400


class NemotronASRCore(nn.Module):
    """The assembled cache-aware streaming pipeline.

    The realtime engine-facing class wraps this core; probes drive it
    directly. Weight names follow the converted tree (rules.py):
    ``featurizer.* / encoder.* / lid.* / predictor.* / joint.*``.
    """

    def __init__(
        self,
        *,
        vocab_size: int,
        att_context: tuple[int, int] = (56, 13),
        enc_hidden: int = 1024,
        n_layers: int = 24,
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
        # n_layers is a checkpoint property, not a fixed default: the
        # served path passes hf_config.n_layers and the dump loader
        # derives it from the converted tensors, so the encoder depth
        # always matches the weights (and the published state manifest).
        self.encoder = FastConformerEncoder(n_layers=n_layers, att_context=att_context)
        self.lid = PromptConditioner(enc_hidden=enc_hidden, num_prompts=num_prompts)
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
    def transcribe_chunk(
        self,
        chunk_conditioned: torch.Tensor,
        state: DecodeState,
    ) -> tuple[list[int], DecodeState]:
        """One streaming chunk-step's decode (D-b queue fill)."""
        return greedy_decode_chunk(chunk_conditioned, self.predictor, self.joint, state)


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
    dump_dir: str | Path,
    *,
    device: torch.device,
    att_context: tuple[int, int] = (56, 13),
    policy: PrecisionPolicy = FP32_BRINGUP,
) -> tuple[NemotronASRCore, dict]:
    """Assemble the core from the offline conversion dump.

    The engine path loads the same converted tree through the standard
    HF loader once the checkpoint is published in HF layout
    (PORT-WGT-001); this loader serves probes and the serving prototype
    from the referee dump meanwhile.
    """
    import json

    from safetensors.torch import load_file

    from vllm_omni.model_executor.models.nemotron_asr.convert import (
        convert_state_dict,
        derive_n_layers,
    )
    from vllm_omni.model_executor.models.nemotron_asr.rules import NEMO_RULES

    dump = Path(dump_dir)
    state = load_file(str(dump / "nemo_state.safetensors"))
    meta = json.loads((dump / "meta.json").read_text())
    converted, _ = convert_state_dict(state, NEMO_RULES)
    core = NemotronASRCore(
        vocab_size=int(meta["vocab_size"]),
        att_context=att_context,
        # Derive the encoder depth from the converted tensors (the same
        # source publish.py uses), never the module default — this
        # loader must build an encoder that matches the dump's weights.
        n_layers=derive_n_layers(converted),
        filterbank=converted["featurizer.fb"][0],
        window=converted["featurizer.window"],
        policy=policy,
    )
    # fb/window enter via the constructor (checkpoint-buffer rule);
    # exclude them here — NeMo's fb carries a leading batch dim.
    loadable = {k: v for k, v in converted.items() if k not in ("featurizer.fb", "featurizer.window")}
    missing, unexpected = core.load_state_dict(loadable, strict=False)
    real_missing = [m for m in missing if not m.endswith(("fb", "window", "pe"))]
    if real_missing or unexpected:
        raise ValueError(f"core load mismatch: missing={real_missing} unexpected={unexpected}")
    apply_policy_dtypes(core)
    core.to(device).eval()
    return core, meta


__all__ = [
    "NemotronASRCore",
    "apply_policy_dtypes",
    "load_core_from_dump",
    "resolve_prompt_index",
]


def build_decode_resolver(hf_config: Any) -> DecodeResolver:
    """Startup decode-dispatch resolution (PORT-DEC-008: no default).

    The SERVED config must declare exactly one of:

    - ``decode_dispatch_arm``: an explicitly configured eager arm for
      the bring-up/correctness lane. More than one ready decode bucket
      still forces the sync-free eager arm — a declared arm is a
      preference, never a license to serialize a busy engine. A
      ``dense-graphed`` declaration is rejected until an actual graph
      binding exists.
    - ``decode_dispatch_table``: a measured, validated table artifact.
      Loading it requires the runtime-fingerprint tooling owned by the
      deferred A100 work packet; until that lands this path fails
      closed rather than validating against a fabricated fingerprint.

    Raises:
        ValueError: neither or both declared, an unknown arm, or the
            not-yet-supported table path.
    """
    from vllm_omni.model_executor.models.nemotron_asr.rnnt import (
        decode_compact_active,
        decode_dense_masked,
    )

    arm = getattr(hf_config, "decode_dispatch_arm", None)
    table_path = getattr(hf_config, "decode_dispatch_table", None)
    if arm and table_path:
        raise ValueError("declare decode_dispatch_arm OR decode_dispatch_table, not both")
    if not arm and not table_path:
        raise ValueError(
            "the served config must declare decode_dispatch_arm or "
            "decode_dispatch_table — decode dispatch is never a "
            "hardcoded default (PORT-DEC-008)"
        )
    if table_path:
        raise ValueError(
            "decode_dispatch_table loading requires the deferred A100 "
            "work packet's runtime-fingerprint tooling; declare "
            "decode_dispatch_arm for the bring-up lane"
        )
    arms = {
        "dense-eager": decode_dense_masked,
        "compact-eager": decode_compact_active,
    }
    if arm not in arms:
        raise ValueError(f"unknown decode_dispatch_arm {arm!r} (known: {sorted(arms)})")

    def resolve(request: DecodeRequest) -> ResolvedDecode:
        if request.graph_covers_decode:
            raise ValueError("graph-covered decode requires an exact padded runner binding; Phase 6c is eager-only")
        if request.ready_decode_buckets > 1 and arm not in SYNC_FREE_ARMS:
            return ResolvedDecode(
                arm="dense-eager",
                decode_fn=arms["dense-eager"],
                override_reason="multi-bucket-serialization-guard",
            )
        return ResolvedDecode(arm=arm, decode_fn=arms[arm])

    return resolve
