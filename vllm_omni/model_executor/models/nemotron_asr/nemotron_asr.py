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
from vllm.multimodal import MULTIMODAL_REGISTRY

from vllm_omni.model_executor.models.nemotron_asr.convert import (
    LID_REQUIRED_PATTERN,
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
from vllm_omni.model_executor.models.nemotron_asr.manifests import (
    FRONTEND_CONSTANTS,
)
from vllm_omni.model_executor.models.nemotron_asr.precision import (
    FP32_BRINGUP,
    PrecisionPolicy,
)
from vllm_omni.model_executor.models.nemotron_asr.processor import (
    NemotronASRDummyInputsBuilder,
    NemotronASRMultiModalProcessor,
    NemotronASRProcessingInfo,
)
from vllm_omni.model_executor.models.nemotron_asr.rnnt import (
    MAX_SYMBOLS_PER_STEP,
    DecodeState,
    Joint,
    Predictor,
    greedy_decode_chunk,
)
from vllm_omni.model_executor.models.nemotron_asr.state_layers import (
    ConvCachePage,
    FrontendBufferPage,
    FrontendCounterPage,
    HybridStateModelMixin,
    LSTMStatePage,
    ReplayQueuePage,
    WindowCachePage,
    register_state_pages,
    state_page_prefixes,
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


@MULTIMODAL_REGISTRY.register_processor(
    NemotronASRMultiModalProcessor,
    info=NemotronASRProcessingInfo,
    dummy_inputs=NemotronASRDummyInputsBuilder,
)
class NemotronASRForRNNT(nn.Module, HybridStateModelMixin):
    """The engine-facing model class (α4 tests-first skeleton).

    Single-stage LLM_AR omni model implementing ``SupportsRealtime`` and
    ``SupportsMultiModal`` (structural protocols — the classvars below
    are what the engine reads, so protocol inheritance is unnecessary)
    over ``NemotronASRCore``; registered as ``Nemotron3_5AsrForRNNT``
    (PORT-INT-001/002). The ``register_processor`` decorator is the one
    top-level vLLM coupling that takes this file off the macOS loader
    path (BU-c2): the pure helpers stay loader-tested, this class is
    ruff/mypy + pod-gated (static-analysis-for-vllm-coupled-code).
    """

    #: Raw-audio multimodal in (one chunk carrier per prompt); read by
    #: ``vllm.multimodal.supports_multimodal`` via ``getattr``.
    supports_multimodal = True
    #: Keep the raw input ids alongside ``inputs_embeds`` — the only
    #: chunk-vs-replay signal at forward (D-BUc-2); read at the runner.
    requires_raw_input_tokens = True
    supports_realtime = True
    #: Secondary framework guard only — the omni realtime route reads
    #: the pipeline's explicit ``max_tokens`` (see pipeline.py); this
    #: classvar is the core-route value (PORT-INT-002), worst case
    #: 14 frames × 10 symbols + park.
    realtime_max_tokens = 141
    #: Engine logit width: tokenizer vocab + the park special token
    #: (checkpoint default; __init__ re-reads it from the config).
    num_logits = 13090

    def __init__(self, *, vllm_config: Any = None, prefix: str = "") -> None:
        """Build the core from the config and register the state pages.

        Reads ``NemotronASRConfig`` off ``vllm_config.model_config``,
        builds ``NemotronASRCore`` at the config's dims, sets
        ``num_logits`` from the config's ``vocab_size``, and constructs
        + registers the four state-page kinds under F4-compliant
        prefixes (``state_page_prefixes``). Weights load afterward via
        ``load_weights``; page pools bind at forward (BU-b).
        """
        super().__init__()
        hf_config = vllm_config.model_config.hf_config
        self.config = hf_config
        # The engine samples over the full logit width (labels + minted
        # park/placeholder specials); the core decodes over V labels.
        self.num_logits = hf_config.vocab_size
        policy = FP32_BRINGUP
        self.core = NemotronASRCore(
            vocab_size=hf_config.num_asr_labels,
            att_context=(
                hf_config.att_context_left,
                hf_config.att_context_right,
            ),
            enc_hidden=hf_config.d_model,
            pred_hidden=hf_config.pred_hidden,
            pred_rnn_layers=hf_config.pred_rnn_layers,
            joint_hidden=hf_config.joint_hidden,
            num_prompts=hf_config.num_prompts,
            filterbank=torch.zeros(_N_MELS, _STFT_FREQ_BINS),
            window=torch.zeros(_WIN_LENGTH),
            policy=policy,
        )
        # The encoder is the source of truth for its layer count.
        n_layers = len(self.core.encoder.layers)
        self._state_pages = self._build_state_pages(
            n_layers, hf_config, policy
        )
        register_state_pages(vllm_config, self._state_pages)
        # Categorize the pages so forward can gather the bound pools by
        # kind (window/conv/lstm/replay); the ordering matches
        # state_page_prefixes.
        kinds = [k for k, _ in state_page_prefixes(n_layers)]
        paired = list(zip(kinds, self._state_pages, strict=True))
        self._window_pages = [p for k, p in paired if k == "window"]
        self._conv_pages = [p for k, p in paired if k == "conv"]
        self._lstm_page = next(p for k, p in paired if k == "lstm")
        self._replay_page = next(p for k, p in paired if k == "replay")
        self._frontend_buffer_page = next(
            p for k, p in paired if k == "frontend_buffer"
        )
        self._frontend_counter_page = next(
            p for k, p in paired if k == "frontend_counter"
        )
        # The pre-encode overlap dropped from non-first chunks
        # (drop_extra); session-first chunks use 0 (run_forward_step).
        self._drop_extra = 2

    def _build_state_pages(
        self, n_layers: int, cfg: Any, policy: PrecisionPolicy
    ) -> list[Any]:
        """The four page kinds under F4-compliant prefixes."""
        pages: list[Any] = []
        for kind, prefix in state_page_prefixes(n_layers):
            if kind == "window":
                pages.append(
                    WindowCachePage(
                        prefix=prefix,
                        window=cfg.att_context_left,
                        d_model=cfg.d_model,
                        policy=policy,
                    )
                )
            elif kind == "conv":
                pages.append(
                    ConvCachePage(
                        prefix=prefix,
                        d_model=cfg.d_model,
                        kernel=cfg.conv_kernel,
                        policy=policy,
                    )
                )
            elif kind == "lstm":
                pages.append(
                    LSTMStatePage(
                        prefix=prefix,
                        pred_rnn_layers=cfg.pred_rnn_layers,
                        pred_hidden=cfg.pred_hidden,
                        policy=policy,
                    )
                )
            elif kind == "replay":
                pages.append(
                    ReplayQueuePage(
                        prefix=prefix,
                        max_symbols_per_step=MAX_SYMBOLS_PER_STEP,
                        max_frames_per_chunk=_MAX_FRAMES_PER_CHUNK,
                        policy=policy,
                    )
                )
            elif kind == "frontend_buffer":
                pages.append(
                    FrontendBufferPage(
                        prefix=prefix,
                        raw_tail=FRONTEND_CONSTANTS["raw_tail_capacity"],
                        n_mels=_N_MELS,
                        policy=policy,
                    )
                )
            elif kind == "frontend_counter":
                pages.append(
                    FrontendCounterPage(prefix=prefix, policy=policy)
                )
        return pages

    def load_weights(self, weights: Any) -> set[str]:
        """Load converted safetensors by the name ledger (PORT-WGT-001).

        Consumes an iterable of ``(name, tensor)`` — the served
        safetensors are already in the port's tree naming (the
        conversion publisher writes them that way), so this is a strict
        consume-exactly-once copy: an unexpected name or a shape
        mismatch hard-fails (no warn-and-proceed, the ``convert``
        posture), and a missing name hard-fails at the end.
        ``prompt_kernel`` absence is called out explicitly and fatal —
        the model never degrades to unconditioned transcription
        (PORT-WGT-003). Returns the set of loaded parameter names.
        """
        expected: dict[str, torch.Tensor] = dict(
            self.core.named_parameters()
        )
        expected.update(self.core.named_buffers())
        # Require only the PERSISTENT state (params + persistent buffers,
        # i.e. state_dict) from the checkpoint; derived buffers registered
        # persistent=False (e.g. the relative-position ``pos_enc.pe``, an
        # init-computed sinusoid, never persisted by NeMo) are valid copy
        # targets but must NOT be demanded of the checkpoint.
        required = set(self.core.state_dict())
        consumed: set[str] = set()
        for name, tensor in weights:
            if name not in expected:
                raise ValueError(
                    f"unexpected weight {name!r}: not in the model's "
                    "parameter/buffer set"
                )
            target = expected[name]
            if tuple(target.shape) != tuple(tensor.shape):
                raise ValueError(
                    f"shape mismatch for {name!r}: expected "
                    f"{tuple(target.shape)}, got {tuple(tensor.shape)}"
                )
            with torch.no_grad():
                target.copy_(tensor)
            consumed.add(name)
        missing = required - consumed
        lid_missing = sorted(
            n for n in missing if LID_REQUIRED_PATTERN.search(n)
        )
        if lid_missing:
            raise ValueError(
                f"LID weights absent ({lid_missing}); the model never "
                "degrades to unconditioned transcription (PORT-WGT-003)"
            )
        if missing:
            raise ValueError(
                f"{len(missing)} expected weights not provided, e.g. "
                f"{sorted(missing)[:3]}"
            )
        # Return MODEL-qualified names: the weights load into ``self.core``,
        # so the engine's post-load audit (``track_weights_loading`` diffs
        # against ``model.named_parameters()``) expects the ``core.`` prefix
        # the checkpoint's core-relative names lack.
        return {f"core.{name}" for name in consumed}

    @classmethod
    async def buffer_realtime_audio(
        cls,
        audio_stream: Any,
        input_stream: Any,
        model_config: Any,
    ) -> Any:
        """The ``SupportsRealtime`` segmenter seam — delegates to the
        engine-free ``buffer_stream`` (PORT-SESS-001/002/003).

        The chunking behaviour lives in ``streaming.buffer_stream`` so it
        stays CPU/loader-tested; this classmethod is the thin protocol
        surface the engine calls (pod-tested).
        """
        from vllm_omni.model_executor.models.nemotron_asr.streaming import (
            buffer_stream,
        )

        async for update in buffer_stream(
            audio_stream, input_stream, model_config
        ):
            yield update

    def embed_multimodal(self, **kwargs: Any) -> Any:
        """Stateless mel front-end → one carrier row per chunk (BU-c1).

        Runs the featurizer on each chunk's audio and packs the mel
        into a single ``inputs_embeds`` row (``pack_audio_carrier``,
        slot 0 = frame count). MUST be pure — the engine content-hash-
        caches this output, so identical chunks must produce identical
        carriers (PORT-INT-003). The mm_kwargs key/shape is the
        processor's contract (BU-c2), pod-verified/adapted.
        """
        from vllm_omni.model_executor.models.nemotron_asr.forward_ops import (
            pack_audio_carrier,
        )

        audios = kwargs.get("audio")
        if audios is None:
            raise ValueError("embed_multimodal expects an 'audio' item")
        # Raw-audio mm fields ride as np arrays, which the batched-field
        # reducer leaves on CPU (it only h2d-moves torch tensors); move to
        # the model device so the featurizer's on-device window/fb match.
        device = next(self.parameters()).device
        rows = []
        for chunk in audios:
            wav = (
                chunk
                if isinstance(chunk, torch.Tensor)
                else torch.as_tensor(getattr(chunk, "audio_arrays", chunk))
            )
            wav = wav.to(device=device, dtype=torch.float32).reshape(1, -1)
            mel, _ = self.core.featurizer(
                wav, torch.tensor([wav.shape[-1]], device=wav.device)
            )
            rows.append(
                pack_audio_carrier(mel[0], hidden_size=self.config.hidden_size)
            )
        # SupportsMultiModal contract (sanity_check_mm_encoder_outputs): a
        # sequence of one 2D (num_tokens, hidden_size) tensor PER audio
        # item — each chunk is a single carrier token, so (1, hidden_size).
        # The runner caches these per item and slices them in
        # _gather_mm_embeddings.
        return [row.unsqueeze(0) for row in rows]

    def embed_input_ids(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings: Any = None,
        *,
        is_multimodal: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Canonical embed seam (``SupportsMultiModal``, D-BUc-2).

        Zero-inits ``(num_tokens, hidden_size)`` — this model has no LM
        table, so replay/flush rows stay zero and carry their id via
        ``input_ids`` — and scatters the carrier rows at the
        ``is_multimodal`` positions. The upstream/core-PR form delegates
        to core ``utils._merge_multimodal_embeddings`` over the zero
        buffer; the port's ``merge_mm_embeddings`` is behaviourally
        identical (native-ness audit) and keeps this loader-testable
        off the engine.
        """
        from vllm_omni.model_executor.models.nemotron_asr.forward_ops import (
            merge_mm_embeddings,
        )

        hidden = self.config.hidden_size
        if multimodal_embeddings is None or is_multimodal is None:
            return torch.zeros(
                input_ids.shape[0], hidden,
                dtype=torch.float32, device=input_ids.device,
            )
        return merge_mm_embeddings(
            input_ids, multimodal_embeddings, is_multimodal,
            hidden_size=hidden,
        )

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        positions: torch.Tensor | None = None,
        intermediate_tensors: Any = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Chunk-ingest or replay step; hidden rows carry decisions.

        Thin wrapper (D-BUc-1): reads the bound page pools and per-row
        ``state_indices`` from ``get_forward_context()`` and calls
        ``run_forward_step`` (the pure pipeline, loader-tested in BU-c1).
        Pod-verified/adapted (BU-c2): the exact metadata field access and
        the ``_p``/``_d`` handling are confirmed against the real
        ``ShortConvAttentionMetadata`` on the engine.
        """
        from vllm.forward_context import get_forward_context

        from vllm_omni.model_executor.models.nemotron_asr.forward_step import (
            run_forward_step,
        )

        assert inputs_embeds is not None and input_ids is not None
        md = get_forward_context().attn_metadata
        if md is None:
            # V1 profiling: compute-shaped, no page IO (hazard 4).
            return torch.zeros(
                inputs_embeds.shape[0], inputs_embeds.shape[1],
                dtype=inputs_embeds.dtype, device=inputs_embeds.device,
            )
        # All four page kinds share one uniform group → one metadata
        # object (ShortConvAttentionMetadata; Any off-engine). The batch
        # is ordered decodes-then-prefills (short_conv splits
        # [num_decode_tokens, num_prefill_tokens]); every row is a single
        # token (replay id, or a chunk placeholder — the first chunk of a
        # session prefills, later chunks decode), so the per-row page
        # index is decode indices then prefill indices concatenated,
        # aligned to ``input_ids``.
        meta = md[self._window_pages[0].prefix]
        parts = [
            t
            for t in (
                meta.state_indices_tensor_d,
                meta.state_indices_tensor_p,
            )
            if t is not None and t.numel()
        ]
        state_indices = (
            torch.cat(parts)
            if parts
            else input_ids.new_zeros(0)
        )

        return run_forward_step(
            self.core, input_ids, inputs_embeds,
            channel_pools=[pg.kv_cache[0] for pg in self._window_pages],
            time_pools=[pg.kv_cache[0] for pg in self._conv_pages],
            len_pools=[pg.kv_cache[1] for pg in self._window_pages],
            h_pool=self._lstm_page.kv_cache[0],
            c_pool=self._lstm_page.kv_cache[1],
            queue_pool=self._replay_page.kv_cache[0],
            book_pool=self._replay_page.kv_cache[1],
            state_indices=state_indices,
            placeholder_id=self.config.audio_chunk_token_id,
            park_id=self.config.eos_token_id,
            feat=_N_MELS,
            drop_extra=self._drop_extra,
        )

    def compute_logits(
        self, hidden_states: torch.Tensor, sampling_metadata: Any = None
    ) -> torch.Tensor:
        """Forced-logits rows from the hidden-row decision carrier.

        The engine's ``logits_indices`` gather hands this exactly the
        last-scheduled-token rows, row-aligned per request in batch
        order (PORT-DEC-002) — ids are read by ROW POSITION, never by
        request id.
        """
        from vllm_omni.model_executor.models.nemotron_asr.rnnt import (
            forced_logits_rows,
            read_decision_carrier,
        )

        ids = read_decision_carrier(hidden_states)
        return forced_logits_rows(ids, num_logits=self.num_logits)
