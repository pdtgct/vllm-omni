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
from vllm.multimodal import MULTIMODAL_REGISTRY

from vllm_omni.model_executor.models.nemotron_asr.advance import (
    ENVELOPE_HEADER_SLOTS,
    DecodeRequest,
    DecodeResolver,
    EmissionAdapter,
    HostStaging,
    ResolvedDecode,
    make_mrv1_adapter,
)
from vllm_omni.model_executor.models.nemotron_asr.commit_sink import (
    BoundedCommitSink,
    resolve_status_reports,
)
from vllm_omni.model_executor.models.nemotron_asr.convert import (
    LID_REQUIRED_PATTERN,
)
from vllm_omni.model_executor.models.nemotron_asr.decode_dispatch import (
    SYNC_FREE_ARMS,
)
from vllm_omni.model_executor.models.nemotron_asr.encoder import (
    FastConformerEncoder,
)
from vllm_omni.model_executor.models.nemotron_asr.encoder_execution import (
    build_encoder_execution,
)
from vllm_omni.model_executor.models.nemotron_asr.featurizer import (
    MelFeaturizer,
)
from vllm_omni.model_executor.models.nemotron_asr.lid import (
    PromptConditioner,
    resolve_prompt_index,
)
from vllm_omni.model_executor.models.nemotron_asr.manifests import (
    CADENCES,
)
from vllm_omni.model_executor.models.nemotron_asr.plan import (
    ObservedRow,
    PlanContextSlot,
    SessionRegistry,
    prepare_plan_context,
    reject_unsupported_outer_graph_mode,
    resolve_row_envelope_header,
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
    DecodeState,
    Joint,
    Predictor,
    greedy_decode_chunk,
)
from vllm_omni.model_executor.models.nemotron_asr.startup import (
    NEMOTRON_PERSISTENT_STATE_STARTUP,
)
from vllm_omni.model_executor.models.nemotron_asr.state_profile import (
    NemotronStatePools,
    build_nemotron_persistent_state_spec,
    project_nemotron_state_pools,
)
from vllm_omni.model_executor.persistent_state import PersistentStateLayerBase

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
    "NemotronASRForRNNT",
    "apply_policy_dtypes",
    "load_core_from_dump",
    "resolve_prompt_index",
]


@MULTIMODAL_REGISTRY.register_processor(
    NemotronASRMultiModalProcessor,
    info=NemotronASRProcessingInfo,
    dummy_inputs=NemotronASRDummyInputsBuilder,
)
class NemotronASRForRNNT(nn.Module):
    """Engine-facing cache-aware RNN-T over one aggregate state page."""

    supports_multimodal = True
    requires_raw_input_tokens = True
    supports_realtime = True
    supports_transcription_only = False
    supports_persistent_state = True
    persistent_state_startup_provider = NEMOTRON_PERSISTENT_STATE_STARTUP
    realtime_max_tokens = 142
    num_logits = 13_092

    def __init__(self, *, vllm_config: Any = None, prefix: str = "") -> None:
        super().__init__()
        if vllm_config is None:
            raise ValueError("NemotronASRForRNNT requires vllm_config")
        hf_config = vllm_config.model_config.hf_config
        from vllm_omni.model_executor.models.nemotron_asr.configuration_nemotron_asr import (
            ensure_prompt_dictionary,
            validate_prompt_dictionary,
        )

        model_path = getattr(vllm_config.model_config, "model", None)
        if model_path:
            ensure_prompt_dictionary(hf_config, model_path)
        validate_prompt_dictionary(
            getattr(hf_config, "prompt_dictionary", None),
            getattr(hf_config, "num_prompts", None),
        )
        from vllm_omni.model_executor.models.nemotron_asr.rnnt import (
            MAX_SYMBOLS_PER_STEP,
        )

        # PORT-DEC-005: the served declaration is the authority; the
        # manifest-derived emission budgets were computed from the
        # module value, so a disagreeing declaration fails closed
        # instead of being silently overridden.
        declared_symbols = getattr(hf_config, "max_symbols_per_step", None)
        if declared_symbols is not None and int(declared_symbols) != MAX_SYMBOLS_PER_STEP:
            raise ValueError(
                "served config declares max_symbols_per_step="
                f"{declared_symbols}, but the manifest-derived emission "
                f"budgets were computed from {MAX_SYMBOLS_PER_STEP}; "
                "requalify the budgets before serving this declaration "
                "(PORT-DEC-005)"
            )
        # PORT-STATE-009: precision is part of the qualified profile.
        # An unqualified --dtype fails loudly, never warn-and-ignore.
        engine_dtype = getattr(vllm_config.model_config, "dtype", None)
        if engine_dtype is not None and engine_dtype != torch.float32:
            raise ValueError(
                f"--dtype {engine_dtype} is not the qualified profile "
                "for this model (float32); a precision change requires "
                "requalification (PORT-STATE-009). If no --dtype was "
                "given, vLLM's auto policy downcasts float32 checkpoints "
                "on SM80+ GPUs and the pipeline's deploy profile "
                "(vllm_omni/deploy/nemotron_asr.yaml) normally pins "
                "float32 - a 'Deploy config not found' warning earlier "
                "in this log means the installation is missing its "
                "deploy data. Pass --dtype float32 or repair the "
                "installation."
            )
        # Pool sizing is resolved after vLLM's real memory profile in the
        # common worker. The model constructor deliberately does not invent
        # physical capacity or mutate ``num_gpu_blocks_override``.
        reject_unsupported_outer_graph_mode(getattr(vllm_config, "compilation_config", None))
        self.config = hf_config
        self.num_logits = int(hf_config.vocab_size)
        policy = FP32_BRINGUP
        self.core = NemotronASRCore(
            vocab_size=hf_config.num_asr_labels,
            att_context=(
                hf_config.att_context_left,
                hf_config.att_context_right,
            ),
            enc_hidden=hf_config.d_model,
            n_layers=hf_config.n_layers,
            pred_hidden=hf_config.pred_hidden,
            pred_rnn_layers=hf_config.pred_rnn_layers,
            joint_hidden=hf_config.joint_hidden,
            num_prompts=hf_config.num_prompts,
            filterbank=torch.zeros(_N_MELS, _STFT_FREQ_BINS),
            window=torch.zeros(_WIN_LENGTH),
            policy=policy,
        )
        state_prefix = f"{prefix}.persistent_state" if prefix else "persistent_state"
        self._persistent_state_layer = PersistentStateLayerBase(
            build_nemotron_persistent_state_spec(hf_config),
            prefix=state_prefix,
            vllm_config=vllm_config,
        )
        self._state_pools_cache: NemotronStatePools | None = None
        self._registry = SessionRegistry()
        self._plan_slot = PlanContextSlot()
        self._plan_step = 0
        self._emission_adapter: EmissionAdapter = make_mrv1_adapter(
            hidden_size=hf_config.hidden_size,
            park_id=hf_config.eos_token_id,
            blank_id=self.core.blank_id,
        )
        self._decode_resolver = build_decode_resolver(hf_config)
        self._encoder_execution = build_encoder_execution(self.core, hf_config)
        self._max_num_seqs = int(vllm_config.scheduler_config.max_num_seqs)
        self._commit_sink: BoundedCommitSink | None = None
        self._host_staging: HostStaging | None = None

    def _state_pools(self) -> NemotronStatePools:
        if self._state_pools_cache is None:
            self._state_pools_cache = project_nemotron_state_pools(
                self._persistent_state_layer.persistent_state_storage,
                n_layers=len(self.core.encoder.layers),
            )
        return self._state_pools_cache

    def load_weights(self, weights: Any) -> set[str]:
        """Strictly load the served tree or the public HF card.

        The served artifact's names load as-is. The public card's
        renamed modules map through ``remap_card_name``; shared leaves
        already match. A card checkpoint persists no featurizer
        buffers, so those two are synthesized after the stream — only
        when card naming was actually seen, never for a served artifact
        (persisted buffers are inputs, not recomputed).
        """
        from vllm_omni.model_executor.models.nemotron_asr.rules import (
            remap_card_name,
        )

        expected: dict[str, torch.Tensor] = dict(self.core.named_parameters())
        expected.update(self.core.named_buffers())
        required = set(self.core.state_dict())
        consumed: set[str] = set()
        card_names_seen = False
        for name, tensor in weights:
            mapped = remap_card_name(name)
            if mapped is not None:
                card_names_seen = True
                name = mapped
            if name not in expected:
                raise ValueError(f"unexpected weight {name!r}")
            target = expected[name]
            tensor = self._normalize_checkpoint_tensor(name, tensor)
            if tuple(target.shape) != tuple(tensor.shape):
                raise ValueError(
                    f"shape mismatch for {name!r}: expected {tuple(target.shape)}, got {tuple(tensor.shape)}"
                )
            with torch.no_grad():
                target.copy_(tensor)
            consumed.add(name)
        missing = required - consumed
        featurizer_buffers = {"featurizer.fb", "featurizer.window"}
        if card_names_seen and featurizer_buffers & missing:
            from vllm_omni.model_executor.models.nemotron_asr.featurizer import (
                synthesize_card_featurizer_buffers,
            )

            fb, window = synthesize_card_featurizer_buffers(
                n_mels=int(self.config.n_mels),
            )
            for name, tensor in (("featurizer.fb", fb), ("featurizer.window", window)):
                if name not in missing:
                    continue
                target = expected[name]
                if tuple(target.shape) != tuple(tensor.shape):
                    raise ValueError(
                        f"synthesized {name!r} shape {tuple(tensor.shape)} "
                        f"does not match the module ({tuple(target.shape)})"
                    )
                with torch.no_grad():
                    target.copy_(tensor)
                consumed.add(name)
            missing = required - consumed
        lid_missing = sorted(name for name in missing if LID_REQUIRED_PATTERN.search(name))
        if lid_missing:
            raise ValueError(f"LID weights absent ({lid_missing}); conditioned transcription never degrades silently")
        if missing:
            raise ValueError(f"{len(missing)} expected weights not provided, e.g. {sorted(missing)[:3]}")
        return {f"core.{name}" for name in consumed}

    @staticmethod
    def _normalize_checkpoint_tensor(
        name: str,
        tensor: torch.Tensor,
    ) -> torch.Tensor:
        """Apply the publisher's one explicit storage-to-runtime reshape."""

        if name == "featurizer.fb" and tensor.ndim == 3 and tensor.shape[0] == 1:
            return tensor[0]
        return tensor

    @classmethod
    async def buffer_realtime_audio(
        cls,
        audio_stream: Any,
        input_stream: Any,
        model_config: Any,
        *,
        observer: Any = None,
        accepted_audio_budget_s: float | None = None,
        session_key: str | None = None,
    ) -> Any:
        from vllm_omni.model_executor.models.nemotron_asr.streaming import (
            buffer_stream,
        )

        async for update in buffer_stream(
            audio_stream,
            input_stream,
            model_config,
            observer=observer,
            accepted_audio_budget_s=accepted_audio_budget_s,
            session_key=session_key,
        ):
            yield update

    def prepare_row_plan_context(
        self,
        *,
        req_ids: Any,
        token_ids_cpu: Any,
        num_computed_tokens_cpu: Any,
        num_scheduled_tokens: Any,
        requests: Any,
        scheduled_encoder_inputs: Any,
    ) -> None:
        """MRV1 oracle projection into the common transaction plan."""

        rows: list[ObservedRow] = []
        for index, request_id in enumerate(req_ids):
            if int(num_scheduled_tokens[index]) != 1:
                raise ValueError("every streaming row must be single-token")
            computed = int(num_computed_tokens_cpu[index])
            token = int(token_ids_cpu[index, computed])
            request = requests[request_id]
            groups = request.block_ids
            if len(groups) != 1 or len(groups[0]) != 1:
                raise ValueError("persistent state requires one slot per request")
            endpoint_mode, endpoint_threshold, endpoint_residue = self._endpoint_controls(request)
            rows.append(
                ObservedRow(
                    request_id=str(request_id),
                    block_id=int(groups[0][0]),
                    scheduled_token_id=token,
                    has_prior_state=computed > 0,
                    envelope_header=resolve_row_envelope_header(
                        scheduled_token_id=token,
                        placeholder_id=int(self.config.audio_chunk_token_id),
                        num_computed_tokens=computed,
                        mm_features=request.mm_features,
                        scheduled_encoder_input_ids=scheduled_encoder_inputs.get(request_id, ()),
                    ),
                    endpoint_mode=endpoint_mode,
                    endpoint_threshold_frames=endpoint_threshold,
                    endpoint_residue_frames=endpoint_residue,
                )
            )
        self._stage_plan(rows, resident_request_ids=tuple(str(value) for value in requests))

    def _stage_plan(
        self,
        rows: list[ObservedRow],
        *,
        resident_request_ids: tuple[str, ...],
    ) -> Any:
        import time

        if self._commit_sink is not None and self._commit_sink.has_staged:
            raise RuntimeError("previous transaction status was not consumed")
        self._plan_step += 1
        context = prepare_plan_context(
            self._registry,
            rows,
            resident_request_ids=resident_request_ids,
            placeholder_id=int(self.config.audio_chunk_token_id),
            num_prompts=int(self.core.lid.num_prompts),
            num_geometries=len(CADENCES),
            now_ns=time.time_ns(),
            step=self._plan_step,
        )
        self._plan_slot.stage(context)
        return context

    @staticmethod
    def _endpoint_controls(request: Any) -> tuple[int, int, int]:
        information = getattr(request, "additional_information", None)
        if not isinstance(information, dict):
            information = getattr(information, "entries", None)
        if not isinstance(information, dict):
            return 0, 0, 0
        policy = information.get("endpoint_policy")
        if not isinstance(policy, dict):
            return 0, 0, 0
        mode_name = policy.get("mode")
        if mode_name not in ("disabled", "greedy_blank"):
            raise ValueError("request carries an unknown endpoint mode")
        mode = 1 if mode_name == "greedy_blank" else 0
        threshold = int(policy.get("threshold_frames", 0))
        residue = int(policy.get("residue_frames", 0))
        if threshold < 0 or residue < 0:
            raise ValueError("request carries an invalid endpoint policy")
        return mode, threshold, residue

    def _stage_v2_projection(self, projection: Any) -> Any:
        input_batch = projection.input_batch
        metadata = projection.request_metadata
        if metadata is None:
            raise RuntimeError("MRv2 projection omitted request metadata")
        rows: list[ObservedRow] = []
        for index, (request_id, binding) in enumerate(zip(projection.req_ids, projection.bindings)):
            if int(input_batch.num_scheduled_tokens[index]) != 1:
                raise ValueError("every streaming row must be single-token")
            request = metadata[request_id]
            token_ids = request.prefill_token_ids or request.prompt_token_ids
            if token_ids is None:
                raise RuntimeError("MRv2 request has no token authority")
            computed = int(input_batch.num_computed_tokens_np[index])
            if not 0 <= computed < len(token_ids):
                raise RuntimeError("MRv2 scheduled token is outside request data")
            token = int(token_ids[computed])
            endpoint_mode, endpoint_threshold, endpoint_residue = self._endpoint_controls(request)
            rows.append(
                ObservedRow(
                    request_id=request_id,
                    block_id=int(binding.slot_id),
                    scheduled_token_id=token,
                    has_prior_state=not bool(binding.fresh),
                    envelope_header=resolve_row_envelope_header(
                        scheduled_token_id=token,
                        placeholder_id=int(self.config.audio_chunk_token_id),
                        num_computed_tokens=computed,
                        mm_features=request.mm_features,
                        scheduled_encoder_input_ids=projection.scheduled_encoder_inputs.get(request_id, ()),
                    ),
                    endpoint_mode=endpoint_mode,
                    endpoint_threshold_frames=endpoint_threshold,
                    endpoint_residue_frames=endpoint_residue,
                )
            )
        storage = self._persistent_state_layer.persistent_state_storage
        for binding in projection.bindings:
            if binding.fresh:
                storage.initialize_fresh_state_slot(
                    int(binding.slot_id),
                    int(binding.generation),
                )
        return self._stage_plan(
            rows,
            resident_request_ids=tuple(projection.req_states.req_id_to_index),
        )

    def warmup_resident_state(self) -> None:
        """Warm every fixed-shape scatter specialization before admission."""

        from vllm_omni.model_executor.models.nemotron_asr.advance import (
            warmup_advance_model_rows_scatter,
        )

        pools = self._state_pools()
        if pools.channel[0].device.type != "cuda":
            return
        warmup_advance_model_rows_scatter(
            channel_pools=list(pools.channel),
            time_pools=list(pools.convolution),
            len_pools=list(pools.valid_length),
            h_pool=pools.predictor_h,
            c_pool=pools.predictor_c,
            queue_pool=pools.replay_queue,
            book_pool=pools.replay_book,
            frontend_raw_pool=pools.frontend_raw,
            frontend_mel_pool=pools.frontend_mel,
            frontend_counter_pool=pools.frontend_counters,
            endpoint_history_pool=pools.endpoint_history,
            endpoint_book_pool=pools.endpoint_book,
        )

    def _ensure_commit_sink(self, device: torch.device) -> BoundedCommitSink:
        if self._commit_sink is None:
            self._commit_sink = BoundedCommitSink(
                self._registry,
                max_rows=self._max_num_seqs,
                device=device,
            )
        return self._commit_sink

    def _ensure_host_staging(self) -> HostStaging:
        if self._host_staging is None:
            self._host_staging = HostStaging(self._max_num_seqs)
        return self._host_staging

    def collect_commit_status(self) -> tuple[dict[str, int], set[str]]:
        sink = self._commit_sink
        if sink is None or not sink.has_staged:
            return {}, set()
        reports, _records, lease_ok = sink.collect()
        return resolve_status_reports(
            self._registry,
            reports,
            lease_ok=lease_ok,
        )

    def embed_multimodal(self, **kwargs: Any) -> Any:
        audios = kwargs.get("audio")
        if audios is None:
            raise ValueError("embed_multimodal expects an 'audio' item")
        device = next(self.parameters()).device
        hidden = int(self.config.hidden_size)
        rows = []
        for chunk in audios:
            envelope = (
                (chunk if isinstance(chunk, torch.Tensor) else torch.as_tensor(getattr(chunk, "audio_arrays", chunk)))
                .reshape(-1)
                .to(dtype=torch.float32)
            )
            if envelope.shape[0] < ENVELOPE_HEADER_SLOTS:
                raise ValueError("audio envelope is smaller than its header")
            if envelope.shape[0] > hidden:
                raise ValueError("audio envelope exceeds carrier width")
            row = torch.zeros(hidden, dtype=torch.float32)
            row[: envelope.shape[0]] = envelope
            rows.append(row.to(device).unsqueeze(0))
        return rows

    def embed_input_ids(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings: Any = None,
        *,
        is_multimodal: torch.Tensor | None = None,
    ) -> torch.Tensor:
        from vllm_omni.model_executor.models.nemotron_asr.forward_ops import (
            merge_mm_embeddings,
        )

        hidden = int(self.config.hidden_size)
        if multimodal_embeddings is None or is_multimodal is None:
            return torch.zeros(
                input_ids.shape[0],
                hidden,
                dtype=torch.float32,
                device=input_ids.device,
            )
        return merge_mm_embeddings(
            input_ids,
            multimodal_embeddings,
            is_multimodal,
            hidden_size=hidden,
        )

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        positions: torch.Tensor | None = None,
        intermediate_tensors: Any = None,
        inputs_embeds: torch.Tensor | None = None,
        persistent_state_projection: Any = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        del positions, intermediate_tensors, kwargs
        from vllm.v1.attention.backends.utils import NULL_BLOCK_ID

        from vllm_omni.model_executor.models.nemotron_asr.advance import (
            advance_model_rows,
        )
        from vllm_omni.model_executor.models.nemotron_asr.plan import (
            build_row_plan,
        )

        if inputs_embeds is None:
            raise RuntimeError("Nemotron forward requires embeddings")
        if persistent_state_projection is not None and persistent_state_projection.no_page_io:
            if not persistent_state_projection.dummy_run:
                raise RuntimeError("no-page-I/O projection must be a dummy invocation")
            if persistent_state_projection.is_profile:
                from vllm_omni.model_executor.models.nemotron_asr.profile_execution import (
                    run_persistent_state_profile,
                )

                run_persistent_state_profile(
                    self,
                    num_rows=len(persistent_state_projection.req_ids),
                    device=inputs_embeds.device,
                )
            return torch.zeros_like(inputs_embeds)
        if input_ids is None:
            raise RuntimeError("Nemotron forward requires raw ids")
        if persistent_state_projection is not None:
            context = self._stage_v2_projection(persistent_state_projection)
            num_decodes = 0
            num_prefills = len(context.request_ids)
        else:
            context = self._plan_slot.consume()
            num_decodes = int(context.has_prior_state.sum().item())
            num_prefills = len(context.request_ids) - num_decodes
        pools = self._state_pools()
        plan = build_row_plan(
            context,
            num_decodes=num_decodes,
            num_prefills=num_prefills,
            null_block_id=int(NULL_BLOCK_ID),
            num_pool_blocks=int(pools.replay_queue.shape[0]),
        )
        return advance_model_rows(
            self.core,
            input_ids.long(),
            inputs_embeds,
            plan,
            channel_pools=list(pools.channel),
            time_pools=list(pools.convolution),
            len_pools=list(pools.valid_length),
            h_pool=pools.predictor_h,
            c_pool=pools.predictor_c,
            queue_pool=pools.replay_queue,
            book_pool=pools.replay_book,
            frontend_raw_pool=pools.frontend_raw,
            frontend_mel_pool=pools.frontend_mel,
            frontend_counter_pool=pools.frontend_counters,
            endpoint_history_pool=pools.endpoint_history,
            endpoint_book_pool=pools.endpoint_book,
            eou_token_id=int(self.config.eou_token_id),
            adapter=self._emission_adapter,
            decode_resolver=self._decode_resolver,
            encoder_transition=self._encoder_execution.transition,
            placeholder_id=int(self.config.audio_chunk_token_id),
            park_id=int(self.config.eos_token_id),
            commit_sink=self._ensure_commit_sink(inputs_embeds.device),
            staging=self._ensure_host_staging(),
        )

    def consume_batch_stats(self) -> list[tuple[str, int]] | None:
        from vllm_omni.model_executor.models.nemotron_asr.advance import (
            consume_batch_stats,
        )

        return consume_batch_stats()

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        sampling_metadata: Any = None,
    ) -> torch.Tensor:
        del sampling_metadata
        from vllm_omni.model_executor.models.nemotron_asr.rnnt import (
            forced_logits_rows,
            read_decision_carrier,
        )

        ids = read_decision_carrier(hidden_states)
        return forced_logits_rows(ids, num_logits=self.num_logits)

    @classmethod
    def get_model_state_cls(cls) -> type[Any]:
        from vllm_omni.model_executor.models.nemotron_asr.model_state_v2 import (
            NemotronASRModelState,
        )

        return NemotronASRModelState


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
        decode_compact_active_frames,
        decode_dense_masked_frames,
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
        "dense-eager": decode_dense_masked_frames,
        "compact-eager": decode_compact_active_frames,
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
