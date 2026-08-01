# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The multimodal processor for the streaming ASR model (BU-c2, D-BUc-2).

This model has **no HF processor**: the mel front-end is a torch module
inside the model (``embed_multimodal`` runs the featurizer +
``pack_audio_carrier``), so the processor's only job is to hand the raw
audio chunk through as the ``audio`` mm-kwarg and register the single
carrier placeholder. One realtime chunk = one prompt = one placeholder
token (``audio_chunk_token_id``) = one carrier row (BU-c1 measured
width). Modeled on the pin's realtime precedent (manual single
placeholder, no placeholder tokens minted into text) and the ASR
field-config precedent (``MultiModalFieldConfig.batched("audio")``).

The exact ``apply``/``_call_hf_processor`` flow for a no-HF-processor
realtime model is confirmed against the live engine (BU-c2 pod round);
this is the first cut against the pinned base API.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING

import numpy as np
import torch
from transformers import BatchFeature
from vllm.inputs import MultiModalDataDict
from vllm.multimodal.inputs import (
    MultiModalFieldConfig,
    MultiModalKwargsItems,
    MultiModalKwargsOptionalItems,
)
from vllm.multimodal.parse import MultiModalDataItems
from vllm.multimodal.processing import (
    BaseProcessingInfo,
    PromptUpdate,
)
from vllm.multimodal.processing.dummy_inputs import BaseDummyInputsBuilder
from vllm.multimodal.processing.processor import (
    BaseMultiModalProcessor,
    PlaceholderFeaturesInfo,
)

if TYPE_CHECKING:
    from vllm.multimodal.cache import BaseMultiModalProcessorCache
    from vllm.multimodal.processing.processor import MultiModalPromptUpdates

#: One second of 16 kHz audio — a safe dummy chunk for warmup profiling
#: (the featurizer only needs a valid non-empty waveform; the realtime
#: route never serves this).
_DUMMY_SAMPLES = 16_000


class NemotronASRProcessingInfo(BaseProcessingInfo):
    """Processing metadata: one audio item per streaming prompt."""

    def get_supported_mm_limits(self) -> Mapping[str, int | None]:
        return {"audio": 1}


class NemotronASRDummyInputsBuilder(BaseDummyInputsBuilder[NemotronASRProcessingInfo]):
    """Warmup inputs: one placeholder-free text + one dummy chunk.

    The carrier placeholder id is a minted special with no text form, so
    the dummy text is empty; the placeholder is registered structurally
    by the processor, not tokenized from text.
    """

    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str:
        return ""

    def get_dummy_mm_data(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
        mm_options: Mapping[str, object] | None = None,
    ) -> MultiModalDataDict:
        num_audios = mm_counts.get("audio", 0)
        return {"audio": self._get_dummy_audios(length=_DUMMY_SAMPLES, num_audios=num_audios)}


class NemotronASRMultiModalProcessor(BaseMultiModalProcessor[NemotronASRProcessingInfo]):
    """Raw-audio passthrough + single carrier placeholder.

    No HF processor: ``_call_hf_processor`` builds the field batch from
    the raw chunk directly (the model's ``embed_multimodal`` owns the
    mel front-end). The realtime prompt carries exactly one placeholder
    token, so placeholder positions are built manually (one carrier row
    per audio) rather than by text-target replacement.
    """

    def __init__(
        self,
        info: NemotronASRProcessingInfo,
        dummy_inputs: BaseDummyInputsBuilder[NemotronASRProcessingInfo],
        *,
        cache: BaseMultiModalProcessorCache | None = None,
    ) -> None:
        # Realtime cannot use the content-hash cache (streams are unique).
        super().__init__(info, dummy_inputs, cache=None)

    def _call_hf_processor(
        self,
        prompt: str,
        mm_data: Mapping[str, object],
        mm_kwargs: Mapping[str, object],
        tok_kwargs: Mapping[str, object],
    ) -> BatchFeature:
        # No HF processor: pass the raw waveform through untouched as the
        # ``audio`` field (the model featurizes it in ``embed_multimodal``;
        # nothing here computes mel), and synthesise the token stream the
        # base ``_apply_hf_processor_text_mm`` pops as ``input_ids``. The
        # streaming prompt is one carrier placeholder per audio chunk;
        # ``requires_raw_input_tokens`` supplies the real ids at serving,
        # so this path only has to hold for the text/profiling render.
        # AudioProcessorItems delivers the batch under the plural key
        # ``audios`` (get_processor_data -> f"{modality}s"); the OUTPUT
        # field stays ``audio`` (embed_multimodal + the field config key).
        audios = mm_data.get("audios", [])
        if not isinstance(audios, list):
            audios = [audios]
        # torch tensors, not np arrays: the engine serializes the mm field
        # via _encode_nested_tensors, which only handles torch tensors
        # (it recurses into an ndarray and chokes on the scalar leaves).
        arrays = [torch.as_tensor(np.asarray(a, dtype=np.float32)) for a in audios]
        placeholder_id = self.info.get_hf_config().audio_chunk_token_id
        input_ids = [placeholder_id] * len(arrays)
        return BatchFeature({"input_ids": [input_ids], "audio": arrays})

    def _get_mm_fields_config(
        self,
        hf_inputs: BatchFeature,
        hf_processor_mm_kwargs: Mapping[str, object],
    ) -> Mapping[str, MultiModalFieldConfig]:
        # One batched field named ``audio`` — the key ``embed_multimodal``
        # reads (``kwargs.get("audio")``).
        return {"audio": MultiModalFieldConfig.batched("audio")}

    def _get_prompt_updates(
        self,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, object],
        out_mm_kwargs: MultiModalKwargsItems,
    ) -> Sequence[PromptUpdate]:
        # The realtime prompt already holds the single placeholder token
        # at the right position; placeholders are built manually in
        # ``_maybe_apply_prompt_updates``, so no text-target replacement.
        return []

    def _maybe_apply_prompt_updates(
        self,
        mm_items: MultiModalDataItems,
        prompt_ids: list[int],
        mm_kwargs: MultiModalKwargsOptionalItems,
        mm_prompt_updates: MultiModalPromptUpdates,
        is_update_applied: bool,
    ) -> tuple[list[int], Mapping[str, list[PlaceholderFeaturesInfo]]]:
        # One carrier row per chunk: the whole prompt is the single
        # placeholder token, so the placeholder spans one position at
        # index 0. ``tokens`` is only used for length accounting.
        features_info = PlaceholderFeaturesInfo(
            modality="audio",
            item_idx=0,
            start_idx=0,
            tokens=[prompt_ids[0]] if prompt_ids else [0],
            is_embed=None,
        )
        return prompt_ids, {"audio": [features_info]}
