# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Per-session state pages via core vLLM's cache-spec system.

The cache-aware streaming capability declares all cross-chunk model
state through `kv_cache_interface` spec pages (PORT-STATE-001/002) —
ALL FOUR kinds as constant-size ``MambaSpec`` pages: the encoder
left-context window (the OPEN-α3-VEHICLE decision, measured
2026-07-14 — paged sliding-window KV evicts live audio under RNN-T
token cadence; see docs/intent/port/decisions/
attention-window-vehicle.md in the notes repo), the depthwise-conv
tails, the predictor LSTM, and the replay queue. ``MambaSpec`` is the
landed mechanism for constant-size per-session state (``ShortConv``
precedent), not a Mamba-specific hack (ledger §8 addendum: the
capability is general; ``MambaSpec`` is today's vehicle).

Each layer implements the ``MambaBase`` contract (get_state_shape /
get_state_dtype / mamba_type), registers itself in the static forward
context via its ``prefix``, and holds one page per session, zeroed at
admission (PORT-STATE-003). Dtypes delegate to the ``PrecisionPolicy``
(PORT-PREC-001/005: recurrent state defaults fp32, never silently
reduced).
"""

from collections.abc import Iterable, Sequence
from typing import Any

import torch

from vllm_omni.model_executor.models.nemotron_asr.precision import (
    PrecisionPolicy,
)

try:  # engine-integration imports; absent in GPU-free unit tests
    from vllm.model_executor.layers.mamba.abstract import MambaBase
except ImportError:  # pragma: no cover - exercised only off-engine
    MambaBase = object  # type: ignore[assignment,misc]


class _StatePage(MambaBase):  # type: ignore[misc]
    """One constant-size per-session state page."""

    def __init__(
        self,
        *,
        prefix: str,
        shapes: tuple[tuple[int, ...], ...],
        tensor_classes: tuple[str, ...],
        policy: PrecisionPolicy,
    ) -> None:
        self.prefix = prefix
        self._shapes = shapes
        self._tensor_classes = tensor_classes
        self._policy = policy
        self.kv_cache: tuple[torch.Tensor, ...] = ()

    def get_state_shape(self) -> Iterable[tuple[int, ...]]:
        return self._shapes

    def get_state_dtype(self) -> tuple[torch.dtype, ...]:
        return tuple(
            self._policy.dtype_for(cls) for cls in self._tensor_classes
        )

    @property
    def mamba_type(self):  # MambaAttentionBackendEnum at runtime
        from vllm.v1.attention.backends.registry import (
            MambaAttentionBackendEnum,
        )

        return MambaAttentionBackendEnum.SHORT_CONV


class ConvCachePage(_StatePage):
    """Depthwise-conv tail per layer: ``(d_model, kernel-1)``."""

    def __init__(
        self,
        *,
        prefix: str,
        d_model: int,
        kernel: int,
        policy: PrecisionPolicy,
    ) -> None:
        super().__init__(
            prefix=prefix,
            shapes=((d_model, kernel - 1),),
            tensor_classes=("conv_state",),
            policy=policy,
        )


class LSTMStatePage(_StatePage):
    """Predictor LSTM ``(h, c)``: two ``(layers, hidden)`` states."""

    def __init__(
        self,
        *,
        prefix: str,
        pred_rnn_layers: int,
        pred_hidden: int,
        policy: PrecisionPolicy,
    ) -> None:
        super().__init__(
            prefix=prefix,
            shapes=(
                (pred_rnn_layers, pred_hidden),
                (pred_rnn_layers, pred_hidden),
            ),
            tensor_classes=("lstm_state", "lstm_state"),
            policy=policy,
        )


class WindowCachePage(_StatePage):
    """The encoder left-context window: the fourth state-page kind.

    The OPEN-α3-VEHICLE decision (docs/intent/port/decisions/
    attention-window-vehicle.md, measured 2026-07-14): the 56-frame
    window lives as constant-size per-session state — NeMo's
    ``cache_last_channel`` form, pre-projection, per layer — NOT as
    paged sliding-window KV, whose positional eviction frees live
    audio blocks under RNN-T token cadence (first evicting burst = 1,
    all five geometries). Shapes: ``(window, d_model)`` plus a
    per-page valid-length scalar (self-contained pages beat a global
    length coupling for ``state_indices`` addressing; every layer's
    stream advance writes the same value). The golden-proven
    ``stream_step`` advances it — the engine path is the oracle path,
    literally.
    """

    def __init__(
        self,
        *,
        prefix: str,
        window: int,
        d_model: int,
        policy: PrecisionPolicy,
    ) -> None:
        super().__init__(
            prefix=prefix,
            shapes=((window, d_model), (1,)),
            tensor_classes=("attention_cache", "queue_state"),
            policy=policy,
        )


class ReplayQueuePage(_StatePage):
    """D-b replay queue + decode bookkeeping (PORT-DEC-002).

    Slots: ``max_symbols_per_step * max_frames_per_chunk`` queued label
    ids, plus a 4-slot bookkeeping vector (queue head, queue length,
    last label, prompt index). Label ids store losslessly in the
    ``queue_state`` tensor class's float dtype (vocab 13088 << 2**24 at
    fp32), keeping the provenance-locked fp32-all policy identifier
    intact rather than introducing an integer dtype axis.
    """

    def __init__(
        self,
        *,
        prefix: str,
        max_symbols_per_step: int,
        max_frames_per_chunk: int,
        policy: PrecisionPolicy,
    ) -> None:
        capacity = max_symbols_per_step * max_frames_per_chunk
        super().__init__(
            prefix=prefix,
            shapes=((capacity,), (4,)),
            tensor_classes=("queue_state", "queue_state"),
            policy=policy,
        )


def register_state_pages(
    vllm_config: Any, pages: Iterable[_StatePage]
) -> None:
    """Register each page in the static forward context by prefix.

    The engine's layer walk (``get_layers_from_vllm_config``) discovers
    spec-emitting layers through
    ``compilation_config.static_forward_context`` — registration is
    what makes a page's ``get_kv_cache_spec`` reachable
    (PORT-STATE-002). A duplicate prefix is a wiring error and raises
    ``ValueError`` — never a silent overwrite.
    """
    context = vllm_config.compilation_config.static_forward_context
    for page in pages:
        if page.prefix in context:
            raise ValueError(
                f"duplicate state-page prefix {page.prefix!r} in the "
                "static forward context"
            )
        context[page.prefix] = page


def zero_state_pages(
    state_tensors: Sequence[torch.Tensor], block_ids: Sequence[int]
) -> None:
    """Zero the given page blocks in every state tensor, in place.

    Recurrent state is read-before-write: a block freed by one session
    and reallocated to another must be zeroed at admission or the new
    session decodes from the old session's state (PORT-STATE-003).
    Only the named blocks are touched.
    """
    ids = list(block_ids)
    if not ids:
        return
    for tensor in state_tensors:
        tensor[ids] = 0


def window_page_channel_view(
    pool: torch.Tensor, *, block_id: int, d_model: int
) -> torch.Tensor:
    """A ``(window, d_model)`` VIEW into one window page block.

    The engine hands the model flat page blocks; the encoder's
    channel cache reads/writes this view in place — never a copy, so
    the golden-proven ``stream_step`` advance IS the page write
    (OPEN-α3-VEHICLE decision).
    """
    width = pool.shape[-1]
    window, remainder = divmod(width - 1, d_model)
    if remainder or window < 1:
        raise ValueError(
            f"page width {width} is not window*d_model+1 for "
            f"d_model={d_model}"
        )
    return pool[block_id, : window * d_model].view(window, d_model)


def window_page_len_slot(
    pool: torch.Tensor, *, block_id: int, d_model: int
) -> torch.Tensor:
    """The window page's trailing valid-length slot, as a view."""
    del d_model  # the slot is positional: always the trailing element
    return pool[block_id, -1:]


class _PagedRows:
    """Per-layer page views behind the stacked-tensor indexing.

    ``stream_step`` reads ``caches.channel[idx]`` and writes
    ``caches.channel[idx] = advanced`` — slice assignment. This facade
    keeps those exact semantics over per-layer page VIEWS: reads
    return the view (batch dim restored), writes ``copy_`` into it —
    so the golden-proven advance writes through to the page pool
    without touching ``stream_step`` at all (OPEN-α3-VEHICLE).
    """

    def __init__(self, views: Sequence[torch.Tensor]) -> None:
        self._views = list(views)

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self._views[idx].unsqueeze(0)

    def __setitem__(self, idx: int, value: torch.Tensor) -> None:
        self._views[idx].copy_(value.squeeze(0))

    @property
    def shape(self) -> tuple[int, ...]:
        head = self._views[0]
        return (len(self._views), 1, *head.shape)


class PagedStreamingCaches:
    """``StreamingCaches``' page-backed twin (one session, B=1).

    Same attribute surface ``stream_step`` consumes — ``channel``,
    ``time``, ``valid``, ``left_context`` — with channel/time as
    :class:`_PagedRows` over window/conv page views and ``valid`` as a
    property writing through to every window page's valid-length slot
    (``stream_step`` REPLACES ``caches.valid`` each step, so a plain
    attribute would silently detach from the pages).
    """

    def __init__(
        self,
        *,
        channel_views: Sequence[torch.Tensor],
        time_views: Sequence[torch.Tensor],
        len_slots: Sequence[torch.Tensor],
        left_context: int,
    ) -> None:
        if not (len(channel_views) == len(time_views) == len(len_slots)):
            raise ValueError("one window/conv/len view per layer")
        self.left_context = left_context
        self.channel = _PagedRows(channel_views)
        self.time = _PagedRows(time_views)
        self._len_slots = list(len_slots)

    @property
    def valid(self) -> torch.Tensor:
        return self._len_slots[0].to(torch.long)

    @valid.setter
    def valid(self, value: torch.Tensor) -> None:
        new = value.reshape(1).to(self._len_slots[0].dtype)
        for slot in self._len_slots:
            slot.copy_(new)


class HybridStateModelMixin:
    """The ``IsHybrid`` conformance surface (consult D-α2a, as
    amended by the OPEN-α3-VEHICLE migration).

    Core makes attention and state pages coexist by pre-equalization,
    never unification: the ``is_hybrid`` flag routes the config
    pre-pass (``HybridAttentionMambaModelConfig``), and the post-load
    platform hook (``_align_hybrid_block_size``) sets
    ``cache_config.mamba_page_size_padded`` from the reference state
    bundle these hooks report. That machinery aligns state pages
    *against an attention spec* — and post-migration this model has
    none: the left-context window is itself a ``MambaSpec`` page (the
    fourth kind), so ``is_hybrid`` is False and the hooks stay as
    dormant documentation of the reference bundle. The model computes
    NO page layout of its own (PORT-STATE-001/002 as amended).

    The reference bundle is the *largest per-layer page kind* — now
    the window page (channel cache + valid-length slot). Geometry
    reads from the model config with the checkpoint's defaults; the
    α4 registration wires the real config through.
    """

    #: is_hybrid is core's CONFIG-ROUTING channel, not an attention
    #: claim: for an arch absent from MODELS_CONFIG_MAP, the
    #: ``model_config.is_hybrid`` branch of try_verify_and_update_config
    #: is the only path to MambaModelConfig's pre-pass, which sets the
    #: ``mamba_block_size`` that ``MambaBase.get_kv_cache_spec``
    #: hard-asserts (abstract.py:45-46 @ the pin) — so the flag is
    #: load-bearing and stays True (α4 consult, correcting the
    #: migration's False). The attention-align phase still never runs:
    #: with every backend SSM, ``_find_non_ssm_backend`` returns None
    #: and the platform hook early-returns before both the block-size
    #: pick and the align (interface.py:608-611), so
    #: ``mamba_page_size_padded`` stays unset and grouping runs over
    #: raw heterogeneous pages (measured
    #: RAW_PURE_STATE_GROUPS_FORMED=True). The upstream-native form is
    #: a one-line MODELS_CONFIG_MAP entry (arch → MambaModelConfig);
    #: this flag is the no-core-change downstream channel until then.
    is_hybrid: bool = True

    _DEFAULT_D_MODEL = 1024
    _DEFAULT_WINDOW = 56

    @classmethod
    def get_mamba_state_shape_from_config(
        cls, vllm_config: Any
    ) -> tuple[tuple[int, ...], ...]:
        """The reference per-layer state bundle (the window page)."""
        hf_config = getattr(
            getattr(vllm_config, "model_config", None), "hf_config", None
        )
        d_model = getattr(hf_config, "d_model", cls._DEFAULT_D_MODEL)
        window = getattr(hf_config, "att_context_left", cls._DEFAULT_WINDOW)
        return ((window, d_model), (1,))

    @classmethod
    def get_mamba_state_dtype_from_config(
        cls, vllm_config: Any
    ) -> tuple[torch.dtype, ...]:
        """Recurrent state defaults fp32 (PORT-PREC-001/005).

        One dtype per tensor in the reference bundle (window channel
        cache + valid-length slot).
        """
        return (torch.float32, torch.float32)

    @classmethod
    def get_mamba_state_copy_func(cls, vllm_config: Any) -> Any:
        """Align-mode prefix caching only; off for this model (v1).

        Raises:
            NotImplementedError: Always, until a session-caching slice
                turns align-mode on — a loud seam, never a silent one.
        """
        raise NotImplementedError(
            "prefix caching is off for the streaming ASR model; "
            "mamba_cache_mode stays 'none'"
        )


def state_page_prefixes(n_encoder_layers: int) -> list[tuple[str, str]]:
    """(kind, prefix) for every state page, F4-compliant.

    ``bind_kv_cache`` runs ``extract_layer_index`` on each registered
    layer name and asserts exactly ONE integer path component
    (``utils.py`` @ the pin). The per-encoder-layer window/conv pages
    carry the layer index natively; the model-global LSTM and replay
    pages get a synthetic ``.0.`` slot so they satisfy the same rule.
    """
    prefixes: list[tuple[str, str]] = []
    for i in range(n_encoder_layers):
        prefixes.append(("window", f"encoder.layers.{i}.window"))
        prefixes.append(("conv", f"encoder.layers.{i}.conv"))
    prefixes.append(("lstm", "predictor.layers.0.lstm_state"))
    prefixes.append(("replay", "decode.layers.0.replay"))
    return prefixes
