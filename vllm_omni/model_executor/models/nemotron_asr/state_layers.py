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


class HybridStateModelMixin:
    """The ``IsHybrid`` conformance surface (consult D-α2a).

    Core makes attention and state pages coexist by pre-equalization,
    never unification: the ``is_hybrid`` flag routes the config
    pre-pass (``HybridAttentionMambaModelConfig``), and the post-load
    platform hook (``_align_hybrid_block_size``) picks the attention
    block size and sets ``cache_config.mamba_page_size_padded`` from
    the reference state bundle these hooks report — after which every
    page's spec is born padded and grouping is uniform by
    construction. The model computes NO page layout of its own
    (PORT-STATE-001/002 as amended).

    The reference bundle is the *largest per-layer page kind* — the
    depthwise-conv tail — because the hook's semantics are "one
    layer's state" and the attention page must dominate it; the
    smaller LSTM/queue pages pad up to the same global value.
    Geometry reads from the model config with the checkpoint's
    defaults; the α4 registration wires the real config through.
    """

    is_hybrid: bool = True

    _DEFAULT_D_MODEL = 1024
    _DEFAULT_CONV_KERNEL = 9

    @classmethod
    def get_mamba_state_shape_from_config(
        cls, vllm_config: Any
    ) -> tuple[tuple[int, ...], ...]:
        """The reference per-layer state bundle (the conv tail)."""
        hf_config = getattr(
            getattr(vllm_config, "model_config", None), "hf_config", None
        )
        d_model = getattr(hf_config, "d_model", cls._DEFAULT_D_MODEL)
        kernel = getattr(
            hf_config, "conv_kernel_size", cls._DEFAULT_CONV_KERNEL
        )
        return ((d_model, kernel - 1),)

    @classmethod
    def get_mamba_state_dtype_from_config(
        cls, vllm_config: Any
    ) -> tuple[torch.dtype, ...]:
        """Recurrent state defaults fp32 (PORT-PREC-001/005)."""
        return (torch.float32,)

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
