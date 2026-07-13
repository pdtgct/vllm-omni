# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Per-session state pages via core vLLM's cache-spec system.

The cache-aware streaming capability declares all cross-chunk model
state through `kv_cache_interface` spec pages (PORT-STATE-001/002):
left-context attention rides paged sliding-window KV (the block-pooled
attention layer); the three non-attention states here ride
``MambaSpec`` pages — the landed mechanism for constant-size
per-session state (``ShortConv`` precedent), not a Mamba-specific hack
(ledger §8 addendum: the capability is general; ``MambaSpec`` is
today's vehicle).

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
    raise NotImplementedError


def zero_state_pages(
    state_tensors: Sequence[torch.Tensor], block_ids: Sequence[int]
) -> None:
    """Zero the given page blocks in every state tensor, in place.

    Recurrent state is read-before-write: a block freed by one session
    and reallocated to another must be zeroed at admission or the new
    session decodes from the old session's state (PORT-STATE-003).
    Only the named blocks are touched.
    """
    raise NotImplementedError
