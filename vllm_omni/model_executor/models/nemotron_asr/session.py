# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The model-local realtime session contract (PORT-RTC-001/002).

The typed object that rides the ``SupportsRealtime.buffer_realtime_audio``
``model_config`` position: admitted geometry, park id, and placeholder
id resolved once from the validated HF configuration and immutable for
the session's life; the session-control prompt selected only through
the checkpoint's own locale authority; and the receipt ledger through
which consumers learn what PORT minted instead of recomputing cadence
arithmetic.

Engine-free (dataclasses/asyncio/manifests only — no ``torch``, no
``vllm``), so it loads and runs on the macOS loader path alongside
``streaming.py``.
"""

from __future__ import annotations

import asyncio
import math
from collections import deque
from dataclasses import dataclass
from typing import Any

from vllm_omni.model_executor.models.nemotron_asr.configuration_nemotron_asr import (
    validate_prompt_dictionary,
)
from vllm_omni.model_executor.models.nemotron_asr.manifests import (
    CADENCES,
    FRONTEND_CONSTANTS,
    RAW_SAMPLES_PER_CHUNK,
)

#: The channel-less standard serving path has no per-session admission
#: channel today, so the package publishes explicit, greppable defaults
#: rather than attribute fallbacks (round-3 gate, decisions 1-2).
DEFAULT_CADENCE = "560ms"
DEFAULT_LOCALE = "auto"

#: Seconds of admitted-cadence audio the ledger will hold as pending
#: carrier tickets before failing the session. A model-package value:
#: the backlog bound is RFC-1's own contract and depends on no consumer
#: invariant.
LEDGER_BACKLOG_S = 30.0

_SAMPLE_RATE_HZ: int = int(FRONTEND_CONSTANTS["sample_rate"])
#: Geometry ids follow manifests.CADENCES order (PORT-SESS-002: the
#: geometry is admission-selected, never invented).
_GEOMETRY_ID_BY_CADENCE = {label: index for index, label in enumerate(CADENCES)}


# @spec PORT-LID-001, PORT-REGIME-003
def normalize_locale_tag(locale: str) -> str:
    """Normalize one BCP-47-shaped locale tag without resolving it."""
    parts = locale.strip().replace("_", "-").split("-")
    normalized = [parts[0].lower()]
    for part in parts[1:]:
        if len(part) == 4 and part.isalpha():
            normalized.append(part.title())
        elif len(part) == 2 and part.isalpha():
            normalized.append(part.upper())
        else:
            normalized.append(part.lower())
    return "-".join(normalized)


# @spec PORT-LID-001, PORT-REGIME-003
def resolve_checkpoint_locale(locale: str, prompts: dict[str, int]) -> str:
    """Resolve a request locale through the checkpoint prompt authority.

    Exact checkpoint keys win. Otherwise locale casing is normalized;
    a two-letter ISO-639-1 code maps only when exactly one checkpoint
    locale has that language prefix. Ambiguous codes require an
    explicit checkpoint locale.

    Args:
        locale: Request locale or ISO-639-1 language code.
        prompts: Validated checkpoint prompt dictionary.

    Returns:
        The exact checkpoint locale key.

    Raises:
        ValueError: If the locale is unknown or an ISO code is
            ambiguous.
    """
    stripped = locale.strip()
    if stripped in prompts:
        return stripped

    candidate = normalize_locale_tag(stripped)
    normalized_matches = [key for key in prompts if normalize_locale_tag(key) == candidate]
    if len(normalized_matches) == 1:
        return normalized_matches[0]
    if len(normalized_matches) > 1:
        raise ValueError(
            f"{locale!r} is ambiguous after locale normalization; use "
            f"an explicit checkpoint locale from {sorted(normalized_matches)}"
        )

    if len(candidate) == 2 and candidate.isalpha():
        iso_matches = [key for key in prompts if normalize_locale_tag(key).split("-", 1)[0] == candidate]
        if len(iso_matches) == 1:
            return iso_matches[0]
        if len(iso_matches) > 1:
            raise ValueError(
                f"ISO-639-1 code {candidate!r} is ambiguous for this "
                "checkpoint; use an explicit checkpoint locale from "
                f"{sorted(iso_matches)}"
            )

    raise ValueError(
        f"{locale!r} is not a locale of the served checkpoint's "
        f"prompt_dictionary; valid locales: {sorted(prompts)} "
        "(PORT-LID-001)"
    )


def _require_token_id(value: object, name: str) -> int:
    """Return a checkpoint token id, failing closed on absence.

    Args:
        value: The candidate value read off the HF configuration.
        name: The configuration field name, for the error message.

    Returns:
        The validated token id.

    Raises:
        ValueError: If the value is missing or is not a non-negative int.
    """
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(
            f"{name} must be a non-negative integer in the served "
            f"checkpoint configuration, got {value!r}; session controls "
            "are read from the validated HF configuration and there are "
            "no magic runtime fallbacks (PORT-RTC-001)"
        )
    return value


@dataclass(frozen=True)
class AdmittedGeometry:
    """The cadence a session was admitted at (PORT-SESS-002)."""

    cadence: str
    chunk_samples: int
    geometry_id: int

    @classmethod
    def from_cadence(cls, cadence: str) -> AdmittedGeometry:
        """Derive the admitted geometry from the published manifests.

        Args:
            cadence: A :data:`manifests.CADENCES` label.

        Returns:
            The geometry for that cadence.

        Raises:
            ValueError: If the label is not an admitted cadence.
        """
        geometry_id = _GEOMETRY_ID_BY_CADENCE.get(cadence)
        if geometry_id is None:
            raise ValueError(
                f"{cadence!r} is not an admitted cadence "
                f"({sorted(_GEOMETRY_ID_BY_CADENCE)}); geometry is "
                "selected at admission, never invented (PORT-SESS-002)"
            )
        return cls(
            cadence=cadence,
            chunk_samples=RAW_SAMPLES_PER_CHUNK[cadence],
            geometry_id=geometry_id,
        )

    @property
    def seconds(self) -> float:
        """The admitted cadence's duration in seconds."""
        return self.chunk_samples / _SAMPLE_RATE_HZ


@dataclass(frozen=True)
class CarrierTicket:
    """PORT's promise that one carrier was minted (PORT-RTC-002).

    Created at ready-stamp time, BEFORE the carrier's prompt leaves the
    segmenter, so the awaitable handle strictly precedes the park it
    will be completed by — a park can never outrun its handle.
    """

    sequence: int
    final_tail: bool
    admission_ms_mod: int
    done: asyncio.Future[Any]


@dataclass(frozen=True)
class PieceReceipt:
    """PORT's acknowledgement of one consumed input frame."""

    samples_consumed: int
    tickets: tuple[CarrierTicket, ...]


class ReceiptLedger:
    """The session's authoritative record of minted carriers.

    PORT mints a ticket per cadence and acknowledges each consumed
    frame; consumers await those statements instead of predicting them
    from cadence arithmetic. All PORT-side calls are synchronous, so
    the segmenter's stamp-then-drain pass gains no ``await``
    (PORT-SESS-001).

    The bounded quantity is pending tickets — minted, not yet completed
    at park. That is the only quantity the ledger itself grows, and the
    bound is the model's own: exceeding it fails the session loudly at
    mint, inside the segmenter and before the prompt is yielded, rather
    than dropping a ticket or growing without limit.
    """

    def __init__(self, *, max_pending_carriers: int) -> None:
        if max_pending_carriers < 1:
            raise ValueError("max_pending_carriers must be positive")
        self._max_pending_carriers = max_pending_carriers
        self._pending: deque[CarrierTicket] = deque()
        self._minted_by_frame: list[CarrierTicket] = []
        self._receipts: deque[PieceReceipt] = deque()
        self._waiter: asyncio.Future[PieceReceipt] | None = None
        self._samples_consumed = 0
        self._sequence = 0
        self._failed: BaseException | None = None

    def _reject_if_failed(self) -> None:
        """Raise once the ledger has been terminally failed."""
        if self._failed is not None:
            raise RuntimeError(
                "the receipt ledger was terminally failed; no further mint/acknowledge/consume is valid (PORT-RTC-002)"
            ) from self._failed

    @property
    def max_pending_carriers(self) -> int:
        """The model-owned cap on pending carrier tickets."""
        return self._max_pending_carriers

    @property
    def pending(self) -> tuple[CarrierTicket, ...]:
        """Minted tickets not yet completed, in mint order."""
        return tuple(self._pending)

    def mint(self, *, final_tail: bool, admission_ms_mod: int) -> CarrierTicket:
        """Record one minted carrier and return its ticket.

        Args:
            final_tail: Whether this carrier is the session's final tail.
            admission_ms_mod: The carrier's ready-stamp (PORT-SESS-001).

        Returns:
            The ticket, whose ``done`` future the consumer completes at
            park.

        Raises:
            RuntimeError: If the pending-ticket cap is already reached,
                or the ledger has been terminally failed.
        """
        self._reject_if_failed()
        if len(self._pending) >= self._max_pending_carriers:
            raise RuntimeError(
                f"pending carrier tickets reached the session's backlog "
                f"bound ({self._max_pending_carriers}); the receipt "
                "ledger fails the session rather than drop a ticket or "
                "grow without bound (PORT-RTC-002)"
            )
        ticket = CarrierTicket(
            sequence=self._sequence,
            final_tail=final_tail,
            admission_ms_mod=admission_ms_mod,
            done=asyncio.get_running_loop().create_future(),
        )
        self._sequence += 1
        self._pending.append(ticket)
        self._minted_by_frame.append(ticket)
        return ticket

    def acknowledge_piece(self, samples: int) -> PieceReceipt:
        """Publish the receipt for one consumed input frame.

        Args:
            samples: Sample count of the frame just consumed.

        Returns:
            The receipt naming exactly the tickets that frame minted
            (empty for a sub-cadence frame).

        Raises:
            RuntimeError: If undrained receipts reach the backlog bound,
                or the ledger has been terminally failed.
        """
        self._reject_if_failed()
        self._samples_consumed += samples
        receipt = PieceReceipt(
            samples_consumed=self._samples_consumed,
            tickets=tuple(self._minted_by_frame),
        )
        self._minted_by_frame.clear()
        waiter = self._waiter
        if waiter is not None and not waiter.done():
            self._waiter = None
            waiter.set_result(receipt)
            return receipt
        # Consumers serialize ``feed``, so at most one acknowledgement
        # is outstanding in practice; the queue keeps the record exact
        # if a driver runs ahead, under the same finite bound.
        if len(self._receipts) >= self._max_pending_carriers:
            raise RuntimeError(
                "undrained piece receipts reached the session's backlog "
                f"bound ({self._max_pending_carriers}) (PORT-RTC-002)"
            )
        self._receipts.append(receipt)
        return receipt

    async def next_piece(self) -> PieceReceipt:
        """Await the receipt for the next consumed frame.

        Returns:
            The oldest unread piece receipt.

        Raises:
            RuntimeError: If another consumer is already awaiting one.
            BaseException: The original failure passed to :meth:`fail`,
                if the ledger has already been (or becomes) terminally
                failed — checked before any queued receipt is served, so
                a terminal failure always dominates a stale queued
                receipt (PORT-RTC-002).
        """
        if self._failed is not None:
            raise self._failed
        if self._receipts:
            return self._receipts.popleft()
        if self._waiter is not None:
            raise RuntimeError(
                "piece acknowledgement is single-slot: feed calls are "
                "serialized, so only one consumer may await (PORT-RTC-002)"
            )
        self._waiter = asyncio.get_running_loop().create_future()
        try:
            return await self._waiter
        finally:
            self._waiter = None

    def complete_next(self, payload: Any) -> CarrierTicket:
        """Complete the oldest pending ticket at an observed park.

        Args:
            payload: An opaque consumer value PORT never reads.

        Returns:
            The completed ticket.

        Raises:
            RuntimeError: If no ticket is pending — the causal chain
                (ticket, then prompt, then park) can only break on a
                protocol error, so this is never a silent skip — or the
                ledger has been terminally failed.
        """
        self._reject_if_failed()
        if not self._pending:
            raise RuntimeError(
                "park observed with no pending carrier ticket; a ticket "
                "always precedes its carrier's prompt, so this means the "
                "causal chain broke (PORT-RTC-002)"
            )
        ticket = self._pending.popleft()
        if not ticket.done.done():
            ticket.done.set_result(payload)
        return ticket

    def fail(self, error: BaseException) -> None:
        """Terminally close the ledger, failing every outstanding waiter.

        Called by the engine binding when generation or rendering ends
        before the session's normal completion: without this, a consumer
        blocked in :meth:`next_piece` (engine died before acknowledging a
        frame) or awaiting a :class:`CarrierTicket`'s ``done`` future
        (engine died between mint and park) would hang forever. ``fail``
        fails the piece waiter and every pending ticket with ``error``,
        discards any already-queued receipts (a terminal failure must
        dominate them — a stale queued receipt is not a successful
        result), and makes later mint/acknowledge/consume calls reject
        (PORT-RTC-002). Idempotent: a second call is a no-op, so the
        binding's success and error paths can both call it defensively.

        Args:
            error: The terminal failure propagated to every waiter.
        """
        if self._failed is not None:
            return
        self._failed = error
        waiter = self._waiter
        if waiter is not None and not waiter.done():
            self._waiter = None
            waiter.set_exception(error)
        while self._pending:
            ticket = self._pending.popleft()
            if not ticket.done.done():
                ticket.done.set_exception(error)
        self._minted_by_frame.clear()
        self._receipts.clear()


# @spec PORT-RTC-001, PORT-LID-001
class NemotronRealtimeSession:
    """One realtime ASR session's model-local contract (PORT-RTC-001).

    Geometry, park id, and placeholder id are immutable for the
    session's life; the session-control prompt is the one mutable
    control and moves only through :meth:`select_prompt`, which
    re-resolves against the checkpoint's validated locale authority
    (PORT-LID-001).
    """

    def __init__(
        self,
        *,
        geometry: AdmittedGeometry,
        park_token_id: int,
        audio_chunk_token_id: int,
        prompts: dict[str, int],
        prompt_index: int,
        ledger: ReceiptLedger | None = None,
    ) -> None:
        self._geometry = geometry
        self._park_token_id = park_token_id
        self._audio_chunk_token_id = audio_chunk_token_id
        self._prompts = dict(prompts)
        self._prompt_index = prompt_index
        self._ledger = ledger

    @classmethod
    def from_model_config(
        cls,
        model_config: Any,
        *,
        cadence: str = DEFAULT_CADENCE,
        locale: str = DEFAULT_LOCALE,
        with_ledger: bool = False,
        max_pending_carriers: int | None = None,
    ) -> NemotronRealtimeSession:
        """Build a session from the served checkpoint's configuration.

        The only place session-control defaults live, and the only
        constructor callers use: an admission value that is missing,
        out of range, or unknown fails here, before a session exists.

        Args:
            model_config: The vLLM ``ModelConfig`` wrapper, or a bare
                ``NemotronASRConfig`` (test paths).
            cadence: A :data:`manifests.CADENCES` label.
            locale: An exact checkpoint locale or an unambiguous
                ISO-639-1 code.
            with_ledger: Whether to arm the receipt ledger.
            max_pending_carriers: Override for the ledger's backlog
                bound; the default is :data:`LEDGER_BACKLOG_S` of
                admitted-cadence audio.

        Returns:
            The typed session.

        Raises:
            ValueError: On any missing or invalid configuration value,
                an unknown cadence, or an unknown/ambiguous admission
                locale.
        """
        hf = getattr(model_config, "hf_config", model_config)
        geometry = AdmittedGeometry.from_cadence(cadence)
        prompts = validate_prompt_dictionary(
            getattr(hf, "prompt_dictionary", None),
            getattr(hf, "num_prompts", None),
        )
        resolved_locale = resolve_checkpoint_locale(locale, prompts)
        ledger = None
        if with_ledger:
            if max_pending_carriers is None:
                max_pending_carriers = math.ceil(LEDGER_BACKLOG_S / geometry.seconds)
            ledger = ReceiptLedger(max_pending_carriers=max_pending_carriers)
        return cls(
            geometry=geometry,
            park_token_id=_require_token_id(getattr(hf, "eos_token_id", None), "eos_token_id"),
            audio_chunk_token_id=_require_token_id(
                getattr(hf, "audio_chunk_token_id", None),
                "audio_chunk_token_id",
            ),
            prompts=prompts,
            prompt_index=prompts[resolved_locale],
            ledger=ledger,
        )

    @property
    def geometry(self) -> AdmittedGeometry:
        """The admitted geometry — immutable for the session's life."""
        return self._geometry

    @property
    def park_token_id(self) -> int:
        """The checkpoint's park id (``eos_token_id``)."""
        return self._park_token_id

    @property
    def audio_chunk_token_id(self) -> int:
        """The minted carrier's placeholder token id."""
        return self._audio_chunk_token_id

    @property
    def prompt_index(self) -> int:
        """The conditioning row the next mint will stamp."""
        return self._prompt_index

    @property
    def ledger(self) -> ReceiptLedger | None:
        """The receipt ledger, or ``None`` when no consumer armed one."""
        return self._ledger

    def select_prompt(self, locale: str) -> int:
        """Select the session-control prompt for a locale.

        Args:
            locale: An exact checkpoint locale or an unambiguous
                ISO-639-1 code.

        Returns:
            The selected conditioning row.

        Raises:
            ValueError: If the locale is unknown or ambiguous — the
                update alone is rejected and the prior selection
                stands (PORT-LID-001).
        """
        resolved_locale = resolve_checkpoint_locale(locale, self._prompts)
        index = self._prompts[resolved_locale]
        self._prompt_index = index
        return index
