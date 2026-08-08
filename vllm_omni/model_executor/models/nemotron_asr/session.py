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

The ``StreamingObserver`` protocol and its ``ChunkReadyHandle`` handle
type are NOT defined here: they live in the neutral, Prometheus-free
``vllm_omni.metrics.streaming_transport`` module (PORT-OBS-003's
"Observer protocol home" decision) so this model package imports a
capability-neutral seam rather than owning it, and the Prometheus
metrics adapter can be statically typed against the same protocol
without this package importing Prometheus.
"""

from __future__ import annotations

import asyncio
import math
import time
from collections import deque
from dataclasses import dataclass
from typing import Any

from vllm_omni.metrics.streaming_transport import (
    ChunkReadyHandle,
    StreamingObserver,
    observe_safely,
)
from vllm_omni.model_executor.models.nemotron_asr.accepted_audio import (
    AcceptedAudioAuthority,
    AcceptedPiece,
)
from vllm_omni.model_executor.models.nemotron_asr.configuration_nemotron_asr import (
    validate_prompt_dictionary,
)
from vllm_omni.model_executor.models.nemotron_asr.endpointing import EndpointPolicy
from vllm_omni.model_executor.models.nemotron_asr.manifests import (
    CADENCES,
    FRONTEND_CONSTANTS,
    RAW_SAMPLES_PER_CHUNK,
    SESSION_LIMITS,
)
from vllm_omni.model_executor.models.nemotron_asr.profiling import phase
from vllm_omni.model_executor.models.nemotron_asr.transcript import (
    BoundedTranscript,
)

__all__ = [
    "ChunkReadyHandle",
    "StreamingObserver",
    "AdmittedGeometry",
    "CarrierTicket",
    "PieceReceipt",
    "ReceiptLedger",
    "NemotronRealtimeSession",
    "DEFAULT_CADENCE",
    "DEFAULT_LOCALE",
    "LEDGER_BACKLOG_S",
    "ACCEPTED_AUDIO_BUDGET_DEFAULT_S",
    "normalize_locale_tag",
    "resolve_checkpoint_locale",
    "cadence_ms_label",
]

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
DEFAULT_MAX_RETAINED_TRANSCRIPT_BYTES = 1 << 20
DEFAULT_TRANSCRIPT_FRAGMENT_OVERHEAD_BYTES = 16
DEFAULT_TRANSCRIPT_TERMINAL_HEADROOM_BYTES = 4_096

#: PORT's per-session accepted-audio queue budget, in seconds of audio
#: (design §Park, Finalization, and Backpressure, Decision 1): "one
#: coherent backpressure budget with the receipt ledger's pending-carrier
#: bound" — hence sharing :data:`LEDGER_BACKLOG_S`'s default. Serving-
#: owned and ENV-configured as a positive finite value per deployment;
#: checked strictly BEFORE acceptance on both native and leased paths.
#: Phase-5 typed stub only: stored on the session, not yet enforced by
#: any acceptance path.
ACCEPTED_AUDIO_BUDGET_DEFAULT_S = LEDGER_BACKLOG_S

_SAMPLE_RATE_HZ: int = int(FRONTEND_CONSTANTS["sample_rate"])
#: Geometry ids follow manifests.CADENCES order (PORT-SESS-002: the
#: geometry is admission-selected, never invented).
_GEOMETRY_ID_BY_CADENCE = {label: index for index, label in enumerate(CADENCES)}


# @spec PORT-OBS-003, PORT-OBS-004, PORT-OBS-006
def cadence_ms_label(cadence: str) -> str:
    """Strip a :data:`manifests.CADENCES` label's ``"ms"`` suffix.

    The observer protocol's ``cadence_ms`` fields (and the Prometheus
    metrics module's bounded-enum guard, ``defs.STREAMING_CADENCE_MS_VALUES``)
    use the bare numeral (``"560"``), never the manifest label
    (``"560ms"``); this is the one place that conversion happens.
    """
    return cadence.removesuffix("ms")


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

    def __init__(
        self,
        *,
        max_pending_carriers: int,
        observer: StreamingObserver | None = None,
        cadence_ms: str | None = None,
    ) -> None:
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
        # The session's correlation key, bound by the owning session at
        # its construction (set-once): the fallback unit_ready below must
        # key by the SESSION's identity, never the ledger's — a ledger-id
        # key would break complete_inflight resolution (PORT-OBS-003).
        self._session_key: str | None = None
        # PORT-OBS-003/005: the ledger's own bound trip (a real, already-
        # implemented backpressure mechanism, PORT-RTC-002) and its own
        # terminal-failure path are the ONLY observation call sites a bare
        # ``ledger.mint``/``ledger.fail`` caller ever reaches — the
        # segmenter's own unit_ready/unit_minted calls (buffer_stream) are
        # a separate, higher-level event stream keyed by the SAME handles
        # when the caller supplies one via :meth:`mint`'s ``handle``
        # argument. Only a caller that mints directly, bypassing
        # buffer_stream (as these two unit-level tests do), reaches the
        # synthesis fallback below — never the real segmenter flow.
        self._observer = observer
        self._cadence_ms = cadence_ms
        self._handle_by_ticket: dict[int, Any] = {}

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

    def mint(
        self,
        *,
        final_tail: bool,
        admission_ms_mod: int,
        handle: Any = None,
    ) -> CarrierTicket:
        """Record one minted carrier and return its ticket.

        Args:
            final_tail: Whether this carrier is the session's final tail.
            admission_ms_mod: The carrier's ready-stamp (PORT-SESS-001).
            handle: The :class:`ChunkReadyHandle` the segmenter's own
                ``unit_ready`` call already minted for this carrier, when
                called from ``buffer_stream``. When omitted (a caller
                minting directly, bypassing the segmenter), and an
                observer is configured, one is synthesized here instead —
                never both, so a segmenter-driven mint never double-fires
                ``unit_ready``.

        Returns:
            The ticket, whose ``done`` future the consumer completes at
            park.

        Raises:
            RuntimeError: If the pending-ticket cap is already reached,
                or the ledger has been terminally failed.
        """
        self._reject_if_failed()
        if len(self._pending) >= self._max_pending_carriers:
            if self._observer is not None:
                observe_safely(self._observer.overflow, kind="carrier")
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
        if handle is None and self._observer is not None and self._cadence_ms is not None:
            handle = observe_safely(
                self._observer.unit_ready,
                session_key=self._session_key or str(id(self)),
                cadence_ms=self._cadence_ms,
                chunk_type="final_tail" if final_tail else "regular",
                ready_stamp_s=time.monotonic(),
            )
        if handle is not None:
            self._handle_by_ticket[ticket.sequence] = handle
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
        # Park observation (unit_parked) is a lease/native-adapter
        # concern, not the ledger's (see session.py's terminal-
        # disposition ownership note); this bookkeeping map only needs
        # tidying so it never grows past the pending bound.
        self._handle_by_ticket.pop(ticket.sequence, None)
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
            # PORT-OBS-003/005: the ledger's own terminal-failure path is
            # the ticket-level clearing authority for still-pending
            # carriers (the leased-path lease consumer's own ledger
            # failure just calls this method — it does not separately
            # re-clear). Guarded by the same idempotency check above:
            # a second ``fail`` call never re-enters this loop.
            handle = self._handle_by_ticket.pop(ticket.sequence, None)
            if handle is not None and self._observer is not None:
                observe_safely(self._observer.unit_cleared, handle, outcome="error")
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
        eou_token_id: int | None,
        flush_token_id: int,
        endpoint_policy: EndpointPolicy,
        accepted_audio: AcceptedAudioAuthority,
        transcript: BoundedTranscript,
        prompts: dict[str, int],
        prompt_index: int,
        ledger: ReceiptLedger | None = None,
        observer: StreamingObserver | None = None,
        accepted_audio_budget_s: float = ACCEPTED_AUDIO_BUDGET_DEFAULT_S,
        session_key: str | None = None,
    ) -> None:
        self._geometry = geometry
        self._park_token_id = park_token_id
        self._audio_chunk_token_id = audio_chunk_token_id
        self._eou_token_id = eou_token_id
        self._flush_token_id = flush_token_id
        self._endpoint_policy = endpoint_policy
        self._accepted_audio = accepted_audio
        self._transcript = transcript
        self._prompts = dict(prompts)
        self._prompt_index = prompt_index
        self._ledger = ledger
        # PORT-OBS-003 (amended): the per-generation correlation key,
        # minted by the caller (connection request id on the native
        # path, lease request id on the leased path) BEFORE session
        # construction. With an observer attached the key is REQUIRED —
        # a keyless observed session would fall back to object identity
        # and silently break every connection/consumer-side correlation
        # (the D6 defect class); keyless construction remains legal only
        # on the unobserved path (review round 2026-07-28, F6).
        if observer is not None and session_key is None:
            raise ValueError(
                "observer-bearing session construction requires the minted "
                "per-generation correlation key (PORT-OBS-003); keyless "
                "construction is only legal without an observer"
            )
        self._session_key = session_key
        # The transport-neutral factory/lease binding injects the same
        # observer at session construction (PORT-OBS-003); buffer_stream
        # reads it back off the session (or an explicit override) to emit
        # ready/minted events, and the lease consumer reads it back to
        # observe terminal disposition.
        self._observer = observer
        self._ready_handles: dict[int, Any] = {}
        # Design §Park, Finalization, and Backpressure (Decision 1): the
        # per-session accepted-audio queue occupancy budget, enforced by
        # buffer_stream on both the native and leased paths regardless of
        # whether an observer is attached.
        if accepted_audio_budget_s <= 0 or not math.isfinite(
            accepted_audio_budget_s
        ):
            raise ValueError("accepted_audio_budget_s must be positive and finite")
        self._accepted_audio_budget_s = accepted_audio_budget_s
        # Bind the correlation key into the armed ledger so its fallback
        # ready events key by the session, never the ledger (PORT-OBS-003).
        if self._ledger is not None:
            self._ledger._session_key = self.session_key
        # Native open is observer-bearing model-session construction
        # after successful validation and before engine request creation
        # (PORT-OBS-006) — every native/leased path funnels through this
        # constructor, so this is the single, un-duplicated open site,
        # keyed by the per-generation correlation key.
        if self._observer is not None:
            observe_safely(
                self._observer.session_opened,
                session_key=self.session_key,
                cadence_ms=cadence_ms_label(geometry.cadence),
            )

    @classmethod
    def from_model_config(
        cls,
        model_config: Any,
        *,
        cadence: str = DEFAULT_CADENCE,
        locale: str = DEFAULT_LOCALE,
        with_ledger: bool = False,
        max_pending_carriers: int | None = None,
        endpoint_policy: EndpointPolicy | None = None,
        endpoint_history_capacity_frames: int | None = None,
        accepted_audio_capacity_samples: int | None = None,
        max_retained_transcript_bytes: int = DEFAULT_MAX_RETAINED_TRANSCRIPT_BYTES,
        transcript_fragment_overhead_bytes: int = DEFAULT_TRANSCRIPT_FRAGMENT_OVERHEAD_BYTES,
        transcript_terminal_headroom_bytes: int = DEFAULT_TRANSCRIPT_TERMINAL_HEADROOM_BYTES,
        request_id: str = "unbound",
        engine_epoch: str = "unbound",
        lease_generation: int = 0,
        max_session_samples: int | None = None,
        observer: StreamingObserver | None = None,
        accepted_audio_budget_s: float = ACCEPTED_AUDIO_BUDGET_DEFAULT_S,
        session_key: str | None = None,
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
            observer: The optional PORT-OBS-003 chunk/session observer.
                Stub: accepted and stored, not yet wired to any event.
            accepted_audio_budget_s: The serving-configured per-session
                accepted-audio queue budget, in seconds (design §Park,
                Finalization, and Backpressure, Decision 1). Stub:
                accepted and stored, not yet enforced by any acceptance
                path.

        Returns:
            The typed session.

        Raises:
            ValueError: On any missing or invalid configuration value,
                an unknown cadence, or an unknown/ambiguous admission
                locale.
        """
        if session_key is None and request_id != "unbound":
            session_key = request_id
        if session_key is not None:
            if request_id != "unbound" and request_id != session_key:
                raise ValueError("request_id and session_key must match")
            request_id = session_key
        hf = getattr(model_config, "hf_config", model_config)
        geometry = AdmittedGeometry.from_cadence(cadence)
        # PORT-SESS-015: the publisher's declared lookahead arms gate
        # cadence admission. A cadence's implied lookahead is its
        # right attention context (frames_per_chunk - 1); a served
        # configuration declaring no set admits every manifest cadence.
        declared_arms = getattr(hf, "supported_num_lookahead_tokens", None)
        if declared_arms is not None:
            implied = CADENCES[cadence][1]
            if implied not in declared_arms:
                raise ValueError(
                    f"cadence {cadence!r} implies lookahead {implied}, "
                    "which is not in the served configuration's declared "
                    f"supported set {sorted(int(v) for v in declared_arms)} "
                    "(PORT-SESS-015)"
                )
        model_path = getattr(model_config, "model", None)
        if model_path and not getattr(hf, "prompt_dictionary", None):
            from vllm_omni.model_executor.models.nemotron_asr.configuration_nemotron_asr import (
                ensure_prompt_dictionary,
            )

            ensure_prompt_dictionary(hf, model_path)
        prompts = validate_prompt_dictionary(
            getattr(hf, "prompt_dictionary", None),
            getattr(hf, "num_prompts", None),
        )
        resolved_locale = resolve_checkpoint_locale(locale, prompts)
        park_token_id = _require_token_id(
            getattr(hf, "eos_token_id", None),
            "eos_token_id",
        )
        audio_chunk_token_id = _require_token_id(
            getattr(hf, "audio_chunk_token_id", None),
            "audio_chunk_token_id",
        )

        installed_capacity = endpoint_history_capacity_frames
        if installed_capacity is None:
            installed_capacity = int(
                getattr(hf, "endpoint_history_capacity_frames", 12)
            )
        if installed_capacity <= 0:
            raise ValueError("endpoint history capacity must be positive")

        explicit_endpointing = endpoint_policy is not None
        eou_value = getattr(hf, "eou_token_id", None)
        flush_value = getattr(hf, "flush_token_id", None)
        if explicit_endpointing or eou_value is not None or flush_value is not None:
            eou_token_id = _require_token_id(eou_value, "eou_token_id")
            flush_token_id = _require_token_id(flush_value, "flush_token_id")
            controls = (
                park_token_id,
                audio_chunk_token_id,
                eou_token_id,
                flush_token_id,
            )
            if len(set(controls)) != len(controls):
                raise ValueError("session control token ids must be distinct")
            num_asr_labels = getattr(hf, "num_asr_labels", None)
            if isinstance(num_asr_labels, int) and any(
                token_id <= num_asr_labels for token_id in controls
            ):
                raise ValueError("session control token id overlaps label space")
            vocab_size = getattr(hf, "vocab_size", None)
            if isinstance(vocab_size, int) and max(controls) >= vocab_size:
                raise ValueError("session control token id exceeds vocabulary")
        else:
            # Compatibility for model-generic callers that do not select
            # endpointing.  New Nemotron serving always supplies all controls.
            eou_token_id = None
            flush_token_id = 0

        if endpoint_policy is None:
            endpoint_policy = EndpointPolicy.resolve(
                mode="disabled",
                stop_history_ms=None,
                residue_frames=0,
                frame_stride_ms=80,
                history_capacity_frames=installed_capacity,
            )
        if endpoint_policy.history_capacity_frames > installed_capacity:
            raise ValueError(
                "endpoint policy exceeds installed history capacity"
            )

        if accepted_audio_capacity_samples is None:
            accepted_audio_capacity_samples = int(
                accepted_audio_budget_s * _SAMPLE_RATE_HZ
            )
        accepted_audio = AcceptedAudioAuthority(
            request_id=request_id,
            engine_epoch=engine_epoch,
            lease_generation=lease_generation,
            chunk_samples=geometry.chunk_samples,
            capacity_samples=accepted_audio_capacity_samples,
            carrier_sequence_modulus=int(
                SESSION_LIMITS["carrier_sequence_modulus"]
            ),
            max_session_samples=max_session_samples,
        )
        accepted_audio.update_locale(resolved_locale)
        transcript = BoundedTranscript(
            max_retained_bytes=max_retained_transcript_bytes,
            fragment_overhead_bytes=transcript_fragment_overhead_bytes,
            terminal_headroom_bytes=transcript_terminal_headroom_bytes,
        )
        ledger = None
        if with_ledger:
            if max_pending_carriers is None:
                max_pending_carriers = math.ceil(LEDGER_BACKLOG_S / geometry.seconds)
            ledger = ReceiptLedger(
                max_pending_carriers=max_pending_carriers,
                observer=observer,
                cadence_ms=cadence_ms_label(geometry.cadence),
            )
        return cls(
            geometry=geometry,
            park_token_id=park_token_id,
            audio_chunk_token_id=audio_chunk_token_id,
            eou_token_id=eou_token_id,
            flush_token_id=flush_token_id,
            endpoint_policy=endpoint_policy,
            accepted_audio=accepted_audio,
            transcript=transcript,
            prompts=prompts,
            prompt_index=prompts[resolved_locale],
            ledger=ledger,
            observer=observer,
            accepted_audio_budget_s=accepted_audio_budget_s,
            session_key=session_key,
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
    def eou_token_id(self) -> int | None:
        """The checkpoint's semantic end-of-utterance control id."""
        return self._eou_token_id

    @property
    def flush_token_id(self) -> int:
        """The checkpoint's finalization control id."""
        return self._flush_token_id

    @property
    def endpoint_policy(self) -> EndpointPolicy:
        """The immutable endpoint policy selected at admission."""
        return self._endpoint_policy

    @property
    def accepted_audio(self) -> AcceptedAudioAuthority:
        """The sole bounded accepted-audio and control authority."""
        return self._accepted_audio

    @property
    def transcript(self) -> BoundedTranscript:
        """The sole bounded terminal-output authority."""
        return self._transcript

    @property
    def prompt_index(self) -> int:
        """The conditioning row the next mint will stamp."""
        return self._prompt_index

    @property
    def ledger(self) -> ReceiptLedger | None:
        """The receipt ledger, or ``None`` when no consumer armed one."""
        return self._ledger

    @property
    def observer(self) -> StreamingObserver | None:
        """The PORT-OBS-003 observer, or ``None`` when none was injected."""
        return self._observer

    @property
    def session_key(self) -> str:
        """This session's stable observer-correlation identity.

        PORT-OBS-003 (amended): the per-generation correlation key — the
        engine request id, minted by the caller before construction and
        passed in as ``session_key`` — on both the native and leased
        paths, so the adapter/consumer resolves handles and terminal
        state with the identity it already owns. Object identity remains
        the fallback for key-less construction only.
        """
        return self._session_key or str(id(self))

    @property
    def accepted_audio_budget_s(self) -> float:
        """The per-session accepted-audio queue budget, in seconds.

        Design §Park, Finalization, and Backpressure (Decision 1) typed
        stub: not yet enforced by ``buffer_stream`` or any acceptance
        path.
        """
        return self._accepted_audio_budget_s

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
        self._accepted_audio.update_locale(resolved_locale)
        index = self._prompts[resolved_locale]
        self._prompt_index = index
        return index

    def accept_audio(self, samples: Any) -> AcceptedPiece:
        """Atomically accept one whole application-audio piece."""
        prior_sequences = {
            unit.logical_sequence for unit in self._accepted_audio.ready_units
        }
        try:
            # Measured at the call site rather than inside the authority:
            # the span then includes any wait for the authority's lock,
            # which is the contention this phase exists to expose.
            with phase("port.ingest"):
                accepted = self._accepted_audio.accept(samples)
        except ValueError as error:
            if (
                self._observer is not None
                and "buffer_overflow" in str(error)
            ):
                observe_safely(self._observer.overflow, kind="input_queue")
            raise
        if self._observer is not None:
            observe_safely(
                self._observer.accepted_audio_seconds,
                cadence_ms=cadence_ms_label(self._geometry.cadence),
                seconds=accepted.samples_accepted / _SAMPLE_RATE_HZ,
            )
            self._observe_new_ready_units(prior_sequences)
        return accepted

    def force_segment(self) -> None:
        """Queue one ordered, zero-sample semantic boundary barrier."""
        if self._eou_token_id is None:
            raise ValueError("endpointing controls are not configured")
        self._accepted_audio.force_segment()

    def begin_finalize(self, *, finalize_at_ns: int | None = None) -> None:
        """Close audio acceptance and queue exactly one final tail."""
        if self._accepted_audio.snapshot().finalizing:
            return
        prior_sequences = {
            unit.logical_sequence for unit in self._accepted_audio.ready_units
        }
        self._accepted_audio.begin_finalize(finalize_at_ns=finalize_at_ns)
        if self._observer is not None:
            self._observe_new_ready_units(prior_sequences)

    def _observe_new_ready_units(self, prior_sequences: set[int]) -> None:
        """Publish newly ready carrier units from the state authority."""
        assert self._observer is not None
        for unit in self._accepted_audio.ready_units:
            if (
                unit.logical_sequence in prior_sequences
                or unit.kind == "forced_eou"
            ):
                continue
            handle = observe_safely(
                self._observer.unit_ready,
                session_key=self.session_key,
                cadence_ms=cadence_ms_label(self._geometry.cadence),
                chunk_type=unit.kind,
                ready_stamp_s=unit.ready_at_ns / 1_000_000_000,
            )
            if handle is not None:
                self._ready_handles[unit.logical_sequence] = handle

    def take_ready_handle(self, logical_sequence: int) -> Any:
        """Consume the observer handle paired with one dispatched unit."""
        return self._ready_handles.pop(logical_sequence, None)

    def ready_handle(self, logical_sequence: int) -> Any:
        """Return the observer handle without advancing mint ownership."""
        return self._ready_handles.get(logical_sequence)


# @spec PORT-RTC-003
def create_nemotron_session_factory(engine: Any) -> Any:
    """Create the public factory with the engine's exact installed service.

    Construction performs no reservation.  Each later ``open`` owns its
    admission operation and lease lifetime.
    """
    from vllm_omni.entrypoints.nemotron_session import NemotronSessionFactory

    return NemotronSessionFactory(
        engine=engine,
        persistent_state_service=engine.get_persistent_state_service(),
    )
