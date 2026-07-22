"""The in-process engine binding for the compute seam (ING-VEH).

``EngineTranscriber`` implements the :class:`~nemotron_asr_ingress.core.
Transcriber` protocol directly over the engine's own streaming-generation
entry point (ING-VEH-004): it drives the model's realtime segmenter seam
(``SupportsRealtime.buffer_realtime_audio``) with a per-session
configuration view (ING-VEH-007), renders each minted prompt through an
injected callable backed by the engine's own renderer and the REAL
process model config (ING-VEH-008), and watches the output stream for
each CHUNK's park token to satisfy ``step``'s cumulative-hypothesis
contract. Everything engine-coupled arrives injected, so this module
stays sans-IO and GPU-free-testable; the real wiring lives in the
serve-tier bootstrap, outside this package.

The bypass reimplements no pacing: the segmenter's own hold-until-park
backpressure stays the mechanism (PORT-REGIME-001), fed here by echoing
every output token batch onto its ``input_stream``. Awaiting a chunk's
own park adds no latency beyond that pacing (PORT-SESS-001).
"""

from __future__ import annotations

import asyncio
from collections import deque
from collections.abc import (
    AsyncIterator,
    Awaitable,
    Callable,
    Mapping,
)
from typing import Any

from nemotron_asr_ingress.core import FloatAudio

#: ``buffer_realtime_audio``-shaped: (audio_stream, input_stream,
#: model_config) -> async iterator of minted prompts.
Segment = Callable[
    [AsyncIterator[FloatAudio], "asyncio.Queue[list[int]]", Any],
    AsyncIterator[Any],
]
#: Prompt -> engine-submittable input (the serving path's parse ->
#: render -> streaming-input wrap, mirrored by the bootstrap).
Render = Callable[[Any], Awaitable[Any]]
#: (rendered prompt stream, request_id) -> engine output stream.
Generate = Callable[[AsyncIterator[Any], str], AsyncIterator[Any]]
#: Frees the engine request's state on a non-normal end.
AbortRequest = Callable[[str], Awaitable[None]]


# @spec ING-VEH-007
class SessionConfigView:
    """A per-session view over the engine's process model config.

    Delegates every attribute it does not override to the base config;
    overrides (the admitted chunk geometry, the mutable session-control
    prompt selection, the park id) live on the instance and never touch
    the shared base. The view goes ONLY to the segmenter seam — the
    render path keeps the real config (ING-VEH-008).
    """

    def __init__(self, base: Any, **overrides: Any) -> None:
        self._base = base
        self.__dict__.update(overrides)

    def __getattr__(self, name: str) -> Any:
        return getattr(self.__dict__["_base"], name)

    def __setattr__(self, name: str, value: Any) -> None:
        # Writes land on the view, never the shared base config.
        self.__dict__[name] = value


# @spec ING-VEH-004, ING-VEH-007, ING-VEH-008, ING-LIFE-010
class EngineTranscriber:
    """``Transcriber`` bound directly over ``engine.generate()``.

    One instance = one session = one long-lived engine request. A
    background consumer drains the generation's outputs, echoes every
    token batch onto the segmenter's ``input_stream`` (feeding its
    hold-until-park backpressure), accumulates the cumulative
    hypothesis, and resolves the oldest pending ``step`` on each park
    (one park per admitted CHUNK, PORT-SESS-001). ``flush`` and the
    skip-flush ``finish`` both end the session through the explicit
    final-tail transaction (PORT-SESS-003) — ``abort`` alone is the
    non-normal teardown (ING-LIFE-010).
    """

    def __init__(
        self,
        *,
        segment: Segment,
        render: Render,
        generate: Generate,
        abort_request: AbortRequest,
        config_view: Any,
        locale_index: Mapping[str, int],
        request_id: str,
    ) -> None:
        park = getattr(config_view, "park_token_id", None)
        if park is None:
            raise ValueError(
                "config_view must carry park_token_id: the park is the "
                "step/CHUNK synchronization signal (PORT-SESS-001)"
            )
        self._park_id = int(park)
        self._segment = segment
        self._render = render
        self._generate = generate
        self._abort_request = abort_request
        self._config_view = config_view
        self._locale_index = locale_index
        self._request_id = request_id
        self._audio: asyncio.Queue[FloatAudio | None] = asyncio.Queue()
        self._input_stream: asyncio.Queue[list[int]] = asyncio.Queue()
        self._waiters: deque[asyncio.Future[str]] = deque()
        self._task: asyncio.Task[None] | None = None
        self._done = asyncio.Event()
        self._text = ""
        self._error: BaseException | None = None
        self._ended = False
        self._audio_closed = False
        self._aborted = False

    async def step(self, chunk: FloatAudio) -> str:
        """Advance one admitted-config CHUNK; await its own park."""
        self._ensure_started()
        waiter: asyncio.Future[str] = (
            asyncio.get_running_loop().create_future()
        )
        self._waiters.append(waiter)
        self._audio.put_nowait(chunk)
        return await waiter

    async def flush(self, residual: FloatAudio) -> str:
        """Finalize with the raw residual; return the final transcript.

        Waits for the generation to END (not merely the final park), so
        any output the engine emits while draining the final-tail still
        lands in the final transcript.
        """
        self._ensure_started()
        if len(residual):
            self._audio.put_nowait(residual)
        await self._drain()
        return self._text

    async def update_locale(self, target_lang: str) -> None:
        """Select the session-control prompt for the NEXT mint."""
        try:
            index = self._locale_index[target_lang]
        except KeyError:
            raise ValueError(
                f"locale {target_lang!r} has no session-control prompt "
                "index; admission validation precedes this seam "
                "(PORT-LID-001)"
            ) from None
        self._config_view.nemotron_prompt_index = index

    async def abort(self) -> None:
        """Free engine state immediately (non-normal end)."""
        if self._aborted:
            return
        self._aborted = True
        while self._waiters:
            self._waiters.popleft().cancel()
        if self._task is None:
            return
        ended = self._ended
        if not self._task.done():
            self._task.cancel()
        await asyncio.gather(self._task, return_exceptions=True)
        if not ended:
            await self._abort_request(self._request_id)

    async def finish(self) -> None:
        """Terminal call on the normal end (ING-LIFE-010).

        After ``flush`` this is a no-op; on the skip-flush path it ends
        the open generation gracefully through the segmenter's explicit
        (zero-sample) final-tail transaction, never an abort.
        """
        if self._task is None or self._aborted:
            return
        await self._drain()

    def _ensure_started(self) -> None:
        if self._task is None:
            self._task = asyncio.get_running_loop().create_task(
                self._consume()
            )

    async def _drain(self) -> None:
        if not self._audio_closed:
            self._audio_closed = True
            self._audio.put_nowait(None)
        await self._done.wait()
        if self._task is not None:
            await asyncio.gather(self._task, return_exceptions=True)
        if self._error is not None:
            raise self._error

    async def _audio_frames(self) -> AsyncIterator[FloatAudio]:
        while True:
            frame = await self._audio.get()
            if frame is None:
                return
            yield frame

    async def _rendered(self) -> AsyncIterator[Any]:
        prompts = self._segment(
            self._audio_frames(), self._input_stream, self._config_view
        )
        async for prompt in prompts:
            yield await self._render(prompt)

    async def _consume(self) -> None:
        try:
            outputs = self._generate(self._rendered(), self._request_id)
            async for output in outputs:
                if getattr(output, "stage_id", 0) not in (None, 0):
                    continue
                stage_outputs = getattr(output, "outputs", None)
                if not stage_outputs:
                    continue
                first = stage_outputs[0]
                ids = list(first.token_ids)
                if ids:
                    self._input_stream.put_nowait(ids)
                self._text += first.text or ""
                if self._park_id in ids and self._waiters:
                    waiter = self._waiters.popleft()
                    if not waiter.done():
                        waiter.set_result(self._text)
            self._ended = True
        except Exception as error:
            self._error = error
        finally:
            self._done.set()
            failure = self._error or RuntimeError(
                "generation ended before the CHUNK's park"
            )
            while self._waiters:
                waiter = self._waiters.popleft()
                if not waiter.done():
                    waiter.set_exception(failure)
