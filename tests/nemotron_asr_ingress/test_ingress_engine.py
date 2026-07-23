"""The in-process EngineTranscriber and its per-session config view.

ING-VEH-004/007/008 + ING-LIFE-010 at the compute seam: the engine
arrives as injected callables (GPU-free), and the integration cases
drive the REAL segmenter (``streaming.buffer_stream``, loaded
engine-free via importlib) against a fake engine to prove the
park-echo pacing loop, whole-piece burst delivery (ING-FE-006), and
the per-mint locale stamp end to end.
"""

import asyncio
import importlib.util
import sys
import types
from collections.abc import AsyncIterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from nemotron_asr_ingress.engine import EngineTranscriber, SessionConfigView

PARK = 13089
CHUNK = 8960
LOCALES = {"en-US": 2, "es-US": 7}
WAIT_S = 2.0


def _load_real_streaming() -> Any:
    """Load the streaming MODULE engine-free (no vllm imports).

    Each call re-executes streaming.py into a fresh module object, so a
    test may monkeypatch module globals (e.g. ``_admission_ms_mod``)
    without leaking into segmenters loaded by other tests.
    """
    base = "vllm_omni.model_executor.models.nemotron_asr"
    pkg = (
        Path(__file__).resolve().parents[2]
        / "vllm_omni/model_executor/models/nemotron_asr"
    )
    for name in (
        "vllm_omni",
        "vllm_omni.model_executor",
        "vllm_omni.model_executor.models",
        base,
    ):
        if name not in sys.modules:
            sys.modules[name] = types.ModuleType(name)
    loaded: dict[str, Any] = {}
    for mod in ("manifests", "streaming"):
        spec = importlib.util.spec_from_file_location(
            f"{base}.{mod}", pkg / f"{mod}.py"
        )
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[f"{base}.{mod}"] = module
        spec.loader.exec_module(module)
        loaded[mod] = module
    return loaded["streaming"]


def _load_real_segmenter() -> Any:
    """Load streaming.buffer_stream engine-free (no vllm imports)."""
    return _load_real_streaming().buffer_stream


def make_output(ids: list[int], text: str = "", stage_id: int = 0) -> Any:
    return SimpleNamespace(
        stage_id=stage_id,
        outputs=[SimpleNamespace(token_ids=list(ids), text=text)],
    )


class FakeSegmenter:
    """Contract double mirroring the real segmenter's shape: frames
    buffer and slice at the admitted cadence, one prompt per completed
    CHUNK plus a final marker, with the segmenter's own
    hold-until-park discipline (PORT-SESS-001)."""

    def __init__(self, chunk_samples: int = CHUNK) -> None:
        self.config: Any = None
        self._chunk_samples = chunk_samples

    def __call__(
        self,
        audio_stream: AsyncIterator[Any],
        input_stream: "asyncio.Queue[list[int]]",
        model_config: Any,
    ) -> AsyncIterator[dict[str, Any]]:
        self.config = model_config
        return self._gen(audio_stream, input_stream, model_config)

    async def _gen(
        self,
        audio_stream: AsyncIterator[Any],
        input_stream: "asyncio.Queue[list[int]]",
        model_config: Any,
    ) -> AsyncIterator[dict[str, Any]]:
        held = 0
        yielded = False
        async for frame in audio_stream:
            held += len(frame)
            while held >= self._chunk_samples:
                held -= self._chunk_samples
                if yielded:
                    await self._hold(input_stream)
                yield self._prompt(model_config, final=False)
                yielded = True
        if yielded:
            await self._hold(input_stream)
        yield self._prompt(model_config, final=True, residual=held)

    @staticmethod
    def _prompt(
        model_config: Any, *, final: bool, residual: int = 0
    ) -> dict[str, Any]:
        return {
            "prompt_index": getattr(
                model_config, "nemotron_prompt_index", 0
            ),
            "final": final,
            "residual": residual,
        }

    @staticmethod
    async def _hold(input_stream: "asyncio.Queue[list[int]]") -> None:
        while True:
            ids = await input_stream.get()
            if PARK in ids:
                return


class FakeEngine:
    """Consumes rendered prompts; emits scripted outputs per prompt."""

    def __init__(
        self, script: Any = None, stop_after: int | None = None
    ) -> None:
        self.prompts: list[Any] = []
        self.aborted: list[str] = []
        self.script = script
        self.stop_after = stop_after
        # Progress marker for clocks tied to engine output (F6/R4).
        self.started_output = False

    async def generate(
        self, prompts: AsyncIterator[Any], request_id: str
    ) -> AsyncIterator[Any]:
        index = 0
        async for prompt in prompts:
            self.prompts.append(prompt)
            if self.script is not None:
                outs = self.script(prompt, index)
            else:
                outs = [make_output([7, PARK], text=f" w{index}")]
            for out in outs:
                self.started_output = True
                yield out
            index += 1
            if self.stop_after is not None and index >= self.stop_after:
                return

    async def abort(self, request_id: str) -> None:
        self.aborted.append(request_id)


def make_view(**overrides: Any) -> SessionConfigView:
    values: dict[str, Any] = {
        "park_token_id": PARK,
        "nemotron_prompt_index": LOCALES["en-US"],
        "nemotron_chunk_samples": CHUNK,
    }
    values.update(overrides)
    return SessionConfigView(SimpleNamespace(base_marker="base"), **values)


def make_transcriber(
    engine: FakeEngine,
    segment: Any = None,
    view: SessionConfigView | None = None,
    rendered: list[Any] | None = None,
) -> tuple[EngineTranscriber, Any, SessionConfigView]:
    seg = segment if segment is not None else FakeSegmenter()
    the_view = view if view is not None else make_view()
    sink = rendered if rendered is not None else []

    async def render(prompt: Any) -> Any:
        sink.append(prompt)
        return {"rendered": prompt}

    transcriber = EngineTranscriber(
        segment=seg,
        render=render,
        generate=engine.generate,
        abort_request=engine.abort,
        config_view=the_view,
        locale_index=dict(LOCALES),
        request_id="rt-test-1",
    )
    return transcriber, seg, the_view


def chunk_audio(n_chunks: float = 1.0) -> Any:
    return np.zeros(int(n_chunks * CHUNK), dtype=np.float32)


# ---- the config view (ING-VEH-007) ---------------------------------------------


def test_config_view_delegates_and_overrides() -> None:
    base = SimpleNamespace(a=1, nemotron_prompt_index=3)
    view = SessionConfigView(base, park_token_id=PARK)
    assert view.a == 1
    assert view.park_token_id == PARK
    assert view.nemotron_prompt_index == 3  # delegated until overridden
    view.nemotron_prompt_index = 9
    assert view.nemotron_prompt_index == 9
    with pytest.raises(AttributeError):
        _ = view.missing
    assert getattr(view, "missing", "d") == "d"


def test_config_view_never_mutates_base() -> None:
    base = SimpleNamespace(nemotron_prompt_index=3)
    view = SessionConfigView(base, park_token_id=PARK)
    view.nemotron_prompt_index = 9
    view.new_field = "x"
    assert base.nemotron_prompt_index == 3
    assert not hasattr(base, "park_token_id")
    assert not hasattr(base, "new_field")


def test_park_id_is_required_at_construction() -> None:
    engine = FakeEngine()
    with pytest.raises(ValueError, match="park_token_id"):
        make_transcriber(
            engine,
            view=SessionConfigView(SimpleNamespace()),
        )


# ---- feed: whole pieces in, one hypothesis per completed CHUNK -----------------


async def test_feed_returns_one_cumulative_hypothesis_per_chunk() -> None:
    engine = FakeEngine()
    transcriber, _, _ = make_transcriber(engine)
    first = await asyncio.wait_for(transcriber.feed(chunk_audio()), WAIT_S)
    assert first == [" w0"]
    second = await asyncio.wait_for(transcriber.feed(chunk_audio()), WAIT_S)
    assert second == [" w0 w1"]


async def test_multi_chunk_piece_reaches_the_segmenter_whole() -> None:
    # ING-FE-006 / PORT-SESS-001: a burst is ONE frame to the
    # segmenter — both hypotheses come back from one feed call.
    engine = FakeEngine()
    transcriber, _, _ = make_transcriber(engine)
    hyps = await asyncio.wait_for(
        transcriber.feed(chunk_audio(2.5)), WAIT_S
    )
    assert hyps == [" w0", " w0 w1"]
    sub_chunk = await asyncio.wait_for(
        transcriber.feed(chunk_audio(0.25)), WAIT_S
    )
    assert sub_chunk == []  # no cadence completed, nothing to await


async def test_feed_includes_all_deltas_through_its_park() -> None:
    def script(prompt: Any, index: int) -> list[Any]:
        return [
            make_output([5], text=" early"),
            make_output([7, PARK], text=" parked"),
        ]

    engine = FakeEngine(script=script)
    transcriber, _, _ = make_transcriber(engine)
    hyps = await asyncio.wait_for(transcriber.feed(chunk_audio()), WAIT_S)
    assert hyps == [" early parked"]


async def test_feed_holds_until_the_park_and_abort_unblocks() -> None:
    def script(prompt: Any, index: int) -> list[Any]:
        return [make_output([5], text=" unparked")]

    engine = FakeEngine(script=script)
    transcriber, _, _ = make_transcriber(engine)
    task = asyncio.ensure_future(transcriber.feed(chunk_audio()))
    done, _ = await asyncio.wait([task], timeout=0.2)
    assert not done  # no park -> the feed must not resolve
    await transcriber.abort()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert engine.aborted == ["rt-test-1"]


async def test_engine_error_fails_the_feed_loudly() -> None:
    def script(prompt: Any, index: int) -> list[Any]:
        raise RuntimeError("engine fell over")

    engine = FakeEngine(script=script)
    transcriber, _, _ = make_transcriber(engine)
    with pytest.raises(RuntimeError, match="engine fell over"):
        await asyncio.wait_for(transcriber.feed(chunk_audio()), WAIT_S)


async def test_feed_after_generation_ended_raises_never_hangs() -> None:
    # The consumer ends after the first prompt (engine died mid
    # stream): the next feed must refuse loudly, never append a
    # waiter no consumer will resolve.
    engine = FakeEngine(stop_after=1)
    transcriber, _, _ = make_transcriber(engine)
    first = await asyncio.wait_for(transcriber.feed(chunk_audio()), WAIT_S)
    assert first == [" w0"]
    await asyncio.wait_for(transcriber._task, WAIT_S)  # type: ignore[arg-type]
    with pytest.raises(RuntimeError, match="ended before"):
        await asyncio.wait_for(transcriber.feed(chunk_audio()), WAIT_S)


async def test_flush_after_generation_ended_raises_never_hangs() -> None:
    engine = FakeEngine(stop_after=1)
    transcriber, _, _ = make_transcriber(engine)
    await asyncio.wait_for(transcriber.feed(chunk_audio()), WAIT_S)
    await asyncio.wait_for(transcriber._task, WAIT_S)  # type: ignore[arg-type]
    with pytest.raises(RuntimeError, match="ended before"):
        await asyncio.wait_for(transcriber.flush(), WAIT_S)


# ---- flush / finish / abort lifecycle (ING-LIFE-010) ---------------------------


async def test_flush_returns_the_final_transcript() -> None:
    def script(prompt: Any, index: int) -> list[Any]:
        if prompt["rendered"]["final"]:
            return [
                make_output([PARK], text=" fin"),
                make_output([9], text=" post"),
            ]
        return [make_output([7, PARK], text=f" w{index}")]

    engine = FakeEngine(script=script)
    transcriber, _, _ = make_transcriber(engine)
    await asyncio.wait_for(transcriber.feed(chunk_audio()), WAIT_S)
    assert await asyncio.wait_for(
        transcriber.feed(chunk_audio(0.25)), WAIT_S
    ) == []
    final = await asyncio.wait_for(transcriber.flush(), WAIT_S)
    # flush waits for STREAM END, not just the park: trailing output
    # after the final park still lands in the final transcript.
    assert final == " w0 fin post"
    assert len(engine.prompts) == 2  # one chunk + the final marker
    assert engine.prompts[-1]["rendered"]["residual"] == CHUNK // 4


async def test_feed_after_flush_raises() -> None:
    engine = FakeEngine()
    transcriber, _, _ = make_transcriber(engine)
    await asyncio.wait_for(transcriber.feed(chunk_audio()), WAIT_S)
    await asyncio.wait_for(transcriber.flush(), WAIT_S)
    with pytest.raises(RuntimeError, match="closed"):
        await transcriber.feed(chunk_audio())


async def test_finish_after_flush_is_a_noop() -> None:
    engine = FakeEngine()
    transcriber, _, _ = make_transcriber(engine)
    await asyncio.wait_for(transcriber.feed(chunk_audio()), WAIT_S)
    await asyncio.wait_for(transcriber.flush(), WAIT_S)
    await asyncio.wait_for(transcriber.finish(), WAIT_S)
    await asyncio.wait_for(transcriber.finish(), WAIT_S)  # idempotent
    assert engine.aborted == []


async def test_finish_without_flush_closes_gracefully() -> None:
    # Defense in depth: should a caller ever skip flush, the open
    # generation still ends via the explicit final-tail transaction
    # (PORT-SESS-003), never an abort.
    engine = FakeEngine()
    transcriber, _, _ = make_transcriber(engine)
    await asyncio.wait_for(transcriber.feed(chunk_audio()), WAIT_S)
    await asyncio.wait_for(transcriber.finish(), WAIT_S)
    assert engine.aborted == []
    assert len(engine.prompts) == 2  # chunk + final marker
    assert engine.prompts[-1]["rendered"]["final"] is True


async def test_finish_before_any_audio_is_a_noop() -> None:
    engine = FakeEngine()
    transcriber, _, _ = make_transcriber(engine)
    await asyncio.wait_for(transcriber.finish(), WAIT_S)
    assert engine.prompts == []
    assert engine.aborted == []


async def test_abort_before_start_never_touches_the_engine() -> None:
    engine = FakeEngine()
    transcriber, _, _ = make_transcriber(engine)
    await transcriber.abort()
    await transcriber.abort()  # idempotent
    assert engine.aborted == []
    assert engine.prompts == []


# ---- the seam contract (ING-VEH-007/008) ---------------------------------------


async def test_view_goes_to_the_segmenter_only() -> None:
    engine = FakeEngine()
    rendered: list[Any] = []
    view = make_view()
    transcriber, seg, _ = make_transcriber(
        engine, view=view, rendered=rendered
    )
    await asyncio.wait_for(transcriber.feed(chunk_audio()), WAIT_S)
    assert seg.config is view  # the segmenter sees exactly the view
    # render sees minted prompts, never the view (ING-VEH-008)
    assert rendered and all(p is not view for p in rendered)
    assert engine.prompts[0] == {"rendered": rendered[0]}


async def test_update_locale_writes_the_view() -> None:
    engine = FakeEngine()
    transcriber, _, view = make_transcriber(engine)
    await transcriber.update_locale("es-US")
    assert view.nemotron_prompt_index == LOCALES["es-US"]
    with pytest.raises(ValueError, match="fr-FR"):
        await transcriber.update_locale("fr-FR")


# ---- integration: the REAL segmenter over a fake engine ------------------------


async def test_real_segmenter_paces_and_stamps_locale_per_mint() -> None:
    buffer_stream = _load_real_segmenter()
    engine = FakeEngine()
    transcriber, _, view = make_transcriber(engine, segment=buffer_stream)
    first = await asyncio.wait_for(transcriber.feed(chunk_audio()), WAIT_S)
    assert first == [" w0"]
    envelope = engine.prompts[0]["rendered"]["multi_modal_data"]["audio"]
    assert envelope[4] == float(LOCALES["en-US"])

    await transcriber.update_locale("es-US")
    second = await asyncio.wait_for(
        transcriber.feed(chunk_audio()), WAIT_S
    )
    assert second == [" w0 w1"]
    envelope = engine.prompts[1]["rendered"]["multi_modal_data"]["audio"]
    assert envelope[4] == float(LOCALES["es-US"])  # next-mint stamp

    assert await asyncio.wait_for(
        transcriber.feed(np.zeros(1279, dtype=np.float32)), WAIT_S
    ) == []
    final = await asyncio.wait_for(transcriber.flush(), WAIT_S)
    assert final == " w0 w1 w2"
    tail = engine.prompts[2]["rendered"]["multi_modal_data"]["audio"]
    assert tail[1] == 1279.0  # the residual rides the final tail as-is
    assert tail[3] == 1.0
    assert tail[4] == float(LOCALES["es-US"])
    assert len(engine.prompts) == 3


async def test_real_segmenter_burst_is_stamped_as_one_frame() -> None:
    # The F6 pin: a burst spanning two cadences reaches buffer_stream
    # as ONE frame, so BOTH chunks are sliced and stamped in one
    # synchronous pass before the first park is awaited. The clock is
    # tied to engine progress — it jumps to LATE once the engine first
    # yields — so a pre-chunking path that stamped chunk 2 only after
    # chunk 1's park (behind the first engine output) would read LATE
    # in the second envelope. Both admission stamps reading BASE pins
    # the pre-park stamping property itself (PORT-SESS-001), not just
    # the (0, 1) sequence numbers the old path also produced.
    streaming = _load_real_streaming()
    engine = FakeEngine()
    base_stamp, late_stamp = 111, 999_777

    def engine_tied_clock() -> int:
        return late_stamp if engine.started_output else base_stamp

    real_clock = streaming._admission_ms_mod
    streaming._admission_ms_mod = engine_tied_clock
    try:
        transcriber, _, _ = make_transcriber(
            engine, segment=streaming.buffer_stream
        )
        hyps = await asyncio.wait_for(
            transcriber.feed(chunk_audio(2.0)), WAIT_S
        )
        assert hyps == [" w0", " w0 w1"]
        first = engine.prompts[0]["rendered"]["multi_modal_data"]["audio"]
        second = engine.prompts[1]["rendered"]["multi_modal_data"]["audio"]
        # BOTH admission stamps pre-date the first engine output.
        assert (first[6], second[6]) == (
            float(base_stamp),
            float(base_stamp),
        )
        assert (first[5], second[5]) == (0.0, 1.0)  # chunk sequence
        final = await asyncio.wait_for(transcriber.flush(), WAIT_S)
        assert final == " w0 w1 w2"
        tail = engine.prompts[2]["rendered"]["multi_modal_data"]["audio"]
        assert tail[1] == 0.0  # exact boundary -> zero-sample final tail
        assert tail[3] == 1.0
        assert tail[6] == float(late_stamp)  # stamped after engine output
    finally:
        streaming._admission_ms_mod = real_clock


async def test_real_segmenter_zero_audio_session() -> None:
    # ING-LIFE-003 through the engine seam: flush with nothing fed
    # still runs the one zero-sample final-tail transaction.
    buffer_stream = _load_real_segmenter()
    engine = FakeEngine()
    transcriber, _, _ = make_transcriber(engine, segment=buffer_stream)
    final = await asyncio.wait_for(transcriber.flush(), WAIT_S)
    assert final == " w0"
    assert len(engine.prompts) == 1
    tail = engine.prompts[0]["rendered"]["multi_modal_data"]["audio"]
    assert tail[1] == 0.0
    assert tail[3] == 1.0
