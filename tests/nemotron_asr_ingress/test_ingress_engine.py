"""The in-process EngineTranscriber and its per-session config view.

ING-VEH-004/007/008 + ING-LIFE-010 at the compute seam: the engine
arrives as injected callables (GPU-free), and the integration cases
drive the REAL segmenter (``streaming.buffer_stream``, loaded
engine-free via importlib) against a fake engine to prove the
park-echo pacing loop and the per-mint locale stamp end to end.
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
LOCALES = {"en-US": 2, "es-US": 7}
WAIT_S = 2.0


def _load_real_segmenter() -> Any:
    """Load streaming.buffer_stream engine-free (no vllm imports)."""
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
    return loaded["streaming"].buffer_stream


def make_output(ids: list[int], text: str = "", stage_id: int = 0) -> Any:
    return SimpleNamespace(
        stage_id=stage_id,
        outputs=[SimpleNamespace(token_ids=list(ids), text=text)],
    )


class FakeSegmenter:
    """Contract double: one prompt per frame + a final marker, with the
    segmenter's own hold-until-park discipline (PORT-SESS-001)."""

    def __init__(self) -> None:
        self.config: Any = None

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
        yielded = False
        async for frame in audio_stream:
            if yielded:
                await self._hold(input_stream)
            yield {
                "chunk": frame,
                "prompt_index": getattr(
                    model_config, "nemotron_prompt_index", 0
                ),
                "final": False,
            }
            yielded = True
        if yielded:
            await self._hold(input_stream)
        yield {
            "chunk": None,
            "prompt_index": getattr(
                model_config, "nemotron_prompt_index", 0
            ),
            "final": True,
        }

    @staticmethod
    async def _hold(input_stream: "asyncio.Queue[list[int]]") -> None:
        while True:
            ids = await input_stream.get()
            if PARK in ids:
                return


class FakeEngine:
    """Consumes rendered prompts; emits scripted outputs per prompt."""

    def __init__(self, script: Any = None) -> None:
        self.prompts: list[Any] = []
        self.aborted: list[str] = []
        self.script = script

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
                yield out
            index += 1

    async def abort(self, request_id: str) -> None:
        self.aborted.append(request_id)


def make_view(**overrides: Any) -> SessionConfigView:
    values: dict[str, Any] = {
        "park_token_id": PARK,
        "nemotron_prompt_index": LOCALES["en-US"],
        "nemotron_chunk_samples": 8960,
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


# ---- step: cumulative hypothesis, one park per chunk ---------------------------


async def test_step_returns_cumulative_hypothesis_per_chunk() -> None:
    engine = FakeEngine()
    transcriber, _, _ = make_transcriber(engine)
    first = await asyncio.wait_for(
        transcriber.step(np.zeros(8960, dtype=np.float32)), WAIT_S
    )
    assert first == " w0"
    second = await asyncio.wait_for(
        transcriber.step(np.zeros(8960, dtype=np.float32)), WAIT_S
    )
    assert second == " w0 w1"


async def test_step_includes_all_deltas_through_its_park() -> None:
    def script(prompt: Any, index: int) -> list[Any]:
        return [
            make_output([5], text=" early"),
            make_output([7, PARK], text=" parked"),
        ]

    engine = FakeEngine(script=script)
    transcriber, _, _ = make_transcriber(engine)
    text = await asyncio.wait_for(
        transcriber.step(np.zeros(8960, dtype=np.float32)), WAIT_S
    )
    assert text == " early parked"


async def test_step_holds_until_the_park_and_abort_unblocks() -> None:
    def script(prompt: Any, index: int) -> list[Any]:
        return [make_output([5], text=" unparked")]

    engine = FakeEngine(script=script)
    transcriber, _, _ = make_transcriber(engine)
    task = asyncio.ensure_future(
        transcriber.step(np.zeros(8960, dtype=np.float32))
    )
    done, _ = await asyncio.wait([task], timeout=0.2)
    assert not done  # no park -> the step must not resolve
    await transcriber.abort()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert engine.aborted == ["rt-test-1"]


async def test_engine_error_fails_the_step_loudly() -> None:
    def script(prompt: Any, index: int) -> list[Any]:
        raise RuntimeError("engine fell over")

    engine = FakeEngine(script=script)
    transcriber, _, _ = make_transcriber(engine)
    with pytest.raises(RuntimeError, match="engine fell over"):
        await asyncio.wait_for(
            transcriber.step(np.zeros(8960, dtype=np.float32)), WAIT_S
        )


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
    await asyncio.wait_for(
        transcriber.step(np.zeros(8960, dtype=np.float32)), WAIT_S
    )
    final = await asyncio.wait_for(
        transcriber.flush(np.zeros(1279, dtype=np.float32)), WAIT_S
    )
    # flush waits for STREAM END, not just the park: trailing output
    # after the final park still lands in the final transcript.
    assert final == " w0 w1 fin post"
    assert len(engine.prompts) == 3  # chunk, residual, final marker


async def test_finish_after_flush_is_a_noop() -> None:
    engine = FakeEngine()
    transcriber, _, _ = make_transcriber(engine)
    await asyncio.wait_for(
        transcriber.step(np.zeros(8960, dtype=np.float32)), WAIT_S
    )
    await asyncio.wait_for(
        transcriber.flush(np.zeros(1279, dtype=np.float32)), WAIT_S
    )
    await asyncio.wait_for(transcriber.finish(), WAIT_S)
    await asyncio.wait_for(transcriber.finish(), WAIT_S)  # idempotent
    assert engine.aborted == []


async def test_finish_without_flush_closes_gracefully() -> None:
    # The ING-LIFE-003 skip-flush path: finalize with an empty residual
    # calls finish() alone. The open generation must end via the
    # explicit final-tail transaction (PORT-SESS-003), NOT an abort.
    engine = FakeEngine()
    transcriber, _, _ = make_transcriber(engine)
    await asyncio.wait_for(
        transcriber.step(np.zeros(8960, dtype=np.float32)), WAIT_S
    )
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
    await asyncio.wait_for(
        transcriber.step(np.zeros(8960, dtype=np.float32)), WAIT_S
    )
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
    first = await asyncio.wait_for(
        transcriber.step(np.zeros(8960, dtype=np.float32)), WAIT_S
    )
    assert first == " w0"
    envelope = engine.prompts[0]["rendered"]["multi_modal_data"]["audio"]
    assert envelope[4] == float(LOCALES["en-US"])

    await transcriber.update_locale("es-US")
    second = await asyncio.wait_for(
        transcriber.step(np.zeros(8960, dtype=np.float32)), WAIT_S
    )
    assert second == " w0 w1"
    envelope = engine.prompts[1]["rendered"]["multi_modal_data"]["audio"]
    assert envelope[4] == float(LOCALES["es-US"])  # next-mint stamp

    final = await asyncio.wait_for(
        transcriber.flush(np.zeros(1279, dtype=np.float32)), WAIT_S
    )
    assert final == " w0 w1 w2"
    tail = engine.prompts[2]["rendered"]["multi_modal_data"]["audio"]
    assert tail[1] == 1279.0  # the residual rides the final tail as-is
    assert tail[3] == 1.0
    assert tail[4] == float(LOCALES["es-US"])
    assert len(engine.prompts) == 3


async def test_real_segmenter_residual_only_session() -> None:
    buffer_stream = _load_real_segmenter()
    engine = FakeEngine()
    transcriber, _, _ = make_transcriber(engine, segment=buffer_stream)
    final = await asyncio.wait_for(
        transcriber.flush(np.zeros(1279, dtype=np.float32)), WAIT_S
    )
    assert final == " w0"
    assert len(engine.prompts) == 1  # one final-tail transaction only
    tail = engine.prompts[0]["rendered"]["multi_modal_data"]["audio"]
    assert tail[1] == 1279.0
    assert tail[3] == 1.0
