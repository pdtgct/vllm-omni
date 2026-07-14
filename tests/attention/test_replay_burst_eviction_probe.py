# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""EVIDENCE PROBE (OPEN-α3-VEHICLE): replay bursts vs the α1 window.

Drives core's REAL sliding-window block bookkeeping with the α1
pooled spec and an interleaved chunk/label scheduler-token stream,
and measures the defect the α3 pre-flight consult identified from
source: `SlidingWindowManager` frees blocks POSITIONALLY in scheduler
tokens, so a burst of emitted-label tokens pushes acoustically-live
audio blocks out of the pooled window — read-after-free of the
encoder left context. The probe emits a JSON trace per chunk geometry
(minimum burst length that evicts a live block, which blocks, the
window-widening cost table) to
``/tmp/replay_burst_eviction_trace.json`` for the decision report.

Under the α1 vehicle this probe's assertion FAILS BY CONSTRUCTION
(that is the evidence); under the recommended state-page vehicle the
scenario cannot arise (labels cost zero KV). The probe is therefore
written as an xfail-style measurement, not a gate: it records, prints,
and asserts only its own bookkeeping consistency.
"""

import json
from math import ceil

import pytest
import torch
from vllm.v1.kv_cache_interface import SlidingWindowSpec

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

WINDOW_FRAMES = 56
HEADS = 8
HEAD_SIZE = 128
#: The five chunk geometries (encoder frames per chunk).
GEOMETRIES = (1, 2, 4, 7, 14)
#: NeMo's per-step symbol budget (labels a chunk may emit).
MAX_SYMBOLS_PER_STEP = 10


def cdiv(a: int, b: int) -> int:
    return -(-a // b)


def pooled_window(bps: int) -> int:
    return cdiv(WINDOW_FRAMES, bps) + 1


def make_spec(bps: int) -> SlidingWindowSpec:
    return SlidingWindowSpec(
        block_size=1,  # post-alignment for bps >= 4; conservative
        num_kv_heads=HEADS * bps,
        head_size=HEAD_SIZE,
        dtype=torch.float32,
        sliding_window=pooled_window(bps),
    )


def make_manager(spec: SlidingWindowSpec, num_blocks: int = 4096):
    """Core's real sliding-window manager over a real block pool.

    Constructor surfaces vary slightly across pins; adapt minimally —
    the semantics under test are ``get_num_skipped_tokens`` /
    ``remove_skipped_blocks``.
    """
    from vllm.v1.core.block_pool import BlockPool
    from vllm.v1.core.single_type_kv_cache_manager import (
        SlidingWindowManager,
    )

    # PIN v0.24.0 drift: BlockPool now requires ``hash_block_size``; it is
    # inert here because caching is disabled (any positive int is fine).
    try:
        pool = BlockPool(
            num_gpu_blocks=num_blocks,
            enable_caching=False,
            hash_block_size=spec.block_size,
        )
    except TypeError:
        pool = BlockPool(num_gpu_blocks=num_blocks, enable_caching=False)
    # PIN v0.24.0 drift: SingleTypeKVCacheManager.__init__ now requires
    # ``enable_caching`` and ``scheduler_block_size`` (LCM of group block
    # sizes; a multiple of this manager's block_size — here block_size=1).
    try:
        manager = SlidingWindowManager(
            kv_cache_spec=spec,
            block_pool=pool,
            enable_caching=False,
            kv_cache_group_id=0,
            scheduler_block_size=spec.block_size,
        )
    except TypeError:
        try:
            manager = SlidingWindowManager(
                kv_cache_spec=spec, block_pool=pool, kv_cache_group_id=0
            )
        except TypeError:
            manager = SlidingWindowManager(spec, pool, 0)
    return manager, pool


def run_geometry(bps: int) -> dict:
    """One geometry's measurement.

    Timeline: C chunk tokens interleaved with label bursts. After the
    final chunk (token index T-1), a burst of k label tokens advances
    ``num_computed_tokens`` to T + k. Acoustically live = the last
    ``cdiv(56, bps)`` chunk tokens (their frames are inside the
    56-frame left context of the NEXT chunk). The probe grows k and
    records the first k at which the manager frees a live chunk
    token's block.
    """
    from vllm.v1.request import Request

    spec = make_spec(bps)
    manager, pool = make_manager(spec)
    live_chunks = cdiv(WINDOW_FRAMES, bps)
    window = spec.sliding_window

    # Build a request with enough chunk history to make eviction
    # possible, then burst labels.
    n_chunks = live_chunks + 4
    trace: dict = {
        "bps": bps,
        "pooled_window_tokens": window,
        "live_chunk_tokens": live_chunks,
        "max_symbols_burst": MAX_SYMBOLS_PER_STEP * bps,
        "first_evicting_burst": None,
        "evicted_live_blocks": [],
        "widened_window_tokens_needed": None,
        "widened_page_factor": None,
    }

    def fresh_request_and_blocks(total_tokens: int):
        # PIN v0.24.0 drift: Request requires exactly one of sampling/
        # pooling params set (ValueError if both unset), and no longer
        # accepts ``eos_token_id``. This is a pooling (encoder) scenario.
        # The Request is inert to the measurement — only its request_id
        # is consumed; the manager is driven directly through
        # ``req_to_blocks`` + ``remove_skipped_blocks``.
        try:
            from vllm.pooling_params import PoolingParams

            req = Request(
                request_id=f"probe-{bps}",
                prompt_token_ids=list(range(total_tokens)),
                sampling_params=None,
                pooling_params=PoolingParams(),
            )
        except TypeError:
            try:
                req = Request(
                    request_id=f"probe-{bps}",
                    prompt_token_ids=list(range(total_tokens)),
                    sampling_params=None,
                    pooling_params=None,
                    eos_token_id=None,
                    arrival_time=0.0,
                )
            except TypeError:
                req = Request(
                    request_id=f"probe-{bps}",
                    prompt_token_ids=list(range(total_tokens)),
                    sampling_params=None,
                    eos_token_id=None,
                    arrival_time=0.0,
                )
        blocks = pool.get_new_blocks(total_tokens)  # block_size=1
        return req, blocks

    # Chunk token positions: 0..n_chunks-1 are chunks (no labels yet);
    # then a burst of k labels occupies positions n_chunks..n_chunks+k-1.
    for k in range(1, MAX_SYMBOLS_PER_STEP * bps + 1):
        total = n_chunks + k
        req, blocks = fresh_request_and_blocks(total)
        try:
            manager.req_to_blocks[req.request_id] = list(blocks)
        except Exception:
            manager.req_to_blocks[getattr(req, "request_id", "probe")] = list(
                blocks
            )
        before = [b.block_id for b in blocks]
        manager.remove_skipped_blocks(req.request_id, total)
        after_blocks = manager.req_to_blocks[req.request_id]
        freed_positions = [
            i
            for i, b in enumerate(after_blocks)
            if b is None or getattr(b, "block_id", None) != before[i]
        ]
        # Live chunk token positions: the last `live_chunks` chunk
        # tokens, i.e. positions n_chunks-live_chunks .. n_chunks-1.
        live_positions = set(range(n_chunks - live_chunks, n_chunks))
        evicted_live = sorted(live_positions.intersection(freed_positions))
        # Release everything for the next iteration.
        for b in blocks:
            if b is not None:
                try:
                    pool.free_blocks([b])
                except Exception:
                    pass
        if evicted_live and trace["first_evicting_burst"] is None:
            trace["first_evicting_burst"] = k
            trace["evicted_live_blocks"] = evicted_live
            break

    # The widening cost: window tokens needed to survive the worst
    # case burst, and the page-memory factor vs the α1 window.
    worst = MAX_SYMBOLS_PER_STEP * bps
    widened = cdiv(WINDOW_FRAMES, bps) * (worst + 2)
    trace["widened_window_tokens_needed"] = widened
    trace["widened_page_factor"] = round(widened / window, 1)
    return trace


def test_replay_burst_eviction_probe():
    traces = []
    for bps in GEOMETRIES:
        traces.append(run_geometry(bps))
    payload = {
        "window_frames": WINDOW_FRAMES,
        "heads": HEADS,
        "head_size": HEAD_SIZE,
        "max_symbols_per_step": MAX_SYMBOLS_PER_STEP,
        "geometries": traces,
    }
    with open("/tmp/replay_burst_eviction_trace.json", "w") as f:
        json.dump(payload, f, indent=2)
    print(json.dumps(payload, indent=2))
    # The probe's own gate is bookkeeping consistency, not the verdict:
    # every geometry produced a measurement.
    assert len(traces) == len(GEOMETRIES)
    # The MEASUREMENT (recorded, not asserted): under α1-as-is, every
    # geometry is expected to show a NeMo-legal burst length that
    # evicts a live block. The decision report consumes the JSON.
