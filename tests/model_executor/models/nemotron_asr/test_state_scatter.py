# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Contract tests for the PORT-owned masked page scatter."""

from __future__ import annotations

import importlib.util
from collections.abc import Callable, Mapping
from pathlib import Path

import pytest
import torch

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

_MODULE = Path(__file__).resolve().parents[4] / "vllm_omni/model_executor/models/nemotron_asr/state_scatter.py"
_SPEC = importlib.util.spec_from_file_location("nemotron_state_scatter", _MODULE)
assert _SPEC is not None and _SPEC.loader is not None
state_scatter = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(state_scatter)


# @spec PORT-ADV-004, PORT-STATE-008
@pytest.mark.parametrize("dtype", [torch.float32, torch.int32, torch.int64])
def test_masked_page_scatter_writes_only_clean_rows(dtype: torch.dtype) -> None:
    pool = torch.arange(6 * 2 * 3).reshape(6, 2, 3).to(dtype)
    before = pool.clone()
    scratch = torch.full((3, 2, 3), 77, dtype=dtype)
    blocks = torch.tensor([4, 2, 5], dtype=torch.long)
    status = torch.tensor([0, 16, 0], dtype=torch.int32)

    state_scatter.masked_page_scatter_(pool, scratch, blocks, status)

    torch.testing.assert_close(pool[4], scratch[0], rtol=0, atol=0)
    torch.testing.assert_close(pool[5], scratch[2], rtol=0, atol=0)
    torch.testing.assert_close(pool[2], before[2], rtol=0, atol=0)
    # The reserved null page and every unrelated page are never targets.
    for block in (0, 1, 3):
        torch.testing.assert_close(pool[block], before[block], rtol=0, atol=0)


# @spec PORT-ADV-004
def test_masked_page_scatter_all_failed_is_exact_noop() -> None:
    pool = torch.randn(5, 7)
    before = pool.clone()
    scratch = torch.randn(3, 7)
    blocks = torch.tensor([4, 1, 3], dtype=torch.long)
    status = torch.tensor([1, 2, 4], dtype=torch.int32)

    state_scatter.masked_page_scatter_(pool, scratch, blocks, status)

    torch.testing.assert_close(pool, before, rtol=0, atol=0)


# @spec PORT-ADV-004, PORT-STATE-008
def test_dirty_rows_may_name_null_without_writing_it() -> None:
    pool = torch.randn(5, 4)
    before = pool.clone()
    scratch = torch.full((3, 4), float("nan"))
    blocks = torch.zeros(3, dtype=torch.long)
    status = torch.ones(3, dtype=torch.int32)

    state_scatter.masked_page_scatter_(pool, scratch, blocks, status)

    torch.testing.assert_close(pool, before, rtol=0, atol=0)


# @spec PORT-STATE-008
def test_masked_page_scatter_supports_padded_block_stride() -> None:
    backing = torch.arange(5 * 11, dtype=torch.float32)
    pool = torch.as_strided(backing, (5, 2, 3), (11, 3, 1))
    before = pool.clone()
    scratch = torch.full((2, 2, 3), 31.0)
    blocks = torch.tensor([3, 1], dtype=torch.long)
    status = torch.tensor([0, 8], dtype=torch.int32)

    state_scatter.masked_page_scatter_(pool, scratch, blocks, status)

    torch.testing.assert_close(pool[3], scratch[0], rtol=0, atol=0)
    torch.testing.assert_close(pool[1], before[1], rtol=0, atol=0)
    torch.testing.assert_close(pool[0], before[0], rtol=0, atol=0)


# @spec PORT-STATE-007, PORT-STATE-008
@pytest.mark.parametrize(
    ("change", "message"),
    [
        (lambda p, s, b, r: (p, s[:1], b, r), "row count"),
        (lambda p, s, b, r: (p, s.double(), b, r), "dtype"),
        (lambda p, s, b, r: (p, s[:, :1], b, r), "page shape"),
        (lambda p, s, b, r: (p, s, b.to(torch.int32), r), "int64"),
        (lambda p, s, b, r: (p, s, b, r.to(torch.int64)), "int32"),
        (
            lambda p, s, b, r: (
                p.transpose(1, 2).contiguous().transpose(1, 2),
                s,
                b,
                r,
            ),
            "row-major",
        ),
    ],
)
def test_masked_page_scatter_rejects_bad_descriptor(
    change: Callable[..., tuple[torch.Tensor, ...]], message: str
) -> None:
    pool = torch.zeros(5, 2, 3)
    scratch = torch.ones(2, 2, 3)
    blocks = torch.tensor([1, 3], dtype=torch.long)
    status = torch.zeros(2, dtype=torch.int32)
    before = pool.clone()
    args = change(pool, scratch, blocks, status)
    with pytest.raises(ValueError, match=message):
        state_scatter.masked_page_scatter_(*args)

    torch.testing.assert_close(pool, before, rtol=0, atol=0)


# @spec PORT-STATE-008
def test_grouped_disjoint_padded_slab_and_overlapping_fallback() -> None:
    slab = torch.zeros(5 * 16)
    pools = [torch.as_strided(slab, (5, 4), (16, 1), i * 4) for i in range(3)]
    scratch = [torch.full((2, 4), float(i + 1)) for i in range(3)]
    assert state_scatter._group_layout_supported(pools, scratch)
    assert not state_scatter._group_layout_supported([pools[0], pools[0]], scratch[:2])
    # Different descriptors may legally share source/destination storage; the
    # original ordered executor must own that dependency rather than race it.
    alias_sources = [scratch[0], pools[0][:2], scratch[2]]
    assert not state_scatter._group_layout_supported(pools, alias_sources)


# @spec PORT-ADV-004, PORT-STATE-008
@pytest.mark.parametrize("dtype", [torch.float32, torch.int32])
def test_grouped_cpu_fallback_rebinding_and_invalid_failed_ids(dtype: torch.dtype) -> None:
    blocks = torch.tensor([3, -999, 10000])
    status = torch.tensor([0, 1, 8], dtype=torch.int32)
    for fill in (5, 19):
        pools = [torch.zeros(5, 4, dtype=dtype) for _ in range(3)]
        sources = [torch.full((3, 4), fill + i, dtype=dtype) for i in range(3)]
        assert state_scatter.prepare_grouped_page_scatter(pools, sources, blocks, status) is None
        for pool, scratch in zip(pools, sources, strict=True):
            state_scatter.masked_page_scatter_(pool, scratch, blocks, status)
            torch.testing.assert_close(pool[3], scratch[0], rtol=0, atol=0)
            assert torch.count_nonzero(pool[[0, 1, 2, 4]]) == 0


# @spec PORT-STATE-008
def test_grouped_layout_rejects_nonhomogeneous_metadata() -> None:
    pools = [torch.zeros(5, 4), torch.zeros(5, 8)]
    sources = [torch.ones(2, 4), torch.ones(2, 8)]
    assert not state_scatter._group_layout_supported(pools, sources)


# @spec PORT-STATE-008
@pytest.mark.parametrize("offset", range(32))
def test_grouped_page_interval_proof_matches_explicit_bytes(offset: int) -> None:
    slab = torch.zeros(128)
    left = torch.as_strided(slab, (4, 3), (16, 1), 4)
    right = torch.as_strided(slab, (3, 5), (16, 1), offset)
    a = {4 + row * 16 + col for row in range(4) for col in range(3)}
    b = {offset + row * 16 + col for row in range(3) for col in range(5)}
    assert state_scatter._page_intervals_overlap(left, right) == bool(a & b)
    assert state_scatter._page_intervals_overlap(right, left) == bool(a & b)


# @spec PORT-ADV-004, PORT-STATE-008, PORT-PERF-001
@pytest.mark.skipif(not torch.cuda.is_available(), reason="grouped kernel requires real SM80 qualification")
@pytest.mark.parametrize("dtype", [torch.float32, torch.int32])
@pytest.mark.parametrize("page_size", [1, 6, 3073])
def test_grouped_cuda_matches_ordered_scalar_and_warms_rebound_pointers(
    dtype: torch.dtype, page_size: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Real package import on CUDA: no local module/package substitution.
    import importlib

    mod = importlib.import_module("vllm_omni.model_executor.models.nemotron_asr.state_scatter")
    if torch.cuda.get_device_capability() != (8, 0):
        pytest.skip("grouped candidate qualification is bounded to SM80")
    layers, rows, slots = 24, 3, 5
    stride = layers * page_size + 7

    def compiled_keys() -> frozenset[object]:
        # Inspect the actual installed Triton cache, outside kernel dispatch.
        # An unsupported runtime API fails qualification instead of silently
        # weakening the no-JIT check to the Python warmed-signature guard.
        caches = getattr(mod._grouped_page_scatter_kernel, "device_caches", None)
        assert isinstance(caches, Mapping), "unsupported Triton device-cache API"
        entry = caches.get(torch.accelerator.current_device_index())
        assert isinstance(entry, (tuple, list)) and entry, "missing Triton device cache"
        binaries = entry[0]
        assert isinstance(binaries, Mapping) and binaries, "no warmed grouped binaries"
        return frozenset(binaries)

    warmed_keys: frozenset[object] | None = None
    for generation in range(2):
        slab = torch.full((slots * stride,), -17, dtype=dtype, device="cuda")
        reference = slab.clone()
        pools = [torch.as_strided(slab, (slots, page_size), (stride, 1), i * page_size) for i in range(layers)]
        refs = [torch.as_strided(reference, p.shape, p.stride(), p.storage_offset()) for p in pools]
        warm = [torch.empty((1, page_size), dtype=dtype, device="cuda") for _ in pools]
        for pool, source in zip(pools, warm, strict=True):
            mod.warmup_masked_page_scatter(pool, source)
        mod.warmup_grouped_page_scatter(pools, warm)
        keys = compiled_keys()
        if warmed_keys is None:
            warmed_keys = keys
        else:
            assert keys == warmed_keys, "new grouped binary after pool/scratch rebinding"
        torch.testing.assert_close(slab, reference, rtol=0, atol=0)
        # Rebind every dynamic source; offset 1 changes pointer alignment,
        # including block/status vectors, without changing layout identity.
        sources = [
            torch.arange(rows * page_size + 1, dtype=dtype, device="cuda")[1:].view(rows, page_size) + generation
            for _ in pools
        ]
        # Preserve the deliberately unaligned data_ptr after generating values.
        sources = [torch.cat((s.new_zeros(1), s.flatten()))[1:].view_as(s) for s in sources]
        blocks = torch.tensor([999, 3, -999, 1], dtype=torch.int64, device="cuda")[1:]
        status = torch.tensor([999, 0, 4, 0], dtype=torch.int32, device="cuda")[1:]
        group = mod.prepare_grouped_page_scatter(pools, sources, blocks, status)
        assert group is not None and group.count == layers
        assert group.pools[0] is pools[0] and group.scratches[0] is sources[0]

        # No Torch tensor construction or registration/preparation in commit.
        def forbidden(*args: object, **kwargs: object) -> None:
            raise AssertionError("allocation or preparation entered commit")

        with monkeypatch.context() as commit:
            for name in ("empty", "empty_like", "zeros", "zeros_like", "ones", "ones_like", "tensor"):
                commit.setattr(torch, name, forbidden)
            commit.setattr(mod, "_ensure_grouped_op_registered", forbidden)
            commit.setattr(mod, "prepare_grouped_page_scatter", forbidden)
            mod._execute_grouped_page_scatter_(group)
        assert compiled_keys() == warmed_keys, "grouped JIT after shifted pointer alignment"
        for pool, source in zip(refs, sources, strict=True):
            mod._execute_masked_page_scatter_(pool, source, blocks, status)
        torch.accelerator.synchronize()
        torch.testing.assert_close(slab, reference, rtol=0, atol=0)
        # All failed rows may carry entirely unusable resident block ids.
        before = slab.clone()
        status.fill_(1)
        blocks.fill_(-10000)
        mod._execute_grouped_page_scatter_(group)
        torch.accelerator.synchronize()
        torch.testing.assert_close(slab, before, rtol=0, atol=0)
        assert compiled_keys() == warmed_keys, "grouped JIT during failed-row execution"


# @spec PORT-STATE-008
@pytest.mark.skipif(not torch.cuda.is_available(), reason="grouped warm guard requires real SM80")
def test_grouped_cuda_unwarmed_selection_fails_before_store(monkeypatch: pytest.MonkeyPatch) -> None:
    import importlib

    mod = importlib.import_module("vllm_omni.model_executor.models.nemotron_asr.state_scatter")
    if torch.cuda.get_device_capability() != (8, 0):
        pytest.skip("grouped candidate qualification is bounded to SM80")
    monkeypatch.setattr(mod, "_GROUPED_WARMED_SPECIALIZATIONS", set())
    pools = [torch.zeros(5, 4, device="cuda") for _ in range(2)]
    sources = [torch.ones(2, 4, device="cuda") for _ in pools]
    blocks = torch.tensor([1, 3], device="cuda")
    status = torch.zeros(2, dtype=torch.int32, device="cuda")
    with pytest.raises(ValueError, match="not warmed"):
        mod.prepare_grouped_page_scatter(pools, sources, blocks, status)
    assert all(torch.count_nonzero(p) == 0 for p in pools)
