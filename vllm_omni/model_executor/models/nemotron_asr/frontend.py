# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The exact bounded incremental frontend (design §Exact Bounded Frontend).

An incremental implementation of the checkpoint's whole-utterance
pre-emphasis and centered STFT (``MelFeaturizer``): it retains only a
bounded raw-sample tail and the committed-boundary mel tail, and on
each CHUNK commits exactly the mel frames whose effective STFT window
is stable given the samples received — proven equal to the
whole-prefix featurizer by differential tests (``test_frontend.py``,
local: this module imports only ``torch``).

Frame geometry (hop 160, ``n_fft`` 512, center=True, constant pad):
frame ``k`` of the whole-signal STFT covers preemphasized samples
``[k*160 - 256, k*160 + 256)``; a frame is STABLE once sample
``k*160 + 255`` exists, i.e. ``k < (N - 256)//160 + 1`` for ``N``
received samples. The final tail instead commits all
``output_lengths(N) = N // 160`` frames, with positions at or past
``N`` contributing zeros — exactly the whole-signal mask + constant
pad. Pre-emphasis continuity costs one extra retained sample
(``y[t] = x[t] - p*x[t-1]``); an absent sample (index < 0) is zero,
which also reproduces ``y[0] = x[0]``.

Batching: rows are grouped by identical segment length and frame
count, so the expensive STFT/mel/log ops run batched per group; only
index assembly loops per row (the design's no-per-session-loop rule
targets the tensor ops).

The frontend counters live in the ``(B, 8)`` int64 tensor whose slot
names/order are pinned by ``manifests.FRONTEND_COUNTER_FIELDS``; the
``CTR_*`` indices below mirror that tuple (cross-pinned by test).
"""

from __future__ import annotations

from typing import Any

import torch

#: Slot indices into the (B, 8) frontend counter tensor — MUST mirror
#: manifests.FRONTEND_COUNTER_FIELDS order (test-pinned).
(
    CTR_TOTAL_VALID_SAMPLES,
    CTR_COMMITTED_MEL_FRAMES,
    CTR_ENCODED_MEL_FRAMES,
    CTR_RAW_TAIL_ORIGIN,
    CTR_RAW_TAIL_LENGTH,
    CTR_MEL_TAIL_LENGTH,
    CTR_EXPECTED_CHUNK_SEQUENCE,
    CTR_FINALIZED,
) = range(8)

#: Mel frames retained before the committed boundary (the encoder's
#: pre-encode cache depth).
MEL_TAIL_FRAMES = 9


def stable_frames(n_samples: int, *, n_fft: int, hop: int) -> int:
    """Frames fully determined by ``n_samples`` received samples."""
    half = n_fft // 2
    if n_samples < half:
        return 0
    return (n_samples - half) // hop + 1


def final_frames(n_samples: int, *, n_fft: int, hop: int) -> int:
    """``MelFeaturizer.output_lengths`` for one element (final tail)."""
    return (n_samples + 2 * (n_fft // 2) - n_fft) // hop


def _gather_absolute(
    *,
    start: int,
    length: int,
    raw_tail: torch.Tensor,
    tail_origin: int,
    tail_length: int,
    new_samples: torch.Tensor,
    new_origin: int,
    valid_limit: int,
) -> torch.Tensor:
    """Samples ``[start, start+length)`` in absolute stream indices.

    Sourced from the retained tail (``[tail_origin, tail_origin +
    tail_length)``), the new samples (``[new_origin, new_origin +
    len)``), and zeros for indices below 0 or at/after
    ``valid_limit`` (the whole-signal mask + constant-pad semantics).
    """
    out = torch.zeros(length, dtype=raw_tail.dtype)
    idx = torch.arange(start, start + length)
    in_tail = (
        (idx >= tail_origin) & (idx < tail_origin + tail_length)
        & (idx < valid_limit) & (idx >= 0)
    )
    if bool(in_tail.any()):
        out[in_tail] = raw_tail[idx[in_tail] - tail_origin]
    n_new = int(new_samples.shape[0])
    in_new = (
        (idx >= new_origin) & (idx < new_origin + n_new)
        & (idx < valid_limit) & (idx >= 0)
    )
    if bool(in_new.any()):
        out[in_new] = new_samples[idx[in_new] - new_origin]
    return out


def cadence_boundary(
    chunk_index: int, *, lookahead: int
) -> int:
    """The approved cumulative encoder boundary after ``chunk_index``
    regular cadence units (design boundary formulas): ``B_k = (8L+1) +
    (k-1)·C`` mel frames with ``C = 8(L+1)`` — i.e. ``kC - 7``. Zero
    before the first unit."""
    if chunk_index <= 0:
        return 0
    cadence = 8 * (lookahead + 1)
    return (8 * lookahead + 1) + (chunk_index - 1) * cadence


def advance_frontend(
    featurizer: Any,
    samples: torch.Tensor,
    valid_samples: torch.Tensor,
    final_tail: torch.Tensor,
    target_frames: torch.Tensor,
    *,
    raw_tail: torch.Tensor,
    mel_tail: torch.Tensor,
    counters: torch.Tensor,
) -> tuple[list[torch.Tensor], torch.Tensor]:
    """Advance the bounded frontend for a batch of CHUNK rows.

    Boundary-capped (the option-(d) resolution, ledger 2026-07-17):
    each row commits mel frames only through its cumulative cadence
    target (``cadence_boundary`` for regular chunks, derived by the
    caller from the admitted geometry and chunk sequence), never
    through every currently stable frame — stability yields a uniform
    6-frame margin over the ``kC-7`` boundary, and the mel tail is
    retained relative to the TARGET so the encoder's pre-encode cache
    always holds the frames before its actual boundary. Frames stable
    beyond the target stay in the raw tail and recompute identically
    on a later update (exactness is preserved). A final tail ignores
    ``target_frames`` and commits through ``final_frames`` under the
    separate residual rules.

    Args:
        featurizer: the checkpoint ``MelFeaturizer`` (supplies
            ``n_fft``/``hop_length``/``preemph``/``fb``/``window``/
            ``log_zero_guard``; the exact op order below matches its
            ``forward``).
        samples: ``(B, S)`` padded raw FP32 rows.
        valid_samples: ``(B,)`` true sample counts.
        final_tail: ``(B,)`` bool final-tail markers.
        target_frames: ``(B,)`` long cumulative cadence targets
            (ignored for final-tail rows).
        raw_tail: ``(B, R)`` retained-tail storage, updated in place.
        mel_tail: ``(B, n_mels, MEL_TAIL_FRAMES)`` committed-boundary
            mel tail, updated in place.
        counters: ``(B, 8)`` int64 counters, updated in place.

    Returns:
        Per-row newly committed mel frames (``(n_mels, frames_i)``
        each; possibly zero-width) and the ``(B,)`` long new-frame
        counts.

    Raises:
        ValueError: on a chunk after finalization (protocol error), a
            regular chunk whose stable frames fall SHORT of its target
            (the design margin was violated — never silently deferred),
            a target below the already-committed boundary, or a
            raw-tail overflow (the derived bound was violated — never
            silently dropped).
    """
    n_fft = featurizer.n_fft
    hop = featurizer.hop_length
    preemph = featurizer.preemph
    half = n_fft // 2
    batch = samples.shape[0]
    capacity = raw_tail.shape[1]

    plans: list[dict[str, int]] = []
    for b in range(batch):
        if int(counters[b, CTR_FINALIZED]):
            raise ValueError(
                f"row {b}: chunk after finalization (a finalized "
                "session accepts no further audio)"
            )
        total_before = int(counters[b, CTR_TOTAL_VALID_SAMPLES])
        committed = int(counters[b, CTR_COMMITTED_MEL_FRAMES])
        n_valid = int(valid_samples[b])
        total = total_before + n_valid
        stable = stable_frames(total, n_fft=n_fft, hop=hop)
        if bool(final_tail[b]):
            target = max(final_frames(total, n_fft=n_fft, hop=hop),
                         committed)
        else:
            target = int(target_frames[b])
            if target < committed:
                raise ValueError(
                    f"row {b}: cadence target {target} is below the "
                    f"committed boundary {committed}"
                )
            if stable < target:
                raise ValueError(
                    f"row {b}: only {stable} stable mel frames for "
                    f"cadence target {target} (design margin "
                    "violated)"
                )
        n_new = target - committed
        # Segment covering frames [committed, target): preemphasized
        # samples [committed*hop - half, (target-1)*hop + half), plus
        # one leading raw sample for pre-emphasis continuity.
        seg_start = committed * hop - half
        seg_len = (n_new - 1) * hop + n_fft if n_new else 0
        plans.append({
            "total_before": total_before,
            "committed": committed,
            "total": total,
            "target": target,
            "n_new": n_new,
            "seg_start": seg_start,
            "seg_len": seg_len,
        })

    counts = torch.zeros(batch, dtype=torch.long)
    out: list[torch.Tensor] = [
        torch.zeros(featurizer.fb.shape[0], 0) for _ in range(batch)
    ]

    # Group rows with identical segment geometry: one batched
    # preemphasis + STFT + mel + log per group.
    groups: dict[tuple[int, int], list[int]] = {}
    for b, plan in enumerate(plans):
        if plan["n_new"]:
            groups.setdefault(
                (plan["seg_len"], plan["n_new"]), []
            ).append(b)

    for (seg_len, n_new), rows in groups.items():
        segs = []
        for b in rows:
            plan = plans[b]
            segs.append(_gather_absolute(
                start=plan["seg_start"] - 1,
                length=seg_len + 1,
                raw_tail=raw_tail[b],
                tail_origin=int(counters[b, CTR_RAW_TAIL_ORIGIN]),
                tail_length=int(counters[b, CTR_RAW_TAIL_LENGTH]),
                new_samples=samples[b, : int(valid_samples[b])],
                new_origin=plan["total_before"],
                valid_limit=plan["total"],
            ))
        x = torch.stack(segs)
        # Masked pre-emphasis: y[t] = x[t] - p*x[t-1]; the leading
        # extra sample supplies x[t-1] across the segment boundary, and
        # positions at/after each row's valid limit arrived as zeros
        # from the gather (the whole-signal mask).
        y = x[:, 1:] - preemph * x[:, :-1]
        # Zero re-mask: y at absolute positions >= the row's valid
        # limit must be exactly 0 (not -p*x[limit-1]).
        for j, b in enumerate(rows):
            plan = plans[b]
            rel_limit = plan["total"] - plan["seg_start"]
            if rel_limit < seg_len:
                y[j, max(rel_limit, 0):] = 0.0
        spec = torch.stft(
            y,
            n_fft=n_fft,
            hop_length=hop,
            win_length=featurizer.win_length,
            center=False,
            window=featurizer.window,
            return_complex=True,
        )
        magnitude = torch.sqrt(torch.view_as_real(spec).pow(2).sum(-1))
        power = magnitude.pow(2.0)
        mel = torch.matmul(featurizer.fb, power)
        logmel = torch.log(mel + featurizer.log_zero_guard)
        for j, b in enumerate(rows):
            out[b] = logmel[j, :, :n_new]
            counts[b] = n_new

    # Commit state: counters, mel tail, raw tail.
    for b, plan in enumerate(plans):
        total, target = plan["total"], plan["target"]
        is_final = bool(final_tail[b])
        if plan["n_new"]:
            keep = min(MEL_TAIL_FRAMES, plan["n_new"])
            mel_tail[b] = torch.roll(mel_tail[b], -keep, dims=1)
            mel_tail[b, :, MEL_TAIL_FRAMES - keep:] = (
                out[b][:, plan["n_new"] - keep:]
            )
        counters[b, CTR_TOTAL_VALID_SAMPLES] = total
        counters[b, CTR_COMMITTED_MEL_FRAMES] = target
        counters[b, CTR_MEL_TAIL_LENGTH] = min(
            MEL_TAIL_FRAMES, target
        )
        if is_final:
            counters[b, CTR_FINALIZED] = 1
            counters[b, CTR_RAW_TAIL_ORIGIN] = total
            counters[b, CTR_RAW_TAIL_LENGTH] = 0
        else:
            # Retain from one sample before the next frame's window.
            keep_from = max(target * hop - half - 1, 0)
            keep_len = total - keep_from
            if keep_len > capacity:
                raise ValueError(
                    f"row {b}: raw-tail overflow ({keep_len} > "
                    f"{capacity}) — the derived bound was violated"
                )
            if keep_len > 0:
                raw_tail[b, :keep_len] = _gather_absolute(
                    start=keep_from,
                    length=keep_len,
                    raw_tail=raw_tail[b].clone(),
                    tail_origin=int(counters[b, CTR_RAW_TAIL_ORIGIN]),
                    tail_length=int(counters[b, CTR_RAW_TAIL_LENGTH]),
                    new_samples=samples[b, : int(valid_samples[b])],
                    new_origin=plan["total_before"],
                    valid_limit=total,
                )
            counters[b, CTR_RAW_TAIL_ORIGIN] = keep_from
            counters[b, CTR_RAW_TAIL_LENGTH] = max(keep_len, 0)

    return out, counts
