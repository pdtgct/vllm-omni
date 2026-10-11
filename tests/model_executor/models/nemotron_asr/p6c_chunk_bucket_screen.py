# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Bounded full-transaction extension of the existing P6B/P6C screen.

Call ``run_full_turn_screen`` from the trained P6B fixture after its recurrent
numerical gate and full-history preparation. All restores and status collection
are outside the synchronized single-call timer. This measures a continuing
CHUNK transaction with the compatibility adapter; it is not service evidence.
"""

import statistics
import time
from dataclasses import replace
from typing import Any

import torch

from vllm_omni.model_executor.models.nemotron_asr import advance, frontend, manifests, plan, rnnt
from vllm_omni.model_executor.models.nemotron_asr.chunk_bucket_graph import capture_chunk_bucket
from vllm_omni.model_executor.models.nemotron_asr.commit_sink import BoundedCommitSink


@torch.inference_mode()
def run_full_turn_screen(
    core: Any,
    seed: advance.SessionStateBatch,
    batch: advance.ChunkBatch,
    *,
    baseline_encoder: Any,
    baseline_decode: Any,
    vllm_config: Any,
    runtime: Any,
    baseline_transaction: Any = advance.advance_model_rows,
) -> dict[str, Any]:
    """B32/B64, geometry 1, split native graphs versus broader bucket graph.

    The seed is a clean full-history state from the existing trained trajectory.
    The replay queue is explicitly seeded drained. Every measured transaction
    starts from this same valid checkpoint; repeated calls never process a
    CHUNK against an outstanding compatibility replay queue.
    """
    population = int(batch.samples.shape[0])
    if population not in (32, 64) or not torch.all(batch.geometry_id == 1):
        raise ValueError("full-turn screen requires a 160-ms B32/B64 fixture")
    device = batch.samples.device
    cap = manifests.SESSION_LIMITS["queue_capacity"]
    park, placeholder = int(core.blank_id) + 1, int(core.blank_id) + 2
    hidden = advance.ENVELOPE_HEADER_SLOTS + max(manifests.RAW_SAMPLES_PER_CHUNK.values())
    env = torch.zeros(population, hidden, device=device)
    env[:, advance.ENV_VERSION] = advance.ENVELOPE_VERSION
    env[:, advance.ENV_VALID_SAMPLES] = batch.valid_samples
    env[:, advance.ENV_GEOMETRY_ID] = batch.geometry_id
    env[:, advance.ENV_FINAL_TAIL] = batch.final_tail
    env[:, advance.ENV_PROMPT_INDEX] = batch.prompt_index
    env[:, advance.ENV_CHUNK_SEQUENCE] = seed.frontend_counters[:, frontend.CTR_EXPECTED_CHUNK_SEQUENCE]
    env[:, advance.ENVELOPE_HEADER_SLOTS : advance.ENVELOPE_HEADER_SLOTS + batch.samples.shape[1]] = batch.samples
    candidate = capture_chunk_bucket(
        core,
        env,
        seed,
        geometry=1,
        admitted_prompt=batch.prompt_index,
        incoming_status=torch.zeros(population, dtype=torch.int32, device=device),
        queue_capacity=cap,
        vllm_config=vllm_config,
        runtime=runtime,
    )

    def resident(value: torch.Tensor) -> torch.Tensor:
        result = value.new_zeros((population + 1, *value.shape[1:]))
        result[1:].copy_(value)
        return result

    pools = dict(
        channel_pools=[resident(value) for value in seed.channel],
        time_pools=[resident(value) for value in seed.time],
        len_pools=[resident(value) for value in seed.window_valid],
        h_pool=resident(seed.h),
        c_pool=resident(seed.c),
        frontend_raw_pool=resident(seed.raw_tail),
        frontend_mel_pool=resident(seed.mel_tail),
        frontend_counter_pool=resident(seed.frontend_counters),
        queue_pool=torch.zeros(population + 1, cap, dtype=torch.int32, device=device),
        book_pool=torch.zeros(population + 1, len(manifests.BOOK_FIELDS), dtype=torch.int32, device=device),
    )
    book_fields = {name: i for i, (name, _) in enumerate(manifests.BOOK_FIELDS)}
    pools["book_pool"][1:, book_fields["last_label"]] = seed.last_label
    pools["book_pool"][1:, book_fields["prompt"]] = batch.prompt_index
    pools["book_pool"][1:, book_fields["geometry"]] = 1
    tensors = tuple(t for value in pools.values() for t in (value if isinstance(value, list) else [value]))
    checkpoint = tuple(value.clone() for value in tensors)
    if device.type == "cuda":
        advance.warmup_advance_model_rows_scatter(**pools)
    registry = plan.SessionRegistry()
    headers = env[:, : advance.ENVELOPE_HEADER_SLOTS].cpu().tolist()
    rows = [
        plan.ObservedRow(f"bucket-screen-{i}", i + 1, placeholder, False, tuple(headers[i])) for i in range(population)
    ]
    common = dict(
        resident_request_ids=[row.request_id for row in rows],
        placeholder_id=placeholder,
        num_prompts=int(core.lid.num_prompts),
        num_geometries=len(manifests.CADENCES),
        now_ns=1,
    )
    # Register identity, then mint the continuing-row authority for seeded pages.
    plan.prepare_plan_context(registry, rows, step=0, **common)
    context = plan.prepare_plan_context(
        registry, [replace(row, has_prior_state=True) for row in rows], step=1, **common
    )
    row_plan = plan.build_row_plan(
        context, num_decodes=0, num_prefills=population, null_block_id=0, num_pool_blocks=population + 1
    )
    sink = BoundedCommitSink(registry, max_rows=population, device=device)
    staging = advance.HostStaging(population)
    adapter = advance.make_mrv1_adapter(hidden_size=hidden, park_id=park, blank_id=int(core.blank_id))
    input_ids = torch.full((population,), placeholder, dtype=torch.long, device=device)

    def invoke(use_candidate: bool) -> torch.Tensor:
        decode = rnnt.decode_dense_masked_frames if use_candidate else baseline_decode
        transaction = advance.advance_model_rows if use_candidate else baseline_transaction
        candidate_kwargs = {"bucket_transition": candidate} if use_candidate else {}
        return transaction(
            core,
            input_ids,
            env,
            row_plan,
            **pools,
            adapter=adapter,
            decode_resolver=lambda _request: advance.ResolvedDecode(arm="dense-eager", decode_fn=decode),
            encoder_transition=None if use_candidate else baseline_encoder,
            **candidate_kwargs,
            placeholder_id=placeholder,
            park_id=park,
            commit_sink=sink,
            staging=staging,
        )

    def reset() -> None:
        for destination, source in zip(tensors, checkpoint, strict=True):
            destination.copy_(source)
        runtime.synchronize(device)

    def collect() -> None:
        statuses, records, lease_ok = sink.collect()
        if not lease_ok or records or len(statuses) != population or any(row.row_status for row in statuses):
            raise ValueError("full-turn checkpoint produced a failed status or lease")
        expected = seed.frontend_counters[:, frontend.CTR_EXPECTED_CHUNK_SEQUENCE] + 1
        if not torch.equal(pools["frontend_counter_pool"][1:, frontend.CTR_EXPECTED_CHUNK_SEQUENCE], expected):
            raise ValueError("full-turn checkpoint did not advance all rows")

    # Compare the complete committed transaction before accepting timings.
    reset()
    expected = invoke(False).clone()
    collect()
    expected_pools = tuple(value.clone() for value in tensors)
    reset()
    actual = invoke(True)
    collect()
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    for actual_pool, expected_pool in zip(tensors, expected_pools, strict=True):
        torch.testing.assert_close(actual_pool, expected_pool, atol=0, rtol=0)

    timings: list[dict[str, Any]] = []
    for arm, use_candidate in (("baseline1", False), ("candidate", True), ("baseline2", False)):
        durations = []
        for sample in range(9):
            reset()
            start = time.perf_counter()
            invoke(use_candidate)
            runtime.synchronize(device)
            elapsed_ms = (time.perf_counter() - start) * 1000
            collect()
            if sample >= 2:
                durations.append(elapsed_ms)
        timings.append(dict(arm=arm, median_ms=statistics.median(durations), samples_ms=durations))
    first, candidate_ms, last = [entry["median_ms"] for entry in timings]
    gain, drift = min(first, last) - candidate_ms, abs(first - last)
    return dict(
        scope="complete advance_model_rows through composite commit stage; excludes pool restore and host status collection",
        population=population,
        geometry=1,
        native_burst=False,
        clean_statuses=True,
        full_transaction_exact=True,
        warmups_per_arm=2,
        timings=timings,
        clock="synchronized wall, one independently restored valid continuing transaction per sample",
        gain_ms=gain,
        drift_ms=drift,
        gain_percent=100 * gain / min(first, last),
        passes=gain > drift and gain / min(first, last) > 0.05,
    )
