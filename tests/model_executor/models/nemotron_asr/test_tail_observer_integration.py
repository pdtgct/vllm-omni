# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Full package integration; requires the matching vLLM runtime."""

from types import SimpleNamespace

import pytest
import torch

from tests.model_executor.models.nemotron_asr import test_batch_substat as fixtures
from vllm_omni.model_executor.models.nemotron_asr.commit_sink import BoundedCommitSink

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _sink():
    registry = SimpleNamespace(validate_lease=lambda bindings: None, lease_is_current=lambda bindings: True)
    return BoundedCommitSink(registry, max_rows=64, conditional_tail_max_invocations=16)


@pytest.mark.parametrize(
    "population,chunk_graph", [(1, True), (31, True), (63, True), (2, False), (32, False), (64, False)]
)
def test_real_transaction_observes_before_rejection_and_collects_once(monkeypatch, population, chunk_graph):
    """Real CPU transaction; graph boundary simulated, CUDA replay untested."""
    advance = fixtures.advance
    sink = _sink()
    original = advance.advance_model_rows
    tier = next(t for t in (1, 2, 4, 8, 16, 32, 64) if t >= population)

    def transition(*args, **kwargs):
        result = advance.advance_chunk_bucket(*args, **kwargs)
        # A downstream rejected lane must still participate in the predicate.
        result.result.frame_emission_counts.zero_()
        result.result.frame_emission_counts[:, 0] = 2
        result.result.frame_valid_lengths.fill_(1)
        result.result.row_status.fill_(512)
        setattr(transition, "replay_count", getattr(transition, "replay_count") + int(chunk_graph))
        return result

    setattr(transition, "replay_count", 0)
    binding = SimpleNamespace(resolve=lambda **kwargs: transition)

    def invoke(*args, **kwargs):
        kwargs["commit_sink"] = sink
        kwargs["decode_resolver"] = lambda request: advance.ResolvedDecode(
            arm="dense-graphed",
            decode_fn=fixtures.rnnt.decode_dense_masked_frames,
            execution_tier=tier,
        )
        kwargs["bucket_transition"] = binding if chunk_graph else transition
        return original(*args, **kwargs)

    monkeypatch.setattr(advance, "advance_model_rows", invoke)
    fixtures._single_geometry_call(num_rows=population, geometry=fixtures.GEOM_REG)
    assert sink.tail_observer.receipt()["pending_invocations"] == 1
    reports, _, _ = sink.collect()
    receipt = sink.tail_observer.receipt()
    assert all(report.row_status != 0 for report in reports)
    assert receipt["estimate_eligible"]
    [record] = receipt["records"]
    assert record["live_rows"] == population and record["decoder_tier"] == tier
    assert record["chunk_graph"] is chunk_graph
    assert record["frames"][0]["valid_lanes"] == population
    assert record["frames"][0]["tail_runs"]
    with pytest.raises(ValueError, match="no staged commit"):
        sink.collect()


def test_disabled_sink_does_not_construct_observer():
    sink = BoundedCommitSink(SimpleNamespace(), max_rows=1)
    assert sink.tail_observer is None


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires real CUDA capture")
@pytest.mark.parametrize(
    "population,chunk_graph", [(1, True), (31, True), (63, True), (2, False), (32, False), (64, False)]
)
@torch.inference_mode()
def test_cuda_replay_observation_preserves_outputs_and_tracks_changed_final_lengths(population, chunk_graph):
    from contextlib import contextmanager
    from dataclasses import fields, replace

    from test_chunk_bucket_graph import _fixture
    from vllm.config import VllmConfig

    from vllm_omni.model_executor.models.nemotron_asr.chunk_bucket_graph import (
        _clone_state,
        _outputs,
        _state_tensors,
        capture_chunk_bucket,
    )
    from vllm_omni.model_executor.models.nemotron_asr.decode_graph import DenseGraphBinding, platform_graph_runtime

    advance = fixtures.advance
    core, env, state, args = _fixture(population)
    device = torch.device("cuda", torch.accelerator.current_device_index())
    # Deterministic cap emissions for valid frames, zeros for the empty final.
    core.joint.joint_net[-1].weight.zero_()
    core.joint.joint_net[-1].bias.zero_()
    core.joint.joint_net[-1].bias[0] = 1
    for module in vars(core).values():
        if isinstance(module, torch.nn.Module):
            module.to(device)
    state = replace(
        state,
        **{
            field.name: [t.to(device) for t in value] if isinstance(value, list) else value.to(device)
            for field in fields(state)
            for value in [getattr(state, field.name)]
        },
    )
    env = env.to(device)
    args = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in args.items()}
    tier = next(t for t in (1, 2, 4, 8, 16, 32, 64) if t >= population)

    @contextmanager
    def capture_stream(_device):
        # Same real platform-wrapper fixture as test_encoder_execution.
        current = torch.cuda.current_stream(device)
        stream = torch.cuda.Stream(device=device)
        stream.wait_stream(current)
        with torch.cuda.stream(stream):
            yield
        current.wait_stream(stream)

    runtime = replace(platform_graph_runtime(), capture_context=capture_stream)
    config = VllmConfig()
    binding = DenseGraphBinding(
        decode_fn=fixtures.rnnt.decode_dense_masked_frames,
        predictor=core.predictor,
        joint=core.joint,
        vllm_config=config,
        frame_widths=(None, 2),
        tiers=(tier,),
        encoder_hidden=32,
        predictor_layers=2,
        predictor_hidden=16,
        blank_id=core.blank_id,
        runtime=runtime,
    )
    binding.warmup(device, torch.float32)
    decode = binding.decode_fn(geometry=1, tier=tier)
    transition = advance.advance_chunk_bucket
    if chunk_graph:
        transition = capture_chunk_bucket(
            core,
            env,
            state,
            **args,
            vllm_config=config,
            runtime=runtime,
            capture_decode_fn=binding.uncaptured_decode_fn(geometry=1, tier=tier),
            admitted_decode_fn=decode,
            decoder_tier=tier,
        )
    registry = SimpleNamespace(validate_lease=lambda bindings: None, lease_is_current=lambda bindings: True)
    sink = BoundedCommitSink(registry, max_rows=population, device=device, conditional_tail_max_invocations=8)
    bindings = fixtures._plan(
        prefills=list(range(1, population + 1)), num_pool_blocks=population + 1, geometries=[1] * population
    ).bindings
    plan = advance.CommitPlan(bindings=bindings, capture=None)
    # The observer is created after warmup/capture: capture must not create an
    # observation. Both cases execute their real captured decoder arithmetic.
    assert sink.tail_observer.receipt()["attempted_invocations"] == 0
    lengths_seen = []
    counts_seen = []
    for final in (False, True, False):
        env[:, advance.ENV_FINAL_TAIL] = int(final)
        env[:, advance.ENV_VALID_SAMPLES] = 0 if final else 2560
        actual_state = _clone_state(state)
        reference_state = _clone_state(state)
        reference = advance.advance_chunk_bucket(
            core,
            env,
            reference_state,
            **args,
            decode_fn=binding.uncaptured_decode_fn(geometry=1, tier=tier),
        )
        # Preserve the same padded-tier arithmetic and own the eager outputs
        # before replay reuses the binding's borrowed workspace.
        reference_values = tuple(t.clone() for t in (*_outputs(reference), *_state_tensors(reference_state)))
        actual = transition(core, env, actual_state, **args, decode_fn=decode)
        snapshots = tuple(t.clone() for t in (*_outputs(actual), *_state_tensors(actual_state)))
        sink.tail_observer.begin()
        sink.tail_observer.observe(
            rows=tuple(range(population)),
            geometry=1,
            tier=tier,
            arm="dense-graphed",
            chunk_graph=chunk_graph,
            counts=actual.result.frame_emission_counts,
            lengths=actual.result.frame_valid_lengths,
            final_tail=actual.batch.final_tail,
        )
        for observed, snapshot, expected in zip(
            (*_outputs(actual), *_state_tensors(actual_state)),
            snapshots,
            reference_values,
            strict=True,
        ):
            torch.testing.assert_close(observed, snapshot, atol=0, rtol=0)
            torch.testing.assert_close(observed, expected, atol=0, rtol=0)
        expected_counts = actual.result.frame_emission_counts.clone()
        expected_lengths = actual.result.frame_valid_lengths.clone()
        # Overwrite the borrowed split workspace BEFORE the summary D2H copy.
        changed_env = env.clone()
        changed_env[:, advance.ENV_FINAL_TAIL] = 1
        changed_env[:, advance.ENV_VALID_SAMPLES] = 0
        transition(core, changed_env, _clone_state(state), **args, decode_fn=decode)
        sink.reserve(plan).stage(actual.result.row_status)
        sink.collect()
        receipt = sink.tail_observer.receipt()
        assert receipt["estimate_eligible"]
        record = receipt["records"][-1]
        assert record["final_tail_rows"] == (population if final else 0)
        valid = torch.arange(2, device=device)[None, :] < expected_lengths.clamp(0, 2)[:, None]
        assert [f["tail_runs"] for f in record["frames"]] == (valid & (expected_counts >= 2)).any(0).tolist()
        assert [f["valid_lanes"] for f in record["frames"]] == valid.sum(0).tolist()
        lengths_seen.append(expected_lengths.tolist())
        counts_seen.append(expected_counts.tolist())
    assert lengths_seen[0] != lengths_seen[1] and lengths_seen[0] == lengths_seen[2]
    assert counts_seen[0] != counts_seen[1] and counts_seen[0] == counts_seen[2]
