# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
from types import SimpleNamespace

import pytest
import torch
from test_persistent_profile_execution import _model_class
from test_persistent_profile_execution import chunk_constructor as chunk_constructor

from vllm_omni.model_executor.models.nemotron_asr.native_burst import NativeBurstHandoff, NativeBurstSampler

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _snapshot(epoch=1):
    batch = SimpleNamespace(req_ids=["a"], idx_mapping_np=[0])
    return SimpleNamespace(
        epoch=epoch,
        req_ids=("a",),
        input_batch=batch,
        bindings=(SimpleNamespace(request_id="a", generation=2, slot_id=3),),
        block_ids=((3,),),
        req_states=SimpleNamespace(req_id_to_index={"a": 0}),
    )


def _payload():
    return SimpleNamespace(
        sampled_token_ids=torch.tensor([[2, 9000, -1]]), num_sampled=torch.tensor([2], dtype=torch.int32)
    )


def test_receipt_single_use_owned_payload_and_native_counts():
    handoff = NativeBurstHandoff()
    snapshot = _snapshot()
    handoff.prepare(snapshot)
    payload = _payload()
    stage = handoff.reserve(payload)
    stage()
    sampler = NativeBurstSampler(SimpleNamespace(), handoff)
    output = sampler(torch.zeros(1, 9001), snapshot.input_batch)
    assert output.sampled_token_ids.tolist() == [[2, 9000, -1]]
    assert output.num_sampled.tolist() == [2]
    assert output.num_rejected.tolist() == [0]
    with pytest.raises(RuntimeError, match="absent"):
        sampler(torch.zeros(1, 9001), snapshot.input_batch)
    handoff.prepare(_snapshot(2))
    assert output.sampled_token_ids.tolist() == [[2, 9000, -1]]


@pytest.mark.parametrize("mutation", ["order", "index", "generation", "batch"])
def test_receipt_rejects_changed_snapshot(mutation):
    handoff = NativeBurstHandoff()
    snapshot = _snapshot()
    handoff.prepare(snapshot)
    handoff.reserve(_payload())()
    batch = snapshot.input_batch
    if mutation == "order":
        batch.req_ids = ["b"]
    elif mutation == "index":
        batch.idx_mapping_np = [1]
    elif mutation == "generation":
        snapshot.bindings[0].generation = 3
    else:
        batch = SimpleNamespace(req_ids=["a"], idx_mapping_np=[0])
    with pytest.raises(RuntimeError, match="snapshot"):
        handoff.consume(batch)


def test_receipt_rejects_overwrite_stale_epoch_and_double_stage():
    handoff = NativeBurstHandoff()
    snapshot = _snapshot()
    handoff.prepare(snapshot)
    with pytest.raises(RuntimeError, match="pending"):
        handoff.prepare(_snapshot(2))
    stage = handoff.reserve(_payload())
    with pytest.raises(RuntimeError, match="reserved"):
        handoff.reserve(_payload())
    stage()
    handoff.consume(snapshot.input_batch)
    with pytest.raises(RuntimeError, match="epoch"):
        handoff.prepare(_snapshot())


@pytest.mark.parametrize("setting", ["async", "spec", "pp", "tp", "dp", "dcp", "pcp", "ep", "processors", "lookahead"])
def test_native_startup_rejects_unqualified_configuration(setting):
    from vllm_omni.model_executor.models.nemotron_asr.native_burst import validate_native_burst_config

    config = SimpleNamespace(
        scheduler_config=SimpleNamespace(async_scheduling=False),
        speculative_config=None,
        model_config=SimpleNamespace(logits_processors=None),
        parallel_config=SimpleNamespace(
            pipeline_parallel_size=1,
            tensor_parallel_size=1,
            data_parallel_size=1,
            decode_context_parallel_size=1,
            prefill_context_parallel_size=1,
            enable_expert_parallel=False,
        ),
    )
    hf = SimpleNamespace(supported_num_lookahead_tokens=[1])
    validate_native_burst_config(config, hf_config=hf)
    if setting == "async":
        config.scheduler_config.async_scheduling = True
    elif setting == "processors":
        config.model_config.logits_processors = ["some.processor"]
    elif setting == "lookahead":
        hf.supported_num_lookahead_tokens = [3]
    elif setting == "spec":
        config.speculative_config = object()
    else:
        field = {
            "pp": "pipeline_parallel_size",
            "tp": "tensor_parallel_size",
            "dp": "data_parallel_size",
            "dcp": "decode_context_parallel_size",
            "pcp": "prefill_context_parallel_size",
            "ep": "enable_expert_parallel",
        }[setting]
        setattr(config.parallel_config, field, 2)
    with pytest.raises(ValueError, match="native burst"):
        validate_native_burst_config(config, hf_config=hf)


@pytest.mark.parametrize(
    "field,value",
    [
        ("repetition_penalty", 1.1),
        ("frequency_penalty", 0.1),
        ("presence_penalty", 0.1),
        ("logprobs", 0),
        ("prompt_logprobs", 0),
        ("logit_bias", {2: 1}),
        ("allowed_token_ids", [2]),
        ("structured_outputs", object()),
        ("min_tokens", 1),
        ("stop", ["x"]),
        ("ignore_eos", True),
        ("max_tokens", 1),
    ],
)
def test_native_admission_rejects_modifiers(field, value):
    from vllm.sampling_params import SamplingParams

    from vllm_omni.model_executor.models.nemotron_asr.native_burst import validate_native_burst_sampling

    params = SamplingParams(temperature=0, max_tokens=100)
    setattr(params, field, value)
    with pytest.raises(ValueError, match="native burst"):
        validate_native_burst_sampling(params, park_id=9000, capacity=48)


@pytest.mark.parametrize("temperature", [0, 1])
def test_native_admission_accepts_plain_greedy(temperature):
    from vllm.sampling_params import SamplingParams

    from vllm_omni.model_executor.models.nemotron_asr.native_burst import validate_native_burst_sampling

    validate_native_burst_sampling(SamplingParams(temperature=temperature, max_tokens=100), park_id=9000, capacity=48)


@pytest.mark.parametrize("mutation", ["blank", "park", "padding", "count", "book"])
def test_native_payload_validator_rejects_corruption(mutation):
    from vllm_omni.model_executor.models.nemotron_asr.advance import ROLE_CHUNK, EmissionContext, EmissionProjection
    from vllm_omni.model_executor.models.nemotron_asr.native_burst import (
        finalize_native_burst,
        native_burst_invariant_rows,
    )

    queue = torch.tensor([[2, 3, 0]], dtype=torch.int32)
    book = torch.tensor([[1, 2, 3, 0, 0, 1, 2]], dtype=torch.int64)
    status = torch.zeros(1, dtype=torch.int32)
    context = EmissionContext(
        torch.tensor([ROLE_CHUNK]),
        torch.tensor([9001]),
        torch.tensor([0]),
        queue.clone(),
        book.clone(),
        torch.tensor([0]),
        status,
    )
    projection = EmissionProjection(torch.tensor([[2.0]]), queue, book, status)
    payload = finalize_native_burst(projection, context, park_id=9000, blank_id=12)
    if mutation == "blank":
        payload.sampled_token_ids[0, 0] = 12
    elif mutation == "park":
        payload.sampled_token_ids[0, 2] = 2
    elif mutation == "padding":
        payload.sampled_token_ids[0, 3] = 2
    elif mutation == "count":
        payload.num_sampled[0] = 2
    else:
        payload.book[0, 5] = 1
    assert native_burst_invariant_rows(projection, context, payload, park_id=9000, blank_id=12).tolist() == [True]


def test_cancelled_receipt_cannot_be_consumed_or_leak_into_next_generation():
    handoff = NativeBurstHandoff()
    snapshot = _snapshot()
    handoff.prepare(snapshot)
    handoff.reserve(_payload())()
    handoff.cancel_requests(frozenset({"a"}))
    with pytest.raises(RuntimeError, match="absent"):
        handoff.consume(snapshot.input_batch)
    new = _snapshot(2)
    new.bindings[0].generation = 3
    handoff.prepare(new)
    handoff.reserve(_payload())()
    assert handoff.consume(new.input_batch).num_sampled.tolist() == [2]


@pytest.mark.parametrize("role", ["chunk", "forced"])
def test_native_eou_is_emitted_before_park_and_echo_is_drained(role):
    from vllm_omni.model_executor.models.nemotron_asr.advance import (
        ROLE_CHUNK,
        ROLE_EOU,
        EmissionContext,
        EmissionProjection,
    )
    from vllm_omni.model_executor.models.nemotron_asr.native_burst import (
        finalize_native_burst,
        native_burst_invariant_rows,
    )

    labels = [2, 15] if role == "chunk" else [15]
    queue = torch.tensor([labels + [0] * (3 - len(labels))], dtype=torch.int32)
    book = torch.tensor([[1, len(labels), 2, 0, 0, 1, labels[0]]], dtype=torch.int64)
    status = torch.zeros(1, dtype=torch.int32)
    context = EmissionContext(
        torch.tensor([ROLE_CHUNK if role == "chunk" else ROLE_EOU]),
        torch.tensor([9001]),
        torch.tensor([0]) if role == "chunk" else torch.empty(0, dtype=torch.int64),
        torch.zeros_like(queue),
        torch.zeros_like(book),
        torch.tensor([0]),
        status,
    )
    source = EmissionProjection(torch.tensor([[float(labels[0])]]), queue, book, status)
    native = finalize_native_burst(source, context, park_id=9000, blank_id=12)
    assert native.sampled_token_ids[0, : native.num_sampled[0]].tolist() == labels + [9000]
    assert native.book[0, 0] == len(labels)
    assert native.book[0, 5] == 0
    assert native.book[0, 6] == 15
    assert native_burst_invariant_rows(
        source, context, native, park_id=9000, blank_id=12, eou_token_id=15
    ).tolist() == [False]


def test_real_pipeline_budget_accepts_declared_160ms_and_rejects_unqualified_1120ms():
    from vllm.sampling_params import SamplingParams

    from vllm_omni.model_executor.models.nemotron_asr.configuration_nemotron_asr import NemotronASRConfig
    from vllm_omni.model_executor.models.nemotron_asr.manifests import SESSION_LIMITS
    from vllm_omni.model_executor.models.nemotron_asr.native_burst import (
        native_burst_token_budget,
        validate_native_burst_sampling,
    )
    from vllm_omni.model_executor.models.nemotron_asr.pipeline import NEMOTRON_ASR_PIPELINE

    params = SamplingParams(**NEMOTRON_ASR_PIPELINE.stages[0].sampling_constraints)
    assert params.max_tokens == 141 and SESSION_LIMITS["queue_capacity"] == 141
    cfg = NemotronASRConfig(supported_num_lookahead_tokens=[1])
    bound = native_burst_token_budget(cfg)
    assert bound == 22  # 2 encoder frames * 10 labels + optional EOU + PARK
    validate_native_burst_sampling(params, park_id=9000, capacity=bound - 1)
    maximum = native_burst_token_budget(NemotronASRConfig())
    assert maximum == 142
    with pytest.raises(ValueError, match="native burst"):
        validate_native_burst_sampling(params, park_id=9000, capacity=maximum - 1)


@pytest.mark.parametrize("prefill_len,max_model_len,valid", [(1, 143, True), (1, 142, False), (2, 143, False)])
def test_native_history_admission_counts_actual_prefill_and_whole_payload(prefill_len, max_model_len, valid):
    from vllm_omni.model_executor.models.nemotron_asr.native_burst import validate_native_burst_history

    request = SimpleNamespace(prefill_token_ids=[9001] * prefill_len, prompt_token_ids=[9001], num_computed_tokens=0)
    if valid:
        validate_native_burst_history(request, max_model_len=max_model_len, burst_tokens=142)
    else:
        with pytest.raises(ValueError, match="history"):
            validate_native_burst_history(request, max_model_len=max_model_len, burst_tokens=142)


def test_dummy_scope_cannot_mask_live_receipt_and_cleans_up_after_error():
    handoff = NativeBurstHandoff()
    handoff.prepare(_snapshot())
    with pytest.raises(RuntimeError, match="overlaps"):
        with handoff.dummy_sampling(park_id=9000):
            pytest.fail("pending live receipt entered dummy scope")
    handoff.clear()
    with pytest.raises(ValueError, match="profile failure"):
        with handoff.dummy_sampling(park_id=9000):
            raise ValueError("profile failure")
    sampler = NativeBurstSampler(object(), handoff)
    with pytest.raises(RuntimeError, match="absent"):
        sampler(torch.zeros(1, 9001), _snapshot().input_batch)


def test_profile_sampler_keeps_native_receipt_scratch_live_with_logits(monkeypatch):
    from vllm_omni.model_executor.models.nemotron_asr.manifests import SESSION_LIMITS

    handoff = NativeBurstHandoff()
    sampler = NativeBurstSampler(object(), handoff)
    logits = torch.zeros(2, 9001)
    allocations = []
    original = torch.empty

    def observe(shape, **kwargs):
        result = original(shape, **kwargs)
        allocations.append((tuple(result.shape), result.dtype))
        return result

    monkeypatch.setattr(torch, "empty", observe)
    with handoff.dummy_sampling(park_id=9000):
        output = sampler(logits, SimpleNamespace(req_ids=["a", "b"]))
    assert allocations == [
        ((2, SESSION_LIMITS["queue_capacity"]), torch.int32),
        ((2, 7), torch.int32),
        ((2,), torch.int32),
    ]
    assert output.sampled_token_ids.shape == (2, SESSION_LIMITS["queue_capacity"] + 1)


def _selected_model_state(chunk_constructor, *, native=True):
    config, _ = chunk_constructor
    config.model_config.hf_config.experimental_native_burst = native
    model = _model_class()(vllm_config=config)
    state = object.__new__(model.get_model_state_cls())
    state.model = model
    state.model_config = SimpleNamespace(max_model_len=256)
    return model, state


@pytest.mark.parametrize("population", [1, 31, 63])
def test_native_compute_logits_preserves_owned_sampler_payload(chunk_constructor, monkeypatch, population):
    # @spec PORT-DEC-012 / PORT-MIG-004
    from vllm_omni.model_executor.models.nemotron_asr import rnnt

    model, state = _selected_model_state(chunk_constructor)
    handoff = model._native_burst_handoff
    sampler, rejection_sampler = state.custom_sampler(object())
    assert isinstance(sampler, NativeBurstSampler) and rejection_sampler is None
    snapshot = _snapshot()
    req_ids = [f"request-{index}" for index in range(population)]
    indices = list(reversed(range(population)))
    snapshot.req_ids = tuple(req_ids)
    snapshot.input_batch.req_ids = req_ids
    snapshot.input_batch.idx_mapping_np = indices
    snapshot.req_states.req_id_to_index = dict(zip(req_ids, indices))
    snapshot.bindings = tuple(
        SimpleNamespace(request_id=req_id, generation=2, slot_id=index + 1) for index, req_id in enumerate(req_ids)
    )
    snapshot.block_ids = tuple((index + 1,) for index in range(population))
    handoff.prepare(snapshot)
    payload = SimpleNamespace(
        sampled_token_ids=torch.tensor([[index + 2, model.config.eos_token_id, -1] for index in range(population)]),
        num_sampled=torch.full((population,), 2, dtype=torch.int32),
    )
    handoff.reserve(payload)()

    def forbidden(*args, **kwargs):
        pytest.fail("native compute_logits invoked compatibility forced_logits_rows")

    monkeypatch.setattr(rnnt, "forced_logits_rows", forbidden)
    hidden_states = torch.arange(population * 2 * 8, dtype=torch.float32).reshape(population * 2, 8)
    # Match MRv2's gather with nontrivial request order and skipped hidden rows.
    logits_indices = torch.tensor(indices) * 2 + 1
    sampled_hidden = hidden_states[logits_indices]
    logits = model.compute_logits(sampled_hidden)
    assert logits.shape == (population, 1) and logits.device == hidden_states.device
    assert logits.dtype == sampled_hidden.dtype and logits.data_ptr() == sampled_hidden.data_ptr()
    assert torch.equal(logits[:, 0], hidden_states[logits_indices, 0])
    # Computing the carrier must not consume the committed receipt.
    assert handoff._payload is payload
    output = sampler(logits, snapshot.input_batch)
    assert output.sampled_token_ids is payload.sampled_token_ids
    assert output.num_sampled is payload.num_sampled
    assert output.num_rejected.tolist() == [0] * population
    assert output.logprobs_tensors is None and output.num_nans is None
    with pytest.raises(RuntimeError, match="absent"):
        sampler(logits, snapshot.input_batch)
    retained = output.sampled_token_ids.clone()
    sampled_hidden.fill_(-123)
    hidden_states.zero_()
    snapshot.epoch += 1
    handoff.prepare(snapshot)
    replacement = SimpleNamespace(sampled_token_ids=torch.zeros_like(retained), num_sampled=payload.num_sampled.clone())
    handoff.reserve(replacement)()
    assert sampler(logits, snapshot.input_batch).sampled_token_ids is replacement.sampled_token_ids
    assert torch.equal(output.sampled_token_ids, retained)


@pytest.mark.parametrize("population", [1, 31, 63])
def test_native_compute_logits_dummy_keeps_device_and_profile_scratch(chunk_constructor, monkeypatch, population):
    # @spec PORT-DEC-012 / PORT-MIG-005
    from vllm_omni.model_executor.models.nemotron_asr import rnnt
    from vllm_omni.model_executor.models.nemotron_asr.manifests import SESSION_LIMITS

    model, state = _selected_model_state(chunk_constructor)
    sampler, _ = state.custom_sampler(object())

    def forbidden(*args, **kwargs):
        pytest.fail("native dummy compute_logits invoked compatibility forced_logits_rows")

    monkeypatch.setattr(rnnt, "forced_logits_rows", forbidden)
    allocations = []
    original = torch.empty

    def observe(shape, **kwargs):
        result = original(shape, **kwargs)
        allocations.append((tuple(result.shape), result.dtype, result.device))
        return result

    monkeypatch.setattr(torch, "empty", observe)
    hidden = torch.zeros(population, 8)
    with model._native_burst_handoff.dummy_sampling(park_id=model.config.eos_token_id):
        logits = model.compute_logits(hidden)
        output = sampler(logits, SimpleNamespace(req_ids=list(range(population))))
    assert logits.shape == (population, 1) and logits.data_ptr() == hidden.data_ptr()
    assert output.sampled_token_ids.device == hidden.device
    assert output.sampled_token_ids.shape == (population, SESSION_LIMITS["queue_capacity"] + 1)
    assert output.sampled_token_ids[:, 0].tolist() == [model.config.eos_token_id] * population
    assert torch.all(output.sampled_token_ids[:, 1:] == -1)
    assert output.num_sampled.tolist() == [1] * population
    assert output.num_rejected.tolist() == [0] * population
    assert allocations == [
        ((population, SESSION_LIMITS["queue_capacity"]), torch.int32, hidden.device),
        ((population, 7), torch.int32, hidden.device),
        ((population,), torch.int32, hidden.device),
    ]
    with pytest.raises(RuntimeError, match="absent"):
        sampler(logits, SimpleNamespace(req_ids=list(range(population))))


def test_compatibility_compute_logits_keeps_forced_fp32_gathered_rows(chunk_constructor):
    # @spec PORT-DEC-002 / PORT-MIG-002
    model, state = _selected_model_state(chunk_constructor, native=False)
    assert state.custom_sampler(object()) is None
    hidden = torch.zeros(5, 8)
    hidden[:, 0] = torch.tensor([2, 3, model.config.eos_token_id, 4, 5])
    indices = torch.tensor([2, 4, 0])
    logits = model.compute_logits(hidden[indices])
    assert logits.shape == (3, model.num_logits) and logits.dtype == torch.float32
    assert logits.device == hidden.device
    assert torch.equal(logits.argmax(dim=1), hidden[indices, 0].long())
    assert torch.count_nonzero(torch.isfinite(logits), dim=1).tolist() == [1, 1, 1]
    assert torch.all(logits[torch.arange(3), hidden[indices, 0].long()] == 0)
    assert torch.all(torch.isneginf(logits) | (logits == 0))


@pytest.mark.parametrize(
    "field,value",
    [("logprobs", 0), ("prompt_logprobs", 0), ("logprob_token_ids", [2]), ("structured_outputs", object())],
)
def test_native_model_state_rejects_logits_consumers(chunk_constructor, field, value):
    # @spec PORT-DEC-011 / PORT-MIG-004
    from vllm.sampling_params import SamplingParams

    model, state = _selected_model_state(chunk_constructor)
    params = SamplingParams(temperature=0, max_tokens=100)
    setattr(params, field, value)
    request = SimpleNamespace(
        sampling_params=params,
        prefill_token_ids=[model.config.audio_chunk_token_id],
        prompt_token_ids=[model.config.audio_chunk_token_id],
        num_computed_tokens=0,
    )
    with pytest.raises(ValueError, match="native burst"):
        state.validate_omni_request(request)
