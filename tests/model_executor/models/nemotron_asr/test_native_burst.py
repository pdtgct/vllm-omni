# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
from types import SimpleNamespace

import pytest
import torch

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


LEGACY_PROFILE = "fp32-sync-single-device-160ms-v1"
FIVE_CADENCE_PROFILE = "fp32-sync-single-device-five-cadence-v1"
LOOKAHEADS = (0, 1, 3, 6, 13)
BOUNDS = (12, 22, 42, 72, 142)


def _runtime_config(hf):
    return SimpleNamespace(
        scheduler_config=SimpleNamespace(async_scheduling=False, max_num_seqs=1),
        speculative_config=None,
        model_config=SimpleNamespace(hf_config=hf, logits_processors=None, dtype=torch.float32),
        parallel_config=SimpleNamespace(
            pipeline_parallel_size=1,
            tensor_parallel_size=1,
            data_parallel_size=1,
            decode_context_parallel_size=1,
            prefill_context_parallel_size=1,
            enable_expert_parallel=False,
        ),
    )


def _native_config(*, profile=None, lookaheads=(1,), enabled=True, precision=None):
    from vllm_omni.model_executor.models.nemotron_asr.configuration_nemotron_asr import NemotronASRConfig

    return NemotronASRConfig(
        experimental_native_burst=enabled,
        experimental_native_burst_profile=profile,
        supported_num_lookahead_tokens=list(lookaheads),
        experimental_encoder_compute_dtype=precision,
        d_model=32,
        n_layers=2,
        conv_kernel=5,
        att_context_left=8,
        pred_hidden=16,
        pred_rnn_layers=2,
        joint_hidden=32,
        num_prompts=4,
        prompt_dictionary={"en-US": 0},
        decode_dispatch_arm="dense-eager",
        encoder_execution_arm="eager",
        eos_token_id=13088,
        audio_chunk_token_id=13089,
        eou_token_id=13090,
        flush_token_id=13091,
        vocab_size=13092,
    )


def _construct_model(monkeypatch, hf):
    from vllm_omni.model_executor.models.nemotron_asr import nemotron_asr as module

    core = torch.nn.Module()
    core.encoder = torch.nn.Linear(2, 2)
    core.lid = torch.nn.Linear(2, 2)
    core.featurizer = torch.nn.Linear(2, 2)
    core.predictor = torch.nn.Linear(2, 2)
    core.joint = torch.nn.Linear(2, 2)
    core.blank_id = 13087

    def construct(*, policy, **kwargs):
        core.policy = policy
        return core

    monkeypatch.setattr(module, "NemotronASRCore", construct)
    monkeypatch.setattr(module, "PersistentStateLayerBase", lambda *a, **kw: None)
    return module.NemotronASRForRNNT(vllm_config=_runtime_config(hf))


@pytest.mark.parametrize("precision", [None, "float32"])
def test_declared_five_cadence_native_profile_starts_with_full_headroom(monkeypatch, precision):
    # @spec PORT-DEC-011, PORT-DEC-014, PORT-INT-002
    hf = _native_config(profile=FIVE_CADENCE_PROFILE, lookaheads=LOOKAHEADS, precision=precision)
    model = _construct_model(monkeypatch, hf)
    assert model._native_burst_handoff.max_tokens == 142
    assert model.get_model_state_cls().num_new_sampled_tokens_per_step == 1


@pytest.mark.parametrize(
    "profile,lookaheads",
    [
        (None, LOOKAHEADS),
        (LEGACY_PROFILE, LOOKAHEADS),
        (FIVE_CADENCE_PROFILE, (1,)),
        (FIVE_CADENCE_PROFILE, (0, 1, 3, 6)),
        (FIVE_CADENCE_PROFILE, tuple(reversed(LOOKAHEADS))),
        ("unknown-native-profile", (1,)),
    ],
)
def test_native_profile_requires_exact_explicit_declaration(monkeypatch, profile, lookaheads):
    # @spec PORT-DEC-011, PORT-DEC-014
    with pytest.raises(ValueError, match="native burst|native_burst"):
        _construct_model(monkeypatch, _native_config(profile=profile, lookaheads=lookaheads))


@pytest.mark.parametrize("profile", [LEGACY_PROFILE, FIVE_CADENCE_PROFILE])
def test_native_profile_cannot_enable_itself_when_boolean_is_false(monkeypatch, profile):
    # @spec PORT-DEC-011, PORT-DEC-014
    with pytest.raises(ValueError, match="native burst|native_burst"):
        _construct_model(monkeypatch, _native_config(profile=profile, enabled=False))


@pytest.mark.parametrize("profile,lookaheads", [(None, (1,)), (FIVE_CADENCE_PROFILE, LOOKAHEADS)])
def test_native_profiles_reject_actual_encoder_fp16_policy(monkeypatch, profile, lookaheads):
    # @spec PORT-DEC-014: engine dtype FP32 alone does not establish FP32 compute.
    with pytest.raises(ValueError, match="FP32|fp32|precision"):
        _construct_model(monkeypatch, _native_config(profile=profile, lookaheads=lookaheads, precision="float16"))


@pytest.mark.parametrize(
    "enabled,profile,lookaheads,lane",
    [
        (False, None, LOOKAHEADS, "compatibility_replay"),
        (True, None, (1,), "native_burst"),
        (True, LEGACY_PROFILE, (1,), "native_burst"),
        (True, FIVE_CADENCE_PROFILE, LOOKAHEADS, "native_burst"),
    ],
)
def test_execution_receipt_retains_immutable_emission_selection(monkeypatch, enabled, profile, lookaheads, lane):
    # @spec PORT-DEC-011: later HF mutation cannot relabel an executing model.
    hf = _native_config(enabled=enabled, profile=profile, lookaheads=lookaheads)
    model = _construct_model(monkeypatch, hf)
    expected = {"lane": lane, "profile": profile or ("fp32-sync-single-device-160ms-v1" if enabled else None)}
    assert model.execution_profile_receipt()["emission"] == expected
    returned = model.execution_profile_receipt()
    returned["emission"]["profile"] = "caller-mutation"
    assert model.execution_profile_receipt()["emission"] == expected
    hf.experimental_native_burst = not enabled
    hf.experimental_native_burst_profile = "mutated-after-startup"
    hf.supported_num_lookahead_tokens = [13]
    assert model.execution_profile_receipt()["emission"] == expected


@pytest.mark.parametrize("lookahead,bound", list(zip(LOOKAHEADS, BOUNDS)))
def test_each_manifest_native_budget_has_optional_eou_and_park_headroom(lookahead, bound):
    # @spec PORT-INT-002
    from vllm.sampling_params import SamplingParams

    from vllm_omni.model_executor.models.nemotron_asr.native_burst import (
        native_burst_token_budget,
        validate_native_burst_sampling,
    )

    assert native_burst_token_budget(_native_config(lookaheads=(lookahead,))) == bound
    validate_native_burst_sampling(SamplingParams(max_tokens=bound), park_id=9000, capacity=bound - 1)
    with pytest.raises(ValueError, match="native burst"):
        validate_native_burst_sampling(SamplingParams(max_tokens=bound - 1), park_id=9000, capacity=bound - 1)


@pytest.mark.parametrize("mutation", ["block", "slot", "request", "residency"])
def test_native_handoff_rejects_changed_reservation_authority(mutation):
    # @spec PORT-DEC-012
    handoff = NativeBurstHandoff()
    snapshot = _snapshot()
    handoff.prepare(snapshot)
    if mutation == "block":
        snapshot.block_ids = ((4,),)
    elif mutation == "slot":
        snapshot.bindings[0].slot_id = 4
    elif mutation == "request":
        snapshot.bindings[0].request_id = "b"
    else:
        snapshot.req_states.req_id_to_index["a"] = 1
    with pytest.raises(RuntimeError, match="snapshot"):
        handoff.reserve(_payload())


@pytest.mark.parametrize("lookahead,bound", list(zip(LOOKAHEADS, BOUNDS)))
def test_each_geometry_rejects_small_budget_and_clamps_large_budget(lookahead, bound):
    # @spec PORT-INT-002: use the existing admission authority.
    from vllm_omni.model_executor.models.nemotron_asr.endpointing import validate_emission_budget

    with pytest.raises(ValueError):
        validate_emission_budget(caller_max_tokens=bound - 1, max_valid_frames=lookahead + 1, max_symbols_per_step=10)
    assert (
        validate_emission_budget(caller_max_tokens=1000, max_valid_frames=lookahead + 1, max_symbols_per_step=10)
        == bound
    )


def test_native_declaration_survives_hf_roundtrip_and_distinguishes_configuration_identity():
    # @spec PORT-DEC-011: effective declarations survive process boundaries.
    from vllm.config import ModelConfig

    from vllm_omni.model_executor.models.nemotron_asr.configuration_nemotron_asr import NemotronASRConfig

    configs = [
        _native_config(enabled=False),
        _native_config(),
        _native_config(profile=FIVE_CADENCE_PROFILE, lookaheads=LOOKAHEADS),
    ]
    fingerprints = []
    for config in configs:
        restored = NemotronASRConfig.from_dict(config.to_dict())
        assert restored.experimental_native_burst is config.experimental_native_burst
        assert restored.experimental_native_burst_profile == config.experimental_native_burst_profile
        assert restored.supported_num_lookahead_tokens == config.supported_num_lookahead_tokens
        runtime = object.__new__(ModelConfig)
        runtime.hf_config = restored
        runtime.multimodal_config = None
        fingerprints.append(runtime.compute_hash())
    assert len(set(fingerprints)) == 3


@pytest.mark.parametrize("profile", [None, LEGACY_PROFILE])
def test_unset_and_explicit_legacy_native_profile_keep_160ms_selection(monkeypatch, profile):
    # @spec PORT-DEC-011, PORT-DEC-014
    model = _construct_model(monkeypatch, _native_config(profile=profile))
    assert model._native_burst_handoff.max_tokens == 22


def test_configuration_hash_distinguishes_only_the_native_profile_field():
    # @spec PORT-DEC-011: no geometry or boolean difference can explain this hash.
    from vllm.config import ModelConfig

    absent = _native_config()
    explicit = _native_config(profile=LEGACY_PROFILE)
    assert {key for key in absent.to_dict() if absent.to_dict()[key] != explicit.to_dict()[key]} == {
        "experimental_native_burst_profile"
    }
    identities = []
    for config in (absent, explicit):
        runtime = object.__new__(ModelConfig)
        runtime.hf_config = config
        runtime.multimodal_config = None
        identities.append(runtime.compute_hash())
    assert identities[0] != identities[1]


@pytest.mark.parametrize(
    "profile,lookaheads", [(None, (1,)), (LEGACY_PROFILE, (1,)), (FIVE_CADENCE_PROFILE, LOOKAHEADS)]
)
def test_native_constructor_rejects_non_fp32_actual_policy_even_with_fp32_hf(monkeypatch, profile, lookaheads):
    # @spec PORT-DEC-014: validate the actual resolved policy, not only an HF knob.
    from vllm_omni.model_executor.models.nemotron_asr import nemotron_asr as module
    from vllm_omni.model_executor.models.nemotron_asr.precision import FP16_ENCODER_EXPERIMENT

    monkeypatch.setattr(module, "resolve_precision_policy", lambda config: FP16_ENCODER_EXPERIMENT)
    with pytest.raises(ValueError, match="FP32|fp32|precision"):
        _construct_model(monkeypatch, _native_config(profile=profile, lookaheads=lookaheads, precision="float32"))


@pytest.mark.parametrize("profile", [None, FIVE_CADENCE_PROFILE])
def test_chunk_graph_pilot_cannot_inherit_five_cadence_native_guard(monkeypatch, profile):
    # @spec PORT-DEC-011, PORT-DEC-014, PORT-PERF-009
    from vllm_omni.model_executor.models.nemotron_asr import chunk_bucket_graph

    def unexpected_binding(*args, **kwargs):
        pytest.fail("unqualified five-cadence CHUNK pilot reached graph binding")

    monkeypatch.setattr(chunk_bucket_graph, "ExactChunkGraphBinding", unexpected_binding)
    hf = _native_config(enabled=False, profile=profile, lookaheads=LOOKAHEADS)
    hf.chunk_bucket_graph_populations = [1]
    with pytest.raises(ValueError, match="160-ms|native burst|native_burst|CHUNK"):
        _construct_model(monkeypatch, hf)


@pytest.mark.parametrize("precision", [None, "float16"])
def test_chunk_graph_pilot_keeps_160ms_compatibility_selection(monkeypatch, precision):
    # @spec PORT-DEC-011, PORT-PERF-009: shared guard must not require native opt-in.
    from vllm_omni.model_executor.models.nemotron_asr import chunk_bucket_graph

    binding = object()
    monkeypatch.setattr(chunk_bucket_graph, "ExactChunkGraphBinding", lambda *args, **kwargs: binding)
    hf = _native_config(enabled=False, lookaheads=(1,), precision=precision)
    hf.chunk_bucket_graph_populations = [1]
    model = _construct_model(monkeypatch, hf)
    assert model._native_burst_handoff is None
    assert model._chunk_bucket_binding is binding
    assert model.core.policy.dtype_for("encoder_compute") == (
        torch.float16 if precision == "float16" else torch.float32
    )


@pytest.mark.parametrize("typed", [False, True], ids=["legacy-stage", "typed-stage"])
def test_five_cadence_model_accepts_real_stage_defaults_at_worker_admission(monkeypatch, typed):
    # @spec PORT-DEC-011, PORT-DEC-012, PORT-INT-002
    from tests.model_executor.models.nemotron_asr.test_emission_sampling_budget import (
        _effective_stage_sampling,
        _request,
    )

    hf = _native_config(profile=FIVE_CADENCE_PROFILE, lookaheads=LOOKAHEADS)
    model = _construct_model(monkeypatch, hf)
    state = object.__new__(model.get_model_state_cls())
    state.model = model
    state.model_config = SimpleNamespace(max_model_len=143)
    effective, _, _ = _effective_stage_sampling(typed)
    state.validate_omni_request(_request(effective))
    assert effective.max_tokens == model._native_burst_handoff.max_tokens == 142
    assert state.num_new_sampled_tokens_per_step == 1
