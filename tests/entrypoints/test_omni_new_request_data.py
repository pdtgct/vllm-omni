from types import SimpleNamespace

import pytest
import torch

from vllm_omni.core.sched.output import OmniNewRequestData

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_omni_new_request_data_copies_payloads():
    prompt_embeds = torch.randn(2, 3)
    additional_information = {
        "speaker": ["test"],
        "codes": torch.tensor([1, 2], dtype=torch.int64),
    }
    request = SimpleNamespace(
        request_id="req-1",
        external_req_id="ext-1",
        prompt_token_ids=[101, 102],
        mm_features=None,
        sampling_params=None,
        pooling_params=None,
        num_computed_tokens=0,
        lora_request=None,
        prompt_embeds=prompt_embeds,
        additional_information=additional_information,
    )

    data = OmniNewRequestData.from_request(request, ([0, 1],), prefill_token_ids=[101, 102])

    assert data.prompt_embeds is prompt_embeds
    assert data.additional_information is additional_information
    assert data.prefill_token_ids == [101, 102]


def test_omni_new_request_data_allows_missing_payloads():
    request = SimpleNamespace(
        request_id="req-2",
        external_req_id="ext-2",
        prompt_token_ids=[201, 202],
        mm_features=None,
        sampling_params=None,
        pooling_params=None,
        num_computed_tokens=0,
        lora_request=None,
        prompt_embeds=None,
        additional_information=None,
    )

    data = OmniNewRequestData.from_request(request, ([0],), prefill_token_ids=None)

    assert data.prompt_embeds is None
    assert data.additional_information is None


def test_omni_new_request_data_rewrap_preserves_v2_prefill_tokens():
    # @spec PORT-MIG-005 / PORT-MIG-006
    prefill_token_ids = [301, 302, 303]
    base = SimpleNamespace(
        req_id="req-v2",
        prompt_token_ids=[301],
        mm_features=[],
        sampling_params=None,
        pooling_params=None,
        block_ids=([7],),
        num_computed_tokens=0,
        lora_request=None,
        prompt_embeds=None,
        prompt_is_token_ids=[True],
        prefill_token_ids=prefill_token_ids,
    )
    request = SimpleNamespace(
        external_req_id="external-v2",
        prompt_embeds=None,
        additional_information={"source": "test"},
    )

    data = OmniNewRequestData.from_new_request_data(base, request)

    assert data.prefill_token_ids is prefill_token_ids
    assert data.external_req_id == "external-v2"
    assert data.additional_information == {"source": "test"}
