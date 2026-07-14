# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""α3 pod-tier: forced emission against the real engine seams.

Specs: PORT-DEC-002/005/007 (forced rows through vLLM's actual greedy
sampler path; the exclusion-mask negative on the real machinery) and
the resumable-stop seam PORT-DEC-003's park path rides. Runs on the
pod tiers (installed vLLM), not on machines without vllm — the α2
engine-binding pattern.
"""

import pytest
import torch
from vllm.v1.request import Request, RequestStatus

from vllm_omni.model_executor.models.nemotron_asr.rnnt import (
    forced_logits_rows,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

_NUM_LOGITS = 13089


def _greedy_metadata(batch: int):
    """SamplingMetadata as the model-pinned params produce it."""
    from vllm.v1.sample.metadata import SamplingMetadata

    return SamplingMetadata(
        temperature=None,  # all-greedy short-circuit
        all_greedy=True,
        all_random=False,
        top_p=None,
        top_k=None,
        generators={},
        max_num_logprobs=None,
        no_penalties=True,
        prompt_token_ids=None,
        frequency_penalties=torch.zeros(batch),
        presence_penalties=torch.zeros(batch),
        repetition_penalties=torch.ones(batch),
        output_token_ids=[[] for _ in range(batch)],
        allowed_token_ids_mask=None,
        bad_words_token_ids={},
        logitsprocs=None,
    )


def test_forced_rows_survive_the_real_greedy_sampler():
    # With the model-pinned SamplingParams every processor falls
    # through and all_greedy short-circuits to argmax — the single
    # finite logit is the emission (D-α3a, proven at the pin; this
    # test pins it against the installed wheel).
    from vllm.v1.sample.sampler import Sampler

    chosen = torch.tensor([7, 13088, 0], dtype=torch.long)
    rows = forced_logits_rows(chosen, num_logits=_NUM_LOGITS)
    out = Sampler().forward(rows, _greedy_metadata(3))
    assert torch.equal(
        out.sampled_token_ids.view(-1).cpu().long(), chosen
    )


def test_exclusion_mask_kills_forced_emission_on_the_real_sampler():
    # The PORT-DEC-007 negative on real machinery: an
    # allowed_token_ids mask that excludes the forced id destroys the
    # emission — the greedy pin is load-bearing, and must be re-applied
    # on every StreamingUpdate (the session-update path replaces
    # sampling_params wholesale).
    from vllm.v1.sample.sampler import Sampler

    chosen = torch.tensor([7], dtype=torch.long)
    rows = forced_logits_rows(chosen, num_logits=_NUM_LOGITS)
    meta = _greedy_metadata(1)
    mask = torch.ones(1, _NUM_LOGITS, dtype=torch.bool)
    mask[0, 7] = False  # mask=True means "not allowed" at the pin
    meta.allowed_token_ids_mask = mask
    out = Sampler().forward(rows, meta)
    assert int(out.sampled_token_ids.view(-1)[0]) != 7


def test_resumable_stop_seam_exists_for_the_park_path():
    # PORT-DEC-003 rides core's resumable stop: EOS →
    # _handle_stopped_request → WAITING_FOR_STREAMING_REQ, park token
    # discarded from the timeline. Seam guard in the α2 style: the
    # status member and the Request streaming surfaces must exist so a
    # core rename breaks loudly here, not at bring-up.
    assert hasattr(RequestStatus, "WAITING_FOR_STREAMING_REQ")
    assert hasattr(Request, "streaming_queue") or (
        "streaming_queue" in getattr(Request.__init__, "__doc__", "")
        or "streaming_queue"
        in Request.__init__.__code__.co_names
        + Request.__init__.__code__.co_varnames
    )
