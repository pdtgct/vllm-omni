# Fusion pre-extraction CPU baseline — 2026-10-15 local

**24 PASS; zero failures, errors or skips.** Existing 13 test families expanded to 24 cases; pytest elapsed 0.42 s, 14 pre-existing deprecation warnings. Exit 0. This is a CPU known-expected/extraction baseline, not CUDA compilation, trained-model, performance or qualification evidence. No new test or production file was written.

## Identity and environment

- Engine HEAD: `f931acd3a696dd4f7dcd1ab41b2291d1eb37dfde`; Git tree: `907eea322f4cd27df8b4ed8870e2d7181843af85`. Clean before work and after this run; unchanged f931 production and tests.
- Python 3.12.10; torch 2.11.0; pytest 9.1.1; local macOS CPU fixtures. Actual target Python3.10/torch2.13/CUDA remains a separate gate.
- Imported Omni package resolved to the exact engine worktree. Core source resolves to `/Users/pete/Documents/Programming/github/pdtgct/.worktrees/vllm-core/v0.30.0`, HEAD `ced6857afa0ea7b2e3f0846a62e1394e90f15607`.
- Local import emits existing unavailable Triton/CPU-extension and duplicate AV-library diagnostics; selected pure tensor fixtures passed. No weights, audio, credentials, network or GPU/provider actions.
- Attempt1 exited4 before collection: configured `--dist=loadgroup` is unavailable without pytest-xdist in this existing environment. Same selected families rerun with `-o addopts=`; no packages or source changed. Both attempts retained.

## Independent assertions preserved

- Corrupt book negative head/oversized length/bad last label, nonnegative corrupt counters and cadence/sequence mismatch: PARK, BOOK_INVARIANT and complete resident pools unchanged.
- Lost incoming transition/adapter status: original ENVELOPE retained plus DECODE_INVARIANT; adapter case PARK/no-store.
- Malformed decisions: NaN, fractional, out-of-range, carrier-tail garbage, echo disagreement and coherent wrong substitution each PARK/DECODE_INVARIANT/no-store. Infinity is not an existing selected case and is not claimed by this baseline.
- Malicious adapter clearing status or mutating FLUSH book/REPLAY queue: nonzero failure, PARK and original pools unchanged, using independent retained authority.
- Proposed book negative-head/armed-blank defects: BOOK_INVARIANT/PARK and unchanged pools.
- Fresh poison: never reads fresh NaN storage; exact zero/fill7 expected tensors, only permitted continuing slot read, independently owned output.
- Failed native row: exact `[PARK,-1,-1,-1]`, count1, unchanged queue/book/status.
- Valid adapter CHUNK/zero burst: explicit first-label/armed-echo versus PARK/unarmed expectations.
- Malformed capture shape: existing precommit-fatal assertion; capture structural rejection remains outside fusion.

## Exact expanded node IDs

All names below are prefixed by `tests/model_executor/models/nemotron_asr/test_advance_model_rows_local.py::`:

- `test_corrupt_book_masks_safely[negative_head]`
- `test_corrupt_book_masks_safely[oversized_length]`
- `test_corrupt_book_masks_safely[bad_last_label]`
- `test_nonnegative_corrupt_replay_counters_mask_without_store`
- `test_replay_counter_sequence_must_match_geometry_cadence`
- `test_transition_validation_preserves_incoming_status_bits`
- `test_adapter_validation_preserves_incoming_status_bits`
- `test_malformed_adapter_decision_masks_without_store[nan]`
- `test_malformed_adapter_decision_masks_without_store[fractional]`
- `test_malformed_adapter_decision_masks_without_store[out_of_range]`
- `test_malformed_adapter_decision_masks_without_store[tail_garbage]`
- `test_malformed_adapter_decision_masks_without_store[echo_disagreement]`
- `test_malformed_adapter_decision_masks_without_store[coherent_substitution]`
- `test_adapter_cannot_mutate_transaction_authority_in_place[clear_status]`
- `test_adapter_cannot_mutate_transaction_authority_in_place[flush_book]`
- `test_adapter_cannot_mutate_transaction_authority_in_place[replay_queue]`
- `test_adapter_proposed_book_defect_masks_without_store[negative_head]`
- `test_adapter_proposed_book_defect_masks_without_store[armed_blank]`
- `test_gather_fresh_poison_is_never_read[False]`
- `test_gather_fresh_poison_is_never_read[True]`
- `test_experimental_native_burst_failed_row_emits_only_park_without_state_change`
- `test_adapter_chunk_burst_first_label_and_armed_echo`
- `test_adapter_zero_burst_parks_without_echo`
- `test_capture_shape_mismatch_is_precommit_fatal`

## Executed successful command

Working directory: `/private/tmp/vllm-omni-cadence80-floor-20261015`.

```sh
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/private/tmp/vllm-omni-cadence80-floor-20261015:/Users/pete/Documents/Programming/github/pdtgct/.worktrees/vllm-core/v0.30.0 /Users/pete/Documents/Programming/github/pdtgct/vllm-omni/.venv/bin/python -m pytest -o addopts= -q -p no:cacheprovider --junitxml=/private/tmp/pr341-followthrough-20261015/fusion-pre-extraction-baseline.xml tests/model_executor/models/nemotron_asr/test_advance_model_rows_local.py::test_corrupt_book_masks_safely tests/model_executor/models/nemotron_asr/test_advance_model_rows_local.py::test_nonnegative_corrupt_replay_counters_mask_without_store tests/model_executor/models/nemotron_asr/test_advance_model_rows_local.py::test_replay_counter_sequence_must_match_geometry_cadence tests/model_executor/models/nemotron_asr/test_advance_model_rows_local.py::test_transition_validation_preserves_incoming_status_bits tests/model_executor/models/nemotron_asr/test_advance_model_rows_local.py::test_adapter_validation_preserves_incoming_status_bits tests/model_executor/models/nemotron_asr/test_advance_model_rows_local.py::test_malformed_adapter_decision_masks_without_store tests/model_executor/models/nemotron_asr/test_advance_model_rows_local.py::test_adapter_cannot_mutate_transaction_authority_in_place tests/model_executor/models/nemotron_asr/test_advance_model_rows_local.py::test_adapter_proposed_book_defect_masks_without_store tests/model_executor/models/nemotron_asr/test_advance_model_rows_local.py::test_gather_fresh_poison_is_never_read tests/model_executor/models/nemotron_asr/test_advance_model_rows_local.py::test_experimental_native_burst_failed_row_emits_only_park_without_state_change tests/model_executor/models/nemotron_asr/test_advance_model_rows_local.py::test_adapter_chunk_burst_first_label_and_armed_echo tests/model_executor/models/nemotron_asr/test_advance_model_rows_local.py::test_adapter_zero_burst_parks_without_echo tests/model_executor/models/nemotron_asr/test_advance_model_rows_local.py::test_capture_shape_mismatch_is_precommit_fatal
```

## SHA256 custody

- `tests/model_executor/models/nemotron_asr/test_advance_model_rows_local.py`: `80728f28db61857461bccee4d529da5b15b7f910bebe572d302107a2a54b9050`
- `vllm_omni/model_executor/models/nemotron_asr/advance.py`: `8596f366587e23b3e634df4f1b4bf78257c4faee8e6d2289f0acf7daf839338b`
- `vllm_omni/model_executor/models/nemotron_asr/native_burst.py`: `a14e3b9a4c731108525947f7942d449e44480ae2f033410d7543b54e2c338841`
- `fusion-pre-extraction-baseline.xml`: `7533a48a43a6ba7e8569602a6dcb05ab569b630b42400b34739146498c76e445`
- `fusion-pre-extraction-baseline.log`: `bf39e137db95328c8611d0b6874958ccc0a95f5deafaff70b6b60284320d468d`
- `fusion-pre-extraction-baseline-attempt1.log`: `beca04a803f9a56f75009fdcba3584cb9e9e661433700d8e501431ca92d5aa59`

Only this compact sanitized receipt and its hash need accompany the eventual tests-only artifact. Raw logs/XML remain scratch evidence; no generated machinery or large repository logs are needed. Preserve the baseline milestone before extraction and compare future known-expected and full-pool results against the existing assertions; shared extracted-helper equivalence alone is insufficient.
