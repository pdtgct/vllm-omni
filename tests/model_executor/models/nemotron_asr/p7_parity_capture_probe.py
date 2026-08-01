# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Task-7 Phase 1: PORT-side parity capture probe (venv-port, dev pod).

Drives a real published checkpoint through the REAL transaction
(``advance_model_rows``, a real ``plan.SessionRegistry``/
``commit_sink.BoundedCommitSink``/``advance.HostStaging`` — the
``p6c_full_turn_probe.py`` pattern) across a full streaming session at
each of the five cadences, with ``capture=True``, and writes the
committed captures into the harness's ``GoldenManifest`` on-disk
schema (``nemotron_omni_port.eval.manifest``) so a later, separate
``make parity CAPTURE=<out>`` run can compare them against the pinned
NeMo-oracle goldens (EVAL-PAR-001/002/003/010).

POD-TIER, CUDA-only, real weights: cannot be collected on macOS or run
without a published checkpoint + GPU (see ``p6c_full_turn_probe.py``'s
own note on the same constraint). This file is the ONLY thing in the
Task-7 probe pair (see also ``p7_capture_manifest.py``) that touches
``torch``/``vllm_omni`` — the chunk-record/delta-geometry arithmetic
lives in the sibling module specifically so it stays locally testable.

Design decisions (Task-7 plan, ledger 2026-07-20 Phase-0 entry):

1. **Direct ``advance_model_rows`` probe, not the real serving API.**
   ``capture=True`` is model-internal; the OpenAI-compatible surface
   never exposes it and ``forward()`` hardcodes ``capture=False``.
   Toggling capture in production serving is a separate, riskier
   decision this probe does not need.
2. **Two-venv boundary respected**: this file does NOT import
   ``nemotron_omni_port`` (the harness package, a different repo/venv)
   — it treats the on-disk ``manifest.json``/``tensors.safetensors``
   schema as a stable file-format contract, the same principle
   env-design.md already applies to goldens (``venv-port``/
   ``venv-oracle`` "coupled ONLY through golden files").
3. **One real session per cadence, batch size 1.** Simplest correct
   design for a first end-to-end pass; multi-session/concurrent
   capture is out of scope here (Phase 2 follow-up territory if ever
   needed).
4. **CHUNK-then-REPLAY-drain per unit.** ``advance_model_rows`` returns
   ONE token id per row per call under the MRV1 adapter (never a
   burst) — the adapter's docstring: "the first burst label as each
   CHUNK row's decision carrier..., the next queued label for a
   validated REPLAY row... park_id for drained[...] rows"
   (advance.py:695-717). So one CHUNK turn's full label burst is only
   observable by repeatedly re-calling with a REPLAY row that echoes
   back the previously emitted id (PORT-DEC-007) until ``park_id``
   comes back — exactly ``p6c_full_turn_probe.py``'s own turn1->turn2
   pattern, generalized into a loop bounded by the queue capacity
   (a bug producing an infinite drain must raise, never hang).

Mel/encoder delta-slicing (why a separate translation module exists at
all) is documented in ``p7_capture_manifest.py``'s module docstring —
read that first if the ``mel_length``/``valid_feature_frames``
distinction below looks surprising.

Usage:
    /opt/venv-port/bin/python p7_parity_capture_probe.py \\
        --checkpoint /workspace/weights/served-nemotron-asr \\
        --clip /workspace/datasets/clips/en-US_sample.wav \\
        --target-lang en-US \\
        --seed 42 \\
        --model-revision <served checkpoint's own revision/tag> \\
        --nemo-commit de242add77945a110568c7c44bdae4891451851e \\
        --out /workspace/evidence/p7-parity-capture/<clip_id> \\
        [--cadences 80ms,160ms,320ms,560ms,1120ms] [--device cuda]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import subprocess
import sys
import time
import traceback
import wave
from pathlib import Path
from typing import Any

import numpy as np
import torch
from p7_capture_manifest import (
    CheckpointIdentity,
    build_chunk_record,
    build_manifest,
    chunk_delta_geometry,
    load_checkpoint_identity,
    sha256_file,
    validate_capture_qualification_inputs,
)
from safetensors.torch import load_file, save_file

from vllm_omni.model_executor.models.nemotron_asr import advance, manifests
from vllm_omni.model_executor.models.nemotron_asr import commit_sink as commit_sink_mod
from vllm_omni.model_executor.models.nemotron_asr import frontend as frontend_mod
from vllm_omni.model_executor.models.nemotron_asr import plan as plan_mod
from vllm_omni.model_executor.models.nemotron_asr.lid import resolve_prompt_index
from vllm_omni.model_executor.models.nemotron_asr.nemotron_asr import NemotronASRCore
from vllm_omni.model_executor.models.nemotron_asr.precision import FP32_BRINGUP

#: Fixed FastConformer architecture constants — NOT checkpoint fields
#: (unlike n_layers/d_model/etc., which come from config.json). Mirrors
#: p6c_full_turn_probe.py's PROD_KERNEL/RAW_TAIL/n_mels literals.
CONV_KERNEL = 9
N_MELS = 128
RAW_TAIL = 1_953
NULL_BLOCK_ID = 0
#: Safety bound on the CHUNK-then-REPLAY-drain loop: a correct session
#: never needs more replay turns than the queue can hold in one chunk
#: (manifests.SESSION_LIMITS["queue_capacity"], PORT-DEC-005's bound);
#: this only guards against looping forever on a real defect.
_DRAIN_SAFETY_MARGIN = 4

BOOK_WIDTH = len(manifests.BOOK_FIELDS)
CTR_WIDTH = len(manifests.FRONTEND_COUNTER_FIELDS)


def _read_wav(path: Path) -> tuple[torch.Tensor, int]:
    """16-bit mono PCM WAV -> normalized float32 samples (p2/p3's own
    helper, duplicated per this test tree's no-cross-probe-import
    convention)."""
    with wave.open(str(path), "rb") as handle:
        assert handle.getsampwidth() == 2 and handle.getnchannels() == 1
        rate = handle.getframerate()
        pcm = np.frombuffer(handle.readframes(handle.getnframes()), dtype=np.int16)
    return torch.from_numpy(pcm.astype(np.float32) / 32768.0), rate


def _fresh_pools(
    *, n_layers: int, window: int, d_model: int, pred_hidden: int, cap: int, num_blocks: int, device: str
) -> dict[str, Any]:
    """Mirrors p6c_full_turn_probe.py's ``_fresh_pools`` at real-checkpoint
    scale (n_layers/window/d_model/pred_hidden/cap sourced from the
    checkpoint config + manifests.SESSION_LIMITS, not hardcoded)."""
    return {
        "channel_pools": [torch.zeros(num_blocks, window, d_model, device=device) for _ in range(n_layers)],
        "time_pools": [torch.zeros(num_blocks, d_model, CONV_KERNEL - 1, device=device) for _ in range(n_layers)],
        "len_pools": [torch.zeros(num_blocks, 1, dtype=torch.int32, device=device) for _ in range(n_layers)],
        "h_pool": torch.zeros(num_blocks, 2, pred_hidden, device=device),
        "c_pool": torch.zeros(num_blocks, 2, pred_hidden, device=device),
        "queue_pool": torch.zeros(num_blocks, cap, dtype=torch.int32, device=device),
        "book_pool": torch.zeros(num_blocks, BOOK_WIDTH, dtype=torch.int32, device=device),
        "frontend_raw_pool": torch.zeros(num_blocks, RAW_TAIL, device=device),
        "frontend_mel_pool": torch.zeros(num_blocks, N_MELS, frontend_mod.MEL_TAIL_FRAMES, device=device),
        "frontend_counter_pool": torch.zeros(num_blocks, CTR_WIDTH, dtype=torch.int64, device=device),
    }


def _header(
    *, valid: int, geometry: int, final: bool, prompt: int, seq: int, admission_ms_mod: int = 0
) -> tuple[float, ...]:
    return (
        float(advance.ENVELOPE_VERSION),
        float(valid),
        float(geometry),
        1.0 if final else 0.0,
        float(prompt),
        float(seq),
        float(admission_ms_mod),
    )


def _carrier(
    samples: torch.Tensor, *, final: bool, seq: int, geometry: int, prompt: int, hidden: int, device: str
) -> torch.Tensor:
    n = samples.shape[0]
    row = torch.zeros(hidden)
    row[advance.ENV_VERSION] = advance.ENVELOPE_VERSION
    row[advance.ENV_VALID_SAMPLES] = n
    row[advance.ENV_GEOMETRY_ID] = geometry
    row[advance.ENV_FINAL_TAIL] = 1.0 if final else 0.0
    row[advance.ENV_PROMPT_INDEX] = prompt
    row[advance.ENV_CHUNK_SEQUENCE] = seq
    row[advance.ENV_ADMISSION_MS_MOD] = 0
    row[advance.ENVELOPE_HEADER_SLOTS : advance.ENVELOPE_HEADER_SLOTS + n] = samples
    return row.to(device)


def _git_tree_identity(start: Path) -> tuple[str, bool]:
    """Return the harness-compatible tracked-content digest and cleanliness.

    This reproduces ``eval.fingerprint.source_tree_identity`` without
    importing across the two-venv boundary: SHA-256 over ``git ls-files
    -s`` plus an exact porcelain-status cleanliness check.
    """
    try:
        tracked = subprocess.run(
            ["git", "ls-files", "-s"],
            cwd=start,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=start,
            capture_output=True,
            text=True,
            check=True,
        ).stdout
        digest = hashlib.sha256(tracked.encode("utf-8")).hexdigest()
        return f"sha256:{digest}", status.strip() == ""
    except (OSError, subprocess.CalledProcessError):
        return "unknown", False


def _build_execution_fingerprint(
    *,
    device: str,
    checkpoint_identity: CheckpointIdentity,
    repo_root: Path,
    container_image_digest: str,
    venv_freeze_hash: str,
) -> dict[str, Any]:
    """Build a fully checkpoint-bound, adoption-eligible fingerprint."""
    tree_digest, tree_clean = _git_tree_identity(repo_root)
    is_cuda = device == "cuda"
    return {
        "platform_backend": "cuda" if is_cuda else "cpu",
        "device_identity": torch.cuda.get_device_name(0) if is_cuda else "cpu",
        "compute_capability": (".".join(str(v) for v in torch.cuda.get_device_capability(0)) if is_cuda else ""),
        "python_build": ".".join(str(v) for v in sys.version_info[:3]),
        "torch_build": torch.__version__,
        "accel_library_versions": ([["cuda", str(torch.version.cuda)]] if is_cuda and torch.version.cuda else []),
        "tf32_matmul_enabled": torch.backends.cuda.matmul.allow_tf32,
        "tf32_cudnn_enabled": torch.backends.cudnn.allow_tf32,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "matmul_precision": torch.get_float32_matmul_precision(),
        "autocast_mode": "off",
        "container_image_digest": container_image_digest,
        "venv_freeze_hash": venv_freeze_hash,
        "source_tree_digest": tree_digest,
        "source_tree_clean": tree_clean,
        # The harness compatibility key defines this as the shared
        # source checkpoint, not the derived PORT safetensors file.
        "model_artifact_digest": checkpoint_identity.source_checkpoint_digest,
        "derived_model_digest": checkpoint_identity.derived_model_digest,
        "precision_policy_id": FP32_BRINGUP.identifier,
        "precision_policy_hash": FP32_BRINGUP.content_hash,
        "checkpoint_profile_id": checkpoint_identity.checkpoint_profile_id,
        "checkpoint_profile_hash": checkpoint_identity.checkpoint_profile_hash,
        "state_manifest_hash": checkpoint_identity.manifest_hashes["state"],
        "geometry_manifest_hash": checkpoint_identity.manifest_hashes["geometry"],
        "transition_manifest_hash": checkpoint_identity.manifest_hashes["transition"],
        "emission_manifest_hash": checkpoint_identity.manifest_hashes["emission"],
    }


def _load_config(checkpoint_dir: Path) -> dict[str, Any]:
    cfg: dict[str, Any] = json.loads((checkpoint_dir / "config.json").read_text())
    return cfg


def _load_core(cfg: dict[str, Any], checkpoint_dir: Path, device: str) -> NemotronASRCore:
    """Real weights, real dims — mirrors NemotronASRForRNNT.__init__'s
    own hf_config -> NemotronASRCore mapping (nemotron_asr.py:445-460)
    AND its load_weights' exact persistent-state-only consume logic
    (nemotron_asr.py:566-620), reproduced directly here (not via
    ``nn.Module.load_state_dict``, whose missing/unexpected semantics
    aren't guaranteed identical) rather than constructing the full
    vLLM-coupled wrapper class, which needs a real ``vllm_config``.

    Deliberately NOT p2_parity_probe.py's ``load_into`` convention:
    that probe loads the featurizer's filterbank/window separately at
    MelFeaturizer construction time from the raw ``.nemo`` dump, never
    through ``load_state_dict`` — so its ``fb``/``window`` exclusion
    from "real missing" does not apply here. This probe zero-
    placeholders them at construction (matching NemotronASRCore's own
    convention, see its ``filterbank``/``window`` params) and expects
    the PUBLISHED checkpoint's persistent state to overwrite them like
    every other tensor; only a genuinely non-persistent buffer (e.g.
    ``pos_enc.pe``, an init-computed sinusoid never persisted by NeMo)
    is legitimately absent from the checkpoint.
    """
    core = NemotronASRCore(
        vocab_size=int(cfg["num_asr_labels"]),
        att_context=(int(cfg["att_context_left"]), int(cfg["att_context_right"])),
        enc_hidden=int(cfg["d_model"]),
        n_layers=int(cfg["n_layers"]),
        pred_hidden=int(cfg["pred_hidden"]),
        pred_rnn_layers=int(cfg["pred_rnn_layers"]),
        joint_hidden=int(cfg["joint_hidden"]),
        num_prompts=int(cfg["num_prompts"]),
        filterbank=torch.zeros(N_MELS, 512 // 2 + 1),
        window=torch.zeros(400),
        policy=FP32_BRINGUP,
    ).to(device)
    core.eval()
    weights = load_file(str(checkpoint_dir / "model.safetensors"))
    expected: dict[str, torch.Tensor] = dict(core.named_parameters())
    expected.update(core.named_buffers())
    # required == state_dict() keys, which already excludes any
    # persistent=False buffer (pos_enc.pe) — no name-suffix guessing.
    required = set(core.state_dict())
    consumed: set[str] = set()
    for name, tensor in weights.items():
        if name not in expected:
            raise RuntimeError(f"unexpected checkpoint weight {name!r}: not in the model")
        target = expected[name]
        if tuple(target.shape) != tuple(tensor.shape):
            raise RuntimeError(
                f"shape mismatch for {name!r}: expected {tuple(target.shape)}, got {tuple(tensor.shape)}"
            )
        with torch.no_grad():
            target.copy_(tensor.to(device))
        consumed.add(name)
    missing = required - consumed
    if missing:
        raise RuntimeError(f"{len(missing)} expected checkpoint weights not provided, e.g. {sorted(missing)[:5]}")
    return core


def _set_determinism(seed: int) -> None:
    """Match the oracle lane's deterministic seed/math contract."""
    required_workspace = ":4096:8"
    configured_workspace = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    if configured_workspace not in (None, required_workspace):
        raise ValueError(f"conflicting CUBLAS_WORKSPACE_CONFIG: {configured_workspace!r} != {required_workspace!r}")
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = required_workspace
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _detokenize(checkpoint_dir: Path, label_ids: list[int]) -> str:
    """Sentencepiece direct decode (p2_parity_probe.py's own approach,
    duplicated per this tree's no-cross-probe-import convention) — the
    published checkpoint is expected to carry ``tokenizer.model``
    alongside the HF-format tokenizer assets (PORT-WGT-001/004); if a
    pod round finds it under a different name this is exactly the kind
    of Phase-2 finding to fix here."""
    import sentencepiece as spm

    model_path = checkpoint_dir / "tokenizer.model"
    if not model_path.exists():
        raise RuntimeError(
            f"{model_path} not found — published checkpoint tokenizer layout "
            "assumption needs revisiting (see _detokenize's docstring)"
        )
    sp = spm.SentencePieceProcessor(str(model_path))
    return str(sp.decode(label_ids))


class _SessionDriver:
    """One real streaming session, batch size 1, for one cadence.

    Owns the registry/pools/sink/staging and the CHUNK-then-REPLAY-
    drain loop; ``run_chunk`` drives exactly one CHUNK (regular or
    final-tail) to legal park and returns its committed capture plus
    the cumulative label sequence so far.
    """

    def __init__(
        self,
        *,
        core: NemotronASRCore,
        cfg: dict[str, Any],
        geometry_id: int,
        request_id: str,
        device: str,
    ) -> None:
        self.core = core
        self.device = device
        self.park_id = int(cfg["eos_token_id"])
        self.placeholder_id = int(cfg["audio_chunk_token_id"])
        self.hidden = int(cfg["hidden_size"])
        self.request_id = request_id
        self.geometry_id = geometry_id
        num_blocks = 2
        self.num_blocks = num_blocks
        self.pools = _fresh_pools(
            n_layers=int(cfg["n_layers"]),
            window=int(cfg["att_context_left"]),
            d_model=int(cfg["d_model"]),
            pred_hidden=int(cfg["pred_hidden"]),
            cap=int(manifests.SESSION_LIMITS["queue_capacity"]),
            num_blocks=num_blocks,
            device=device,
        )
        advance.warmup_advance_model_rows_scatter(**self.pools)
        self.registry = plan_mod.SessionRegistry()
        self.sink = commit_sink_mod.BoundedCommitSink(
            self.registry,
            max_rows=1,
            max_capture_rows=1,
            max_capture_bytes=1 << 30,
            device=torch.device(device),
        )
        self.staging = advance.HostStaging(1)
        self.adapter = advance.make_mrv1_adapter(hidden_size=self.hidden, park_id=self.park_id, blank_id=core.blank_id)
        self.drain_cap = int(manifests.SESSION_LIMITS["queue_capacity"]) + _DRAIN_SAFETY_MARGIN
        self._now_ns = 0
        self._step = 0
        self._registered = False
        # Whether the registry has already admitted this session's
        # first CHUNK — False only for run_chunk's very first call
        # (Task-7 Phase-2 finding, see run_chunk's own comment: a
        # session is registered atomically AT its first minted CHUNK,
        # plan.py's SessionRegistry docstring — there is no separate
        # pre-admission step).
        self._admitted = False

    def _replay_turn(self, row: plan_mod.ObservedRow) -> tuple[int, list[advance.CaptureRecord]]:
        """One REPLAY-role turn: no audio (PORT-POOL-001 — only a
        fresh CHUNK carries a real carrier, built by ``run_chunk``
        itself), one decode row, no prefill."""
        self._now_ns += 1
        self._step += 1
        context = plan_mod.prepare_plan_context(
            self.registry,
            [row],
            resident_request_ids=[row.request_id],
            placeholder_id=self.placeholder_id,
            num_prompts=self._num_prompts,
            num_geometries=len(manifests.CADENCES),
            now_ns=self._now_ns,
            step=self._step,
        )
        rp = plan_mod.build_row_plan(
            context,
            num_decodes=1,
            num_prefills=0,
            null_block_id=NULL_BLOCK_ID,
            num_pool_blocks=self.num_blocks,
        )
        input_ids = torch.tensor([row.scheduled_token_id], dtype=torch.long, device=self.device)
        embeds = torch.zeros(1, self.hidden, device=self.device)
        projected = advance.advance_model_rows(
            self.core,
            input_ids,
            embeds,
            rp,
            adapter=self.adapter,
            decode_resolver=self._resolver,
            placeholder_id=self.placeholder_id,
            park_id=self.park_id,
            commit_sink=self.sink,
            capture=True,
            graph_covers_decode=False,
            staging=self.staging,
            **self.pools,
        )
        # advance_model_rows returns an (N, H) decision-carrier, not a
        # per-row scalar (advance.py:2057-2059) — the emitted label
        # lives at column 0 of each row (advance.py:818-824, "Decision
        # carrier: slot 0 of each runner row"); projected[0] alone is
        # the whole H-wide row, not a scalar (Task-7 Phase-2 finding).
        emitted = int(projected[0, 0].item())
        reports, records, lease_ok = self.sink.collect()
        if not lease_ok or reports[0].row_status != 0:
            raise RuntimeError(
                f"{self.request_id}: non-clean turn (row_status={reports[0].row_status}, "
                f"lease_ok={lease_ok}) — aborting rather than emitting a corrupt capture"
            )
        return emitted, records

    def register_fresh(self, *, geometry: int, prompt: int, num_prompts: int) -> None:
        """Record this session's admission parameters for later calls.

        Deliberately does NOT call ``registry.bind_rows`` itself (an
        earlier version did — Task-7 Phase-2 finding): the registry
        admits a session atomically at its first minted CHUNK
        (``SessionRegistry``'s own docstring), and that SAME call is
        also what ``advance_model_rows`` uses to decide a row needs
        fresh pool/book initialization (``RowPlan.has_initial_states_p``,
        sourced from the identical ``ObservedRow.has_prior_state`` flag
        — plan.py:643). A separate pre-registration step here would
        admit the session into the registry without ever running the
        transaction's fresh-row pool init, so the session's real first
        ``run_chunk`` call performs both at once instead (unlike
        ``p6c_full_turn_probe.py``'s own ``_register_fresh``, which
        pairs registry-only admission with MANUAL pool seeding to build
        a synthetic mid-stream fixture — this probe drives a real
        session-first CHUNK live, so it needs the real init to run).
        """
        del geometry, prompt  # carried per-call by run_chunk, not stored
        self._num_prompts = num_prompts
        self._registered = True

    def run_chunk(
        self,
        *,
        samples: torch.Tensor,
        seq: int,
        is_final_tail: bool,
        prompt: int,
    ) -> tuple[advance.CaptureRecord, list[int]]:
        """Drive one CHUNK (regular or final-tail) to legal park.

        Returns the CHUNK turn's own committed ``CaptureRecord`` (this
        function raises rather than returning if the sink yields
        anything other than exactly one — PORT-HOOK-001 guarantees one
        per CHUNK, so a different count is a hard failure, never a
        None to propagate) and every label emitted this chunk, in
        order (park excluded).
        """
        if not self._registered:
            raise RuntimeError("register_fresh must run before the first chunk")
        embeds_row = _carrier(
            samples,
            final=is_final_tail,
            seq=seq,
            geometry=self.geometry_id,
            prompt=prompt,
            hidden=self.hidden,
            device=self.device,
        )
        # has_prior_state is False on exactly this driver's first
        # run_chunk call (the session's real first CHUNK — the atomic
        # registry admission AND the transaction's fresh pool/book init
        # both key off this one flag, plan.py:643/advance.py:2228-2242)
        # and True on every later call (Task-7 Phase-2 finding: a
        # separate pre-registration step, tried first, admitted the
        # registry without ever running fresh-row pool init — see
        # register_fresh's docstring).
        row = plan_mod.ObservedRow(
            self.request_id,
            1,
            self.placeholder_id,
            self._admitted,
            _header(
                valid=int(samples.shape[0]), geometry=self.geometry_id, final=is_final_tail, prompt=prompt, seq=seq
            ),
        )
        self._admitted = True
        self._now_ns += 1
        self._step += 1
        context = plan_mod.prepare_plan_context(
            self.registry,
            [row],
            resident_request_ids=[self.request_id],
            placeholder_id=self.placeholder_id,
            num_prompts=self._num_prompts,
            num_geometries=len(manifests.CADENCES),
            now_ns=self._now_ns,
            step=self._step,
        )
        rp = plan_mod.build_row_plan(
            context,
            num_decodes=0,
            num_prefills=1,
            null_block_id=NULL_BLOCK_ID,
            num_pool_blocks=self.num_blocks,
        )
        input_ids = torch.tensor([self.placeholder_id], dtype=torch.long, device=self.device)
        projected = advance.advance_model_rows(
            self.core,
            input_ids,
            embeds_row.unsqueeze(0),
            rp,
            adapter=self.adapter,
            decode_resolver=self._resolver,
            placeholder_id=self.placeholder_id,
            park_id=self.park_id,
            commit_sink=self.sink,
            capture=True,
            graph_covers_decode=False,
            staging=self.staging,
            **self.pools,
        )
        # advance_model_rows returns an (N, H) decision-carrier, not a
        # per-row scalar (advance.py:2057-2059) — the emitted label
        # lives at column 0 of each row (advance.py:818-824, "Decision
        # carrier: slot 0 of each runner row"); projected[0] alone is
        # the whole H-wide row, not a scalar (Task-7 Phase-2 finding).
        emitted = int(projected[0, 0].item())
        reports, records, lease_ok = self.sink.collect()
        if not lease_ok or reports[0].row_status != 0:
            raise RuntimeError(
                f"{self.request_id} chunk {seq}: non-clean CHUNK turn "
                f"(row_status={reports[0].row_status}, lease_ok={lease_ok})"
            )
        if len(records) != 1:
            raise RuntimeError(
                f"{self.request_id} chunk {seq}: expected exactly one committed "
                f"capture record for a CHUNK row, got {len(records)} "
                "(PORT-HOOK-001 is violated or the sink was misconfigured)"
            )
        capture_record = records[0]
        labels: list[int] = []
        drained = 0
        current = emitted
        while current != self.park_id:
            labels.append(current)
            drained += 1
            if drained > self.drain_cap:
                raise RuntimeError(
                    f"{self.request_id} chunk {seq}: replay drain exceeded "
                    f"{self.drain_cap} turns without reaching park_id — "
                    "likely an echo/queue defect, not a slow-but-valid burst"
                )
            replay_row = plan_mod.ObservedRow(self.request_id, 1, current, True, None)
            current, _replay_records = self._replay_turn(replay_row)
        return capture_record, labels

    @staticmethod
    def _resolver(request: advance.DecodeRequest) -> advance.ResolvedDecode:
        from vllm_omni.model_executor.models.nemotron_asr import rnnt

        del request
        return advance.ResolvedDecode(arm="dense-eager", decode_fn=rnnt.decode_dense_masked)


def _run_cadence(
    *,
    core: NemotronASRCore,
    cfg: dict[str, Any],
    cadence: str,
    samples: torch.Tensor,
    prompt_index: int,
    device: str,
    checkpoint_dir: Path,
    clip_path: Path,
    clip_id: str,
    out_root: Path,
    model_revision: str,
    nemo_commit: str,
    repo_root: Path,
    container_image_digest: str,
    venv_freeze_hash: str,
    checkpoint_identity: CheckpointIdentity,
    seed: int,
) -> dict[str, Any]:
    geometry_id = list(manifests.CADENCES).index(cadence)
    att_context = manifests.CADENCES[cadence]
    cadence_samples = manifests.RAW_SAMPLES_PER_CHUNK[cadence]
    total = int(samples.shape[0])
    num_regular = total // cadence_samples
    residual = total - num_regular * cadence_samples

    driver = _SessionDriver(
        core=core,
        cfg=cfg,
        geometry_id=geometry_id,
        request_id=f"p7-{clip_id}-{cadence}",
        device=device,
    )
    driver.register_fresh(geometry=geometry_id, prompt=prompt_index, num_prompts=int(cfg["num_prompts"]))

    chunk_records: list[dict[str, Any]] = []
    chunk_timing_ms: list[float] = []
    partial_transcripts: list[str] = []
    tensors: dict[str, torch.Tensor] = {}
    cumulative_labels: list[int] = []

    def _commit_chunk(*, seq: int, chunk_samples: torch.Tensor, is_final_tail: bool) -> None:
        t0 = time.perf_counter()
        record, labels = driver.run_chunk(
            samples=chunk_samples,
            seq=seq,
            is_final_tail=is_final_tail,
            prompt=prompt_index,
        )
        chunk_timing_ms.append((time.perf_counter() - t0) * 1000.0)
        cumulative_labels.extend(labels)

        mel_length = int(record.mel_length.item())
        encoder_length = int(record.encoder_length.item())
        geom = chunk_delta_geometry(
            chunk_sequence=seq,
            mel_length=mel_length,
            encoder_length=encoder_length,
            is_final_tail=is_final_tail,
        )
        chunk_index = len(chunk_records)
        chunk_records.append(
            build_chunk_record(
                chunk_index=chunk_index,
                geometry=geom,
                encoder_length=encoder_length,
                is_final_tail=is_final_tail,
            )
        )
        if geom.has_tensors:
            mel = record.frontend_mel[:, geom.mel_start : geom.mel_start + geom.new_mel_frames]
            enc_raw = record.encoder_raw[:encoder_length, :].transpose(0, 1).contiguous()
            enc_cond = record.encoder_conditioned[:encoder_length, :].transpose(0, 1).contiguous()
            tensors[f"frontend_mel/{chunk_index:05d}"] = mel.to(torch.float32).cpu()
            tensors[f"encoder_raw/{chunk_index:05d}"] = enc_raw.to(torch.float32).cpu()
            tensors[f"encoder_conditioned/{chunk_index:05d}"] = enc_cond.to(torch.float32).cpu()
        partial_transcripts.append(_detokenize(checkpoint_dir, cumulative_labels))

    for i in range(num_regular):
        start = i * cadence_samples
        _commit_chunk(
            seq=i,
            chunk_samples=samples[start : start + cadence_samples],
            is_final_tail=False,
        )
    final_samples = samples[num_regular * cadence_samples :]
    assert int(final_samples.shape[0]) == residual
    _commit_chunk(seq=num_regular, chunk_samples=final_samples, is_final_tail=True)

    out_dir = out_root / clip_id / cadence
    out_dir.mkdir(parents=True, exist_ok=True)
    tensors_path = out_dir / "tensors.safetensors"
    save_file(tensors, str(tensors_path))
    tensors_digest = sha256_file(tensors_path)
    tensors_size_bytes = tensors_path.stat().st_size

    fingerprint = _build_execution_fingerprint(
        device=device,
        checkpoint_identity=checkpoint_identity,
        repo_root=repo_root,
        container_image_digest=container_image_digest,
        venv_freeze_hash=venv_freeze_hash,
    )
    manifest = build_manifest(
        model_revision=model_revision,
        nemo_commit=nemo_commit,
        precision_policy_id=FP32_BRINGUP.identifier,
        execution_fingerprint=fingerprint,
        seed=seed,
        cadence=cadence,
        clip_checksum=sha256_file(clip_path),
        att_context_size=att_context,
        tensors_digest=tensors_digest,
        tensors_size_bytes=tensors_size_bytes,
        partial_transcripts=partial_transcripts,
        final_transcript=partial_transcripts[-1] if partial_transcripts else "",
        chunk_timing_ms=chunk_timing_ms,
        chunk_records=chunk_records,
    )
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return {
        "cadence": cadence,
        "num_chunks": len(chunk_records),
        "final_transcript": manifest["final_transcript"],
        "tensors_size_bytes": tensors_size_bytes,
        "out_dir": str(out_dir),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--clip", type=Path, required=True)
    ap.add_argument("--target-lang", default="en-US")
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--model-revision", required=True)
    ap.add_argument("--nemo-commit", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument(
        "--cadences",
        default="80ms,160ms,320ms,560ms,1120ms",
        help="Comma-separated subset of the five PORT-EVAL cadences.",
    )
    ap.add_argument("--container-image-digest", required=True)
    ap.add_argument("--venv-freeze-hash", required=True)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    repo_root = Path(__file__).resolve()
    for _ in range(6):  # walk up to the fork root (tests/.../nemotron_asr/<file>)
        if (repo_root / ".git").exists():
            break
        repo_root = repo_root.parent
    _set_determinism(args.seed)
    _tree_digest, tree_clean = _git_tree_identity(repo_root)
    validate_capture_qualification_inputs(
        seed=args.seed,
        container_image_digest=args.container_image_digest,
        venv_freeze_hash=args.venv_freeze_hash,
        source_tree_clean=tree_clean,
        cublas_workspace_config=os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
    )
    if args.device == "cuda" and not torch.cuda.is_available():
        print("FATAL: --device cuda but no CUDA device", flush=True)
        sys.exit(2)

    checkpoint_identity = load_checkpoint_identity(args.checkpoint)
    cfg = _load_config(args.checkpoint)
    core = _load_core(cfg, args.checkpoint, args.device)
    prompt_index = resolve_prompt_index(checkpoint_identity.prompt_dictionary, args.target_lang)
    waveform, rate = _read_wav(args.clip)
    if rate != 16000:
        print(f"FATAL: clip sample rate {rate} != 16000", flush=True)
        sys.exit(2)
    clip_id = args.clip.stem
    cadences = [c.strip() for c in args.cadences.split(",") if c.strip()]
    for cadence in cadences:
        if cadence not in manifests.CADENCES:
            print(f"FATAL: unknown cadence {cadence!r}, expected one of {list(manifests.CADENCES)}", flush=True)
            sys.exit(2)

    args.out.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {"probe": "p7_parity_capture", "clip_id": clip_id, "cadences": {}}
    failures: list[str] = []
    for cadence in cadences:
        try:
            report["cadences"][cadence] = _run_cadence(
                core=core,
                cfg=cfg,
                cadence=cadence,
                samples=waveform,
                prompt_index=prompt_index,
                device=args.device,
                checkpoint_dir=args.checkpoint,
                clip_path=args.clip,
                clip_id=clip_id,
                out_root=args.out,
                model_revision=args.model_revision,
                nemo_commit=args.nemo_commit,
                repo_root=repo_root,
                container_image_digest=args.container_image_digest,
                venv_freeze_hash=args.venv_freeze_hash,
                checkpoint_identity=checkpoint_identity,
                seed=args.seed,
            )
        except Exception as err:  # a cadence's failure must not lose the others' evidence
            report["cadences"][cadence] = {
                "error": f"{type(err).__name__}: {err}",
                "traceback": traceback.format_exc(),
            }
            failures.append(cadence)
        (args.out / "report.json").write_text(json.dumps(report, indent=2, default=str))

    report["pass"] = not failures
    (args.out / "report.json").write_text(json.dumps(report, indent=2, default=str))
    print(json.dumps(report, indent=2, default=str))
    print(
        f"{'PASS' if report['pass'] else 'FAIL'}: {len(cadences) - len(failures)}/{len(cadences)} "
        f"cadences captured cleanly",
        flush=True,
    )
    sys.exit(0 if report["pass"] else 1)


if __name__ == "__main__":
    main()
