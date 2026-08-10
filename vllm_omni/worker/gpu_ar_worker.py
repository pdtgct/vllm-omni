import gc
import os

import torch
from vllm.config.compilation import CompilationMode
from vllm.logger import init_logger
from vllm.model_executor.warmup.kernel_warmup import kernel_warmup
from vllm.platforms import current_platform
from vllm.tracing import instrument
from vllm.utils.gc_utils import freeze_gc_heap, maybe_attach_gc_debug_callback
from vllm.utils.gpu_sync_debug import enable_gpu_sync_check
from vllm.utils.mem_utils import MemorySnapshot, format_gib
from vllm.utils.torch_utils import set_random_seed
from vllm.v1.utils import report_usage_stats
from vllm.v1.worker.gpu_worker import init_worker_distributed_environment
from vllm.v1.worker.worker_base import CompilationTimes
from vllm.v1.worker.workspace import init_workspace_manager

from vllm_omni.diffusion.data import OmniACK, OmniSleepTask, OmniWakeTask
from vllm_omni.platforms import current_omni_platform
from vllm_omni.worker.base import OmniGPUWorkerBase
from vllm_omni.worker.gpu_ar_model_runner import GPUARModelRunner
from vllm_omni.worker.gpu_ar_model_runner_v2 import GPUARModelRunnerV2
from vllm_omni.worker.memory_utils import request_memory_tolerant
from vllm_omni.worker.mixins import OmniWorkerMixin

logger = init_logger(__name__)


class GPUARWorker(OmniWorkerMixin, OmniGPUWorkerBase):
    """GPU worker for autoregressive omni model stages.

    Extends the base GPUWorker to initialize and manage autoregressive
    model runners for text generation stages (e.g., thinker stages).
    """

    def _has_persistent_only_cache(self) -> bool:
        """Return whether this runner has state but no token-cache group."""

        storage = getattr(
            self.model_runner,
            "_persistent_state_storage",
            None,
        )
        config = getattr(self.model_runner, "kv_cache_config", None)
        groups = None if config is None else config.kv_cache_groups
        return storage is not None and groups == []

    def _has_persistent_cache(self) -> bool:
        """Return whether the runner owns an allocated state group."""

        return (
            getattr(
                self.model_runner,
                "_persistent_state_storage",
                None,
            )
            is not None
        )

    def persistent_state_warmup_attestation(self) -> bool:
        """Return whether this worker completed resident-scatter warmup."""

        return bool(getattr(self, "_persistent_state_warmup_complete", False))

    @instrument(span_name="Warmup persistent-only model (GPU)")
    def _compile_or_warm_up_persistent_only_model(self) -> CompilationTimes:
        """Finish eager worker warmup without inventing token-cache requests.

        Core MRv2's final ``warmup_kernels`` pass constructs synthetic text
        requests and divides capacity by their attention/Mamba block demand.
        A persistent-state-only model deliberately has no such cache group;
        its canonical maximum-shape profile has already executed the model and
        sampler. Preserve core's model-neutral kernel warmup and operational
        postamble while omitting only that incompatible synthetic-request pass.

        This specialization is intentionally limited to the already-enforced
        eager, non-compiled lane. A future execution-mode expansion must first
        qualify the corresponding core warmup/capture behavior.
        """

        # @spec PORT-ADV-003, ENV-MIG-012
        self._persistent_state_warmup_complete = False
        if not self.model_config.enforce_eager:
            raise RuntimeError(
                "persistent-only warmup requires eager execution"
            )
        if self.compilation_config.mode != CompilationMode.NONE:
            raise RuntimeError(
                "persistent-only warmup does not support model compilation"
            )

        self.model_runner.maybe_remove_all_loras(
            self.model_runner.lora_config
        )
        kernel_warmup(self)

        warmup_resident_state = getattr(
            self.model_runner.model,
            "warmup_resident_state",
            None,
        )
        if not callable(warmup_resident_state):
            raise RuntimeError(
                "persistent-only model does not expose resident-state warmup"
            )
        warmup_resident_state()
        self._persistent_state_warmup_complete = True

        # Profiling and warmup must not perturb request-time randomness.
        set_random_seed(self.model_config.seed)

        from vllm.utils.jit_monitor import activate as activate_jit_monitor

        activate_jit_monitor(
            mode=self.observability_config.jit_monitor_mode,
            verbose=self.observability_config.jit_monitor_verbose,
        )
        freeze_gc_heap()
        maybe_attach_gc_debug_callback()
        enable_gpu_sync_check()
        return CompilationTimes(
            language_model=self.compilation_config.compilation_time,
            encoder=self.compilation_config.encoder_compilation_time,
        )

    def compile_or_warm_up_model(self) -> CompilationTimes:
        """Warm ordinary runners normally and persistent-only runners natively."""

        if self._has_persistent_only_cache():
            runner = self.model_runner
            model = runner.model
            model_state = getattr(runner, "model_state", None)
            storage = runner._persistent_state_storage
            state_spec = getattr(storage, "spec", None)
            logger.info(
                "Persistent-state execution fingerprint: worker=%s "
                "runner=%s model=%s model_state=%s state_spec=%s "
                "is_hybrid=%s eager=%s",
                type(self).__name__,
                type(runner).__name__,
                type(model).__name__,
                type(model_state).__name__,
                type(state_spec).__name__,
                bool(getattr(type(model), "is_hybrid", False)),
                bool(self.model_config.enforce_eager),
            )
            return self._compile_or_warm_up_persistent_only_model()
        if self._has_persistent_cache():
            raise RuntimeError(
                "mixed persistent and token-cache warmup is not qualified"
            )
        return super().compile_or_warm_up_model()

    @instrument(span_name="Init device")
    def init_device(self):
        if self.device_config.device_type in ("cuda", "musa"):
            # This env var set by Ray causes exceptions with graph building.
            os.environ.pop("NCCL_ASYNC_ERROR_HANDLING", None)
            parallel_config = self.parallel_config
            if (
                parallel_config.distributed_executor_backend not in ("ray", "external_launcher")
                and parallel_config.data_parallel_backend != "ray"
                and parallel_config.nnodes_within_dp == 1
            ):
                # Use local DP rank if available, otherwise use global DP rank.
                dp_local_rank = self.parallel_config.data_parallel_rank_local
                if dp_local_rank is None:
                    dp_local_rank = self.parallel_config.data_parallel_index

                tp_pp_world_size = (
                    self.parallel_config.pipeline_parallel_size * self.parallel_config.tensor_parallel_size
                )

                # DP_LOCAL_RANK * TP_PP_WORLD_SIZE + TP_LOCAL_RANK
                self.local_rank += dp_local_rank * tp_pp_world_size

            # Publish the logical-to-physical mapping for topology queries
            # such as NIC affinity and P2P checks (upstream PR #45026).
            assigned_physical_gpu_ids = parallel_config.assigned_physical_gpu_ids
            if assigned_physical_gpu_ids is not None:
                from vllm.platforms.interface import set_assigned_physical_gpu_ids

                set_assigned_physical_gpu_ids(assigned_physical_gpu_ids)
                assert self.local_rank < len(assigned_physical_gpu_ids), (
                    f"local_rank {self.local_rank} is out of bounds for "
                    f"assigned_physical_gpu_ids {assigned_physical_gpu_ids}"
                )
                if parallel_config.distributed_executor_backend not in ("ray", "external_launcher"):
                    assert self.parallel_config.local_world_size <= len(assigned_physical_gpu_ids), (
                        f"local_world_size ({self.parallel_config.local_world_size}) "
                        "exceeds assigned_physical_gpu_ids count "
                        f"({len(assigned_physical_gpu_ids)})"
                    )
            else:
                assert self.local_rank < torch.accelerator.device_count(), (
                    f"DP adjusted local rank {self.local_rank} is out of "
                    f"bounds for {torch.accelerator.device_count()} devices."
                )

            visible_device_index = current_platform.logical_device_id_to_visible_device_id(self.local_rank)
            self.device = current_omni_platform.get_torch_device(visible_device_index)
            torch.accelerator.set_device_index(self.device)

            current_platform.check_if_supports_dtype(self.model_config.dtype)

            # Initialize the distributed environment BEFORE taking
            # memory snapshot
            # This ensures NCCL buffers are allocated before we measure
            # available memory
            init_worker_distributed_environment(
                self.vllm_config,
                self.rank,
                self.distributed_init_method,
                self.local_rank,
                current_platform.dist_backend,
            )

            # Set random seed.
            set_random_seed(self.model_config.seed)

            # Now take memory snapshot after NCCL is initialized
            gc.collect()
            torch.accelerator.empty_cache()

            # take current memory snapshot
            self.init_snapshot = init_snapshot = MemorySnapshot(device=self.device)
            self.requested_memory = request_memory_tolerant(init_snapshot, self.cache_config)
            logger.debug("worker init memory snapshot: %r", self.init_snapshot)
            logger.debug("worker requested memory: %sGiB", format_gib(self.requested_memory))
        else:
            raise RuntimeError(f"Not support device type: {self.device_config.device}")

        # Initialize workspace manager
        num_ubatches = 2 if self.vllm_config.parallel_config.enable_dbo else 1
        init_workspace_manager(self.device, num_ubatches)

        # Construct the model runner
        runner_cls = (
            GPUARModelRunnerV2
            if self.use_v2_model_runner
            else GPUARModelRunner
        )
        self.model_runner = runner_cls(self.vllm_config, self.device)

        if self.rank == 0:
            # If usage stat is enabled, collect relevant info.
            report_usage_stats(self.vllm_config)

    def handle_sleep_task(self, task: OmniSleepTask | dict) -> OmniACK:
        """
        Explicitly handle sleep commands.
        Calls the implementation in the base class OmniGPUWorkerBase.
        """
        logger.debug(f"[AR Worker {self.rank}] Resolving handle_sleep_task dispatch")
        if isinstance(task, dict):
            task = OmniSleepTask(**task)
        return super().handle_sleep_task(task)

    def handle_wake_task(self, task: OmniWakeTask | dict) -> OmniACK:
        """
        Explicitly handle wake-up commands.
        Calls the implementation in the base class OmniGPUWorkerBase.
        """
        logger.debug(f"[AR Worker {self.rank}] Resolving handle_wake_task dispatch")
        if isinstance(task, dict):
            task = OmniWakeTask(**task)
        return super().handle_wake_task(task)
