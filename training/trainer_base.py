"""
Shared training infrastructure for all nanoGPT-Modern scripts.

Provides:
  - Distributed setup/teardown helpers
  - Seed management
  - Model checkpoint loading
  - AMP context + GradScaler construction
  - A CheckpointManager that persists full training state
  - A lightweight BaseTrainer class that concrete trainers can subclass
"""

import os
import random
import threading
import warnings
from abc import ABC, abstractmethod
from contextlib import nullcontext
from functools import partial
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP


from model.baseline_gpt import BaselineGPT, BaselineGPTConfig
from model.modern_gpt import ModernGPT, ModernGPTConfig
from utils.checkpoint import (
    build_checkpoint,
    write_checkpoint,
    stage_to_cpu,
    load_checkpoint,
    get_rng_state,
    set_rng_state,
)
from utils.config import to_dict
from utils.experiment import ExperimentRecorder
from utils.logging import Logger
from utils.monitoring import GradientNormMonitor, LossSpikeDetector


def enable_tf32() -> None:
    """Enable TF32 for CUDA matmul and cuDNN (A100/H100 fast path).

    Idempotent and safe on CPU-only builds — the backend flags are
    accepted regardless of CUDA availability, so all trainer scripts
    call this at module level (FR8-AC3: TF32 coverage across all
    trainers, previously pretrain-only).
    """
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


def save_run_config(args, out_dir, filename="config.yaml"):
    """Save the merged CLI + YAML configuration to the output directory.

    The saved file can be reused with ``--config`` to reproduce the run.
    """
    import yaml

    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, filename)
    cfg = to_dict(args)
    # Filter non-serializable values.
    clean = {}
    for k, v in cfg.items():
        try:
            # Trigger serialization check.
            _ = yaml.safe_dump({k: v})
            clean[k] = v
        except Exception:
            clean[k] = str(v)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(clean, f, sort_keys=True)
    return path


def setup_distributed(backend="nccl"):
    """Initialize distributed process group if launched with torchrun.

    Returns
    -------
    rank, local_rank, world_size, distributed
    """
    rank = int(os.environ.get("RANK", -1))
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    distributed = rank != -1

    if distributed:
        if not dist.is_initialized():
            dist.init_process_group(backend=backend)

    return rank, local_rank, world_size, distributed


def cleanup_distributed():
    """Destroy process group if it was initialized."""
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def set_seed(seed, rank=0):
    """Set random seeds for torch/numpy/random with rank-based offset."""
    seed = int(seed) + int(rank)
    # NumPy requires a non-negative seed below 2**32.
    seed = seed % 2**32
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _seed_worker(worker_id, base_seed=0, rank=0):
    # In non-distributed mode ``rank`` is -1; treat it as 0 for seeding.
    effective_rank = rank if rank >= 0 else 0
    worker_seed = base_seed + effective_rank * 1000 + worker_id
    set_seed(worker_seed)


def make_worker_init_fn(base_seed=0, rank=0):
    """Return a DataLoader ``worker_init_fn`` for deterministic per-worker seeds.

    Usage::

        DataLoader(..., worker_init_fn=make_worker_init_fn(args.seed, rank))

    Returns a ``functools.partial`` over a module-level function so that it
    remains picklable under Windows spawn-based DataLoader workers.
    """
    return partial(_seed_worker, base_seed=base_seed, rank=rank)


def infer_device(device_arg, local_rank):
    """Resolve the actual torch device from CLI arg and local rank."""
    if device_arg.startswith("cuda") and local_rank >= 0:
        return f"cuda:{local_rank}"
    return device_arg


def load_model_from_checkpoint(path, device="cpu", model_type="auto", strict=True):
    """Load a BaselineGPT or ModernGPT checkpoint and return (model, ckpt_extra).

    Parameters
    ----------
    path : str
        Checkpoint path.
    device : str
        map_location for loading.
    model_type : {"auto", "baseline", "modern"}
        If "auto", infer from saved config.
    strict : bool
        Passed to load_state_dict.

    Returns
    -------
    model : nn.Module
        Model loaded on `device`.
    ckpt_extra : dict
        Extra checkpoint state (iter_num, best_val_loss, scaler, scheduler, etc.).
    """
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    raw_config = checkpoint.get("config")

    if raw_config is None:
        if model_type == "modern" or model_type == "auto":
            config = ModernGPTConfig()
        else:
            config = BaselineGPTConfig()
    elif isinstance(raw_config, dict):
        config_dict = dict(raw_config)
        saved_model_type = config_dict.pop("model_type", None)
        if model_type == "auto" and saved_model_type is not None:
            model_type = saved_model_type

        if model_type == "baseline":
            config = BaselineGPTConfig.from_dict(config_dict)
        else:
            # Modern is the default / fallback
            config = ModernGPTConfig.from_dict(config_dict)
    else:
        config = raw_config

    if model_type == "baseline":
        model = BaselineGPT(config)
    else:
        model = ModernGPT(config)

    model.load_state_dict(checkpoint["model"], strict=strict)
    model = model.to(device)

    return model, checkpoint


def build_amp_context(device, use_bf16=True):
    """Build an autocast context for mixed precision training.

    Returns
    -------
    ctx : context manager
    scaler : GradScaler or None
    dtype : torch.dtype
    """
    if device == "cpu":
        return nullcontext(), None, torch.float32

    if use_bf16 and torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        dtype = torch.bfloat16
    else:
        dtype = torch.float16

    ctx = torch.amp.autocast(device_type="cuda", dtype=dtype)
    # fp16 needs GradScaler; bf16 does not.
    scaler = torch.amp.GradScaler("cuda", enabled=(dtype == torch.float16))
    return ctx, scaler, dtype


class CheckpointManager:
    """Thin wrapper around utils.checkpoint that tracks training state."""

    def __init__(
        self,
        out_dir,
        model,
        optimizer,
        config,
        scaler=None,
        scheduler=None,
        ema_shadow=None,
        resume_offset=0,
        keep_last_n=0,
    ):
        self.out_dir = out_dir
        self.model = model
        self.optimizer = optimizer
        self.config = config
        self.scaler = scaler
        self.scheduler = scheduler
        self.ema_shadow = ema_shadow
        self.resume_offset = resume_offset
        self.keep_last_n = keep_last_n
        self._saved = []  # FIFO of prunable checkpoint filenames
        self._writer_thread = None  # in-flight background write, if any

    def _is_protected(self, filename):
        """Best, final, and EMA checkpoints are never pruned."""
        protected = {"best_ckpt.pt", "final_ckpt.pt", "ema_ckpt.pt", "latest_ckpt.pt"}
        if filename in protected:
            return True
        name = filename.lower()
        return name.startswith(("best_", "final_", "ema_"))

    def save(
        self,
        filename,
        iter_num,
        best_metric,
        rng_state=None,
        resume_offset=None,
        ema_shadow=None,
        extra_state=None,
        async_save=True,
    ):
        """Save a checkpoint with full training state.

        Optional overrides for ``rng_state``, ``resume_offset`` and
        ``ema_shadow`` are useful when the trainer keeps these values outside
        of the manager (e.g. ``PretrainTrainer.resume_offset``).
        ``extra_state`` carries additional resume keys
        (``early_stop_counter`` / ``consumed_chunks`` / ``world_size`` /
        ``rank``) that are persisted when present and not None.

        By default the payload is staged to CPU on the caller thread and
        serialized by a background writer (tmp file + atomic rename), so
        the training loop never blocks on disk I/O.  Saves are strictly
        serial: a new save first waits for the previous background write
        to finish.  Call :meth:`flush` (or let ``train()`` /
        ``cleanup()`` do it) before reading the file.  Pass
        ``async_save=False`` for a fully synchronous write.
        """
        # Serialize with any in-flight background write before staging.
        self.flush()

        model_src = self.model
        # If model is wrapped by DDP, unwrap for checkpointing.
        if isinstance(model_src, DDP):
            model_src = model_src.module

        # Inject model_type into the config dict so reloaders know which class to build.
        config_data = (
            self.config.to_dict() if hasattr(self.config, "to_dict") else self.config
        )
        model_type = (
            "baseline" if type(model_src).__name__ == "BaselineGPT" else "modern"
        )
        if isinstance(config_data, dict):
            config_data = dict(config_data)
            config_data["model_type"] = model_type
        else:
            # If config is a dataclass-like object, try to set model_type as an attribute.
            if not hasattr(config_data, "model_type"):
                try:
                    object.__setattr__(config_data, "model_type", model_type)
                except Exception:
                    pass

        extra_kwargs = {}
        if extra_state:
            for key in ("early_stop_counter", "consumed_chunks", "world_size", "rank"):
                if key in extra_state and extra_state[key] is not None:
                    extra_kwargs[key] = extra_state[key]

        checkpoint = build_checkpoint(
            model_src,
            self.optimizer,
            iter_num,
            best_metric,
            config_data,
            scaler=self.scaler,
            scheduler=self.scheduler,
            ema_shadow=ema_shadow if ema_shadow is not None else self.ema_shadow,
            rng_state=rng_state if rng_state is not None else get_rng_state(),
            resume_offset=(
                resume_offset if resume_offset is not None else self.resume_offset
            ),
            **extra_kwargs,
        )
        # Deep-copy the payload to CPU storage on the caller thread so
        # the background writer never races in-place parameter/optimizer
        # mutation from continued training (on CPU devices .cpu() alone
        # would alias live state, hence the explicit copy).
        checkpoint = stage_to_cpu(checkpoint)

        if async_save:
            self._writer_thread = threading.Thread(
                target=self._write_payload,
                args=(checkpoint, self.out_dir, filename),
                daemon=True,
                name="ckpt-writer",
            )
            self._writer_thread.start()
        else:
            self._write_payload(checkpoint, self.out_dir, filename)

        path = os.path.join(self.out_dir, filename)

        if self.keep_last_n > 0 and not self._is_protected(filename):
            self._saved.append(filename)
            while len(self._saved) > self.keep_last_n:
                old = self._saved.pop(0)
                old_path = os.path.join(self.out_dir, old)
                try:
                    if os.path.exists(old_path):
                        os.remove(old_path)
                except OSError:
                    pass

        return path

    def _write_payload(self, checkpoint, out_dir, filename):
        """Serialize a staged payload; WARN + sync retry on failure."""
        try:
            write_checkpoint(checkpoint, out_dir, filename)
        except Exception:
            warnings.warn(
                f"async checkpoint write failed for {filename}; "
                "retrying synchronously",
                RuntimeWarning,
            )
            try:
                write_checkpoint(checkpoint, out_dir, filename)
            except Exception:
                warnings.warn(
                    f"synchronous checkpoint retry failed for {filename}",
                    RuntimeWarning,
                )

    def flush(self):
        """Block until any in-flight background write has finished."""
        if self._writer_thread is not None:
            self._writer_thread.join()
            self._writer_thread = None

    def load(self, path, strict=True):
        """Load a checkpoint and return auxiliary state.

        Handles unwrapping DDP models automatically.  When the checkpoint
        carries an ``rng_state`` bundle, it is restored via
        ``set_rng_state`` so a resumed run continues the exact RNG stream
        of the save point.
        """
        model_src = self.model
        if isinstance(model_src, DDP):
            model_src = model_src.module

        extra = load_checkpoint(
            path,
            model_src,
            self.optimizer,
            device=next(model_src.parameters()).device,
            strict=strict,
        )

        if "scaler" in extra and self.scaler is not None:
            self.scaler.load_state_dict(extra["scaler"])
        if (
            "scheduler" in extra
            and self.scheduler is not None
            and hasattr(self.scheduler, "load_state_dict")
        ):
            self.scheduler.load_state_dict(extra["scheduler"])
        if "ema_shadow" in extra and self.ema_shadow is not None:
            # If the caller keeps a reference to the dict, update in place.
            self.ema_shadow.clear()
            self.ema_shadow.update(extra["ema_shadow"])
        if "rng_state" in extra:
            set_rng_state(extra["rng_state"])
        if "resume_offset" in extra:
            self.resume_offset = extra["resume_offset"]

        return extra


class BaseTrainer(ABC):
    """Lightweight base trainer for nanoGPT-Modern scripts.

    Subclasses implement `train()`; shared setup (distributed, seed, model,
    optimizer, scheduler, logger, AMP, checkpointing) is handled here.
    """

    def __init__(self, args):
        self.args = args
        self.rank, self.local_rank, self.world_size, self.distributed = (
            setup_distributed(backend=getattr(args, "backend", "nccl"))
        )
        self.master_process = self.rank <= 0
        self.device = infer_device(args.device, self.local_rank)

        if self.distributed and self.device.startswith("cuda"):
            torch.cuda.set_device(self.device)

        # Use the same base seed across all ranks so that model initialization is
        # identical in DDP/FSDP.  Data-order randomness is provided by
        # DistributedSampler / per-worker seeds, not by the global seed.
        set_seed(args.seed)

        self.model = None
        self.raw_model = None
        self.optimizer = None
        self.scheduler = None
        self.scaler = None
        self.ctx = nullcontext()
        self.logger = None
        self.ckpt_manager = None
        self.recorder = None
        self.spike_detector = None
        self.grad_monitor = None

        # Template-method orchestration: subclasses override the protected hooks
        # below instead of re-implementing __init__.  This guarantees that data,
        # model, optimizer, scheduler, AMP, checkpointing, logger, and resume are
        # always built in the correct order.
        self._init_state()
        self._build_data()
        self._build_model()
        self._build_optimizer()
        self._build_scheduler()
        self._setup_amp()
        self._setup_checkpointing()
        self._setup_logger()
        self._maybe_resume()
        self._setup_recorder()
        self._setup_monitors()

    def _init_state(self):
        """Initialize training state variables.  Subclasses may override."""
        pass

    def _build_data(self):
        """Build datasets and dataloaders.  Subclasses may override."""
        pass

    def _build_model(self):
        """Build or load the model.  Subclasses may override."""
        pass

    def _build_optimizer(self):
        """Build the optimizer.  Subclasses may override."""
        pass

    def _build_scheduler(self):
        """Build the LR scheduler.  Subclasses may override."""
        pass

    def _setup_amp(self, use_bf16=True):
        """Initialize mixed precision context and GradScaler."""
        self.setup_amp(use_bf16=use_bf16)

    def _setup_checkpointing(self):
        """Create the CheckpointManager.  Subclasses may override."""
        pass

    def _setup_logger(self):
        """Create the Logger.  Subclasses may override."""
        pass

    def _maybe_resume(self):
        """Load checkpoint if --resume is set.  Subclasses may override."""
        pass

    def _setup_recorder(self):
        """Create the ExperimentRecorder for structured metrics (master only).

        Metrics land in ``<out_dir>/experiments/<exp_id>/`` where ``exp_id``
        defaults to the basename of ``out_dir``.  Recorder failures degrade
        to a warning and never block training.
        """
        self.recorder = None
        if not self.master_process:
            return
        if getattr(self.args, "no_metrics", False):
            return
        out_dir = str(getattr(self.args, "out_dir", "out"))
        exp_id = getattr(self.args, "exp_id", None) or os.path.basename(out_dir) or "exp"
        exp_dir = os.path.join(out_dir, "experiments", exp_id)
        resume = bool(getattr(self.args, "resume", None))
        try:
            self.recorder = ExperimentRecorder(
                exp_dir=exp_dir,
                exp_id=exp_id,
                config=to_dict(self.args),
                resume=resume,
            )
        except Exception as exc:  # recording must never block training
            warnings.warn(
                f"ExperimentRecorder setup failed ({exc}); "
                "continuing without structured metrics.",
                RuntimeWarning,
            )
            self.recorder = None

    def _setup_monitors(self):
        """Build the loss-spike / grad-norm monitors (master only).

        ``--no_monitoring`` leaves both as None (spike detection disabled);
        construction failures degrade to a warning and never block training.
        """
        self.spike_detector = None
        self.grad_monitor = None
        if not self.master_process:
            return
        if getattr(self.args, "no_monitoring", False):
            return
        try:
            self.spike_detector = LossSpikeDetector()
        except Exception as exc:
            warnings.warn(
                f"LossSpikeDetector setup failed ({exc}); "
                "continuing without spike detection.",
                RuntimeWarning,
            )
            self.spike_detector = None
        try:
            self.grad_monitor = GradientNormMonitor(self.logger)
        except Exception as exc:
            warnings.warn(
                f"GradientNormMonitor setup failed ({exc}); "
                "continuing without grad-norm monitoring.",
                RuntimeWarning,
            )
            self.grad_monitor = None

    def on_step_end(self, step, loss, lr=None, grad_norm=None, tokens_per_sec=None):
        """Monitoring + structured-recording hook for one logged train step.

        Order (design.md §4.2 flow B): grad-norm acquisition -> spike
        detection (with optional LR auto-reduction) -> record_step.  Every
        monitoring failure degrades to a warning; recording and training are
        never interrupted.
        """
        if not self.master_process:
            return
        loss = float(loss)

        # --- monitoring path ---
        if grad_norm is None and self.model is not None:
            if self.grad_monitor is not None:
                try:
                    norms = self.grad_monitor.compute(self.model, step)
                    grad_norm = float(norms.get("total_norm", 0.0))
                except Exception as exc:
                    warnings.warn(
                        f"GradientNormMonitor.compute failed ({exc}); "
                        "grad_norm omitted for this step.",
                        RuntimeWarning,
                    )
                    grad_norm = None

        if self.spike_detector is not None:
            try:
                result = self.spike_detector.update(loss, step)
                if result.get("spike_detected"):
                    print(f"[WARN] {result.get('message', 'loss spike detected')}")
                    factor = result.get("lr_multiplier")
                    if factor is not None and self.optimizer is not None:
                        # Multiply the *current* lr; the next scheduler.step()
                        # overwrites it with the scheduled value.
                        for group in self.optimizer.param_groups:
                            group["lr"] = group["lr"] * factor
                    if self.recorder is not None:
                        self.recorder.record_event(
                            "loss_spike",
                            step,
                            loss=loss,
                            z_score=result.get("z_score"),
                            lr_reduce_factor=factor,
                            window_mean=result.get("window_mean"),
                            window_std=result.get("window_std"),
                        )
            except Exception as exc:
                warnings.warn(
                    f"LossSpikeDetector.update failed ({exc}); "
                    "spike detection skipped for this step.",
                    RuntimeWarning,
                )

        # --- record path ---
        if self.recorder is not None:
            fields = {"loss": loss}
            if lr is not None:
                fields["lr"] = lr
            if grad_norm is not None:
                fields["grad_norm"] = grad_norm
            if tokens_per_sec is not None:
                fields["tokens_per_sec"] = tokens_per_sec
            if torch.cuda.is_available():
                fields["mem_gb"] = round(
                    torch.cuda.max_memory_allocated() / 1024**3, 3
                )
            self.recorder.record_step(step, **fields)

    def setup_amp(self, use_bf16=True):
        """Initialize mixed precision context and GradScaler."""
        self.ctx, self.scaler, self.amp_dtype = build_amp_context(
            self.device, use_bf16=use_bf16
        )

    def build_logger(self, project_name, run_name, config=None):
        """Create a Logger (only on master process)."""
        if not self.master_process:
            return None
        log_dir = os.path.join(getattr(self.args, "out_dir", "out"), "logs")
        config = to_dict(config) if config is not None else to_dict(self.args)
        # Persist merged config so runs are reproducible.
        save_run_config(self.args, os.path.dirname(log_dir))
        self.logger = Logger(
            project_name=project_name,
            run_name=run_name,
            log_dir=log_dir,
            use_wandb=getattr(self.args, "use_wandb", False),
            use_tensorboard=True,
            config=config,
        )
        return self.logger

    def wrap_distributed(self, model, compile_model=False):
        """Optionally compile and wrap model for distributed training.

        Returns the wrapped model and sets self.raw_model to the unwrapped one.
        """
        if compile_model:
            model = torch.compile(model)

        if self.distributed:
            if getattr(self.args, "fsdp", False):
                from torch.distributed.fsdp import (
                    FullyShardedDataParallel as FSDP,
                    ShardingStrategy,
                    MixedPrecision,
                )

                shard_map = {
                    "full": ShardingStrategy.FULL_SHARD,
                    "grad": ShardingStrategy.SHARD_GRAD_OP,
                    "no": ShardingStrategy.NO_SHARD,
                }
                mp_policy = MixedPrecision(
                    param_dtype=(
                        torch.bfloat16
                        if (
                            torch.cuda.is_available() and torch.cuda.is_bf16_supported()
                        )
                        else torch.float16
                    ),
                    reduce_dtype=torch.float32,
                    buffer_dtype=torch.float32,
                )
                model = FSDP(
                    model,
                    sharding_strategy=shard_map[
                        getattr(self.args, "fsdp_sharding_strategy", "full")
                    ],
                    mixed_precision=mp_policy,
                    device_id=(
                        torch.cuda.current_device()
                        if torch.cuda.is_available()
                        else None
                    ),
                )
                self.raw_model = model.module
            else:
                # FR9a: static_graph (fixed model structure, enables DDP
                # graph analysis) + gradient_as_bucket_view (gradients
                # alias the communication buckets, saves a full-size
                # gradient buffer).
                model = DDP(
                    model,
                    device_ids=[self.local_rank],
                    static_graph=True,
                    gradient_as_bucket_view=True,
                )
                self.raw_model = model.module
        else:
            self.raw_model = model

        self.model = model
        return model

    def configure_checkpointing(self, config, ema_shadow=None):
        """Create the CheckpointManager after optimizer/scheduler are built."""
        self.ckpt_manager = CheckpointManager(
            out_dir=getattr(self.args, "out_dir", "out"),
            model=self.model,
            optimizer=self.optimizer,
            config=config,
            scaler=self.scaler,
            scheduler=self.scheduler,
            ema_shadow=ema_shadow,
            resume_offset=0,
            keep_last_n=getattr(self.args, "keep_last_n", 0),
        )

    # Eval tags dual-written to the ExperimentRecorder as eval events
    # (tag -> metrics.jsonl field name).
    _RECORDER_EVAL_TAGS = {
        "val/loss": "val_loss",
        "val/mean_reward": "mean_reward",
        "val/accuracy": "val_accuracy",
    }

    def log_scalars(self, scalars, step):
        if self.logger is not None:
            self.logger.log_scalars(scalars, step)
        if self.recorder is not None:
            for tag, value in scalars.items():
                field = self._RECORDER_EVAL_TAGS.get(tag)
                if field:
                    self.recorder.record_eval(step, **{field: float(value)})

    def log_scalar(self, tag, value, step):
        if self.logger is not None:
            self.logger.log_scalar(tag, value, step)
        if self.recorder is not None:
            field = self._RECORDER_EVAL_TAGS.get(tag)
            if field:
                self.recorder.record_eval(step, **{field: float(value)})

    def log_text(self, tag, text, step):
        if self.logger is not None:
            self.logger.log_text(tag, text, step)

    def log_memory_stats(self, step):
        if self.logger is not None:
            self.logger.log_memory_stats(step=step)

    def log_grad_norms(self, step):
        if self.logger is not None and self.model is not None:
            self.logger.log_grad_norms(self.model, step)

    def save_checkpoint(self, filename, step, best_metric):
        if self.ckpt_manager is None:
            raise RuntimeError(
                "configure_checkpointing() must be called before save_checkpoint()"
            )
        return self.ckpt_manager.save(filename, step, best_metric)

    def load_checkpoint(self, path, strict=True):
        if self.ckpt_manager is None:
            raise RuntimeError(
                "configure_checkpointing() must be called before load_checkpoint()"
            )
        return self.ckpt_manager.load(path, strict=strict)

    def finalize_recorder(self, status="completed", final_metrics=None):
        """Finalize the ExperimentRecorder; no-op when recording is disabled.

        ``wall_time_s`` is filled in automatically (since the original start)
        when not supplied by the caller.
        """
        if self.recorder is None or self.recorder.finalized:
            return
        final_metrics = dict(final_metrics or {})
        if "wall_time_s" not in final_metrics:
            wall = self.recorder.wall_time()
            if wall is not None:
                final_metrics["wall_time_s"] = wall
        self.recorder.finalize(status, final_metrics)

    @abstractmethod
    def train(self):
        """Main training loop. Must be implemented by subclasses."""
        raise NotImplementedError

    def cleanup(self):
        # Drain any pending async checkpoint write first so files are
        # on disk before the process tears down.
        if self.ckpt_manager is not None and hasattr(self.ckpt_manager, "flush"):
            self.ckpt_manager.flush()
        # A recorder still in "running" state at cleanup time means train()
        # ended without finalize() (interrupt or crash) — mark it as such.
        if self.recorder is not None and not self.recorder.finalized:
            self.recorder.finalize("interrupted", {})
        if self.logger is not None:
            self.logger.close()
        cleanup_distributed()


def maybe_warn_dropout(model):
    """Warn if model has dropout > 0 and will be used for GRPO-style logprob ratios."""
    dp = getattr(getattr(model, "config", None), "dropout", 0.0)
    if dp > 0:
        warnings.warn(
            f"Model dropout={dp} > 0. When computing old/new logprob ratios "
            "(e.g. GRPO), eval-mode and train-mode dropout masks will differ, "
            "biasing the ratio. Set dropout=0.0 for reliable RL training.",
            UserWarning,
        )
