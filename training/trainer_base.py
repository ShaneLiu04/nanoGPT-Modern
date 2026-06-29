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
import warnings
from abc import ABC, abstractmethod
from contextlib import nullcontext

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP


from model.baseline_gpt import BaselineGPT, BaselineGPTConfig
from model.modern_gpt import ModernGPT, ModernGPTConfig
from utils.checkpoint import save_checkpoint, load_checkpoint, get_rng_state
from utils.config import to_dict
from utils.logging import Logger


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
    seed = seed + rank
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_worker_init_fn(base_seed=0, rank=0):
    """Return a DataLoader ``worker_init_fn`` for deterministic per-worker seeds.

    Usage::

        DataLoader(..., worker_init_fn=make_worker_init_fn(args.seed, rank))
    """
    def _worker_init(worker_id):
        worker_seed = base_seed + rank * 1000 + worker_id
        set_seed(worker_seed)
    return _worker_init


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

    def __init__(self, out_dir, model, optimizer, config, scaler=None,
                 scheduler=None, ema_shadow=None, resume_offset=0, keep_last_n=0):
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

    def _is_protected(self, filename):
        """Best, final, and EMA checkpoints are never pruned."""
        protected = {"best_ckpt.pt", "final_ckpt.pt", "ema_ckpt.pt", "latest_ckpt.pt"}
        if filename in protected:
            return True
        name = filename.lower()
        return name.startswith(("best_", "final_", "ema_"))

    def save(self, filename, iter_num, best_metric, rng_state=None, resume_offset=None, ema_shadow=None):
        """Save a checkpoint with full training state.

        Optional overrides for ``rng_state``, ``resume_offset`` and
        ``ema_shadow`` are useful when the trainer keeps these values outside
        of the manager (e.g. ``PretrainTrainer.resume_offset``).
        """
        model_src = self.model
        # If model is wrapped by DDP, unwrap for checkpointing.
        if isinstance(model_src, DDP):
            model_src = model_src.module

        # Inject model_type into the config dict so reloaders know which class to build.
        config_data = self.config.to_dict() if hasattr(self.config, "to_dict") else self.config
        model_type = "baseline" if type(model_src).__name__ == "BaselineGPT" else "modern"
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

        path = save_checkpoint(
            model_src,
            self.optimizer,
            iter_num,
            best_metric,
            config_data,
            self.out_dir,
            filename,
            scaler=self.scaler,
            scheduler=self.scheduler,
            ema_shadow=ema_shadow if ema_shadow is not None else self.ema_shadow,
            rng_state=rng_state if rng_state is not None else get_rng_state(),
            resume_offset=resume_offset if resume_offset is not None else self.resume_offset,
        )

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

    def load(self, path, strict=True):
        """Load a checkpoint and return auxiliary state.

        Handles unwrapping DDP models automatically.
        """
        model_src = self.model
        if isinstance(model_src, DDP):
            model_src = model_src.module

        extra = load_checkpoint(path, model_src, self.optimizer,
                                device=next(model_src.parameters()).device,
                                strict=strict)

        if "scaler" in extra and self.scaler is not None:
            self.scaler.load_state_dict(extra["scaler"])
        if "scheduler" in extra and self.scheduler is not None and hasattr(self.scheduler, "load_state_dict"):
            self.scheduler.load_state_dict(extra["scheduler"])
        if "ema_shadow" in extra and self.ema_shadow is not None:
            # If the caller keeps a reference to the dict, update in place.
            self.ema_shadow.clear()
            self.ema_shadow.update(extra["ema_shadow"])
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
        self.rank, self.local_rank, self.world_size, self.distributed = setup_distributed(
            backend=getattr(args, "backend", "nccl")
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

    def setup_amp(self, use_bf16=True):
        """Initialize mixed precision context and GradScaler."""
        self.ctx, self.scaler, self.amp_dtype = build_amp_context(self.device, use_bf16=use_bf16)

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
                    param_dtype=torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else torch.float16,
                    reduce_dtype=torch.float32,
                    buffer_dtype=torch.float32,
                )
                model = FSDP(
                    model,
                    sharding_strategy=shard_map[getattr(self.args, "fsdp_sharding_strategy", "full")],
                    mixed_precision=mp_policy,
                    device_id=torch.cuda.current_device() if torch.cuda.is_available() else None,
                )
                self.raw_model = model.module
            else:
                model = DDP(model, device_ids=[self.local_rank])
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

    def log_scalars(self, scalars, step):
        if self.logger is not None:
            self.logger.log_scalars(scalars, step)

    def log_scalar(self, tag, value, step):
        if self.logger is not None:
            self.logger.log_scalar(tag, value, step)

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
            raise RuntimeError("configure_checkpointing() must be called before save_checkpoint()")
        return self.ckpt_manager.save(filename, step, best_metric)

    def load_checkpoint(self, path, strict=True):
        if self.ckpt_manager is None:
            raise RuntimeError("configure_checkpointing() must be called before load_checkpoint()")
        return self.ckpt_manager.load(path, strict=strict)

    @abstractmethod
    def train(self):
        """Main training loop. Must be implemented by subclasses."""
        raise NotImplementedError

    def cleanup(self):
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
