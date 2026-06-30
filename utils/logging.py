"""
Unified logging utility supporting wandb and tensorboard.

Features
--------
* Graceful fallback if ``wandb`` or ``tensorboard`` init fails.
* Scalar, histogram, text, and memory logging.
* Per-step gradient norm / parameter histogram tracking (optional).
* Configurable log level for console output.
"""

import os
import logging
from typing import Any, Dict, Optional


class Logger:
    """Multi-backend logger with robust fallback behavior.

    Parameters
    ----------
    project_name : str
        Project name used for wandb.
    run_name : str
        Run name used for wandb and tensorboard sub-directory.
    log_dir : str
        Root directory for tensorboard logs.
    use_wandb : bool
        Whether to attempt wandb logging.
    use_tensorboard : bool
        Whether to attempt tensorboard logging.
    config : dict or None
        Run configuration to log once at init.
    log_level : int
        Console logging level (default: ``logging.INFO``).
    """

    def __init__(
        self,
        project_name: str,
        run_name: str,
        log_dir: str = "logs",
        use_wandb: bool = False,
        use_tensorboard: bool = True,
        config: Optional[Dict[str, Any]] = None,
        log_level: int = logging.INFO,
    ):
        self.use_wandb = use_wandb
        self.use_tensorboard = use_tensorboard
        self.writer = None
        self.run = None

        # Console logger for warnings / fallback messages.
        self._console = logging.getLogger(f"nanogpt_logger_{run_name}")
        self._console.setLevel(log_level)
        if not self._console.handlers:
            handler = logging.StreamHandler()
            handler.setLevel(log_level)
            handler.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
            self._console.addHandler(handler)

        # TensorBoard
        if use_tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter

                full_log_dir = os.path.join(log_dir, run_name)
                os.makedirs(full_log_dir, exist_ok=True)
                self.writer = SummaryWriter(log_dir=full_log_dir)
            except Exception as e:
                self._console.warning(
                    f"TensorBoard init failed: {e}. Falling back to no TensorBoard."
                )
                self.writer = None

        # Weights & Biases
        if use_wandb:
            try:
                import wandb

                # Allow offline mode if the network is unavailable.
                os.environ.setdefault("WANDB_MODE", "online")
                self.run = wandb.init(
                    project=project_name,
                    name=run_name,
                    config=config,
                    reinit=True,
                )
                if self.run is None:
                    raise RuntimeError("wandb.init returned None")
            except Exception as e:
                self._console.warning(
                    f"wandb init failed: {e}. Continuing without wandb logging."
                )
                self.run = None
                self.use_wandb = False

    def log_scalar(self, tag: str, value: float, step: int):
        """Log a single scalar value."""
        if self.writer is not None:
            self.writer.add_scalar(tag, value, step)
        if self.run is not None:
            self.run.log({tag: value}, step=step)

    def log_scalars(self, scalars: Dict[str, float], step: int):
        """Log a dictionary of scalars."""
        if self.writer is not None:
            for tag, value in scalars.items():
                self.writer.add_scalar(tag, value, step)
        if self.run is not None:
            self.run.log(scalars, step=step)

    def log_histogram(self, tag: str, values, step: int):
        """Log a histogram (e.g. gradient norms or parameter values)."""
        if self.writer is not None:
            self.writer.add_histogram(tag, values, step)
        if self.run is not None:
            import wandb

            self.run.log({tag: wandb.Histogram(values)}, step=step)

    def log_text(self, tag: str, text: str, step: int):
        """Log a text sample (e.g. generated response).

        TensorBoard writes to the ``text`` plugin; wandb writes a ``tag`` key
        containing the raw string.
        """
        if self.writer is not None:
            self.writer.add_text(tag, text, step)
        if self.run is not None:
            self.run.log({tag: text}, step=step)

    def log_memory_stats(self, tag_prefix: str = "system", step: int = 0):
        """Log CUDA memory statistics if available."""
        if not torch_cuda_available():
            return
        import torch

        allocated = torch.cuda.memory_allocated() / (1024**2)
        reserved = torch.cuda.memory_reserved() / (1024**2)
        max_allocated = torch.cuda.max_memory_allocated() / (1024**2)
        self.log_scalars(
            {
                f"{tag_prefix}/memory_allocated_mb": allocated,
                f"{tag_prefix}/memory_reserved_mb": reserved,
                f"{tag_prefix}/max_memory_allocated_mb": max_allocated,
            },
            step,
        )

    def log_grad_norms(self, model, step: int, tag_prefix: str = "grad"):
        """Log total gradient norm and per-layer gradient norm histogram.

        Should be called after ``loss.backward()`` and (for fp16) after
        ``scaler.unscale_(optimizer)`` so that the gradients are in fp32.
        """
        import torch

        total_norm = 0.0
        norms = []
        for name, p in model.named_parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2).item()
                norms.append(param_norm)
                total_norm += param_norm**2
        total_norm = total_norm**0.5
        self.log_scalar(f"{tag_prefix}/total_norm", total_norm, step)
        if norms:
            self.log_histogram(f"{tag_prefix}/layer_norms", torch.tensor(norms), step)

    def log_model_weights(self, model, step: int, tag_prefix: str = "param"):
        """Log per-layer parameter value histograms."""
        import torch

        for name, p in model.named_parameters():
            # Sanitize tag name.
            safe_name = name.replace(".", "/")
            self.log_histogram(f"{tag_prefix}/{safe_name}", p.data, step)

    def log_config(self, config: Dict[str, Any]):
        """Log a configuration dictionary to all active backends."""
        if self.run is not None:
            self.run.config.update(config, allow_val_change=True)
        if self.writer is not None:
            # TensorBoard does not have a native config object; dump as text.
            import json

            self.writer.add_text("config", json.dumps(config, indent=2, default=str), 0)

    def close(self):
        if self.writer is not None:
            self.writer.close()
            self.writer = None
        if self.run is not None:
            import wandb

            wandb.finish()
            self.run = None


def torch_cuda_available() -> bool:
    """Safe check for torch CUDA availability without importing torch eagerly."""
    try:
        import torch

        return torch.cuda.is_available()
    except Exception:
        return False
