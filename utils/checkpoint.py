"""
Checkpoint save/load utilities.

Supports full training-state checkpointing (model, optimizer, scheduler,
scaler, EMA shadow weights, RNG state) while remaining backward compatible
with older checkpoints that only store the minimal fields.
"""

import os
import torch


def _state_dict(model_or_state):
    """Return a state dict whether a module or a raw state dict is passed."""
    if isinstance(model_or_state, dict):
        return model_or_state
    return model_or_state.state_dict()


def save_checkpoint(
    model,
    optimizer,
    iter_num,
    best_val_loss,
    config,
    out_dir,
    filename="ckpt.pt",
    scaler=None,
    scheduler=None,
    ema_shadow=None,
    rng_state=None,
    resume_offset=0,
):
    """Save a training checkpoint.

    Parameters
    ----------
    model : torch.nn.Module or dict
        The model to checkpoint.  A raw ``state_dict`` can be passed for
        FSDP compatibility.
    optimizer : torch.optim.Optimizer
    iter_num : int
    best_val_loss : float
    config : object or dict
    out_dir : str
    filename : str
    scaler : torch.cuda.amp.GradScaler, optional
    scheduler : callable/object with ``state_dict()``, optional
    ema_shadow : dict[str, torch.Tensor], optional
        EMA shadow weights (as produced by ``ModernGPT.init_ema``).
    rng_state : dict, optional
        RNG state bundle produced by ``get_rng_state``.
    resume_offset : int, optional
        Data loader resume offset.

    Returns
    -------
    str
        Path to the saved checkpoint.
    """
    os.makedirs(out_dir, exist_ok=True)
    config_data = config.to_dict() if hasattr(config, "to_dict") else config

    checkpoint = {
        "model": _state_dict(model),
        "optimizer": optimizer.state_dict(),
        "iter_num": iter_num,
        "best_val_loss": best_val_loss,
        "config": config_data,
    }

    if scaler is not None:
        checkpoint["scaler"] = scaler.state_dict()
    if scheduler is not None and hasattr(scheduler, "state_dict"):
        checkpoint["scheduler"] = scheduler.state_dict()
    if ema_shadow is not None:
        checkpoint["ema_shadow"] = ema_shadow
    if rng_state is not None:
        checkpoint["rng_state"] = rng_state
    if resume_offset:
        checkpoint["resume_offset"] = resume_offset

    path = os.path.join(out_dir, filename)
    # Write to a temporary file and atomically rename to avoid corrupt
    # checkpoints if the process is interrupted mid-write.
    tmp_path = path + ".tmp"
    torch.save(checkpoint, tmp_path)
    os.replace(tmp_path, path)
    return path


def load_checkpoint(path, model, optimizer=None, device="cpu", strict=True):
    """Load a checkpoint and return auxiliary training state.

    Parameters
    ----------
    path : str
    model : torch.nn.Module
    optimizer : torch.optim.Optimizer, optional
    device : str
    strict : bool
        Passed to ``model.load_state_dict``.

    Returns
    -------
    dict
        Always contains ``iter_num`` and ``best_val_loss``.  May also
        contain ``scaler``, ``scheduler``, ``ema_shadow``, ``rng_state``,
        and ``resume_offset`` when present in the checkpoint.
    """
    checkpoint = torch.load(path, map_location=device, weights_only=False)

    model.load_state_dict(checkpoint["model"], strict=strict)

    if optimizer is not None and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])

    extra = {
        "iter_num": checkpoint.get("iter_num", 0),
        "best_val_loss": checkpoint.get("best_val_loss", float("inf")),
    }

    for key in ("scaler", "scheduler", "ema_shadow", "rng_state", "resume_offset"):
        if key in checkpoint:
            extra[key] = checkpoint[key]

    return extra


def get_rng_state():
    """Capture the RNG state of torch, numpy, and Python's random module."""
    import random
    import numpy as np

    return {
        "torch": torch.get_rng_state(),
        "torch_cuda": (
            torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
        ),
        "numpy": np.random.get_state(),
        "random": random.getstate(),
    }


def set_rng_state(rng_state):
    """Restore RNG state from a bundle produced by ``get_rng_state``."""
    import random
    import numpy as np

    if "torch" in rng_state:
        # ``torch.set_rng_state`` expects a legacy CPU ByteTensor.  Loading a
        # checkpoint with map_location='cuda' can produce a plain uint8 Tensor,
        # so force it to the expected type.
        torch_state = rng_state["torch"].cpu().type(torch.ByteTensor)
        torch.set_rng_state(torch_state)
    if "torch_cuda" in rng_state and torch.cuda.is_available():
        # set_rng_state_all accepts CPU ByteTensors and moves them to the GPU.
        cuda_states = [t.cpu().type(torch.ByteTensor) for t in rng_state["torch_cuda"]]
        torch.cuda.set_rng_state_all(cuda_states)
    if "numpy" in rng_state:
        np.random.set_state(rng_state["numpy"])
    if "random" in rng_state:
        random.setstate(rng_state["random"])


def save_checkpoint_raw(
    state_dict, optimizer, iter_num, best_val_loss, config, out_dir, filename
):
    """Backward-compatible wrapper that accepts a raw state_dict.

    Prefer :func:`save_checkpoint`; this wrapper exists for legacy callers
    that already compute an FSDP state dict.
    """
    return save_checkpoint(
        state_dict,
        optimizer,
        iter_num,
        best_val_loss,
        config,
        out_dir,
        filename,
    )
