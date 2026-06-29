"""
Learning-rate schedulers for pre-training, SFT, and GRPO.

All schedulers implement the same interface:

    scheduler(step: int) -> float

so they can be dropped into any training loop.
"""
import math


class LRScheduler:
    """Unified LR scheduler factory.

    Parameters
    ----------
    schedule : str
        One of `"cosine"`, `"linear"`, `"wsd"`, `"constant"`.
    learning_rate : float
        Peak / constant learning rate.
    min_lr : float
        Minimum learning rate (used for cosine/linear/wsd decay).
    warmup_iters : int
        Number of linear warmup steps (0 = no warmup).
    lr_decay_iters : int
        Total number of steps after which the LR reaches `min_lr`.
        For `"wsd"` this is the *decay phase* length (warmup+stable steps
        should be computed from `max_iters - lr_decay_iters`).
    max_iters : int
        Total training steps.  Only used by `"wsd"`.
    """

    def __init__(self, schedule, learning_rate, min_lr, warmup_iters,
                 lr_decay_iters, max_iters=None):
        schedule = schedule.lower()
        if schedule not in ("cosine", "linear", "wsd", "constant"):
            raise ValueError(f"Unknown schedule: {schedule}")

        self.schedule = schedule
        self.lr = learning_rate
        self.min_lr = min_lr
        self.warmup_iters = warmup_iters
        self.decay_iters = lr_decay_iters
        self.max_iters = max_iters

    def __call__(self, step):
        # --- warmup ---
        if step < self.warmup_iters:
            return self.lr * (step + 1) / max(self.warmup_iters, 1)

        # --- post-decay floor ---
        if self.schedule == "constant":
            return self.lr

        if self.schedule == "wsd":
            stable_end = self.max_iters - self.decay_iters
            if step < stable_end:
                return self.lr
            # decay phase
            ratio = (step - stable_end) / max(self.decay_iters, 1)
            ratio = min(ratio, 1.0)
            # cosine decay in decay phase
            coeff = 0.5 * (1.0 + math.cos(math.pi * ratio))
            return self.min_lr + coeff * (self.lr - self.min_lr)

        # --- beyond decay horizon ---
        if step >= self.decay_iters:
            return self.min_lr

        # --- decay phase ---
        ratio = (step - self.warmup_iters) / max(self.decay_iters - self.warmup_iters, 1)
        if self.schedule == "cosine":
            coeff = 0.5 * (1.0 + math.cos(math.pi * ratio))
        elif self.schedule == "linear":
            coeff = 1.0 - ratio
        else:
            coeff = 1.0  # fallback (shouldn't reach here)
        return self.min_lr + coeff * (self.lr - self.min_lr)

    def state_dict(self):
        """Return a serializable snapshot of the scheduler hyper-parameters."""
        return {
            "schedule": self.schedule,
            "lr": self.lr,
            "min_lr": self.min_lr,
            "warmup_iters": self.warmup_iters,
            "decay_iters": self.decay_iters,
            "max_iters": self.max_iters,
        }

    def load_state_dict(self, state):
        """Restore scheduler hyper-parameters from a checkpoint."""
        self.schedule = state.get("schedule", self.schedule)
        self.lr = state.get("lr", self.lr)
        self.min_lr = state.get("min_lr", self.min_lr)
        self.warmup_iters = state.get("warmup_iters", self.warmup_iters)
        self.decay_iters = state.get("decay_iters", self.decay_iters)
        self.max_iters = state.get("max_iters", self.max_iters)
