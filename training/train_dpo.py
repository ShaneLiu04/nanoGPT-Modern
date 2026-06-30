"""Direct Preference Optimization (DPO) / IPO / KTO trainer.

Complete implementation including:
  * Preference pair construction from GRPO group_size rollouts (winner vs loser).
  * Full training loop with evaluation, checkpointing, and metric logging.
  * Win-rate evaluation against the reference (SFT) checkpoint.

Usage
-----
>>> python training/train_dpo.py \
...     --init_from out/sft/best_sft-only.pt \
...     --ref_from out/sft/best_sft-only.pt \
...     --preference_source grpo \
...     --grpo_checkpoint out/grpo/best_grpo_g4.pt \
...     --out_dir out/dpo
"""

from __future__ import annotations

import argparse
import json
import os
import random
from contextlib import nullcontext
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, Dataset

from model.attention_utils import print_attention_backend, set_attention_backend
from model.modern_gpt import ModernGPTConfig
from training.trainer_base import (
    BaseTrainer,
    load_model_from_checkpoint,
    make_worker_init_fn,
    maybe_warn_dropout,
)
from utils.config import parse_args_with_config, to_dict
from utils.dpo_utils import (
    compute_dpo_loss,
    compute_ipo_loss,
    compute_kto_loss,
    compute_sequence_logprob,
)
from utils.lr_scheduler import LRScheduler


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--init_from",
        type=str,
        required=True,
        help="Policy checkpoint (SFT or previous DPO)",
    )
    parser.add_argument(
        "--ref_from",
        type=str,
        default=None,
        help="Reference checkpoint; defaults to init_from if not provided",
    )
    parser.add_argument("--out_dir", type=str, default="out/dpo")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=1e-6)
    parser.add_argument("--min_lr", type=float, default=1e-7)
    parser.add_argument(
        "--lr_schedule",
        type=str,
        default="cosine",
        choices=["cosine", "linear", "wsd", "constant"],
    )
    parser.add_argument(
        "--warmup_iters", type=int, default=0, help="Linear warmup steps (0 = none)"
    )
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument(
        "--beta", type=float, default=0.1, help="DPO/IPO/KTO temperature"
    )
    parser.add_argument(
        "--preference_loss", type=str, default="dpo", choices=["dpo", "ipo", "kto"]
    )
    parser.add_argument(
        "--label_smoothing", type=float, default=0.0, help="DPO label smoothing"
    )
    parser.add_argument("--eval_interval", type=int, default=50)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--backend", type=str, default="nccl")
    parser.add_argument("--keep_last_n", type=int, default=0)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument(
        "--attn_backend",
        type=str,
        default="auto",
        choices=["auto", "flash", "mem_efficient", "math", "default"],
    )
    # Dataset sources
    parser.add_argument(
        "--preference_source",
        type=str,
        default="synthetic",
        choices=["synthetic", "grpo", "jsonl"],
    )
    parser.add_argument(
        "--grpo_checkpoint",
        type=str,
        default=None,
        help="GRPO checkpoint path used to load policy for GRPO-based preference generation",
    )
    parser.add_argument(
        "--grpo_group_size",
        type=int,
        default=4,
        help="Group size for GRPO rollout when constructing preference pairs",
    )
    parser.add_argument("--num_train", type=int, default=256)
    parser.add_argument("--num_val", type=int, default=64)
    parser.add_argument("--vocab_size", type=int, default=50257)
    # Win-rate evaluation
    parser.add_argument(
        "--win_rate_eval",
        action="store_true",
        help="Run win-rate evaluation against the reference model during training",
    )
    parser.add_argument(
        "--win_rate_interval",
        type=int,
        default=100,
        help="Run win-rate evaluation every N steps",
    )
    parser.add_argument(
        "--win_rate_samples",
        type=int,
        default=50,
        help="Number of prompts for win-rate evaluation",
    )
    return parse_args_with_config(parser)


def _build_synthetic_preference_dataset(
    num_samples: int, seq_len: int, vocab_size: int, seed: int
):
    """Create a synthetic preference dataset for testing/demonstration."""
    torch.manual_seed(seed)
    chosen = torch.randint(0, vocab_size, (num_samples, seq_len))
    rejected = torch.randint(0, vocab_size, (num_samples, seq_len))
    return TensorDataset(chosen, rejected)


class PreferenceDataset(Dataset):
    """Simple wrapper around a list of (chosen, rejected) token-id tensors."""

    def __init__(self, pairs: List[Tuple[torch.Tensor, torch.Tensor]]):
        self.pairs = pairs

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.pairs[idx]


def _build_preference_dataset_from_grpo(
    grpo_trainer,
    num_prompts: int,
    max_length: int,
    seed: int,
) -> PreferenceDataset:
    """Construct preference pairs from GRPO ``group_size`` rollouts.

    For each prompt, the GRPO policy generates ``group_size`` responses.
    The highest-reward response becomes ``chosen`` and the lowest-reward
    response becomes ``rejected``.  Both are padded / truncated to
    ``max_length`` token ids.

    Parameters
    ----------
    grpo_trainer : GRPOTrainer-like object
        Must have ``sample_group`` and ``tokenizer`` attributes.
    num_prompts : int
    max_length : int
    seed : int

    Returns
    -------
    PreferenceDataset
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Sample prompts from the GRPO trainer's training data.
    data = grpo_trainer.train_data
    if len(data) < num_prompts:
        num_prompts = len(data)
    prompts_batch = random.sample(data, num_prompts)
    prompts = [b["prompt"] for b in prompts_batch]
    answers = [b["answer"] for b in prompts_batch]

    pairs = []
    for prompt, answer in zip(prompts, answers):
        # Single-prompt rollout with group_size responses.
        rollout = grpo_trainer.sample_group([prompt], [answer])
        rewards = rollout["rewards"]  # [G, 1]
        response_ids = rollout["response_ids"]  # list[G] of list[1] of list[int]
        # rewards shape: [group_size, batch_size=1]
        g_size = rewards.shape[0]
        # Flatten rewards and responses.
        flat_rewards = [rewards[g, 0] for g in range(g_size)]
        flat_responses = [response_ids[g][0] for g in range(g_size)]

        # Winner = highest reward, loser = lowest reward.
        winner_idx = int(np.argmax(flat_rewards))
        loser_idx = int(np.argmin(flat_rewards))
        winner_ids = flat_responses[winner_idx]
        loser_ids = flat_responses[loser_idx]

        # Pad / truncate to max_length.
        def pad_or_trim(ids: List[int]) -> torch.Tensor:
            if len(ids) >= max_length:
                return torch.tensor(ids[:max_length], dtype=torch.long)
            return torch.tensor(ids + [0] * (max_length - len(ids)), dtype=torch.long)

        pairs.append((pad_or_trim(winner_ids), pad_or_trim(loser_ids)))

    return PreferenceDataset(pairs)


class DPOTrainer(BaseTrainer):
    """Direct Preference Optimization trainer with full loop and eval."""

    def __init__(self, args):
        super().__init__(args)

    def _init_state(self):
        os.makedirs(self.args.out_dir, exist_ok=True)
        self.start_epoch = 0
        self.global_step = 0
        self.best_metric = float("inf")

    def _build_data(self):
        args = self.args
        vocab_size = getattr(args, "vocab_size", 50257)
        if args.preference_source == "synthetic":
            train_ds = _build_synthetic_preference_dataset(
                num_samples=getattr(args, "num_train", 256),
                seq_len=args.max_length,
                vocab_size=vocab_size,
                seed=args.seed,
            )
            val_ds = _build_synthetic_preference_dataset(
                num_samples=getattr(args, "num_val", 64),
                seq_len=args.max_length,
                vocab_size=vocab_size,
                seed=args.seed + 1000,
            )
        elif args.preference_source == "grpo":
            if args.grpo_checkpoint is None:
                raise ValueError(
                    "--grpo_checkpoint is required when --preference_source=grpo"
                )
            # Load GRPO trainer to generate preference pairs.
            from training.train_grpo import GRPOTrainer, get_args as grpo_get_args

            # Build a minimal GRPO args object from the loaded checkpoint.
            grpo_ckpt = torch.load(
                args.grpo_checkpoint, map_location=self.device, weights_only=False
            )
            grpo_config = grpo_ckpt.get("config", {})
            # Build synthetic GRPO trainer for rollouts.
            grpo_args = grpo_get_args()
            # Override defaults with checkpoint config.
            for k, v in grpo_config.items():
                if hasattr(grpo_args, k):
                    setattr(grpo_args, k, v)
            grpo_args.device = self.device
            grpo_trainer = GRPOTrainer(grpo_args)
            # Load the checkpoint into the GRPO policy so rollouts use the trained policy.
            grpo_trainer.load_checkpoint(args.grpo_checkpoint)
            train_ds = _build_preference_dataset_from_grpo(
                grpo_trainer,
                num_prompts=getattr(args, "num_train", 256),
                max_length=args.max_length,
                seed=args.seed,
            )
            val_ds = _build_preference_dataset_from_grpo(
                grpo_trainer,
                num_prompts=getattr(args, "num_val", 64),
                max_length=args.max_length,
                seed=args.seed + 1000,
            )
        else:
            raise ValueError(f"Unknown preference_source: {args.preference_source}")

        worker_init = make_worker_init_fn(args.seed, self.rank)
        self.train_loader = DataLoader(
            train_ds,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=True,
            worker_init_fn=worker_init,
        )
        self.val_loader = DataLoader(
            val_ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True,
            worker_init_fn=worker_init,
        )

    def _build_model(self):
        args = self.args
        policy, _ = load_model_from_checkpoint(args.init_from, device=self.device)
        ref_path = args.ref_from if args.ref_from is not None else args.init_from
        ref_model, _ = load_model_from_checkpoint(ref_path, device=self.device)
        ref_model.eval()
        for p in ref_model.parameters():
            p.requires_grad = False

        maybe_warn_dropout(policy)
        self.ref_model = ref_model
        self.policy_model = policy
        self.wrap_distributed(policy)

    def _build_optimizer(self):
        args = self.args
        self.optimizer = self.raw_model.configure_optimizers(
            weight_decay=args.weight_decay,
            learning_rate=args.learning_rate,
            betas=(0.9, 0.95),
            device_type="cuda" if self.device.startswith("cuda") else "cpu",
        )

    def _build_scheduler(self):
        args = self.args
        total_steps = (
            len(self.train_loader) * args.epochs // args.gradient_accumulation_steps
        )
        self.scheduler = LRScheduler(
            schedule=args.lr_schedule,
            learning_rate=args.learning_rate,
            min_lr=args.min_lr,
            warmup_iters=getattr(args, "warmup_iters", 0),
            lr_decay_iters=total_steps,
            max_iters=total_steps,
        )

    def _setup_checkpointing(self):
        self.configure_checkpointing(self.raw_model.config)

    def _setup_logger(self):
        args = self.args
        self.build_logger(
            project_name="nanogpt-modern-dpo",
            run_name=f"dpo_{args.preference_loss}",
            config=to_dict(args),
        )

    def _maybe_resume(self):
        if self.args.resume:
            extra = self.load_checkpoint(self.args.resume)
            self.start_epoch = extra.get("epoch", 0) + 1
            self.global_step = extra.get("iter_num", 0)
            self.best_metric = extra.get("best_metric", float("inf"))

    def _compute_logprobs(self, model: nn.Module, tokens: torch.Tensor) -> torch.Tensor:
        """Return per-sequence log-prob for ``tokens`` under ``model``."""
        with torch.no_grad() if model is self.ref_model else nullcontext():
            logits, _, _ = model(tokens)
        return compute_sequence_logprob(logits, tokens)

    def _compute_loss(
        self, chosen: torch.Tensor, rejected: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        policy_chosen_logp = self._compute_logprobs(self.model, chosen)
        policy_rejected_logp = self._compute_logprobs(self.model, rejected)
        ref_chosen_logp = self._compute_logprobs(self.ref_model, chosen)
        ref_rejected_logp = self._compute_logprobs(self.ref_model, rejected)

        if self.args.preference_loss == "dpo":
            loss, metrics = compute_dpo_loss(
                policy_chosen_logp,
                policy_rejected_logp,
                ref_chosen_logp,
                ref_rejected_logp,
                beta=self.args.beta,
                label_smoothing=self.args.label_smoothing,
            )
        elif self.args.preference_loss == "ipo":
            loss, metrics = compute_ipo_loss(
                policy_chosen_logp,
                policy_rejected_logp,
                ref_chosen_logp,
                ref_rejected_logp,
                beta=self.args.beta,
            )
        else:  # kto
            policy_logp = torch.cat([policy_chosen_logp, policy_rejected_logp])
            ref_logp = torch.cat([ref_chosen_logp, ref_rejected_logp])
            is_desirable = torch.cat(
                [
                    torch.ones_like(policy_chosen_logp, dtype=torch.bool),
                    torch.zeros_like(policy_rejected_logp, dtype=torch.bool),
                ]
            )
            loss, metrics = compute_kto_loss(
                policy_logp, ref_logp, is_desirable, beta=self.args.beta
            )
        return loss, metrics

    def _set_lr(self) -> float:
        """Apply the scheduler's current LR to all optimizer param groups."""
        lr = self.scheduler(self.global_step)
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr
        return lr

    def _run_epoch(self, epoch: int, is_train: bool = True) -> float:
        """Run one epoch over the preference dataset."""
        loader = self.train_loader if is_train else self.val_loader
        self.model.train(is_train)
        total_loss = 0.0
        num_batches = 0
        for chosen, rejected in loader:
            chosen = chosen.to(self.device, non_blocking=True)
            rejected = rejected.to(self.device, non_blocking=True)

            with self.ctx:
                loss, metrics = self._compute_loss(chosen, rejected)

            if is_train:
                loss = loss / self.args.gradient_accumulation_steps
                if self.scaler is not None:
                    self.scaler.scale(loss).backward()
                else:
                    loss.backward()

                if (num_batches + 1) % self.args.gradient_accumulation_steps == 0:
                    if self.scaler is not None:
                        self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.args.grad_clip
                    )
                    if self.scaler is not None:
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        self.optimizer.step()
                    self.optimizer.zero_grad(set_to_none=True)
                    lr = self._set_lr()
                    self.global_step += 1

                    if self.master_process:
                        self.log_scalars({"train/lr": lr}, self.global_step)

                    if self.global_step % self.args.eval_interval == 0:
                        val_loss = self._evaluate()
                        if self.master_process:
                            print(f"step {self.global_step}: val_loss={val_loss:.4f}")
                            self.log_scalars({"val/loss": val_loss}, self.global_step)
                            if val_loss < self.best_metric:
                                self.best_metric = val_loss
                                self.save_checkpoint(
                                    "best_ckpt.pt", self.global_step, self.best_metric
                                )

            total_loss += loss.detach().item()
            num_batches += 1

            if self.master_process and is_train and num_batches % 10 == 0:
                self.log_scalars(
                    {f"train/{k}": v for k, v in metrics.items()}, self.global_step
                )

        return total_loss / max(num_batches, 1)

    def _evaluate(self) -> float:
        """Evaluate on the validation preference dataset."""
        with torch.no_grad():
            return self._run_epoch(0, is_train=False)

    def _evaluate_win_rate(self, num_samples: int = 50) -> Dict[str, Any]:
        """Compare policy vs reference on a held-out prompt set and return win rate.

        Uses the ``evaluation.eval_win_rate`` module with a rule-based judge.
        """
        try:
            import tiktoken

            tokenizer = tiktoken.get_encoding("gpt2")
        except Exception:
            return {"error": "tiktoken not available for win-rate evaluation"}

        try:
            from evaluation.eval_win_rate import WinRateEvaluator, RuleJudge

            prompts = [f"What is {i} + {i+1}?" for i in range(num_samples)]
            evaluator = WinRateEvaluator(
                policy_model=self.raw_model,
                ref_model=self.ref_model,
                tokenizer=tokenizer,
                judge=RuleJudge(),
                device=self.device,
            )
            results = evaluator.evaluate(
                prompts,
                n_samples=1,
                max_new_tokens=32,
                temperature=1.0,
            )
            return {
                "win_rate": results.get("win_rate", 0.0),
                "tie_rate": results.get("tie_rate", 0.0),
                "policy_mean_score": results.get("policy_mean_score", 0.0),
                "ref_mean_score": results.get("ref_mean_score", 0.0),
            }
        except Exception as e:
            return {"error": str(e)}

    def train(self) -> None:
        """Full DPO training loop with periodic evaluation, checkpointing, and optional win-rate metrics."""
        args = self.args
        for epoch in range(self.start_epoch, args.epochs):
            train_loss = self._run_epoch(epoch, is_train=True)
            if self.master_process:
                print(f"epoch {epoch}: train_loss={train_loss:.4f}")
                self.save_checkpoint(
                    "latest_ckpt.pt",
                    self.global_step,
                    self.best_metric,
                )

            # Win-rate evaluation at the end of each epoch or by interval.
            if self.master_process and args.win_rate_eval:
                if (epoch + 1) % max(
                    1, args.win_rate_interval // len(self.train_loader)
                ) == 0 or epoch == args.epochs - 1:
                    win_rate_metrics = self._evaluate_win_rate(
                        num_samples=args.win_rate_samples
                    )
                    print(f"win_rate metrics: {win_rate_metrics}")
                    self.log_scalars(
                        {
                            f"win_rate/{k}": v
                            for k, v in win_rate_metrics.items()
                            if isinstance(v, (int, float))
                        },
                        self.global_step,
                    )
                    # Save win-rate results to JSON.
                    wr_path = os.path.join(
                        args.out_dir, f"win_rate_step{self.global_step}.json"
                    )
                    with open(wr_path, "w", encoding="utf-8") as f:
                        json.dump(win_rate_metrics, f, indent=2)

        if self.master_process:
            self.save_checkpoint("final_ckpt.pt", self.global_step, self.best_metric)
            print(f"Training complete. Best val loss: {self.best_metric:.4f}")


def main():
    args = get_args()
    set_attention_backend(args.attn_backend)
    print_attention_backend()
    trainer = DPOTrainer(args)
    try:
        trainer.train()
    finally:
        trainer.cleanup()


if __name__ == "__main__":
    main()
