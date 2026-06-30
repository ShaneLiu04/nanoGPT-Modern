"""
Supervised Fine-Tuning (SFT) on synthetic arithmetic data.

Uses the shared BaseTrainer infrastructure: AMP, GradScaler, gradient
accumulation, LR scheduler, and full-state checkpointing/resume.
"""

import os
import time
import argparse

import numpy as np
import torch
from torch.utils.data import DataLoader


from data.arithmetic import (
    generate_easy,
    generate_medium,
    generate_hard,
    ArithmeticDataset,
    collate_fn,
)
from model.attention_utils import set_attention_backend, print_attention_backend
from training.trainer_base import (
    BaseTrainer,
    load_model_from_checkpoint,
    maybe_warn_dropout,
    make_worker_init_fn,
)
from utils.lr_scheduler import LRScheduler
from utils.config import parse_args_with_config, to_dict
import tiktoken


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--init_from", type=str, required=True, help="Path to pretrain checkpoint"
    )
    parser.add_argument("--out_dir", type=str, default="out/sft")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of forward/backward passes before one optimizer step",
    )
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--min_lr", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--eval_interval", type=int, default=100)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--backend", type=str, default="nccl")
    parser.add_argument(
        "--variant", type=str, default="sft-only", choices=["sft-only", "sft-continued"]
    )
    parser.add_argument(
        "--num_train",
        type=int,
        default=5000,
        help="Samples per difficulty level for training",
    )
    parser.add_argument(
        "--num_val",
        type=int,
        default=500,
        help="Samples per difficulty level for validation",
    )
    parser.add_argument(
        "--lr_schedule",
        type=str,
        default="cosine",
        choices=["cosine", "linear", "wsd", "constant"],
        help="LR schedule type",
    )
    parser.add_argument(
        "--keep_last_n",
        type=int,
        default=0,
        help="Keep only the last N non-best checkpoints (0=keep all)",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Resume from a checkpoint saved by this script",
    )
    parser.add_argument(
        "--config", type=str, default=None, help="Path to YAML config file"
    )
    parser.add_argument(
        "--attn_backend",
        type=str,
        default="auto",
        choices=["auto", "flash", "mem_efficient", "math", "default"],
        help="Force SDPA attention backend (auto lets PyTorch choose)",
    )
    return parse_args_with_config(parser)


class SFTTrainer(BaseTrainer):
    def __init__(self, args):
        super().__init__(args)

    def _init_state(self):
        os.makedirs(self.args.out_dir, exist_ok=True)
        self.start_epoch = 0
        self.global_step = 0
        self.best_val_loss = float("inf")
        self.tokenizer = tiktoken.get_encoding("gpt2")

    def _build_data(self):
        args = self.args
        train_data = (
            generate_easy(args.num_train, seed=args.seed)
            + generate_medium(args.num_train, seed=args.seed + 1)
            + generate_hard(args.num_train, seed=args.seed + 2)
        )
        val_data = (
            generate_easy(args.num_val, seed=args.seed + 100)
            + generate_medium(args.num_val, seed=args.seed + 101)
            + generate_hard(args.num_val, seed=args.seed + 102)
        )

        train_ds = ArithmeticDataset(
            train_data, self.tokenizer, max_length=args.max_length, pre_tokenize=True
        )
        val_ds = ArithmeticDataset(
            val_data, self.tokenizer, max_length=args.max_length, pre_tokenize=True
        )

        # DistributedSampler for multi-GPU SFT (shuffle handled by dataset if needed)
        worker_init = make_worker_init_fn(args.seed, self.rank)
        if self.distributed:
            from torch.utils.data.distributed import DistributedSampler

            train_sampler = DistributedSampler(
                train_ds, num_replicas=self.world_size, rank=self.rank, shuffle=True
            )
            self.train_loader = DataLoader(
                train_ds,
                batch_size=args.batch_size,
                sampler=train_sampler,
                collate_fn=collate_fn,
                num_workers=0,
                pin_memory=True,
                worker_init_fn=worker_init,
            )
            self.train_sampler = train_sampler
        else:
            self.train_loader = DataLoader(
                train_ds,
                batch_size=args.batch_size,
                shuffle=True,
                collate_fn=collate_fn,
                num_workers=0,
                pin_memory=True,
                worker_init_fn=worker_init,
            )
            self.train_sampler = None

        self.val_loader = DataLoader(
            val_ds,
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=0,
            pin_memory=True,
            worker_init_fn=worker_init,
        )

    def _build_model(self):
        args = self.args
        if args.resume:
            self.model, _ = load_model_from_checkpoint(args.resume, device=self.device)
        else:
            self.model, _ = load_model_from_checkpoint(
                args.init_from, device=self.device
            )
        self.model = self.model.to(self.device)
        self.wrap_distributed(self.model)
        maybe_warn_dropout(self.raw_model)

    def _build_optimizer(self):
        args = self.args
        self.optimizer = self.raw_model.configure_optimizers(
            args.weight_decay,
            args.learning_rate,
            (0.9, 0.95),
            device_type="cuda" if "cuda" in str(self.device) else "cpu",
        )

    def _build_scheduler(self):
        args = self.args
        steps_per_epoch = len(self.train_loader)
        total_steps = steps_per_epoch * args.epochs // args.gradient_accumulation_steps
        self.scheduler = LRScheduler(
            schedule=args.lr_schedule,
            learning_rate=args.learning_rate,
            min_lr=args.min_lr,
            warmup_iters=max(1, total_steps // 20),  # 5% warmup
            lr_decay_iters=total_steps,
            max_iters=total_steps,
        )

    def _setup_checkpointing(self):
        self.configure_checkpointing(self.raw_model.config)

    def _setup_logger(self):
        args = self.args
        self.build_logger(
            project_name="nanogpt-modern-sft",
            run_name=args.variant,
            config=to_dict(args),
        )

    def _maybe_resume(self):
        args = self.args
        if args.resume:
            extra = self.load_checkpoint(args.resume)
            self.global_step = extra.get("iter_num", 0)
            self.best_val_loss = extra.get("best_val_loss", float("inf"))

    def _run_eval(self):
        self.model.eval()
        val_losses = []
        with torch.no_grad():
            for vbatch in self.val_loader:
                input_ids = vbatch["input_ids"].to(self.device)
                labels = vbatch["labels"].to(self.device)
                with self.ctx:
                    _, loss, _ = self.model(input_ids, targets=labels)
                val_losses.append(loss.item())
        self.model.train()
        return float(np.mean(val_losses)) if val_losses else float("inf")

    def train(self):
        args = self.args
        accum_steps = args.gradient_accumulation_steps
        t0 = time.time()

        for epoch in range(self.start_epoch, args.epochs):
            if self.train_sampler is not None:
                self.train_sampler.set_epoch(epoch)

            for batch in self.train_loader:
                input_ids = batch["input_ids"].to(self.device)
                labels = batch["labels"].to(self.device)

                with self.ctx:
                    logits, loss, _ = self.model(input_ids, targets=labels)

                if accum_steps > 1:
                    loss = loss / accum_steps

                if self.scaler is not None and self.scaler.is_enabled():
                    self.scaler.scale(loss).backward()
                else:
                    loss.backward()

                # Gradient accumulation: only step every accum_steps micro batches
                if (self.global_step + 1) % accum_steps == 0:
                    if args.grad_clip > 0:
                        if self.scaler is not None and self.scaler.is_enabled():
                            self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(), args.grad_clip
                        )

                    if self.scaler is not None and self.scaler.is_enabled():
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        self.optimizer.step()

                    self.optimizer.zero_grad(set_to_none=True)

                # LR scheduling
                lr = self.scheduler(self.global_step // accum_steps)
                for pg in self.optimizer.param_groups:
                    pg["lr"] = lr

                # Logging (use raw per-micro-batch loss)
                raw_loss = loss.item() * accum_steps
                if self.global_step % 10 == 0 and self.master_process:
                    t1 = time.time()
                    _dt = t1 - t0
                    t0 = t1
                    print(f"step {self.global_step}: loss {raw_loss:.4f}, lr {lr:.2e}")
                    self.log_scalars(
                        {
                            "train/loss": raw_loss,
                            "train/lr": lr,
                        },
                        self.global_step,
                    )

                # Evaluation + checkpointing
                if self.global_step % args.eval_interval == 0 and self.master_process:
                    avg_val = self._run_eval()
                    print(f"step {self.global_step}: val loss {avg_val:.4f}")
                    self.log_scalar("val/loss", avg_val, self.global_step)
                    if avg_val < self.best_val_loss:
                        self.best_val_loss = avg_val
                        self.save_checkpoint(
                            f"best_{args.variant}.pt",
                            self.global_step,
                            self.best_val_loss,
                        )

                self.global_step += 1

        if self.master_process:
            self.save_checkpoint(
                f"final_{args.variant}.pt", self.global_step, self.best_val_loss
            )


def main():
    args = get_args()
    set_attention_backend(args.attn_backend)
    print_attention_backend()
    trainer = SFTTrainer(args)
    try:
        trainer.train()
    finally:
        trainer.cleanup()


if __name__ == "__main__":
    main()
