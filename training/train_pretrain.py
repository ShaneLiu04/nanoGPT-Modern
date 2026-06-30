"""
Pretraining script for BaselineGPT and ModernGPT on OpenWebText.

Refactored to use the shared BaseTrainer infrastructure: DDP/FSDP, AMP,
GradScaler, gradient accumulation, LR scheduler, EMA, checkpointing, and
resumable training.
"""

import os
import time
import argparse

import numpy as np
import torch
import torch.distributed as dist


from model.attention_utils import set_attention_backend, print_attention_backend
from model.baseline_gpt import BaselineGPT, BaselineGPTConfig
from model.modern_gpt import ModernGPT, ModernGPTConfig
from data.openwebtext import get_dataloader
from training.trainer_base import (
    BaseTrainer,
    load_model_from_checkpoint,
    get_rng_state,
    make_worker_init_fn,
)
from utils.lr_scheduler import LRScheduler
from utils.config import parse_args_with_config


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", type=str, default="modern", choices=["baseline", "modern"]
    )
    parser.add_argument("--data_dir", type=str, default="data/openwebtext")
    parser.add_argument("--out_dir", type=str, default="out/pretrain")
    parser.add_argument("--batch_size", type=int, default=12)
    parser.add_argument("--block_size", type=int, default=1024)
    parser.add_argument("--n_layer", type=int, default=9)
    parser.add_argument("--n_head", type=int, default=8)
    parser.add_argument("--n_embd", type=int, default=512)
    parser.add_argument(
        "--n_kv_head",
        type=int,
        default=None,
        help="Number of KV heads for GQA (default: n_head for MHA)",
    )
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--learning_rate", type=float, default=6e-4)
    parser.add_argument("--max_iters", type=int, default=18000)
    parser.add_argument("--warmup_iters", type=int, default=2000)
    parser.add_argument("--lr_decay_iters", type=int, default=18000)
    parser.add_argument("--min_lr", type=float, default=6e-5)
    parser.add_argument(
        "--lr_schedule",
        type=str,
        default="cosine",
        choices=["cosine", "linear", "wsd", "constant"],
        help="LR schedule type",
    )
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--eval_interval", type=int, default=1000)
    parser.add_argument("--eval_iters", type=int, default=200)
    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--backend", type=str, default="nccl")
    parser.add_argument(
        "--fsdp",
        action="store_true",
        help="Use Fully Sharded Data Parallel instead of DDP",
    )
    parser.add_argument(
        "--fsdp_sharding_strategy",
        type=str,
        default="full",
        choices=["full", "grad", "no"],
        help="FSDP sharding: full (params+grads+opt), grad (grads+opt), no (DDP-like)",
    )
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of forward/backward passes before one optimizer step",
    )
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument(
        "--use_ema",
        action="store_true",
        help="Enable Exponential Moving Average of weights",
    )
    parser.add_argument("--ema_decay", type=float, default=0.999, help="EMA decay rate")
    parser.add_argument(
        "--early_stopping_patience",
        type=int,
        default=0,
        help="Stop if val loss does not improve for N evals (0=disabled)",
    )
    parser.add_argument(
        "--keep_last_n",
        type=int,
        default=0,
        help="Keep only the last N non-best checkpoints (0=keep all)",
    )
    parser.add_argument(
        "--use_packing",
        action="store_true",
        help="Use document packing with cross-document attention mask",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Enable gradient checkpointing to trade compute for memory",
    )
    parser.add_argument(
        "--shuffle_buffer",
        type=int,
        default=None,
        help="MemmapDataset shuffle buffer size (default: 10000 or dataset size)",
    )
    parser.add_argument("--resume", type=str, default=None)
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
    parser.add_argument(
        "--use_ring_attention",
        action="store_true",
        help="Use the pure-PyTorch blockwise (Ring) attention fallback",
    )
    parser.add_argument(
        "--ring_block_size_q",
        type=int,
        default=64,
        help="Query block size for ring attention",
    )
    parser.add_argument(
        "--ring_block_size_kv",
        type=int,
        default=64,
        help="KV block size for ring attention",
    )
    return parse_args_with_config(parser)


class PretrainTrainer(BaseTrainer):
    """Pretraining trainer for BaselineGPT / ModernGPT on OpenWebText."""

    def __init__(self, args):
        super().__init__(args)

    def _init_state(self):
        if self.master_process:
            os.makedirs(self.args.out_dir, exist_ok=True)
        self.iter_num = 0
        self.best_val_loss = float("inf")
        self.early_stop_counter = 0
        self.resume_offset = 0

    def _build_data(self):
        args = self.args
        resume_offset = getattr(self, "resume_offset", 0)
        worker_init = (
            make_worker_init_fn(args.seed, self.rank) if args.num_workers > 0 else None
        )
        self.train_loader = get_dataloader(
            args.data_dir,
            "train",
            args.batch_size,
            args.block_size,
            args.num_workers,
            resume_offset=resume_offset,
            worker_init_fn=worker_init,
            use_packing=args.use_packing,
            shuffle_buffer=args.shuffle_buffer,
        )
        self.val_loader = get_dataloader(
            args.data_dir,
            "val",
            args.batch_size,
            args.block_size,
            args.num_workers,
            worker_init_fn=worker_init,
            use_packing=args.use_packing,
        )
        self.train_iter = iter(self.train_loader)

    def _build_model(self):
        args = self.args
        if args.resume:
            self.model, _ = load_model_from_checkpoint(args.resume, device=self.device)
        else:
            if args.model == "baseline":
                config = BaselineGPTConfig(
                    block_size=args.block_size,
                    n_layer=args.n_layer,
                    n_head=args.n_head,
                    n_embd=args.n_embd,
                    dropout=args.dropout,
                    attention_backend="sdpa",
                    gradient_checkpointing=args.gradient_checkpointing,
                )
                self.model = BaselineGPT(config)
            else:
                config = ModernGPTConfig(
                    block_size=args.block_size,
                    n_layer=args.n_layer,
                    n_head=args.n_head,
                    n_embd=args.n_embd,
                    n_kv_head=args.n_kv_head,
                    dropout=args.dropout,
                    gradient_checkpointing=args.gradient_checkpointing,
                    use_ring_attention=args.use_ring_attention,
                    ring_block_size_q=args.ring_block_size_q,
                    ring_block_size_kv=args.ring_block_size_kv,
                )
                self.model = ModernGPT(config)
            self.model = self.model.to(self.device)

        self.config = self.model.config
        self.wrap_distributed(self.model, compile_model=args.compile)

        if args.model == "modern" and self.master_process:
            print(f"[Config] {self.raw_model.config.describe()}")
        elif self.master_process:
            print(f"[Config] BaselineGPT: n_params={self.raw_model.get_num_params():,}")

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
        self.scheduler = LRScheduler(
            schedule=args.lr_schedule,
            learning_rate=args.learning_rate,
            min_lr=args.min_lr,
            warmup_iters=args.warmup_iters,
            lr_decay_iters=args.lr_decay_iters,
            max_iters=args.max_iters,
        )

    def _setup_amp(self, use_bf16=True):
        args = self.args
        if args.use_ema:
            self.raw_model.init_ema(decay=args.ema_decay)
        super()._setup_amp(use_bf16=use_bf16)

    def _setup_checkpointing(self):
        self.configure_checkpointing(
            self.config,
            ema_shadow=getattr(self.raw_model, "ema_shadow", None),
        )

    def _setup_logger(self):
        args = self.args
        self.build_logger(
            project_name="nanogpt-modern-pretrain",
            run_name=f"{args.model}_l{args.n_layer}_h{args.n_head}_d{args.n_embd}",
        )

    def _maybe_resume(self):
        args = self.args
        if args.resume:
            extra = self.load_checkpoint(args.resume)
            self.iter_num = extra.get("iter_num", 0)
            self.best_val_loss = extra.get("best_val_loss", float("inf"))
            self.resume_offset = extra.get("resume_offset", 0)
            # Data loader needs to be rebuilt with the resumed offset.
            self._build_data()

    def _forward(self, batch):
        """Forward pass compatible with both model types and optional document_ids."""
        if len(batch) == 3:
            input_ids, targets, document_ids = batch
            document_ids = document_ids.to(self.device)
        else:
            input_ids, targets = batch
            document_ids = None
        if self.args.model == "modern":
            return self.model(input_ids, targets=targets, document_ids=document_ids)
        # BaselineGPT returns (logits, loss); pad to the 3-tuple ModernGPT API.
        logits, loss = self.model(input_ids, targets)
        return logits, loss, None

    @torch.no_grad()
    def estimate_loss(self):
        self.model.eval()
        out = {}
        for split, loader in [("train", self.train_loader), ("val", self.val_loader)]:
            losses = torch.zeros(self.args.eval_iters)
            for k, batch in enumerate(loader):
                if k >= self.args.eval_iters:
                    break
                batch = [t.to(self.device) for t in batch]
                with self.ctx:
                    _, loss, _ = self._forward(batch)
                losses[k] = loss.item()
            out[split] = losses.mean().item()
        self.model.train()
        return out

    def _get_model_state_dict(self):
        """Return a state dict suitable for checkpointing (handles FSDP)."""
        if self.args.fsdp and self.distributed:
            from torch.distributed.fsdp import (
                FullyShardedDataParallel as FSDP,
                StateDictType,
            )

            FSDP.set_state_dict_type(self.model, StateDictType.FULL_STATE_DICT)
            return self.model.state_dict()
        return self.raw_model.state_dict()

    def _save_checkpoint(self, filename):
        ema_shadow = (
            self.raw_model.ema_shadow
            if (self.args.use_ema and hasattr(self.raw_model, "ema_shadow"))
            else None
        )
        return self.ckpt_manager.save(
            filename,
            self.iter_num,
            self.best_val_loss,
            rng_state=get_rng_state(),
            resume_offset=self.resume_offset,
            ema_shadow=ema_shadow,
        )

    def _save_ema_checkpoint(self):
        if not (self.args.use_ema and hasattr(self.raw_model, "ema_shadow")):
            return
        self.raw_model.apply_ema_weights()
        self._save_checkpoint("ema_ckpt.pt")
        self.raw_model.restore_ema_weights()

    def train(self):
        args = self.args
        accum_steps = args.gradient_accumulation_steps
        micro_step = 0
        t0 = time.time()
        running_loss = 0.0
        running_count = 0

        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        while self.iter_num < args.max_iters:
            lr = self.scheduler(self.iter_num)
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = lr

            try:
                batch = next(self.train_iter)
            except StopIteration:
                self.train_iter = iter(self.train_loader)
                batch = next(self.train_iter)

            batch = [t.to(self.device) for t in batch]

            with self.ctx:
                _, loss, _ = self._forward(batch)

            if accum_steps > 1:
                loss = loss / accum_steps

            if self.scaler is not None and self.scaler.is_enabled():
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

            if self.distributed:
                dist.all_reduce(loss.detach() * accum_steps, op=dist.ReduceOp.AVG)

            micro_step += 1
            raw_loss = loss.item() * accum_steps if accum_steps > 1 else loss.item()
            running_loss += raw_loss
            running_count += 1

            if micro_step % accum_steps == 0:
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

                if args.use_ema and hasattr(self.raw_model, "update_ema"):
                    self.raw_model.update_ema()

            if self.iter_num % args.log_interval == 0 and self.master_process:
                lossf = running_loss / max(running_count, 1)
                t1 = time.time()
                dt = t1 - t0
                t0 = t1
                tokens_per_sec = (
                    (args.batch_size * args.block_size * running_count)
                    / dt
                    / max(self.world_size, 1)
                )
                msg = (
                    f"iter {self.iter_num}: loss {lossf:.4f}, lr {lr:.2e}, "
                    f"tok/s {tokens_per_sec:.2f}"
                )
                if accum_steps > 1:
                    msg += f" (accum={accum_steps})"
                print(msg)
                self.log_scalars(
                    {
                        "train/loss": lossf,
                        "train/lr": lr,
                        "train/tokens_per_sec": tokens_per_sec,
                    },
                    self.iter_num,
                )
                self.log_memory_stats(self.iter_num)
                running_loss = 0.0
                running_count = 0

            if (
                self.iter_num > 0
                and self.iter_num % args.eval_interval == 0
                and self.master_process
            ):
                losses = self.estimate_loss()
                print(
                    f"iter {self.iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}"
                )
                self.log_scalars(
                    {
                        "val/loss": losses["val"],
                        "train/avg_loss": losses["train"],
                    },
                    self.iter_num,
                )

                if losses["val"] < self.best_val_loss:
                    self.best_val_loss = losses["val"]
                    self._save_checkpoint("best_ckpt.pt")
                    self.early_stop_counter = 0
                else:
                    self.early_stop_counter += 1

                self._save_checkpoint("latest_ckpt.pt")
                self._save_ema_checkpoint()

                if (
                    args.early_stopping_patience > 0
                    and self.early_stop_counter >= args.early_stopping_patience
                ):
                    print(
                        f"Early stopping triggered at iter {self.iter_num} "
                        f"(val loss did not improve for {args.early_stopping_patience} evals)"
                    )
                    return

            self.iter_num += 1

        if self.master_process:
            losses = self.estimate_loss()
            print(
                f"final: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}"
            )
            self.log_scalars({"val/loss": losses["val"]}, self.iter_num)
            self._save_checkpoint("latest_ckpt.pt")
            if losses["val"] < self.best_val_loss:
                self.best_val_loss = losses["val"]
                self._save_checkpoint("best_ckpt.pt")


def main():
    args = get_args()
    set_attention_backend(args.attn_backend)
    print_attention_backend()
    trainer = PretrainTrainer(args)
    try:
        trainer.train()
    finally:
        trainer.cleanup()


if __name__ == "__main__":
    main()
