"""Shared argparse assembly for the five trainer scripts (FR11a).

``add_common_trainer_args`` registers the 20 flags that are duplicated
across train_pretrain / train_sft / train_grpo / train_dpo /
iterative_grpo.  The module is pure assembly: it only calls
``parser.add_argument`` and never parses anything, so it stays unit
testable (each script keeps its own ``parse_args_with_config`` call and
its script-specific flags/defaults).
"""

import argparse

import torch

LR_SCHEDULE_CHOICES = ["cosine", "linear", "wsd", "constant"]
ATTN_BACKEND_CHOICES = ["auto", "flash", "mem_efficient", "math", "default"]


def add_common_trainer_args(parser: argparse.ArgumentParser) -> None:
    """Add the trainer flags shared by all five training scripts."""
    parser.add_argument("--seed", type=int, default=1337, help="Random seed for reproducibility")
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Compute device (default: cuda if available, else cpu)",
    )
    parser.add_argument(
        "--backend", type=str, default="nccl", help="Distributed communication backend"
    )
    parser.add_argument(
        "--grad_clip",
        type=float,
        default=1.0,
        help="Gradient clipping magnitude (0 = disabled)",
    )
    parser.add_argument(
        "--use_wandb",
        action="store_true",
        help="Log metrics to Weights & Biases",
    )
    parser.add_argument(
        "--exp_id",
        type=str,
        default=None,
        help="Experiment ID for structured metrics (default: basename of --out_dir)",
    )
    parser.add_argument(
        "--no_metrics",
        action="store_true",
        help="Disable structured metrics.jsonl / manifest.json recording",
    )
    parser.add_argument(
        "--no_monitoring",
        action="store_true",
        help="Disable loss-spike detection and grad-norm monitoring",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Resume from a checkpoint saved by this script",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML config file",
    )
    parser.add_argument(
        "--ckpt_interval",
        type=int,
        default=0,
        help="Save latest_ckpt.pt every N iters between eval points " "(0 = follow eval_interval)",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of accumulation steps before one optimizer step",
    )
    parser.add_argument(
        "--lr_schedule",
        type=str,
        default="cosine",
        choices=LR_SCHEDULE_CHOICES,
        help="LR schedule type",
    )
    parser.add_argument(
        "--keep_last_n",
        type=int,
        default=0,
        help="Keep only the last N non-best checkpoints (0=keep all)",
    )
    parser.add_argument(
        "--attn_backend",
        type=str,
        default="auto",
        choices=ATTN_BACKEND_CHOICES,
        help="Force SDPA attention backend (auto lets PyTorch choose)",
    )
    parser.add_argument(
        "--group_size",
        type=int,
        default=4,
        help="GRPO group size (rollouts per prompt)",
    )
    parser.add_argument(
        "--num_steps",
        type=int,
        default=1000,
        help="Total number of training steps",
    )
    parser.add_argument(
        "--max_prompt_len",
        type=int,
        default=128,
        help="Maximum prompt length in tokens",
    )
    parser.add_argument(
        "--max_response_len",
        type=int,
        default=128,
        help="Maximum response length in tokens",
    )
    parser.add_argument(
        "--eps",
        type=float,
        default=0.2,
        help="PPO clipping epsilon",
    )
