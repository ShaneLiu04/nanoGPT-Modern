"""
Iterative GRPO with RLHF: reference model updates and rejection sampling.

Builds on the optimized GRPOTrainer in ``training/train_grpo.py`` and adds:
  1. Iterative RLHF — periodically update the reference model from the current
     best policy (EMA-style mixing).
  2. Rejection Sampling — periodically sample a large batch per prompt, keep
     only the top-K highest-reward responses, and run a short SFT phase on the
     filtered data.

All GRPO optimizations are preserved: batched rollouts, batched logprobs,
mixed precision, gradient accumulation, and LR scheduling.
"""

import os
import argparse

import torch


from data.arithmetic import (
    generate_easy,
    generate_medium,
    generate_hard,
    ArithmeticDataset,
    collate_fn,
)
from rewards.rule_reward import compute_reward_batch
from training.train_grpo import GRPOTrainer
from utils.config import parse_args_with_config


def get_args():
    parser = argparse.ArgumentParser()
    # Base GRPO arguments
    parser.add_argument(
        "--init_from", type=str, required=True, help="SFT checkpoint path"
    )
    parser.add_argument(
        "--ref_from",
        type=str,
        required=True,
        help="Reference (frozen SFT) checkpoint path",
    )
    parser.add_argument("--out_dir", type=str, default="out/iterative_grpo")
    parser.add_argument("--group_size", type=int, default=4)
    parser.add_argument("--num_steps", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of group rollouts before one optimizer step",
    )
    parser.add_argument("--max_prompt_len", type=int, default=128)
    parser.add_argument("--max_response_len", type=int, default=128)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--min_lr", type=float, default=1e-6)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument(
        "--beta", type=float, default=0.04, help="KL penalty coefficient"
    )
    parser.add_argument("--eps", type=float, default=0.2, help="PPO clipping epsilon")
    parser.add_argument(
        "--lr_schedule",
        type=str,
        default="cosine",
        choices=["cosine", "linear", "wsd", "constant"],
    )
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--backend", type=str, default="nccl")
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--eval_interval", type=int, default=100)
    parser.add_argument("--num_train", type=int, default=2000)
    parser.add_argument("--num_val", type=int, default=300)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument(
        "--config", type=str, default=None, help="Path to YAML config file"
    )

    # Iterative RLHF args
    parser.add_argument(
        "--ref_update_interval",
        type=int,
        default=250,
        help="Steps between reference model updates (0 = disable)",
    )
    parser.add_argument(
        "--ref_update_ratio",
        type=float,
        default=0.5,
        help="EMA mixing ratio for ref update: ref = ratio*policy + (1-ratio)*ref",
    )

    # Rejection sampling args
    parser.add_argument(
        "--rejection_interval",
        type=int,
        default=200,
        help="Steps between rejection sampling rounds (0 = disable)",
    )
    parser.add_argument(
        "--rejection_samples",
        type=int,
        default=64,
        help="Number of samples per prompt during rejection sampling",
    )
    parser.add_argument(
        "--rejection_top_k",
        type=int,
        default=8,
        help="Number of top-K high-reward responses to keep per prompt",
    )
    parser.add_argument(
        "--rejection_sft_steps",
        type=int,
        default=50,
        help="SFT fine-tuning steps on rejection-sampled data",
    )
    parser.add_argument(
        "--rejection_sft_lr",
        type=float,
        default=1e-4,
        help="Learning rate for rejection sampling SFT phase",
    )
    parser.add_argument(
        "--rejection_batch_size",
        type=int,
        default=None,
        help="Batch size for rejection SFT (defaults to batch_size)",
    )

    return parse_args_with_config(parser)


class IterativeGRPOTrainer(GRPOTrainer):
    """Iterative GRPO trainer with ref-model updates and rejection sampling."""

    def __init__(self, args):
        self.ref_updates_done = 0
        self.rejection_rounds_done = 0
        self.best_policy_state = None
        super().__init__(args)

    def _build_data(self):
        super()._build_data()
        args = self.args
        self.rejection_pool = (
            generate_easy(args.num_train, seed=args.seed + 200)
            + generate_medium(args.num_train, seed=args.seed + 201)
            + generate_hard(args.num_train, seed=args.seed + 202)
        )

    def _setup_checkpointing(self):
        super()._setup_checkpointing()
        # best_policy_state is allocated lazily when a new best policy is found.

    # ------------------------------------------------------------------
    #  Iterative RLHF: update reference model
    # ------------------------------------------------------------------
    def update_reference_model(self, ratio=None):
        """Update reference model from current policy (EMA-style).

        Two strategies:
          1. Hard update (ratio=1.0): ref <- policy
          2. Soft/EMA update (0 < ratio < 1): ref <- ratio*policy + (1-ratio)*ref
        """
        if ratio is None:
            ratio = self.args.ref_update_ratio

        if ratio >= 1.0:
            self.ref.load_state_dict(
                {k: v.clone() for k, v in self.raw_model.state_dict().items()}
            )
        else:
            policy_sd = self.raw_model.state_dict()
            for k, v in self.ref.state_dict().items():
                v.copy_(ratio * policy_sd[k] + (1.0 - ratio) * v)

        self.ref.eval()
        for p in self.ref.parameters():
            p.requires_grad = False

    # ------------------------------------------------------------------
    #  Rejection Sampling: sample many, keep top-K, SFT fine-tune
    # ------------------------------------------------------------------
    def rejection_sampling_sft(self, data_pool):
        """Rejection sampling + SFT fine-tuning.

        1. For each prompt, sample N responses and compute rewards.
        2. Keep only top-K highest-reward responses per prompt.
        3. Run a few SFT fine-tuning steps on the filtered (prompt, response) pairs.
        """
        args = self.args
        device = self.device
        tokenizer = self.tokenizer
        sft_batch_size = args.rejection_batch_size or args.batch_size

        print(
            f"\n--- Rejection Sampling: sampling {args.rejection_samples}x per prompt, "
            f"keeping top-{args.rejection_top_k} ---"
        )

        # Phase 1: Sample a large batch per prompt using batched generation.
        sft_data = []
        self.policy.eval()

        prompts = [item["prompt"] for item in data_pool]
        answers = [item["answer"] for item in data_pool]
        prompt_tokens = self._encode_prompts(prompts)

        with torch.no_grad():
            for prompt, answer, ptoks in zip(prompts, answers, prompt_tokens):
                # Repeat the prompt for rejection_samples and generate in one batch.
                repeated_prompts = [ptoks for _ in range(args.rejection_samples)]
                responses, response_ids_list = self._generate_responses_from_tokens(
                    self.policy,
                    repeated_prompts,
                    max_response_len=args.max_response_len,
                    gen_batch_size=min(sft_batch_size, args.rejection_samples),
                )

                rewards, _, _, _ = compute_reward_batch(
                    responses, [answer] * len(responses)
                )
                samples = list(zip(responses, rewards, response_ids_list))
                samples.sort(key=lambda x: x[1], reverse=True)
                top_samples = samples[: args.rejection_top_k]

                for resp_text, r, _ in top_samples:
                    if r > 0:
                        sft_data.append(
                            {"prompt": prompt, "response": resp_text, "reward": r}
                        )

        if len(sft_data) == 0:
            print("  No high-reward samples found, skipping SFT phase.")
            return 0.0

        print(f"  Collected {len(sft_data)} high-reward samples for SFT.")

        # Phase 2: SFT fine-tuning on rejection-sampled data.
        sft_dataset = ArithmeticDataset(
            sft_data,
            tokenizer,
            max_length=args.max_prompt_len + args.max_response_len,
            pre_tokenize=True,
        )
        sft_loader = torch.utils.data.DataLoader(
            sft_dataset,
            batch_size=sft_batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=0,
            pin_memory=True,
        )

        sft_optimizer = self.raw_model.configure_optimizers(
            args.weight_decay,
            args.rejection_sft_lr,
            (0.9, 0.95),
            device_type="cuda" if "cuda" in str(device) else "cpu",
        )

        self.policy.train()
        total_sft_loss = 0.0
        sft_steps = 0

        for batch in sft_loader:
            if sft_steps >= args.rejection_sft_steps:
                break
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)

            with self.ctx:
                logits, loss, _ = self.policy(input_ids, targets=labels)

            sft_optimizer.zero_grad(set_to_none=True)
            if self.scaler is not None and self.scaler.is_enabled():
                self.scaler.scale(loss).backward()
                self.scaler.step(sft_optimizer)
                self.scaler.update()
            else:
                loss.backward()
                if args.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.policy.parameters(), args.grad_clip
                    )
                sft_optimizer.step()

            total_sft_loss += loss.item()
            sft_steps += 1

        avg_sft_loss = total_sft_loss / max(sft_steps, 1)
        print(f"  SFT phase complete: {sft_steps} steps, avg loss {avg_sft_loss:.4f}")
        return avg_sft_loss

    # ------------------------------------------------------------------
    #  Training hooks
    # ------------------------------------------------------------------
    def on_step_end(self, step, metrics, rollout):
        """Periodic ref-model updates and rejection-sampling SFT."""
        args = self.args
        extra_metrics = {}

        if (
            args.ref_update_interval > 0
            and step > 0
            and step % args.ref_update_interval == 0
        ):
            self.ref_updates_done += 1
            print(
                f"\n--- Updating reference model (round {self.ref_updates_done}, step {step}) ---"
            )
            self.update_reference_model()
            extra_metrics["iter/ref_updates"] = self.ref_updates_done

        if (
            args.rejection_interval > 0
            and step > 0
            and step % args.rejection_interval == 0
        ):
            self.rejection_rounds_done += 1
            print(
                f"\n--- Rejection sampling round {self.rejection_rounds_done} (step {step}) ---"
            )
            sft_loss = self.rejection_sampling_sft(self.rejection_pool)
            extra_metrics["rejection/sft_loss"] = sft_loss
            extra_metrics["rejection/rounds"] = self.rejection_rounds_done

        return extra_metrics if extra_metrics else None

    def on_eval_end(self, step, mean_val_reward, improved):
        """Track best policy state for iterative RLHF ref updates."""
        if improved:
            self.best_policy_state = {
                k: v.clone().cpu() for k, v in self.raw_model.state_dict().items()
            }

    # ------------------------------------------------------------------
    #  Finalize
    # ------------------------------------------------------------------
    def finalize(self):
        """Restore best policy before final checkpoint (mirrors legacy behavior)."""
        if self.best_policy_state is not None:
            self.raw_model.load_state_dict(self.best_policy_state)
            if self.distributed:
                # DDP broadcasts the restored weights on the next forward; here we
                # just ensure the underlying module is updated.
                pass


def main():
    args = get_args()
    trainer = IterativeGRPOTrainer(args)
    try:
        trainer.train()
    finally:
        trainer.finalize()
        trainer.save_checkpoint(
            f"final_iterative_grpo_g{args.group_size}.pt",
            trainer.global_step,
            trainer.best_reward,
        )
        trainer.cleanup()


if __name__ == "__main__":
    main()
