"""
Group Relative Policy Optimization (GRPO) for RL alignment.

Improvements over the naive implementation
------------------------------------------
* Batched rollout generation (prompts grouped by length, KV-cache enabled).
* Batched log-probability computation across all group rollouts.
* Prompt tokenization is performed once per training step and reused for
  generation, reference logprobs, and policy logprobs.
* Mixed precision (bf16/fp16 + GradScaler).
* Gradient accumulation across multiple group rollouts.
* Warmup + decay learning-rate schedule (cosine / linear / wsd / constant).
* Full distributed-training, checkpointing, and resume support via BaseTrainer.
"""

import os
import time
import random
import argparse

import numpy as np
import torch


from collections import defaultdict

from data.arithmetic import generate_easy, generate_medium, generate_hard
from model.attention_utils import set_attention_backend, print_attention_backend
from rewards.rule_reward import compute_reward_batch
from training.trainer_base import (
    BaseTrainer,
    load_model_from_checkpoint,
    maybe_warn_dropout,
)
from utils.lr_scheduler import LRScheduler
from utils.rl_utils import compute_kl_divergence, compute_token_logprobs
from utils.config import parse_args_with_config, to_dict
import tiktoken


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--init_from", type=str, required=True, help="SFT checkpoint path"
    )
    parser.add_argument(
        "--ref_from",
        type=str,
        required=True,
        help="Reference (frozen SFT) checkpoint path",
    )
    parser.add_argument("--out_dir", type=str, default="out/grpo")
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
        "--adv_norm",
        type=str,
        default="group",
        choices=["group", "batch", "none"],
        help="Advantage normalization scope: "
        "'group' (per-prompt baseline, original GRPO), "
        "'batch' (normalize over the whole GxB batch), "
        "'none' (raw centered rewards, no std division)",
    )
    parser.add_argument(
        "--adv_clip",
        type=float,
        default=0.0,
        help="Symmetric advantage clip magnitude (e.g. 3.0); " "0 disables clipping",
    )
    parser.add_argument(
        "--lr_schedule",
        type=str,
        default="cosine",
        choices=["cosine", "linear", "wsd", "constant"],
    )
    parser.add_argument(
        "--keep_last_n",
        type=int,
        default=0,
        help="Keep only the last N non-best checkpoints (0=keep all)",
    )
    parser.add_argument(
        "--allow_dropout",
        action="store_true",
        help="Allow dropout > 0 in GRPO (old/new logprob ratios will be biased)",
    )
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--backend", type=str, default="nccl")
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--eval_interval", type=int, default=100)
    parser.add_argument(
        "--num_train",
        type=int,
        default=2000,
        help="Samples per difficulty level for training",
    )
    parser.add_argument(
        "--num_val",
        type=int,
        default=300,
        help="Samples per difficulty level for validation",
    )
    parser.add_argument(
        "--prompt_diversity",
        type=int,
        default=1,
        help="Prompt template diversity: 0=original only, "
        "1=all 12 templates, N=first N templates",
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
    return parse_args_with_config(parser)


class GRPOTrainer(BaseTrainer):
    """Group Relative Policy Optimization trainer.

    Important invariant
    -------------------
    ``old_logprobs`` are recorded under ``policy.eval() + no_grad()``
    (dropout=0), while ``new_logprobs`` are computed under ``policy.train()``.
    For the PPO ratio ``exp(new_logp - old_logp)`` to be unbiased, the model
    MUST use ``dropout=0.0``.  If you set ``dropout > 0``, sampling and loss
    will use different dropout masks and the advantage estimate will be noisy.
    """

    def __init__(self, args):
        super().__init__(args)

    def _init_state(self):
        os.makedirs(self.args.out_dir, exist_ok=True)
        self.tokenizer = tiktoken.get_encoding("gpt2")
        self.eot_token = self.tokenizer.eot_token
        self.global_step = 0
        self.best_reward = -float("inf")

    def _build_data(self):
        args = self.args
        # Different ranks get different synthetic data so that the global pool
        # is more diverse in distributed training.
        rank_offset = self.rank * 1000 if self.rank > 0 else 0
        pd = getattr(args, "prompt_diversity", 1)
        train_data = (
            generate_easy(
                args.num_train, seed=args.seed + rank_offset, prompt_diversity=pd
            )
            + generate_medium(
                args.num_train, seed=args.seed + 1 + rank_offset, prompt_diversity=pd
            )
            + generate_hard(
                args.num_train, seed=args.seed + 2 + rank_offset, prompt_diversity=pd
            )
        )
        val_data = (
            generate_easy(
                args.num_val, seed=args.seed + 100 + rank_offset, prompt_diversity=pd
            )
            + generate_medium(
                args.num_val, seed=args.seed + 101 + rank_offset, prompt_diversity=pd
            )
            + generate_hard(
                args.num_val, seed=args.seed + 102 + rank_offset, prompt_diversity=pd
            )
        )
        self.train_data = train_data
        self.val_data = val_data

    def _build_model(self):
        args = self.args
        if args.resume:
            self.policy, _ = load_model_from_checkpoint(args.resume, device=self.device)
        else:
            self.policy, _ = load_model_from_checkpoint(
                args.init_from, device=self.device
            )
        self.policy = self.policy.to(self.device)
        maybe_warn_dropout(self.policy)

        dp = getattr(self.policy.config, "dropout", 0.0)
        if dp > 0 and not args.allow_dropout:
            raise ValueError(
                f"GRPO requires dropout=0.0 for unbiased old/new logprob ratios, "
                f"but loaded model has dropout={dp}. Set --allow_dropout to override "
                f"(ratios will be biased) or re-train the SFT checkpoint with dropout=0.0."
            )

        self.ref, _ = load_model_from_checkpoint(args.ref_from, device=self.device)
        self.ref = self.ref.to(self.device)
        self.ref.eval()
        for p in self.ref.parameters():
            p.requires_grad = False

        self.wrap_distributed(self.policy)

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
        total_steps = args.num_steps // args.gradient_accumulation_steps
        self.scheduler = LRScheduler(
            schedule=args.lr_schedule,
            learning_rate=args.learning_rate,
            min_lr=args.min_lr,
            warmup_iters=max(1, total_steps // 20),
            lr_decay_iters=total_steps,
            max_iters=total_steps,
        )

    def _setup_checkpointing(self):
        self.configure_checkpointing(self.raw_model.config)

    def _setup_logger(self):
        args = self.args
        self.build_logger(
            project_name="nanogpt-modern-grpo",
            run_name=f"grpo_g{args.group_size}_beta{args.beta}",
            config=to_dict(args),
        )

    def _maybe_resume(self):
        args = self.args
        if args.resume:
            extra = self.load_checkpoint(args.resume)
            self.global_step = extra.get("iter_num", 0)
            self.best_reward = extra.get("best_val_loss", -float("inf"))

    # ------------------------------------------------------------------
    #  Prompt / response encoding helpers
    # ------------------------------------------------------------------
    def _encode_prompts(self, prompts):
        """Encode a list of prompts to token-id lists, truncating to ``max_prompt_len``."""
        max_len = self.args.max_prompt_len
        out = []
        for p in prompts:
            toks = self.tokenizer.encode(p)
            if len(toks) > max_len:
                toks = toks[-max_len:]
            out.append(toks)
        return out

    def _generate_responses_from_tokens(
        self, model, prompt_tokens_list, max_response_len, gen_batch_size
    ):
        """Generate responses from pre-tokenized prompt lists.

        Prompts are grouped by length so each batch fed to ``model.generate`` is
        rectangular and can use the KV cache.  Returns decoded response strings
        and the raw response token-id lists.
        """
        eos_token_id = self.eot_token

        by_length = defaultdict(list)
        for i, toks in enumerate(prompt_tokens_list):
            by_length[len(toks)].append((i, toks))

        response_ids = [None] * len(prompt_tokens_list)
        model.eval()
        with torch.no_grad():
            for length, items in by_length.items():
                for start in range(0, len(items), gen_batch_size):
                    batch_items = items[start : start + gen_batch_size]
                    idx = torch.tensor(
                        [toks for _, toks in batch_items],
                        dtype=torch.long,
                        device=self.device,
                    )
                    generated = model.generate(
                        idx,
                        max_new_tokens=max_response_len,
                        temperature=1.0,
                        top_k=50,
                        use_cache=True,
                        eos_token_id=eos_token_id,
                    )
                    for j, (orig_i, toks) in enumerate(batch_items):
                        resp = generated[j, len(toks) :].tolist()
                        if eos_token_id in resp:
                            resp = resp[: resp.index(eos_token_id)]
                        response_ids[orig_i] = resp

        responses = [self.tokenizer.decode(r) for r in response_ids]
        return responses, response_ids

    # ------------------------------------------------------------------
    #  Batched logprob computation
    # ------------------------------------------------------------------
    def _batch_logprobs(self, model, sequences, prompt_lens, response_lens):
        """Compute per-token log-probs for a list of variable-length sequences.

        Parameters
        ----------
        model : nn.Module
        sequences : list[list[int]]
            Each item is prompt + response token ids.
        prompt_lens : list[int]
        response_lens : list[int]

        Returns
        -------
        token_logprobs : torch.Tensor [B, T]
            Log-prob for each target token (shifted by one). Padded positions
            contain 0.
        mask : torch.Tensor [B, T]
            1 for response tokens, 0 otherwise (including padding).
        """
        return self._batch_logprobs_impl(model, sequences, prompt_lens, response_lens)

    def _batch_logprobs_ref(self, ref_model, sequences, prompt_lens, response_lens):
        """Compute reference-model log-probs under an explicit no_grad guard.

        The reference model is always frozen (``eval()`` mode), so its forward
        must not build a computation graph.  This wrapper makes that invariant
        explicit and avoids the conditional ``no_grad`` branch used for the
        policy model.
        """
        with torch.no_grad():
            return self._batch_logprobs_impl(
                ref_model, sequences, prompt_lens, response_lens
            )

    def _batch_logprobs_impl(self, model, sequences, prompt_lens, response_lens):
        """Shared implementation for policy/ref log-prob computation."""
        B = len(sequences)
        if B == 0:
            return torch.zeros(0, 0, device=self.device), torch.zeros(
                0, 0, dtype=torch.bool, device=self.device
            )

        max_len = max(len(s) for s in sequences)
        pad_id = self.eot_token

        input_ids = torch.full(
            (B, max_len), pad_id, dtype=torch.long, device=self.device
        )
        targets = torch.full((B, max_len), pad_id, dtype=torch.long, device=self.device)
        attention_mask = torch.zeros((B, max_len), dtype=torch.bool, device=self.device)
        response_mask = torch.zeros((B, max_len), dtype=torch.bool, device=self.device)

        for i, seq in enumerate(sequences):
            L = len(seq)
            if L < 2:
                seq = seq + [pad_id]
                L = len(seq)
            input_ids[i, : L - 1] = torch.tensor(
                seq[:-1], dtype=torch.long, device=self.device
            )
            targets[i, : L - 1] = torch.tensor(
                seq[1:], dtype=torch.long, device=self.device
            )
            attention_mask[i, : L - 1] = True
            p_len = prompt_lens[i]
            r_len = response_lens[i]
            # response tokens start after the prompt; seq = prompt + response,
            # targets is seq shifted left by one, so response targets occupy
            # positions [p_len - 1, p_len + r_len - 1).
            resp_start = max(0, p_len - 1)
            resp_end = max(0, p_len + r_len - 1)
            response_mask[i, resp_start:resp_end] = True

        with self.ctx:
            logits = model(input_ids, attention_mask=attention_mask)[0]
        token_logprobs = compute_token_logprobs(logits, targets, mask=attention_mask)

        token_logprobs = token_logprobs * response_mask.float()
        return token_logprobs, response_mask

    # ------------------------------------------------------------------
    #  Sampling + training step
    # ------------------------------------------------------------------
    def sample_group(self, prompts, answers, prompt_tokens=None):
        """Sample ``group_size`` responses per prompt and compute old/ref logprobs.

        Parameters
        ----------
        prompts : list[str]
        answers : list[str]
        prompt_tokens : list[list[int]] or None
            Pre-tokenized prompts.  If None, prompts are encoded on the fly.

        Returns
        -------
        rollout : dict
            * responses: list[list[str]]          shape [G, B]
            * response_ids: list[list[list[int]]] shape [G, B]
            * rewards: np.ndarray                 shape [G, B]
            * old_logprobs: torch.Tensor          shape [G*B, T]
            * ref_logprobs: torch.Tensor          shape [G*B, T]
            * masks: torch.Tensor                 shape [G*B, T]
            * prompt_lens: list[list[int]]        shape [G, B]
            * response_lens: list[list[int]]      shape [G, B]
            * prompt_tokens: list[list[int]]      shape [B]
        """
        args = self.args
        group_size = args.group_size

        if prompt_tokens is None:
            prompt_tokens = self._encode_prompts(prompts)

        self.policy.eval()

        all_responses = []
        all_response_ids = []
        all_rewards = []
        all_prompt_lens = []
        all_response_lens = []

        with torch.no_grad():
            for g in range(group_size):
                responses, response_tokens_list = self._generate_responses_from_tokens(
                    self.raw_model,
                    prompt_tokens,
                    max_response_len=args.max_response_len,
                    gen_batch_size=args.batch_size,
                )

                rewards, fmt_scores, proc_scores, acc_scores = compute_reward_batch(
                    responses, answers
                )
                all_rewards.append(rewards)
                all_responses.append(responses)
                all_response_ids.append(response_tokens_list)
                # process_scores are available for logging/analysis if needed.
                _ = proc_scores

                prompt_lens_g = [len(toks) for toks in prompt_tokens]
                response_lens_g = [len(resp_ids) for resp_ids in response_tokens_list]
                all_prompt_lens.append(prompt_lens_g)
                all_response_lens.append(response_lens_g)

        # Flatten all group sequences for a single batched logprob forward pass.
        flat_sequences = []
        flat_prompt_lens = []
        flat_response_lens = []
        for g in range(group_size):
            for b, toks in enumerate(prompt_tokens):
                full_ids = toks + all_response_ids[g][b]
                if len(full_ids) < 2:
                    full_ids = full_ids + [self.eot_token]
                flat_sequences.append(full_ids)
                flat_prompt_lens.append(all_prompt_lens[g][b])
                flat_response_lens.append(all_response_lens[g][b])

        with torch.no_grad():
            old_logprobs, masks = self._batch_logprobs(
                self.model, flat_sequences, flat_prompt_lens, flat_response_lens
            )
        ref_logprobs, _ = self._batch_logprobs_ref(
            self.ref, flat_sequences, flat_prompt_lens, flat_response_lens
        )

        rewards_arr = np.array(all_rewards, dtype=np.float32)

        return {
            "responses": all_responses,
            "response_ids": all_response_ids,
            "rewards": rewards_arr,
            "old_logprobs": old_logprobs,
            "ref_logprobs": ref_logprobs,
            "masks": masks,
            "prompt_lens": all_prompt_lens,
            "response_lens": all_response_lens,
            "prompt_tokens": prompt_tokens,
        }

    def on_step_end(self, step, metrics, rollout):
        """Hook called after each training step (before evaluation).

        Subclasses may override this to perform extra per-step work and return
        a dict of scalars to log.
        """
        return None

    def on_eval_end(self, step, mean_val_reward, improved):
        """Hook called after each evaluation phase.

        Subclasses may override this to react to validation metrics.
        """
        pass

    # ------------------------------------------------------------------
    #  Advantage computation (factored out for testability)
    # ------------------------------------------------------------------
    @staticmethod
    def compute_advantages(
        rewards_t: torch.Tensor,
        adv_norm: str = "group",
        adv_clip: float = 0.0,
    ) -> torch.Tensor:
        """Compute GRPO advantages from a reward tensor.

        Parameters
        ----------
        rewards_t : Tensor of shape ``[G, B]`` (group_size x batch_size).
        adv_norm  : One of ``"group"``, ``"batch"`` or ``"none"``.
                    * ``group``: per-prompt baseline (original GRPO).
                    * ``batch``: normalize over the whole ``[G, B]`` batch.
                    * ``none`` : center only (no std division).
        adv_clip  : Symmetric clip magnitude; ``0`` disables clipping.
        """
        if adv_norm == "group":
            mean_r = rewards_t.mean(dim=0, keepdim=True)
            std_r = rewards_t.std(dim=0, keepdim=True).clamp_min(1e-8)
            adv = (rewards_t - mean_r) / std_r
        elif adv_norm == "batch":
            mean_r = rewards_t.mean()
            std_r = rewards_t.std().clamp_min(1e-8)
            adv = (rewards_t - mean_r) / std_r
        elif adv_norm == "none":
            mean_r = rewards_t.mean(dim=0, keepdim=True)
            adv = rewards_t - mean_r
        else:
            raise ValueError(
                f"adv_norm must be one of 'group', 'batch', 'none'; got {adv_norm!r}"
            )
        if adv_clip > 0.0:
            adv = torch.clamp(adv, -adv_clip, adv_clip)
        return adv

    def compute_grpo_loss(self, rollout):
        """Compute the GRPO clipped surrogate + KL penalty loss.

        The forward pass for ``new_logprobs`` is performed once for all
        ``group_size * batch_size`` sequences, then reshaped to compute the
        group-relative advantage per prompt.
        """
        args = self.args
        device = self.device

        rewards_t = torch.from_numpy(rollout["rewards"]).to(device)  # [G, B]
        group_size, batch_size = rewards_t.shape

        # Advantage normalization.  ``group`` keeps the original GRPO
        # per-prompt baseline; ``batch`` normalizes over the whole GxB batch
        # (useful for small group_size); ``none`` centers rewards without
        # dividing by std.  See ``GRPOTrainer.compute_advantages`` for details.
        adv_mode = getattr(args, "adv_norm", "group")
        adv_clip = float(getattr(args, "adv_clip", 0.0))
        advantages = self.compute_advantages(rewards_t, adv_mode, adv_clip)

        # Recompute new logprobs in train mode (one batched forward).
        self.model.train()
        flat_sequences = []
        flat_prompt_lens = []
        flat_response_lens = []
        for g in range(group_size):
            for b, toks in enumerate(rollout["prompt_tokens"]):
                full_ids = toks + rollout["response_ids"][g][b]
                if len(full_ids) < 2:
                    full_ids = full_ids + [self.eot_token]
                flat_sequences.append(full_ids)
                flat_prompt_lens.append(rollout["prompt_lens"][g][b])
                flat_response_lens.append(rollout["response_lens"][g][b])

        new_logprobs, masks = self._batch_logprobs(
            self.model, flat_sequences, flat_prompt_lens, flat_response_lens
        )

        # Reshape back to [G, B, T].
        T = new_logprobs.size(1)
        old_logp = (
            rollout["old_logprobs"].detach().to(device).view(group_size, batch_size, T)
        )
        ref_logp = (
            rollout["ref_logprobs"].detach().to(device).view(group_size, batch_size, T)
        )
        new_logp = new_logprobs.view(group_size, batch_size, T)
        mask = masks.view(group_size, batch_size, T)
        adv = advantages.unsqueeze(-1)  # [G, B, 1]

        # PPO ratio and clipped surrogate.
        ratio = torch.exp(new_logp - old_logp)
        surr1 = ratio * adv
        surr2 = torch.clamp(ratio, 1.0 - args.eps, 1.0 + args.eps) * adv
        pg_loss = -torch.min(surr1, surr2)

        # KL penalty: numerically stable reverse KL(ref || policy) = ref - new.
        kl_term = compute_kl_divergence(ref_logp, new_logp, mask=mask, reduction="none")

        total_tokens = mask.sum()
        if total_tokens > 0:
            loss = ((pg_loss + args.beta * kl_term) * mask.float()).sum() / total_tokens
        else:
            loss = torch.tensor(0.0, device=device, requires_grad=True)

        # Per-component metrics (token-averaged).
        with torch.no_grad():
            policy_loss = (pg_loss * mask.float()).sum() / total_tokens.clamp_min(1)
            kl_loss = compute_kl_divergence(
                ref_logp, new_logp, mask=mask, reduction="mean"
            )

        metrics = {
            "policy_loss": policy_loss.item(),
            "kl_loss": kl_loss.item(),
            "mean_reward": rewards_t.mean().item(),
            "std_reward": rewards_t.std().item(),
            "mean_advantage": advantages.mean().item(),
            "std_advantage": advantages.std().item(),
        }
        return loss, metrics

    # ------------------------------------------------------------------
    #  Training loop
    # ------------------------------------------------------------------
    def train(self):
        args = self.args
        accum_steps = args.gradient_accumulation_steps
        t0 = time.time()

        self.optimizer.zero_grad(set_to_none=True)

        while self.global_step < args.num_steps:
            # --- sample a rollout (one micro-batch) ---
            batch = random.sample(self.train_data, args.batch_size)
            prompts = [b["prompt"] for b in batch]
            answers = [b["answer"] for b in batch]
            prompt_tokens = self._encode_prompts(prompts)

            rollout = self.sample_group(prompts, answers, prompt_tokens=prompt_tokens)
            loss, metrics = self.compute_grpo_loss(rollout)

            # --- gradient accumulation scaling ---
            if accum_steps > 1:
                loss = loss / accum_steps

            if self.scaler is not None and self.scaler.is_enabled():
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

            # --- optimizer step only after accumulation is complete ---
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

            # --- LR schedule (per optimizer step) ---
            opt_step = self.global_step // accum_steps
            lr = self.scheduler(opt_step)
            for pg in self.optimizer.param_groups:
                pg["lr"] = lr

            # --- logging (use the raw per-rollout loss) ---
            raw_loss = loss.item() * accum_steps if accum_steps > 1 else loss.item()
            if self.global_step % 10 == 0 and self.master_process:
                t1 = time.time()
                _dt = t1 - t0
                t0 = t1
                print(
                    f"step {self.global_step}: loss {raw_loss:.4f}, "
                    f"mean_reward {metrics['mean_reward']:.4f}, lr {lr:.2e}"
                )
                self.log_scalars(
                    {
                        "train/loss": raw_loss,
                        "train/policy_loss": metrics["policy_loss"],
                        "train/kl_loss": metrics["kl_loss"],
                        "train/mean_reward": metrics["mean_reward"],
                        "train/mean_advantage": metrics.get("mean_advantage", 0.0),
                        "train/std_advantage": metrics.get("std_advantage", 0.0),
                        "train/lr": lr,
                    },
                    self.global_step,
                )

            # --- hooks for extensions (e.g. iterative GRPO) ---
            hook_metrics = self.on_step_end(self.global_step, metrics, rollout)
            if hook_metrics and self.master_process:
                self.log_scalars(hook_metrics, self.global_step)

            # --- evaluation ---
            improved = False
            if self.global_step % args.eval_interval == 0 and self.master_process:
                self.model.eval()
                val_batch = random.sample(
                    self.val_data, min(args.batch_size * 2, len(self.val_data))
                )
                val_prompts = [b["prompt"] for b in val_batch]
                val_answers = [b["answer"] for b in val_batch]
                val_rollout = self.sample_group(val_prompts, val_answers)
                val_rewards_t = torch.from_numpy(val_rollout["rewards"]).float()
                mean_val_reward = val_rewards_t.mean().item()
                print(f"step {self.global_step}: val mean reward {mean_val_reward:.4f}")
                self.log_scalar("val/mean_reward", mean_val_reward, self.global_step)
                if mean_val_reward > self.best_reward:
                    self.best_reward = mean_val_reward
                    improved = True
                    self.save_checkpoint(
                        f"best_grpo_g{args.group_size}.pt",
                        self.global_step,
                        self.best_reward,
                    )
                self.model.train()

                self.on_eval_end(self.global_step, mean_val_reward, improved)

            self.global_step += 1

        if self.master_process:
            self.save_checkpoint(
                f"final_grpo_g{args.group_size}.pt", self.global_step, self.best_reward
            )


def main():
    args = get_args()
    set_attention_backend(args.attn_backend)
    print_attention_backend()
    trainer = GRPOTrainer(args)
    try:
        trainer.train()
    finally:
        trainer.cleanup()


if __name__ == "__main__":
    main()
