"""Win-rate evaluation for aligned models (DPO / GRPO / SFT).

Compares a policy model against a reference (typically the SFT checkpoint) by
sampling completions for the same prompts and scoring them with a rule-based or
model-based judge.  The win rate is the fraction of head-to-head comparisons
where the policy is preferred.

Usage
-----
>>> from evaluation.eval_win_rate import WinRateEvaluator, RuleJudge
>>> evaluator = WinRateEvaluator(policy_model, ref_model, tokenizer, judge=RuleJudge())
>>> results = evaluator.evaluate(prompts, n_samples=4, max_new_tokens=128)
>>> print(results["win_rate"])
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn


class Judge(ABC):
    """Abstract judge that scores a (prompt, response) pair."""

    @abstractmethod
    def score(self, prompt: str, response: str) -> float:
        """Return a scalar score; higher is better."""
        raise NotImplementedError


class RuleJudge(Judge):
    """Simple rule-based judge for arithmetic / reasoning tasks.

    Expects the response to contain a numeric answer.  The score is 1.0 if the
    answer is correct, 0.0 otherwise.  For open-ended prompts, falls back to a
    length-normalised heuristic (shorter = better, with a small penalty for
    very short answers).
    """

    def __init__(self, correct_answer: Optional[str] = None):
        self.correct_answer = correct_answer

    def score(self, prompt: str, response: str) -> float:
        if self.correct_answer is not None:
            return 1.0 if self.correct_answer in response else 0.0
        # Heuristic: prefer concise but non-empty answers.
        length = len(response.strip().split())
        if length == 0:
            return 0.0
        return max(0.0, 1.0 - (length - 20) / 200.0)


class LengthPenaltyJudge(Judge):
    """Judge that rewards correct answers and penalises excessive length."""

    def __init__(self, correct_answer: Optional[str] = None, target_length: int = 30):
        self.correct_answer = correct_answer
        self.target_length = target_length

    def score(self, prompt: str, response: str) -> float:
        correct = 1.0
        if self.correct_answer is not None:
            correct = 1.0 if self.correct_answer in response else 0.0
        length = len(response.strip().split())
        length_penalty = max(0.0, 1.0 - abs(length - self.target_length) / 100.0)
        return correct * 0.8 + length_penalty * 0.2


class WinRateEvaluator:
    """Evaluate win rate of a policy model versus a reference model.

    Parameters
    ----------
    policy_model : nn.Module
    ref_model : nn.Module
    tokenizer : Any
    judge : Judge
    device : str or torch.device
    """

    def __init__(
        self,
        policy_model: nn.Module,
        ref_model: nn.Module,
        tokenizer: Any,
        judge: Judge,
        device: str = "cuda",
    ):
        self.policy_model = policy_model
        self.ref_model = ref_model
        self.tokenizer = tokenizer
        self.judge = judge
        self.device = device

    def _generate(
        self,
        model: nn.Module,
        prompt_ids: torch.Tensor,
        max_new_tokens: int,
        temperature: float,
    ) -> str:
        """Greedy or sampled generation wrapper."""
        model.eval()
        with torch.no_grad():
            out = model.generate(
                prompt_ids,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=50,
                use_cache=True,
            )
        generated_ids = out[0, prompt_ids.shape[1] :].tolist()
        return self.tokenizer.decode(generated_ids)

    def evaluate(
        self,
        prompts: List[str],
        n_samples: int = 4,
        max_new_tokens: int = 128,
        temperature: float = 1.0,
    ) -> Dict[str, Any]:
        """Run head-to-head comparison for each prompt.

        Parameters
        ----------
        prompts : list[str]
        n_samples : int
            Number of completions sampled per prompt per model.
        max_new_tokens : int
        temperature : float

        Returns
        -------
        results : dict
            Contains ``win_rate``, ``policy_scores``, ``ref_scores``, and raw
            comparison records.
        """
        records = []
        policy_scores_all = []
        ref_scores_all = []

        for prompt in prompts:
            prompt_ids = torch.tensor(
                [self.tokenizer.encode(prompt)], dtype=torch.long, device=self.device
            )
            policy_scores = []
            ref_scores = []
            for _ in range(n_samples):
                policy_resp = self._generate(
                    self.policy_model, prompt_ids, max_new_tokens, temperature
                )
                ref_resp = self._generate(
                    self.ref_model, prompt_ids, max_new_tokens, temperature
                )
                ps = self.judge.score(prompt, policy_resp)
                rs = self.judge.score(prompt, ref_resp)
                policy_scores.append(ps)
                ref_scores.append(rs)
                records.append(
                    {
                        "prompt": prompt,
                        "policy_response": policy_resp,
                        "ref_response": ref_resp,
                        "policy_score": ps,
                        "ref_score": rs,
                        "policy_wins": ps > rs,
                        "tie": ps == rs,
                    }
                )
            policy_scores_all.extend(policy_scores)
            ref_scores_all.extend(ref_scores)

        wins = sum(bool(r["policy_wins"]) for r in records)
        ties = sum(bool(r["tie"]) for r in records)
        total = len(records)
        win_rate = wins / total if total > 0 else 0.0
        tie_rate = ties / total if total > 0 else 0.0

        return {
            "win_rate": win_rate,
            "tie_rate": tie_rate,
            "policy_mean_score": (
                sum(policy_scores_all) / len(policy_scores_all)
                if policy_scores_all
                else 0.0
            ),
            "ref_mean_score": (
                sum(ref_scores_all) / len(ref_scores_all) if ref_scores_all else 0.0
            ),
            "total_comparisons": total,
            "records": records,
        }

    def evaluate_and_save(
        self,
        prompts: List[str],
        output_json: str,
        n_samples: int = 4,
        max_new_tokens: int = 128,
        temperature: float = 1.0,
    ) -> Dict[str, Any]:
        """Run evaluation and persist results to ``output_json``."""
        results = self.evaluate(prompts, n_samples, max_new_tokens, temperature)
        with open(output_json, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        return results


def build_arithmetic_prompts(num_samples: int = 100, seed: int = 42) -> List[str]:
    """Generate a simple arithmetic prompt set for quick win-rate smoke tests."""
    import random

    random.seed(seed)
    prompts = []
    for _ in range(num_samples):
        a = random.randint(1, 100)
        b = random.randint(1, 100)
        op = random.choice(["+", "-", "*"])
        prompts.append(f"What is {a} {op} {b}?")
    return prompts
