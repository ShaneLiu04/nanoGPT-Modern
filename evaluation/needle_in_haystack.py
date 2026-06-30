"""Needle-in-Haystack long-context evaluation for nanoGPT-Modern.

Tests whether a model can retrieve a single "needle" fact buried deep inside a
long "haystack" of irrelevant text.  This is the standard stress test for
NTK-aware RoPE, RingAttention, and long-context training.

Usage
-----
>>> from evaluation.needle_in_haystack import NeedleEvaluator
>>> evaluator = NeedleEvaluator(model, tokenizer, max_context_length=4096)
>>> results = evaluator.evaluate(depths=[0.0, 0.5, 1.0], num_trials=5)
"""

from __future__ import annotations

import random
import re
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn


class NeedleEvaluator:
    """Evaluate long-context retrieval with the Needle-in-Haystack protocol.

    Parameters
    ----------
    model : nn.Module
    tokenizer : Any
        Must support ``encode`` and ``decode``.
    max_context_length : int
        Maximum prompt length (including needle and haystack).
    needle_text : str
        The secret fact to hide in the context.
    question_text : str
        The question that should trigger retrieval of the needle.
    expected_answer : str
        The answer the model should produce.
    haystack_char : str
        Filler token(s) used to build the haystack.
    """

    def __init__(
        self,
        model: nn.Module,
        tokenizer: Any,
        max_context_length: int = 4096,
        needle_text: str = "The special magic number is 87342.",
        question_text: str = "What is the special magic number?",
        expected_answer: str = "87342",
        haystack_char: str = " ",
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.max_context_length = max_context_length
        self.needle_text = needle_text
        self.question_text = question_text
        self.expected_answer = expected_answer
        self.haystack_char = haystack_char
        self.device = next(model.parameters()).device

    def _build_prompt(self, depth: float, context_len: int) -> str:
        """Build a prompt with the needle at ``depth`` fraction of ``context_len``.

        ``depth`` = 0.0 means the needle is at the very beginning,
        ``depth`` = 1.0 means it is at the very end.
        """
        needle_toks = self.tokenizer.encode(self.needle_text)
        question_toks = self.tokenizer.encode(self.question_text)
        needle_len = len(needle_toks)
        question_len = len(question_toks)
        available = context_len - needle_len - question_len - 5  # safety margin
        if available <= 0:
            raise ValueError(
                f"Context length {context_len} too short for needle + question"
            )

        # Build haystack as filler text that tokenizes to roughly the needed length.
        haystack_text = "The sky is blue. The grass is green. " * (available // 10)
        haystack_toks = self.tokenizer.encode(haystack_text)
        # Trim or pad haystack to exact available length.
        if len(haystack_toks) > available:
            haystack_toks = haystack_toks[:available]
            haystack_text = self.tokenizer.decode(haystack_toks)
        elif len(haystack_toks) < available:
            pad = self.tokenizer.encode(" ") * (available - len(haystack_toks))
            haystack_toks += pad

        needle_pos = int(depth * len(haystack_toks))
        # Insert needle into haystack at token position.
        before = haystack_toks[:needle_pos]
        after = haystack_toks[needle_pos:]
        full_tokens = before + needle_toks + after + question_toks
        prompt = self.tokenizer.decode(full_tokens)
        return prompt

    def _generate_answer(self, prompt: str, max_new_tokens: int = 20) -> str:
        """Run the model on the prompt and return the generated text."""
        self.model.eval()
        toks = self.tokenizer.encode(prompt)
        idx = torch.tensor([toks], dtype=torch.long, device=self.device)
        with torch.no_grad():
            out = self.model.generate(
                idx,
                max_new_tokens=max_new_tokens,
                temperature=0.0,  # greedy for reproducibility
                top_k=None,
                use_cache=True,
            )
        generated = out[0, len(toks) :].tolist()
        return self.tokenizer.decode(generated)

    def evaluate(
        self,
        depths: Optional[List[float]] = None,
        context_lengths: Optional[List[int]] = None,
        num_trials: int = 3,
        max_new_tokens: int = 20,
    ) -> Dict[str, Any]:
        """Run the needle-in-haystack evaluation grid.

        Parameters
        ----------
        depths : list[float] or None
            Depth fractions to test (0.0 = start, 1.0 = end).  Defaults to a
            coarse grid ``[0.0, 0.25, 0.5, 0.75, 1.0]``.
        context_lengths : list[int] or None
            Context lengths to test.  Defaults to ``[512, 1024, 2048, 4096]``
            truncated to ``max_context_length``.
        num_trials : int
            Number of random trials per (depth, length) cell.
        max_new_tokens : int
            Generation budget for the answer.

        Returns
        -------
        results : dict
            Contains ``scores`` (list of dicts), ``summary`` (accuracy per depth
            and per length), and ``raw`` (generated answers).
        """
        if depths is None:
            depths = [0.0, 0.25, 0.5, 0.75, 1.0]
        if context_lengths is None:
            context_lengths = [512, 1024, 2048, 4096]
        context_lengths = [c for c in context_lengths if c <= self.max_context_length]

        scores = []
        raw = []
        for length in context_lengths:
            for depth in depths:
                for _ in range(num_trials):
                    prompt = self._build_prompt(depth, length)
                    answer = self._generate_answer(prompt, max_new_tokens)
                    correct = self.expected_answer in answer
                    scores.append(
                        {
                            "context_length": length,
                            "depth": depth,
                            "correct": correct,
                        }
                    )
                    raw.append(
                        {
                            "context_length": length,
                            "depth": depth,
                            "answer": answer,
                            "correct": correct,
                        }
                    )

        # Summary.
        by_depth: Dict[float, List[bool]] = {d: [] for d in depths}
        by_length: Dict[int, List[bool]] = {l: [] for l in context_lengths}
        for s in scores:
            by_depth[float(s["depth"])].append(bool(s["correct"]))
            by_length[int(s["context_length"])].append(bool(s["correct"]))

        summary = {
            "overall_accuracy": (
                sum(s["correct"] for s in scores) / len(scores) if scores else 0.0
            ),
            "by_depth": {d: sum(v) / len(v) for d, v in by_depth.items()},
            "by_length": {l: sum(v) / len(v) for l, v in by_length.items()},
        }

        return {
            "scores": scores,
            "summary": summary,
            "raw": raw,
        }

    def evaluate_single(
        self,
        depth: float = 0.5,
        context_length: Optional[int] = None,
        max_new_tokens: int = 20,
    ) -> Dict[str, Any]:
        """Quick single-shot evaluation for CI / smoke tests."""
        length = context_length or self.max_context_length
        prompt = self._build_prompt(depth, length)
        answer = self._generate_answer(prompt, max_new_tokens)
        correct = self.expected_answer in answer
        return {
            "context_length": length,
            "depth": depth,
            "correct": correct,
            "answer": answer,
        }
