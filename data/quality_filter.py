"""Quality scoring filter for document-level pre-training data.

Provides ``QualityScoreFilter``, a ``QualityFilter`` subclass that returns a
continuous quality score in ``[0, 1]`` instead of a binary keep/drop decision.
Supports both FastText-based and rule-based scoring, and exposes stratified
sampling utilities so high-quality documents can be up-sampled.
"""
from __future__ import annotations

import math
import random
import re
import warnings
from collections import Counter
from typing import Any, Dict, Optional, Sequence

from data.filter import QualityFilter


class QualityScoreFilter(QualityFilter):
    """Score documents on a 0-1 quality scale and support stratified sampling.

    The scoring model combines multiple signals:

    - **length score**: rewards moderate-length documents, penalises very short
      or extremely long text.
    - **repetition score**: penalises high character n-gram repetition.
    - **entropy score**: rewards high lexical diversity (character entropy).
    - **FastText score** (optional): if a FastText quality model is provided,
      its prediction is blended into the final score.

    Parameters
    ----------
    fasttext_model_path:
        Path to a FastText ``.bin`` quality model.  If ``None``, only rule-based
        signals are used.
    fasttext_weight:
        Blending weight for the FastText score when a model is available.
        Must be in ``[0, 1]``.
    length_range:
        ``(min_chars, max_chars)`` considered ideal.  Documents outside this
        range are penalised linearly.
    repetition_n:
        Character n-gram size for repetition detection.
    repetition_max_ratio:
        Maximum allowed repetition ratio before the score drops to zero.
    weights:
        Per-component weights for the rule-based ensemble.  Keys are
        ``"length"``, ``"repetition"``, ``"entropy"``.  Values must sum to
        a positive number; they are normalised internally.
    """

    def __init__(
        self,
        fasttext_model_path: Optional[str] = None,
        fasttext_weight: float = 0.5,
        length_range: tuple[int, int] = (100, 100_000),
        repetition_n: int = 10,
        repetition_max_ratio: float = 0.3,
        weights: Optional[Dict[str, float]] = None,
    ):
        if not 0.0 <= fasttext_weight <= 1.0:
            raise ValueError("fasttext_weight must be in [0, 1]")
        self._fasttext_weight = fasttext_weight
        self._length_range = length_range
        self._repetition_n = repetition_n
        self._repetition_max_ratio = repetition_max_ratio

        default_weights = {"length": 0.3, "repetition": 0.4, "entropy": 0.3}
        self._weights = {**default_weights, **(weights or {})}
        total = sum(self._weights.values())
        if total <= 0:
            raise ValueError("weights must sum to a positive value")
        self._weights = {k: v / total for k, v in self._weights.items()}

        self._fasttext_model: Optional[Any] = None
        self._fasttext_available = False
        if fasttext_model_path is not None:
            try:
                import fasttext  # type: ignore[import-untyped]
                self._fasttext_model = fasttext.load_model(fasttext_model_path)
                self._fasttext_available = True
            except Exception as exc:
                warnings.warn(
                    f"FastText quality model unavailable ({exc}). "
                    "Falling back to rule-only scoring."
                )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def score_document(self, text: str) -> float:
        """Return a quality score in ``[0, 1]`` for *text*.

        Higher is better.  The score is a convex combination of rule-based
        sub-scores and an optional FastText score.
        """
        rule_score = self._rule_score(text)
        if not self._fasttext_available or self._fasttext_model is None:
            return rule_score

        ft_score = self._fasttext_score(text)
        # Blend FastText and rule scores.
        w = self._fasttext_weight
        return w * ft_score + (1.0 - w) * rule_score

    def sample_probability(self, score: float, temperature: float = 1.0) -> float:
        """Convert a quality score to a sampling probability.

        Parameters
        ----------
        score:
            Document quality score in ``[0, 1]``.
        temperature:
            ``t > 1`` makes the mapping more uniform (democratic),
            ``t < 1`` makes it more peaked (elitist).  ``t = 1`` is linear.

        Returns
        -------
        prob:
            A value in ``[0, 1]`` that can be compared against a uniform
            random draw to decide whether to keep the document.
        """
        if not 0.0 <= score <= 1.0:
            raise ValueError("score must be in [0, 1]")
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        # Apply temperature scaling to the score.
        adjusted = math.pow(score, 1.0 / temperature)
        return adjusted

    def __call__(self, text: str) -> bool:
        """Binary compatibility: keep documents with score >= 0.5."""
        return self.score_document(text) >= 0.5

    def describe(self) -> str:
        parts = [f"QualityScoreFilter(weights={self._weights})"]
        if self._fasttext_available:
            parts.append(f"fasttext_weight={self._fasttext_weight}")
        return ", ".join(parts)

    # ------------------------------------------------------------------
    # Rule-based sub-scores
    # ------------------------------------------------------------------

    def _rule_score(self, text: str) -> float:
        scores = {
            "length": self._length_score(text),
            "repetition": self._repetition_score(text),
            "entropy": self._entropy_score(text),
        }
        return sum(self._weights[k] * scores[k] for k in self._weights)

    def _length_score(self, text: str) -> float:
        n = len(text)
        lo, hi = self._length_range
        if lo <= n <= hi:
            return 1.0
        if n < lo:
            return max(0.0, n / lo)
        # n > hi
        return max(0.0, 1.0 - (n - hi) / hi)

    def _repetition_score(self, text: str) -> float:
        n = self._repetition_n
        if len(text) < n:
            return 1.0
        positions = len(text) - n + 1
        counts = Counter(text[i : i + n] for i in range(positions))
        if not counts:
            return 1.0
        max_count = max(counts.values())
        ratio = max_count / positions
        if ratio >= self._repetition_max_ratio:
            return 0.0
        return 1.0 - ratio / self._repetition_max_ratio

    def _entropy_score(self, text: str) -> float:
        if not text:
            return 0.0
        counts = Counter(text)
        total = len(text)
        entropy = -sum(
            (c / total) * math.log2(c / total) for c in counts.values()
        )
        # Normalise against the maximum possible entropy for printable ASCII.
        max_entropy = math.log2(min(95, total))
        if max_entropy <= 0:
            return 0.0
        return min(1.0, entropy / max_entropy)

    def _fasttext_score(self, text: str) -> float:
        if self._fasttext_model is None:
            return 0.5
        line = " ".join(text.split()).replace("\n", " ")
        if not line:
            return 0.0
        try:
            labels, scores = self._fasttext_model.predict(line, k=1)
            score = float(scores[0])
            # Normalise to [0, 1] assuming FastText returns probabilities.
            return min(1.0, max(0.0, score))
        except Exception:
            return 0.5


class StratifiedSampler:
    """Stratified document sampler driven by ``QualityScoreFilter``.

    Parameters
    ----------
    scorer:
        ``QualityScoreFilter`` instance.
    temperature:
        Sampling temperature (see ``QualityScoreFilter.sample_probability``).
    seed:
        Random seed for reproducibility.
    """

    def __init__(
        self,
        scorer: QualityScoreFilter,
        temperature: float = 1.0,
        seed: int = 42,
    ):
        self.scorer = scorer
        self.temperature = temperature
        self._rng = random.Random(seed)

    def __call__(self, text: str) -> bool:
        """Return ``True`` if the document should be sampled."""
        score = self.scorer.score_document(text)
        prob = self.scorer.sample_probability(score, self.temperature)
        return self._rng.random() < prob


# ---------------------------------------------------------------------------
# Unit-test stubs
# ---------------------------------------------------------------------------
def _test_quality_score_filter():
    """Quick smoke test for the rule-based scoring path."""
    f = QualityScoreFilter(fasttext_model_path=None)
    assert 0.0 <= f.score_document("Hello world") <= 1.0
    assert f.score_document("") < 0.5
    assert f.sample_probability(1.0) == 1.0
    assert f.sample_probability(0.0) == 0.0
    print("quality_filter smoke tests passed")


if __name__ == "__main__":
    _test_quality_score_filter()
