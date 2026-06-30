"""Document-level quality filters for the pre-training data pipeline.

All filters implement a common callable interface::

    keep = filter_fn(text: str) -> bool

and can be combined via ``CompositeFilter``.  The ``fasttext`` quality
classifier is optional: if the library or model is unavailable the filter
falls back to ``True`` so the pipeline keeps running.
"""

from __future__ import annotations

import re
import warnings
from abc import ABC, abstractmethod
from collections import Counter
from typing import Any, Dict, List, Optional, Sequence


class QualityFilter(ABC):
    """Base class for a document-level text filter."""

    @abstractmethod
    def __call__(self, text: str) -> bool:
        """Return ``True`` if the document should be kept."""
        ...

    def describe(self) -> str:
        """Short human-readable description for logging."""
        return self.__class__.__name__


class LengthFilter(QualityFilter):
    """Keep documents whose character length is within ``[min_chars, max_chars]``.

    A value of ``None`` means no bound.
    """

    def __init__(
        self,
        min_chars: Optional[int] = None,
        max_chars: Optional[int] = None,
    ):
        if min_chars is not None and min_chars < 0:
            raise ValueError("min_chars must be >= 0")
        if max_chars is not None and max_chars < 0:
            raise ValueError("max_chars must be >= 0")
        if min_chars is not None and max_chars is not None and min_chars > max_chars:
            raise ValueError("min_chars must not exceed max_chars")
        self.min_chars = min_chars
        self.max_chars = max_chars

    def __call__(self, text: str) -> bool:
        n = len(text)
        if self.min_chars is not None and n < self.min_chars:
            return False
        if self.max_chars is not None and n > self.max_chars:
            return False
        return True

    def describe(self) -> str:
        return f"LengthFilter(min={self.min_chars}, max={self.max_chars})"


class RepetitionFilter(QualityFilter):
    """Drop documents with excessive character n-gram repetition.

    The filter computes the most frequent character n-gram and rejects the
    document if its frequency exceeds ``max_repetition_ratio`` of all
    n-gram positions.  This catches low-information text such as repeated
    placeholders or spam.
    """

    def __init__(self, n: int = 10, max_repetition_ratio: float = 0.3):
        if n < 1:
            raise ValueError("n must be >= 1")
        if not 0.0 <= max_repetition_ratio <= 1.0:
            raise ValueError("max_repetition_ratio must be in [0, 1]")
        self.n = n
        self.max_repetition_ratio = max_repetition_ratio

    def __call__(self, text: str) -> bool:
        if len(text) < self.n:
            return True
        positions = len(text) - self.n + 1
        counts = Counter(text[i : i + self.n] for i in range(positions))
        if not counts:
            return True
        max_count = max(counts.values())
        return max_count / positions <= self.max_repetition_ratio

    def describe(self) -> str:
        return f"RepetitionFilter(n={self.n}, max_ratio={self.max_repetition_ratio})"


class RegexFilter(QualityFilter):
    """Keep documents only if they match ``require`` and do not match ``reject``.

    Both patterns are optional.  ``reject`` is useful for dropping boilerplate
    markers such as "This page has been removed".
    """

    def __init__(
        self,
        require: Optional[str] = None,
        reject: Optional[str] = None,
        flags: int = re.IGNORECASE,
    ):
        self.require = re.compile(require, flags) if require is not None else None
        self.reject = re.compile(reject, flags) if reject is not None else None

    def __call__(self, text: str) -> bool:
        if self.require is not None and not self.require.search(text):
            return False
        if self.reject is not None and self.reject.search(text):
            return False
        return True

    def describe(self) -> str:
        parts = []
        if self.require is not None:
            parts.append(f"require={self.require.pattern}")
        if self.reject is not None:
            parts.append(f"reject={self.reject.pattern}")
        return f"RegexFilter({', '.join(parts)})"


class FastTextQualityFilter(QualityFilter):
    """Optional fasttext language / quality classifier.

    Parameters
    ----------
    model_path:
        Path to a fasttext ``.bin`` model.  Common choices are the fasttext
        language-id model or a quality classifier such as CCNet's model.
    threshold:
        Minimum predicted score for a document to be kept.
    label:
        If provided, require the top-1 label to equal this value.  For
        language ID this is usually ``__label__en``.
    """

    def __init__(
        self,
        model_path: str,
        threshold: float = 0.5,
        label: Optional[str] = None,
    ):
        self.model_path = model_path
        self.threshold = threshold
        self.label = label
        self._model: Optional[Any] = None
        self._available = True
        try:
            import fasttext  # type: ignore[import-untyped]

            self._model = fasttext.load_model(model_path)
        except Exception as exc:
            self._available = False
            warnings.warn(
                f"FastTextQualityFilter disabled: could not load {model_path} ({exc}). "
                "All documents will pass this filter."
            )

    def __call__(self, text: str) -> bool:
        if not self._available or self._model is None:
            return True
        # fasttext expects a single line; normalize whitespace.
        line = " ".join(text.split()).replace("\n", " ")
        if not line:
            return False
        labels, scores = self._model.predict(line, k=1)
        score = float(scores[0])
        if score < self.threshold:
            return False
        if self.label is not None and labels[0] != self.label:
            return False
        return True

    def describe(self) -> str:
        return (
            f"FastTextQualityFilter(model={self.model_path}, "
            f"threshold={self.threshold}, label={self.label})"
        )


class CompositeFilter(QualityFilter):
    """Apply multiple filters in sequence; a document is kept only if all pass."""

    def __init__(self, filters: Sequence[QualityFilter]):
        self.filters = list(filters)

    def __call__(self, text: str) -> bool:
        return all(f(text) for f in self.filters)

    def describe(self) -> str:
        return f"CompositeFilter([{', '.join(f.describe() for f in self.filters)}])"

    def stats(self, texts: Sequence[str]) -> Dict[str, int]:
        """Return how many documents each individual filter rejects."""
        rejected: Dict[str, int] = {f.describe(): 0 for f in self.filters}
        for text in texts:
            for f in self.filters:
                if not f(text):
                    rejected[f.describe()] += 1
                    break
        return rejected
