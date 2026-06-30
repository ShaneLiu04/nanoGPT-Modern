"""Multilingual data pipeline: tokenisers, language detection, and mixed datasets.

Supports both ``tiktoken`` (GPT-2 BPE) and ``sentencepiece`` backends, plus a
lightweight language detector that tries ``langdetect`` before falling back to
heuristic rules.
"""

from __future__ import annotations

import json
import re
import warnings
from abc import ABC, abstractmethod
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Optional

import numpy as np
from torch.utils.data import IterableDataset

# ---------------------------------------------------------------------------
# Tokeniser abstraction
# ---------------------------------------------------------------------------


class TokenizerBackend(ABC):
    """Abstract tokeniser backend."""

    @abstractmethod
    def encode(self, text: str) -> List[int]: ...

    @abstractmethod
    def decode(self, tokens: List[int]) -> str: ...

    @property
    @abstractmethod
    def vocab_size(self) -> int: ...


class TiktokenBackend(TokenizerBackend):
    """Thin wrapper around ``tiktoken.Encoding``."""

    def __init__(self, encoding_name: str = "gpt2"):
        import tiktoken

        self._enc = tiktoken.get_encoding(encoding_name)

    def encode(self, text: str) -> List[int]:
        return self._enc.encode(text)

    def decode(self, tokens: List[int]) -> str:
        return self._enc.decode(tokens)

    @property
    def vocab_size(self) -> int:
        return self._enc.n_vocab


class SentencePieceBackend(TokenizerBackend):
    """Thin wrapper around ``sentencepiece.SentencePieceProcessor``."""

    def __init__(self, model_path: str):
        import sentencepiece as spm  # type: ignore[import-untyped]

        self._sp = spm.SentencePieceProcessor(model_file=model_path)

    def encode(self, text: str) -> List[int]:
        return self._sp.encode(text, out_type=int)

    def decode(self, tokens: List[int]) -> str:
        return self._sp.decode(tokens)

    @property
    def vocab_size(self) -> int:
        return self._sp.vocab_size()


class MultilingualTokenizer:
    """Unified tokeniser that can switch between tiktoken and SentencePiece.

    Parameters
    ----------
    backend:
        Either ``"tiktoken"`` or ``"sentencepiece"``.
    model_path:
        Required for SentencePiece; ignored for tiktoken.
    encoding_name:
        tiktoken encoding name (default ``"gpt2"``).
    """

    def __init__(
        self,
        backend: str = "tiktoken",
        model_path: Optional[str] = None,
        encoding_name: str = "gpt2",
    ):
        backend = backend.lower()
        if backend == "tiktoken":
            self._backend: TokenizerBackend = TiktokenBackend(encoding_name)
        elif backend == "sentencepiece":
            if model_path is None:
                raise ValueError("model_path is required for SentencePiece backend")
            self._backend = SentencePieceBackend(model_path)
        else:
            raise ValueError(f"Unsupported backend: {backend}")

    def encode(self, text: str) -> List[int]:
        return self._backend.encode(text)

    def decode(self, tokens: List[int]) -> str:
        return self._backend.decode(tokens)

    @property
    def vocab_size(self) -> int:
        return self._backend.vocab_size


# ---------------------------------------------------------------------------
# Language detection
# ---------------------------------------------------------------------------


class LanguageDetector:
    """Detect the dominant language of a text string.

    Tries ``langdetect`` first, then falls back to a heuristic based on Unicode
    character ranges.
    """

    def __init__(self):
        self._langdetect_available = False
        try:
            from langdetect import detect  # type: ignore[import-untyped]

            self._detect = detect
            self._langdetect_available = True
        except Exception as exc:
            warnings.warn(f"langdetect unavailable ({exc}); using heuristic fallback.")

    def detect(self, text: str) -> str:
        """Return a 2-letter ISO language code (e.g. ``"en"``, ``"zh"``)."""
        if self._langdetect_available:
            try:
                return self._detect(text[:1000])  # truncate for speed
            except Exception:
                pass
        return self._heuristic_detect(text)

    def _heuristic_detect(self, text: str) -> str:
        # Count characters in major language blocks.
        counts: Dict[str, int] = {
            "en": len(re.findall(r"[a-zA-Z]", text)),
            "zh": len(re.findall(r"[\u4e00-\u9fff]", text)),
            "ja": len(re.findall(r"[\u3040-\u309f\u30a0-\u30ff]", text)),
            "ko": len(re.findall(r"[\uac00-\ud7af]", text)),
            "ar": len(re.findall(r"[\u0600-\u06ff]", text)),
            "ru": len(re.findall(r"[\u0400-\u04ff]", text)),
            "de": len(re.findall(r"[äöüß]", text, re.IGNORECASE)),
            "fr": len(re.findall(r"[àâçéèêëïîôùûü]", text, re.IGNORECASE)),
            "es": len(re.findall(r"[áéíóúñ]", text, re.IGNORECASE)),
        }
        if not counts:
            return "en"
        best = max(counts, key=lambda k: counts[k])
        return best if counts[best] > 0 else "en"


# ---------------------------------------------------------------------------
# Multilingual dataset mixer
# ---------------------------------------------------------------------------


class MultilingualDataset(IterableDataset):
    """Mixed-language dataset that samples from language-specific sources.

    Each source is an iterable that yields examples.  The mixer samples a
    language on every step according to the configured proportions, then
    yields the next example from that language's iterator.  Examples are
    annotated with ``__language__``.

    Parameters
    ----------
    sources:
        Mapping from language code (e.g. ``"en"``, ``"zh"``) to iterable.
    weights:
        Raw sampling weights per language.
    temperature:
        Mixture temperature.
    total_examples:
        Total examples to yield.  ``None`` means run until exhaustion.
    seed:
        Random seed.
    stop_on_exhaustion:
        If ``True``, stop when any source runs out.
    """

    def __init__(
        self,
        sources: Mapping[str, Iterable[Any]],
        weights: Mapping[str, float],
        temperature: float = 1.0,
        total_examples: Optional[int] = None,
        seed: int = 0,
        stop_on_exhaustion: bool = False,
    ):
        if not sources:
            raise ValueError("sources must not be empty")
        if set(sources.keys()) != set(weights.keys()):
            raise ValueError("keys of sources and weights must match")

        self._names = list(weights.keys())
        raw = np.array([float(weights[name]) for name in self._names], dtype=np.float64)
        if raw.sum() <= 0:
            raise ValueError("weights must sum to a positive value")
        if temperature != 1.0:
            raw = np.power(raw, 1.0 / temperature)
        self._probabilities = raw / raw.sum()
        self._sources = dict(sources)
        self._total = total_examples
        self._rng = np.random.default_rng(seed)
        self._stop_on_exhaustion = stop_on_exhaustion
        self._yielded = 0

    def __iter__(self) -> Iterator[Any]:
        iterators = {name: iter(src) for name, src in self._sources.items()}
        active = set(iterators.keys())
        resume = self._yielded
        yielded = 0
        while (self._total is None or yielded < self._total) and active:
            name = self._rng.choice(self._names, p=self._probabilities)
            if name not in active:
                continue
            try:
                example = next(iterators[name])
            except StopIteration:
                if self._stop_on_exhaustion:
                    return
                active.remove(name)
                continue
            if isinstance(example, dict):
                example["__language__"] = name
                example["__mix_source__"] = name
            yielded += 1
            if yielded <= resume:
                continue
            self._yielded = yielded
            yield example

    def state_dict(self) -> Dict[str, Any]:
        return {
            "yielded": self._yielded,
            "total": self._total,
            "probabilities": dict(zip(self._names, self._probabilities.tolist())),
            "rng_state": self._rng.bit_generator.state,
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        self._yielded = int(state.get("yielded", 0))
        self._total = state.get("total", self._total)
        rng_state = state.get("rng_state")
        if rng_state is not None:
            self._rng.bit_generator.state = rng_state


def load_multilingual_config(path: str) -> Dict[str, Any]:
    """Load a multilingual mixture config JSON.

    Example::

        {
          "languages": {"en": 0.6, "zh": 0.3, "ja": 0.1},
          "temperature": 0.8,
          "seed": 42
        }
    """
    with open(path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    if "languages" not in cfg:
        raise ValueError("multilingual config must contain 'languages'")
    return cfg


# ---------------------------------------------------------------------------
# Unit-test stubs
# ---------------------------------------------------------------------------
def _test_multilingual_tokenizer():
    tok = MultilingualTokenizer(backend="tiktoken")
    ids = tok.encode("hello")
    assert isinstance(ids, list)
    assert tok.decode(ids) == "hello"
    print("MultilingualTokenizer smoke test passed")


def _test_language_detector():
    det = LanguageDetector()
    # Test heuristic fallback directly (bypass langdetect for determinism).
    assert det._heuristic_detect("The quick brown fox jumps over the lazy dog.") == "en"
    assert det._heuristic_detect("你好世界，这是一段中文文本。") == "zh"
    assert det._heuristic_detect("こんにちは世界") == "ja"
    assert det._heuristic_detect("안녕하세요 세계") == "ko"
    print("LanguageDetector smoke test passed")


def _test_multilingual_dataset():
    ds = MultilingualDataset(
        sources={
            "en": iter([{"text": "a"}, {"text": "b"}, {"text": "c"}]),
            "zh": iter([{"text": "一"}, {"text": "二"}, {"text": "三"}]),
        },
        weights={"en": 1.0, "zh": 1.0},
        total_examples=4,
        seed=42,
    )
    items = list(ds)
    assert len(items) == 4
    assert all("__language__" in item for item in items)
    print("MultilingualDataset smoke test passed")


if __name__ == "__main__":
    _test_multilingual_tokenizer()
    _test_language_detector()
    _test_multilingual_dataset()
