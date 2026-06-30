"""Code pre-training dataset loader.

Supports loading from local source files or HuggingFace ``bigcode/the-stack``.
Includes AST-based syntax filtering and language-group sampling.
"""
from __future__ import annotations

import ast
import json
import os
import random
import warnings
from pathlib import Path
from typing import Any, Dict, Iterator, List, Mapping, Optional, Sequence, Union

import torch
from torch.utils.data import IterableDataset


# ---------------------------------------------------------------------------
# AST filtering
# ---------------------------------------------------------------------------

def _is_valid_python(source: str) -> bool:
    """Return ``True`` if *source* is syntactically valid Python."""
    try:
        ast.parse(source)
        return True
    except SyntaxError:
        return False


def _is_valid_javascript(source: str) -> bool:
    """Heuristic JS syntax check (falls back to brace balance)."""
    braces = {"{": 0, "[": 0, "(": 0}
    closers = {"}": "{", "]": "[", ")": "("}
    in_string: Optional[str] = None
    for ch in source:
        if in_string is None:
            if ch in '"\'':
                in_string = ch
            elif ch in braces:
                braces[ch] += 1
            elif ch in closers:
                braces[closers[ch]] -= 1
                if braces[closers[ch]] < 0:
                    return False
        else:
            if ch == in_string and (len(source) == 1 or source[source.index(ch) - 1] != "\\"):
                in_string = None
    return all(v == 0 for v in braces.values())


def _is_valid_cpp(source: str) -> bool:
    """Heuristic C++ syntax check (brace balance + no obvious errors)."""
    return _is_valid_javascript(source)  # brace balance is identical


_AST_FILTERS: Dict[str, Any] = {
    "python": _is_valid_python,
    "js": _is_valid_javascript,
    "javascript": _is_valid_javascript,
    "cpp": _is_valid_cpp,
    "c++": _is_valid_cpp,
    "c": _is_valid_javascript,
}


# ---------------------------------------------------------------------------
# Local file loader
# ---------------------------------------------------------------------------

_EXTENSION_MAP: Dict[str, str] = {
    ".py": "python",
    ".js": "js",
    ".ts": "js",
    ".jsx": "js",
    ".tsx": "js",
    ".cpp": "cpp",
    ".cc": "cpp",
    ".cxx": "cpp",
    ".hpp": "cpp",
    ".h": "cpp",
    ".c": "c",
    ".java": "java",
    ".go": "go",
    ".rs": "rust",
    ".rb": "ruby",
}


def _detect_language(path: Union[str, Path]) -> str:
    ext = Path(path).suffix.lower()
    return _EXTENSION_MAP.get(ext, "unknown")


def _read_local_files(
    root: Union[str, Path],
    languages: Optional[Sequence[str]] = None,
    max_files: Optional[int] = None,
) -> Iterator[Dict[str, Any]]:
    """Yield code examples from a local directory tree."""
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(f"Code root not found: {root}")
    count = 0
    for file_path in root.rglob("*"):
        if not file_path.is_file():
            continue
        lang = _detect_language(file_path)
        if languages is not None and lang not in languages:
            continue
        try:
            source = file_path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        # AST filter
        checker = _AST_FILTERS.get(lang)
        if checker is not None and not checker(source):
            continue
        yield {
            "content": source,
            "language": lang,
            "file_path": str(file_path),
        }
        count += 1
        if max_files is not None and count >= max_files:
            break


# ---------------------------------------------------------------------------
# HuggingFace The Stack loader
# ---------------------------------------------------------------------------

def _load_hf_stack(
    language: str,
    split: str = "train",
    max_files: Optional[int] = None,
    streaming: bool = True,
) -> Iterator[Dict[str, Any]]:
    """Yield examples from ``bigcode/the-stack`` via ``datasets``."""
    try:
        from datasets import load_dataset  # type: ignore[import-untyped]
    except Exception as exc:
        raise ImportError(f"datasets library is required for HF loading: {exc}")

    ds = load_dataset("bigcode/the-stack", data_dir=f"data/{language}", split=split, streaming=streaming)
    count = 0
    for row in ds:
        source = row.get("content", "")
        # Optional AST filter
        checker = _AST_FILTERS.get(language)
        if checker is not None and not checker(source):
            continue
        yield {
            "content": source,
            "language": language,
            "file_path": row.get("hexsha", "unknown"),
        }
        count += 1
        if max_files is not None and count >= max_files:
            break


# ---------------------------------------------------------------------------
# CodeDataset
# ---------------------------------------------------------------------------

class CodeDataset(IterableDataset):
    """Iterable dataset for code pre-training.

    Supports two source modes:

    - **local**: read from a directory tree on disk.
    - **huggingface**: stream from ``bigcode/the-stack``.

    Each yielded example is a dict with at least ``content``, ``language``,
    and ``file_path``.  When ``tokenize`` is enabled, the dict also contains
    ``input_ids`` and ``labels``.

    Parameters
    ----------
    source:
        ``"local"`` or ``"huggingface"``.
    path_or_language:
        For local mode: root directory.  For HF mode: language slug.
    languages:
        Filter to specific languages (e.g. ``["python", "js"]``).  ``None``
        keeps all.
    tokenizer:
        Optional tokenizer with ``encode(text) -> List[int]``.
    max_length:
        Block size for tokenization.
    max_files:
        Cap the number of files read.
    split:
        HuggingFace split name.
    ast_filter:
        If ``True`` (default), run AST-based syntax checks.
    """

    def __init__(
        self,
        source: str = "local",
        path_or_language: str = ".",
        languages: Optional[Sequence[str]] = None,
        tokenizer: Optional[Any] = None,
        max_length: int = 1024,
        max_files: Optional[int] = None,
        split: str = "train",
        ast_filter: bool = True,
    ):
        self.source = source
        self.path_or_language = path_or_language
        self.languages = list(languages) if languages is not None else None
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.max_files = max_files
        self.split = split
        self.ast_filter = ast_filter

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        if self.source == "local":
            raw = _read_local_files(
                self.path_or_language,
                languages=self.languages,
                max_files=self.max_files,
            )
        elif self.source == "huggingface":
            raw = _load_hf_stack(
                self.path_or_language,
                split=self.split,
                max_files=self.max_files,
            )
        else:
            raise ValueError(f"Unknown source: {self.source}")

        for item in raw:
            # If language filtering is requested but not already applied
            if self.languages is not None and item["language"] not in self.languages:
                continue
            example = {
                "content": item["content"],
                "language": item["language"],
                "file_path": item["file_path"],
                "__mix_source__": item["language"],
            }
            if self.tokenizer is not None:
                tokens = self.tokenizer.encode(item["content"])
                if len(tokens) > self.max_length:
                    tokens = tokens[: self.max_length]
                input_ids = torch.tensor(tokens[:-1], dtype=torch.long)
                labels = torch.tensor(tokens[1:], dtype=torch.long)
                example["input_ids"] = input_ids
                example["labels"] = labels
            yield example

    def state_dict(self) -> Dict[str, Any]:
        return {}

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        pass


class GroupedCodeSampler:
    """Sample code files by language group with fixed probabilities.

    Wraps multiple ``CodeDataset`` instances and yields from them according
    to configured language weights.  This is a thin compatibility layer over
    ``data.mixer.MixedIterableDataset``.
    """

    def __init__(
        self,
        datasets: Mapping[str, CodeDataset],
        weights: Mapping[str, float],
        temperature: float = 1.0,
        seed: int = 42,
        total_examples: Optional[int] = None,
    ):
        self.datasets = dict(datasets)
        self.weights = dict(weights)
        self.temperature = temperature
        self.seed = seed
        self.total_examples = total_examples

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        # Lazy import to avoid circular dependency.
        from data.mixer import MixedIterableDataset
        return iter(
            MixedIterableDataset(
                sources=self.datasets,
                weights=self.weights,
                temperature=self.temperature,
                seed=self.seed,
                total_examples=self.total_examples,
            )
        )


# ---------------------------------------------------------------------------
# Unit-test stubs
# ---------------------------------------------------------------------------
def _test_code_dataset():
    # Create a temporary directory with a valid Python file.
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        py_file = Path(tmpdir) / "test.py"
        py_file.write_text("def foo():\n    return 42\n", encoding="utf-8")
        ds = CodeDataset(source="local", path_or_language=tmpdir, languages=["python"])
        items = list(ds)
        assert len(items) == 1
        assert items[0]["language"] == "python"
        assert "content" in items[0]
        print("CodeDataset smoke test passed")


def _test_grouped_sampler():
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        py_file = Path(tmpdir) / "a.py"
        py_file.write_text("x = 1", encoding="utf-8")
        js_file = Path(tmpdir) / "b.js"
        js_file.write_text("var x = 1;", encoding="utf-8")
        ds_py = CodeDataset(source="local", path_or_language=tmpdir, languages=["python"])
        ds_js = CodeDataset(source="local", path_or_language=tmpdir, languages=["js"])
        sampler = GroupedCodeSampler(
            datasets={"python": ds_py, "js": ds_js},
            weights={"python": 0.5, "js": 0.5},
            seed=42,
            total_examples=4,
        )
        items = list(sampler)
        assert 1 <= len(items) <= 2
        assert all("__mix_source__" in item for item in items)
        print("GroupedCodeSampler smoke test passed")


if __name__ == "__main__":
    _test_code_dataset()
    _test_grouped_sampler()
