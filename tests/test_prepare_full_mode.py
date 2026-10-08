"""Interface tests for ``data/prepare.py`` explicit full-dataset mode (FR2).

Covers the T002 reproduction trap: ``--max_docs`` defaults to ``None`` while
``_prepare_map_mode`` / ``_prepare_streaming_mode`` silently fall back to
1,000,000 (train) / 10,000 (val) docs, so a bare ``--split train`` run
produces a 1M-doc subset instead of the full 7,913,597-doc dataset.

Contracts verified here (per design.md §4.3.6 / §6.2):

* ``--full`` flag exists and parses to ``args.full is True``;
* ``resolve_max_docs(split, max_docs, full)`` implements the priority
  explicit ``--max_docs``  >  ``--full`` (no limit, ``None``)  >  default
  (train 1M / val 10K);
* ``--help`` documents the default 1M semantics next to ``--full``.

This module covers FR2 of AR-v0.5.0-hpc-readiness.
"""

import contextlib
import io
import re

import pytest

from data.prepare import get_args


def test_full_flag_parses():
    """FR2: `--full` is a recognized flag and parses to True."""
    args = get_args(["--split", "train", "--full"])
    assert args.full is True


def test_resolve_max_docs_priority():
    """FR2: resolve_max_docs priority = explicit --max_docs > --full > default."""
    from data.prepare import resolve_max_docs

    # Explicit --max_docs wins even when --full is set.
    assert resolve_max_docs("train", 123, True) == 123
    # --full with no explicit cap means no limit (None).
    assert resolve_max_docs("train", None, True) is None
    assert resolve_max_docs("val", None, True) is None
    # Default fallback: 1M for train, 10K for val.
    assert resolve_max_docs("train", None, False) == 1_000_000
    assert resolve_max_docs("val", None, False) == 10_000


def _capture_help() -> str:
    stdout = io.StringIO()
    with contextlib.redirect_stdout(stdout):
        with pytest.raises(SystemExit):
            get_args(["--help"])
    return stdout.getvalue()


def test_max_docs_help_documents_default_semantics():
    """FR2: --help surfaces both the --full flag and the default 1M semantics."""
    help_text = _capture_help()

    assert "--full" in help_text, (
        "--help output must document the --full flag; the bare "
        "`--split train` 1M-subset trap must be visible"
    )
    assert re.search(r"1,000,000|1M", help_text), (
        "--help must spell out the default max_docs magnitude "
        "('1,000,000' or '1M')"
    )
    assert "train" in help_text, (
        "--help must tie the default docs cap to the train split"
    )


def test_full_mode_help_mentioned():
    """FR2: the --full help line carries a full-dataset semantics description."""
    help_text = _capture_help()

    # Option-description lines only: usage lines like `[--max_docs MAX_DOCS]
    # [--full] ...` also contain the token but carry no semantics.
    full_lines = [
        line
        for line in help_text.splitlines()
        if "--full" in line and line.lstrip().startswith("--full")
    ]
    assert full_lines, "--help output has no --full line"

    for line in full_lines:
        # Strip the flag token itself so a bare `--full` (no description)
        # cannot pass; the remainder must still say "full" or 「全量」.
        remainder = line.replace("--full", "")
        assert re.search(r"full|全量", remainder, re.IGNORECASE), (
            f"--full help line lacks full-dataset semantics: {line!r}"
        )
