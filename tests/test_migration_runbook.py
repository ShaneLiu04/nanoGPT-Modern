"""Contract tests for the FR2 migration runbook and reproduction commands.

Verifies T002 of AR-v0.5.0-hpc-readiness:

* ``docs/MIGRATION.md`` exists as a non-empty runbook and covers the
  required sections (must-bring files, env rebuild via ``uv sync``, data
  validation via ``validate.py``, smoke run, ablation orchestrator, and
  redundant-artifact cleanup disposition);
* every ``prepare.py ... --split`` reproduction command in README.md,
  docs/EXPERIMENTS.md and docs/TECH_REPORT.md is explicit about dataset
  scope (``--full`` or an explicit ``--max_docs``), so following the docs
  can no longer silently produce a 1M-doc subset of the 7,913,597-doc
  on-disk dataset (FR2-AC2).

Only stdlib + pytest are used; tests are mutually independent.
"""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

MIGRATION_PATH = ROOT / "docs" / "MIGRATION.md"

REPRODUCTION_DOCS = [
    ROOT / "README.md",
    ROOT / "docs" / "EXPERIMENTS.md",
    ROOT / "docs" / "TECH_REPORT.md",
]


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def test_migration_doc_exists():
    """FR2-AC1: docs/MIGRATION.md exists and is non-empty."""
    assert MIGRATION_PATH.is_file(), "docs/MIGRATION.md does not exist"
    text = _read(MIGRATION_PATH).strip()
    assert text, "docs/MIGRATION.md is empty"


def test_migration_doc_covers_required_sections():
    """FR2-AC1: MIGRATION.md covers the required runbook sections."""
    assert MIGRATION_PATH.is_file(), "docs/MIGRATION.md does not exist"
    text = _read(MIGRATION_PATH)

    required = {
        "must-bring checklist (「必带」/ must-bring)": (
            lambda: ("必带" in text) or bool(re.search(r"must[- ]bring", text, re.IGNORECASE))
        ),
        "train.bin": lambda: "train.bin" in text,
        "val.bin": lambda: "val.bin" in text,
        "train.bin.shuffle.idx": lambda: "train.bin.shuffle.idx" in text,
        "env rebuild via uv sync": lambda: "uv sync" in text,
        "data validation via validate.py": lambda: "validate.py" in text,
        "smoke run (「冒烟」/ smoke)": (
            lambda: ("冒烟" in text) or bool(re.search(r"smoke", text, re.IGNORECASE))
        ),
        "ablation orchestrator (run_component_ablations)": (
            lambda: "run_component_ablations" in text
        ),
        "redundant train_shards cleanup disposition": lambda: "train_shards" in text,
        "redundant owt_parquet cleanup disposition": lambda: "owt_parquet" in text,
    }

    missing = [name for name, check in required.items() if not check()]
    assert not missing, f"docs/MIGRATION.md is missing required sections: {missing}"


def test_reproduction_commands_use_full_mode():
    """FR2-AC2: no doc reproduces with a bare `prepare.py --split` command.

    Every ``prepare.py ... --split`` line in README.md / docs/EXPERIMENTS.md /
    docs/TECH_REPORT.md must be explicit about dataset scope: either use
    ``--full``, or pass an explicit numeric ``--max_docs``.  Comment lines
    (starting with ``#``) that explain the default semantics are exempt.
    """
    offenders = []
    for doc in REPRODUCTION_DOCS:
        assert doc.is_file(), f"reproduction doc missing: {doc}"
        for lineno, line in enumerate(_read(doc).splitlines(), start=1):
            if not re.search(r"prepare\.py[^\n]*--split", line):
                continue
            if "--full" in line:
                continue
            if re.search(r"--max_docs\s+\d+", line):
                continue
            stripped = line.lstrip()
            if stripped.startswith("#") and re.search(r"默认|default", line, re.IGNORECASE):
                continue
            offenders.append(f"{doc.relative_to(ROOT)}:{lineno}: {line.strip()}")

    assert not offenders, (
        "Reproduction commands run a bare `prepare.py --split` without "
        "--full or an explicit --max_docs (silent 1M-doc subset trap); "
        f"offending lines: {offenders}"
    )
