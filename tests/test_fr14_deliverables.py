"""FR14 (AR-v0.5.0-hpc-readiness T021) Red phase: docs deliverables do not exist yet.

Contracts covered (design FR14 / FR14-AC2):

* ``docs/ARCHITECTURE.md`` exists as the training-foundation architecture
  doc: >= 100 lines, covers the BaseTrainer lifecycle hooks, the training
  mechanisms (DDP / FSDP / AMP / EMA / Checkpoint), the monitoring
  degradation philosophy, all five trainer entry points, and cross-links
  with ``docs/MIGRATION.md``;
* ``docs/GRPO_NOTES.md`` exists (>= 60 lines) and covers the three GRPO
  algorithm dimensions (advantage / reference model / KL) plus the GRPO
  family members (GRPOTrainer / IterativeGRPOTrainer / DPOTrainer and the
  hydra shims);
* README.md's documentation index lists both new docs;
* ``model/modern_gpt.py`` public API is fully documented (FR14-AC2:
  100% docstring coverage on public methods/classes/functions via ast),
  the module docstring mentions GQA / MoE / speculative decoding / KV
  cache, and the source carries four ``SECTION:`` navigation banners.

Only stdlib + pytest are used; tests are mutually independent.
"""

import ast
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

ARCHITECTURE_PATH = ROOT / "docs" / "ARCHITECTURE.md"
GRPO_NOTES_PATH = ROOT / "docs" / "GRPO_NOTES.md"
MIGRATION_PATH = ROOT / "docs" / "MIGRATION.md"
README_PATH = ROOT / "README.md"
MODERN_GPT_PATH = ROOT / "model" / "modern_gpt.py"


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def _read_strict(path: Path) -> str:
    return path.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# A. ARCHITECTURE.md contracts
# ---------------------------------------------------------------------------


def test_architecture_doc_exists():
    """ARCHITECTURE.md exists and is a substantial doc (>= 100 lines)."""
    assert ARCHITECTURE_PATH.is_file(), "docs/ARCHITECTURE.md does not exist"
    n_lines = len(_read(ARCHITECTURE_PATH).splitlines())
    assert n_lines >= 100, f"docs/ARCHITECTURE.md too short: {n_lines} lines (< 100)"


def test_architecture_doc_covers_trainer_lifecycle():
    """ARCHITECTURE.md documents the BaseTrainer lifecycle hooks in order."""
    assert ARCHITECTURE_PATH.is_file(), "docs/ARCHITECTURE.md does not exist"
    text = _read(ARCHITECTURE_PATH)
    hooks = [
        "_init_state",
        "_build_data",
        "_build_model",
        "_build_optimizer",
        "_build_scheduler",
        "_maybe_resume",
        "on_step_end",
        "train",
    ]
    missing = [hook for hook in hooks if hook not in text]
    assert not missing, f"docs/ARCHITECTURE.md missing lifecycle hooks: {missing}"


def test_architecture_doc_covers_mechanisms():
    """ARCHITECTURE.md covers DDP / FSDP / AMP / EMA / Checkpoint mechanisms."""
    assert ARCHITECTURE_PATH.is_file(), "docs/ARCHITECTURE.md does not exist"
    text = _read(ARCHITECTURE_PATH)
    mechanisms = ["DDP", "FSDP", "AMP", "EMA", "Checkpoint"]
    missing = [m for m in mechanisms if m not in text]
    assert not missing, f"docs/ARCHITECTURE.md missing training mechanisms: {missing}"


def test_architecture_doc_covers_degradation():
    """ARCHITECTURE.md explains the monitoring degradation philosophy."""
    assert ARCHITECTURE_PATH.is_file(), "docs/ARCHITECTURE.md does not exist"
    text = _read(ARCHITECTURE_PATH)
    assert "降级" in text, "docs/ARCHITECTURE.md does not mention 降级 (degradation)"
    lower = text.lower()
    assert ("monitor" in lower) or ("degradation" in lower), (
        "docs/ARCHITECTURE.md does not mention monitor/degradation"
    )


def test_architecture_doc_covers_five_trainers():
    """ARCHITECTURE.md covers all five trainer entry points."""
    assert ARCHITECTURE_PATH.is_file(), "docs/ARCHITECTURE.md does not exist"
    text = _read(ARCHITECTURE_PATH)
    trainers = ["train_pretrain", "train_sft", "train_grpo", "train_dpo", "iterative_grpo"]
    missing = [t for t in trainers if t not in text]
    assert not missing, f"docs/ARCHITECTURE.md missing trainer entry points: {missing}"


def test_architecture_doc_cross_references():
    """ARCHITECTURE.md <-> MIGRATION.md cross-link each other."""
    assert ARCHITECTURE_PATH.is_file(), "docs/ARCHITECTURE.md does not exist"
    assert MIGRATION_PATH.is_file(), "docs/MIGRATION.md does not exist"
    arch_text = _read(ARCHITECTURE_PATH)
    assert "MIGRATION.md" in arch_text, (
        "docs/ARCHITECTURE.md does not link back to docs/MIGRATION.md"
    )
    migration_text = _read(MIGRATION_PATH)
    assert "ARCHITECTURE.md" in migration_text, (
        "docs/MIGRATION.md does not link to docs/ARCHITECTURE.md"
    )


def test_readme_docs_index_lists_new_docs():
    """README.md documentation index lists ARCHITECTURE.md and MIGRATION.md."""
    assert README_PATH.is_file(), "README.md does not exist"
    readme = _read(README_PATH)
    # Prefer the documentation-index section; fall back to the whole file.
    match = re.search(r"文档索引(.*?)(?=\n## |\Z)", readme, re.DOTALL)
    scope = match.group(1) if match else readme
    missing = [doc for doc in ("ARCHITECTURE.md", "MIGRATION.md") if doc not in scope]
    assert not missing, f"README.md docs index missing links to: {missing}"


# ---------------------------------------------------------------------------
# B. GRPO family notes contracts
# ---------------------------------------------------------------------------


def test_grpo_notes_exist():
    """docs/GRPO_NOTES.md exists and is substantial (>= 60 lines)."""
    assert GRPO_NOTES_PATH.is_file(), "docs/GRPO_NOTES.md does not exist"
    n_lines = len(_read(GRPO_NOTES_PATH).splitlines())
    assert n_lines >= 60, f"docs/GRPO_NOTES.md too short: {n_lines} lines (< 60)"


def test_grpo_notes_cover_algorithm_dimensions():
    """GRPO_NOTES.md covers advantage / reference model / KL dimensions."""
    assert GRPO_NOTES_PATH.is_file(), "docs/GRPO_NOTES.md does not exist"
    text = _read(GRPO_NOTES_PATH)
    assert "advantage" in text.lower(), "docs/GRPO_NOTES.md missing advantage dimension"
    assert ("参考模型" in text) or ("reference" in text.lower()), (
        "docs/GRPO_NOTES.md missing reference-model dimension"
    )
    assert "KL" in text, "docs/GRPO_NOTES.md missing KL dimension"


def test_grpo_notes_cover_family_members():
    """GRPO_NOTES.md covers the GRPO family classes and the hydra shims."""
    assert GRPO_NOTES_PATH.is_file(), "docs/GRPO_NOTES.md does not exist"
    text = _read(GRPO_NOTES_PATH)
    members = ["GRPOTrainer", "IterativeGRPOTrainer", "DPOTrainer"]
    missing = [m for m in members if m not in text]
    assert not missing, f"docs/GRPO_NOTES.md missing family members: {missing}"
    assert "hydra" in text.lower(), "docs/GRPO_NOTES.md does not mention the hydra shims"


# ---------------------------------------------------------------------------
# C. modern_gpt.py docstring coverage (FR14-AC2)
# ---------------------------------------------------------------------------


def _parse_modern_gpt() -> ast.Module:
    assert MODERN_GPT_PATH.is_file(), "model/modern_gpt.py does not exist"
    return ast.parse(_read_strict(MODERN_GPT_PATH))


def test_modern_gpt_public_methods_docstring_coverage():
    """FR14-AC2: 100% docstring coverage on public (non-dunder) methods."""
    tree = _parse_modern_gpt()
    missing = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef):
            continue
        for child in node.body:
            if not isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            if child.name.startswith("_"):
                continue
            if not ast.get_docstring(child):
                missing.append(f"{node.name}.{child.name}")
    assert not missing, (
        "model/modern_gpt.py public methods missing docstrings "
        f"({len(missing)}): {missing}"
    )


def test_modern_gpt_public_classes_and_functions_have_docstring():
    """Top-level public classes and functions must have docstrings."""
    tree = _parse_modern_gpt()
    missing = []
    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            if not ast.get_docstring(node):
                missing.append(f"class {node.name}")
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if not node.name.startswith("_") and not ast.get_docstring(node):
                missing.append(f"function {node.name}")
    assert not missing, (
        "model/modern_gpt.py top-level public classes/functions missing "
        f"docstrings: {missing}"
    )


def test_modern_gpt_module_docstring_current():
    """Module docstring mentions GQA / MoE / speculative decoding / KV cache."""
    tree = _parse_modern_gpt()
    doc = ast.get_docstring(tree)
    assert doc, "model/modern_gpt.py has no module docstring"
    lower = doc.lower()
    assert "gqa" in lower, "module docstring does not mention GQA"
    assert "moe" in lower, "module docstring does not mention MoE"
    assert ("投机" in doc) or ("speculative" in lower), (
        "module docstring does not mention speculative decoding"
    )
    assert "kv" in lower, "module docstring does not mention KV cache"


# ---------------------------------------------------------------------------
# D. modern_gpt.py section navigation
# ---------------------------------------------------------------------------


def test_modern_gpt_has_section_navigation():
    """Source carries >= 4 SECTION banners covering the four themes."""
    assert MODERN_GPT_PATH.is_file(), "model/modern_gpt.py does not exist"
    src = _read_strict(MODERN_GPT_PATH)
    section_lines = [line for line in src.splitlines() if "SECTION:" in line]
    assert len(section_lines) >= 4, (
        f"model/modern_gpt.py has only {len(section_lines)} SECTION: banners (< 4)"
    )
    joined = "\n".join(section_lines).upper()
    themes = {
        "architecture components": ("ARCHITECTURE COMPONENTS" in joined) or ("架构组件" in joined),
        "KV cache": "KV CACHE" in joined,
        "speculative decoding": ("SPECULATIVE" in joined) or ("投机" in joined),
        "generation": ("GENERATION" in joined) or ("生成" in joined),
    }
    missing = [name for name, ok in themes.items() if not ok]
    assert not missing, f"model/modern_gpt.py SECTION banners missing themes: {missing}"
