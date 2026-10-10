"""FR10 repo layout tests (T016 Red phase).

Structural assertions only: no model loading, no training, fast (seconds).
Green-phase contract:
- Migrate export_gguf / export_to_hf / export_to_onnx / load_from_hf to tools/export/,
  profile_training / profile_inference to tools/profiling/.
- Root keeps exactly: api_server.py, run_ablations.py (deprecated), run_component_ablations.py,
  run_full_evaluation.py.
- Delete run_inference_ablation.py, generate_experiment_log.py, logs_prepare.txt,
  logs_prepare_train.txt.
- pyproject.toml console scripts and mypy overrides synced to tools.export.* paths.
- .github/workflows/ci.yml no longer references root-level export_to_hf.py / export_to_onnx.py.
"""

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python < 3.11 fallback
    import tomli as tomllib


def _pyproject():
    with open(REPO_ROOT / "pyproject.toml", "rb") as f:
        return tomllib.load(f)


def test_root_python_whitelist():
    root_py = sorted(p.name for p in REPO_ROOT.glob("*.py"))
    assert root_py == [
        "api_server.py",
        "run_ablations.py",
        "run_component_ablations.py",
        "run_full_evaluation.py",
    ]


def test_export_scripts_moved():
    export_dir = REPO_ROOT / "tools" / "export"
    for name in ("export_gguf.py", "export_to_hf.py", "export_to_onnx.py", "load_from_hf.py"):
        assert (export_dir / name).is_file(), f"missing tools/export/{name}"
    assert (REPO_ROOT / "tools" / "__init__.py").is_file()
    assert (export_dir / "__init__.py").is_file()


def test_profiling_scripts_moved():
    profiling_dir = REPO_ROOT / "tools" / "profiling"
    for name in ("profile_training.py", "profile_inference.py"):
        assert (profiling_dir / name).is_file(), f"missing tools/profiling/{name}"
    assert (REPO_ROOT / "tools" / "__init__.py").is_file()
    assert (profiling_dir / "__init__.py").is_file()


def test_dead_files_removed():
    for name in (
        "run_inference_ablation.py",
        "generate_experiment_log.py",
        "logs_prepare.txt",
        "logs_prepare_train.txt",
    ):
        assert not (REPO_ROOT / name).exists(), f"dead file still present: {name}"


def test_run_ablations_deprecated():
    src = (REPO_ROOT / "run_ablations.py").read_text(encoding="utf-8").lower()
    assert "deprecated" in src
    assert "run_component_ablations" in src


def test_console_scripts_and_mypy_synced():
    pyproject = _pyproject()
    scripts = pyproject["project"]["scripts"]
    assert scripts["nanogpt-export-gguf"] == "tools.export.export_gguf:main"
    assert scripts["nanogpt-export-hf"] == "tools.export.export_to_hf:main"
    assert scripts["nanogpt-load-hf"] == "tools.export.load_from_hf:main"
    # AC2 pin: unchanged (already correct in current pyproject).
    assert scripts["nanogpt-ablations"] == "run_component_ablations:main"

    # mypy overrides: bare "export_to_onnx" replaced by "tools.export.export_to_onnx".
    module_lists = [
        o["module"] if isinstance(o["module"], list) else [o["module"]]
        for o in pyproject.get("tool", {}).get("mypy", {}).get("overrides", [])
    ]
    all_modules = [m for lst in module_lists for m in lst]
    assert "export_to_onnx" not in all_modules
    assert "tools.export.export_to_onnx" in all_modules


def test_mypy_override_updated():
    pyproject = _pyproject()
    module_lists = [
        o["module"] if isinstance(o["module"], list) else [o["module"]]
        for o in pyproject.get("tool", {}).get("mypy", {}).get("overrides", [])
    ]
    all_modules = [m for lst in module_lists for m in lst]
    assert "tools.export.export_to_onnx" in all_modules
    assert "export_to_onnx" not in all_modules


def test_ci_paths_synced():
    ci_text = (REPO_ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    # Root-path references use " export_to_hf.py" / " export_to_onnx.py" (space-preceded
    # in the mypy/black/ruff commands). The new path "tools/export/export_to_hf.py"
    # contains "export_to_hf.py" as a substring, so use a negative lookbehind on "/"
    # plus a word boundary to only flag references NOT already under tools/export/.
    for stem in ("export_to_hf", "export_to_onnx"):
        pattern = r"(?<!/)\b" + re.escape(stem) + r"\.py"
        offending = re.findall(pattern, ci_text)
        assert not offending, f"ci.yml still references root-level {stem}.py"
    assert "tools/export/" in ci_text


def test_moved_modules_importable():
    import importlib

    for module in (
        "tools.export.export_to_hf",
        "tools.export.load_from_hf",
        "tools.profiling.profile_training",
        "tools.profiling.profile_inference",
    ):
        importlib.import_module(module)
