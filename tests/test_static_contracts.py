"""Static contract tests for repository configuration files.

Verifies that packaging/dependency/infra declarations stay consistent:
pyproject extras, torch pin, uv.lock provides-extras coverage, requirements.txt
derivation annotation, docker-compose GPU syntax, CI lock-check & Docker build
jobs, and api_server importability with the `server` extra.

This module currently covers FR1 of AR-v0.5.0-hpc-readiness; it will later
host additional static contracts (e.g. FR13).
"""

import re
import tomllib
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]

EXPECTED_EXTRAS = {"eval", "quant", "hf", "plot", "dev", "server", "validate"}


def _load_pyproject():
    with open(ROOT / "pyproject.toml", "rb") as f:
        return tomllib.load(f)


def _load_uv_lock():
    with open(ROOT / "uv.lock", "rb") as f:
        return tomllib.load(f)


def test_pyproject_declares_server_and_validate_extras():
    """FR1-AC1: pyproject declares `server` (fastapi/uvicorn/pydantic) and `validate` (pydantic) extras."""
    pyproject = _load_pyproject()
    optional = pyproject["project"]["optional-dependencies"]

    assert "server" in optional, "missing [project.optional-dependencies].server extra"
    server_deps = " ".join(optional["server"]).lower()
    for pkg in ("fastapi", "uvicorn", "pydantic"):
        assert pkg in server_deps, f"extra `server` must depend on {pkg}"

    assert "validate" in optional, "missing [project.optional-dependencies].validate extra"
    validate_deps = " ".join(optional["validate"]).lower()
    assert "pydantic" in validate_deps, "extra `validate` must depend on pydantic"


def test_torch_pin_lower_bound_2_5():
    """FR1: torch dependency is pinned to the ~=2.5 compatible-release series."""
    pyproject = _load_pyproject()
    deps = pyproject["project"]["dependencies"]

    torch_specs = [d for d in deps if re.match(r"^\s*torch(\[|[<>=!~]|$)", d)]
    assert torch_specs, "no torch entry found in project dependencies"
    assert any(
        re.search(r"~=\s*2\.5(\.\d+)?", spec) for spec in torch_specs
    ), f"torch must be pinned with ~=2.5, got {torch_specs}"


def test_uv_lock_provides_extras_complete():
    """FR1-AC3: uv.lock provides-extras covers every extra declared in pyproject."""
    pyproject = _load_pyproject()
    declared = set(pyproject["project"]["optional-dependencies"].keys())

    lock = _load_uv_lock()
    root_pkgs = [p for p in lock.get("package", []) if p.get("name") == "nanogpt-modern"]
    assert root_pkgs, "root package 'nanogpt-modern' not found in uv.lock"
    root_pkg = root_pkgs[0]
    provides = set(root_pkg.get("provides-extras", [])) | set(
        root_pkg.get("metadata", {}).get("provides-extras", [])
    )

    missing = declared - provides
    assert not missing, (
        f"uv.lock provides-extras is missing extras declared in pyproject: {sorted(missing)}; "
        f"provides-extras={sorted(provides)}"
    )
    assert EXPECTED_EXTRAS <= provides, (
        f"expected extras {sorted(EXPECTED_EXTRAS)} not fully covered by provides-extras "
        f"{sorted(provides)}"
    )


def test_requirements_txt_derived_annotation_or_removed():
    """FR1: requirements.txt is either removed or explicitly marked as derived from pyproject.toml."""
    req_path = ROOT / "requirements.txt"
    if not req_path.exists():
        pytest.skip("requirements.txt removed; single-source pyproject.toml")
    head = req_path.read_text(encoding="utf-8", errors="replace")[:2048]
    assert re.search(r"derived from pyproject|派生", head, re.IGNORECASE), (
        "requirements.txt exists but has no 'derived from pyproject' / 「派生」 annotation; "
        "mark it as derived or delete it"
    )


def test_docker_compose_uses_modern_gpu_syntax():
    """FR1: docker-compose uses modern `gpus:` syntax instead of deprecated `runtime: nvidia`."""
    compose_path = ROOT / "docker-compose.yml"
    compose = yaml.safe_load(compose_path.read_text(encoding="utf-8"))
    services = compose.get("services", {})
    assert services, "docker-compose.yml declares no services"

    gpu_services = ("train", "api")
    for name, svc in services.items():
        assert svc.get("runtime") != "nvidia", (
            f"service '{name}' uses deprecated 'runtime: nvidia'; use 'gpus:' instead"
        )
        if name in gpu_services:
            assert "gpus" in svc, (
                f"GPU service '{name}' must use the modern 'gpus:' key"
            )


def test_ci_has_lock_check_and_docker_build():
    """FR1-AC2: CI contains a uv lock check step and a Docker build job."""
    ci_path = ROOT / ".github" / "workflows" / "ci.yml"
    ci = yaml.safe_load(ci_path.read_text(encoding="utf-8"))

    def all_runs(doc):
        runs = []
        for job in doc.get("jobs", {}).values():
            for step in job.get("steps", []):
                run = step.get("run")
                if run:
                    runs.append(run)
        return runs

    runs = all_runs(ci)
    assert runs, "CI workflow declares no steps with 'run'"

    lock_check = any(
        re.search(r"uv\s+lock\s+--(check|locked)", run) for run in runs
    )
    assert lock_check, "no step runs 'uv lock --check' (or 'uv lock --locked')"

    docker_build = any(
        re.search(r"docker\s+(compose\s+)?build", run) for run in runs
    )
    assert docker_build, "no step runs 'docker build' / 'docker compose build'"


def test_api_server_importable_with_server_extra():
    """FR1-AC1: with the `server` extra installed, `import api_server` succeeds.

    Skips if fastapi is not installed locally (E8 graceful-degradation semantics).
    """
    pytest.importorskip("fastapi", reason="server extra (fastapi) not installed")
    import api_server  # noqa: F401
