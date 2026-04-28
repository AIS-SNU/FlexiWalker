"""End-to-end test fixtures. All tests in this module require a GPU."""

import shutil
import subprocess
from pathlib import Path

import pytest


def _has_gpu() -> bool:
    """True iff nvidia-smi is present and exits 0."""
    if shutil.which("nvidia-smi") is None:
        return False
    try:
        r = subprocess.run(
            ["nvidia-smi"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        return r.returncode == 0
    except (subprocess.TimeoutExpired, OSError):
        return False


def pytest_collection_modifyitems(config, items):
    """Skip the whole e2e module if no GPU is visible."""
    if _has_gpu():
        return
    skip_marker = pytest.mark.skip(reason="No GPU detected; skipping e2e tests.")
    for item in items:
        if "tests/e2e" in str(item.fspath):
            item.add_marker(skip_marker)


@pytest.fixture(scope="session")
def built_flowwalker(repo_root) -> Path:
    """
    Ensure build/bin/flowwalker exists, building it if missing. Returns
    the path to the executable.
    """
    binary = repo_root / "build" / "bin" / "flowwalker"
    if binary.exists():
        return binary

    # Run the pipeline first (dependency of the build) then cmake + make.
    r = subprocess.run(
        ["python3", "run_pipeline.py"],
        cwd=repo_root,
        capture_output=True,
        text=True,
    )
    if r.returncode != 0:
        pytest.fail(
            f"Pipeline failed during built_flowwalker fixture.\n"
            f"STDOUT:\n{r.stdout}\nSTDERR:\n{r.stderr}"
        )

    build_dir = repo_root / "build"
    build_dir.mkdir(exist_ok=True)
    r = subprocess.run(
        ["cmake", ".."],
        cwd=build_dir,
        capture_output=True,
        text=True,
    )
    if r.returncode != 0:
        pytest.fail(f"cmake failed.\nSTDOUT:\n{r.stdout}\nSTDERR:\n{r.stderr}")
    r = subprocess.run(
        ["make", "-j"],
        cwd=build_dir,
        capture_output=True,
        text=True,
    )
    if r.returncode != 0:
        pytest.fail(f"make failed.\nSTDOUT:\n{r.stdout}\nSTDERR:\n{r.stderr}")

    assert binary.exists(), f"Expected {binary} to exist after build."
    return binary
