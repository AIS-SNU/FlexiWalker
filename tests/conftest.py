"""Shared fixtures for FlexiWalker test suite."""

import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import pytest


def _find_repo_root() -> Path:
    """Walk upward from this file to find the repo root (contains run_pipeline.py)."""
    here = Path(__file__).resolve()
    for ancestor in [here, *here.parents]:
        if (ancestor / "run_pipeline.py").is_file():
            return ancestor
    raise RuntimeError("Could not locate repo root containing run_pipeline.py")


@pytest.fixture(scope="session")
def repo_root() -> Path:
    """Absolute path to the repository root."""
    return _find_repo_root()


@pytest.fixture(scope="module")
def pipeline_workspace(tmp_path_factory, repo_root):
    """
    Create an isolated workspace that mirrors the repo structure enough to
    run the pipeline without touching the real include/generated/ directory.

    Strategy: we cannot trivially relocate the whole project (CMake, tools,
    config files all assume known paths), so instead we back up the live
    include/generated/ and artifacts/ directories, run the pipeline, and
    restore the backup on teardown. This keeps the developer's live
    include/generated/ intact while tests run.

    Module-scoped so an expensive pipeline run can be shared across all
    parametrized tests in a module. Uses tmp_path_factory (session-scoped)
    for its backup storage since function-scoped tmp_path is incompatible.

    Yields:
        A dict with keys:
          - 'repo_root': Path
          - 'generated_dir': Path (include/generated, post-run contents)
          - 'artifacts_dir': Path (artifacts, post-run contents)
    """
    generated_dir = repo_root / "include" / "generated"
    artifacts_dir = repo_root / "artifacts"

    # Snapshot existing state to a tmpdir so we can restore on teardown.
    backup_root = tmp_path_factory.mktemp("pipeline_workspace_backup")
    gen_backup = backup_root / "generated"
    art_backup = backup_root / "artifacts"
    if generated_dir.exists():
        shutil.copytree(generated_dir, gen_backup)
    if artifacts_dir.exists():
        shutil.copytree(artifacts_dir, art_backup)

    try:
        yield {
            "repo_root": repo_root,
            "generated_dir": generated_dir,
            "artifacts_dir": artifacts_dir,
        }
    finally:
        # Restore the original state.
        if generated_dir.exists():
            shutil.rmtree(generated_dir)
        if gen_backup.exists():
            shutil.copytree(gen_backup, generated_dir)
        if artifacts_dir.exists():
            shutil.rmtree(artifacts_dir)
        if art_backup.exists():
            shutil.copytree(art_backup, artifacts_dir)


def run_pipeline_clean(repo_root: Path) -> subprocess.CompletedProcess:
    """Run the full pipeline with --clean from the repo root."""
    return subprocess.run(
        ["python3", "run_pipeline.py", "--clean"],
        cwd=repo_root,
        capture_output=True,
        text=True,
    )
