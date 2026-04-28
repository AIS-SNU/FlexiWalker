"""
Golden-file tests for the FlexiWalker compiler.

For each of the 5 shipped walkers, runs the full pipeline in an isolated
workspace and asserts that every generated .cuh file and every intermediate
JSON (filtered to the walker) matches a committed snapshot.

A mismatch indicates either (a) an unintended regression in the compiler,
or (b) a legitimate output change that should be captured by regenerating
with `--update-golden`.
"""

import json
import shutil
from pathlib import Path

import pytest

from .conftest import (
    SHIPPED_WALKERS,
    GENERATED_FILES,
    INTERMEDIATE_JSONS,
    assert_file_matches_golden,
)

# Import via the tests package so conftest.run_pipeline_clean is available.
from tests.conftest import run_pipeline_clean


pytestmark = pytest.mark.compiler


def _filter_json_by_walker(src_json: Path, walker: str, dest_json: Path) -> None:
    """
    Write a walker-scoped slice of src_json to dest_json.

    The upstream JSONs are keyed by walker class name at the top level (or
    contain per-walker substructures). We keep only the entry for the given
    walker so each walker's golden is self-contained and immune to changes
    in unrelated walkers.
    """
    data = json.loads(src_json.read_text())
    if isinstance(data, dict) and walker in data:
        out = {walker: data[walker]}
    else:
        # Some JSONs (e.g., graph_fields.json) are globally structured; emit
        # as-is so the golden still captures their full content.
        out = data
    dest_json.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")


def _collect_outputs_for_walker(
    repo_root: Path,
    walker: str,
    dest_dir: Path,
) -> list[tuple[Path, Path]]:
    """
    Copy the generated .cuh files and walker-filtered intermediate JSONs
    for `walker` into `dest_dir`. Returns a list of (source, dest) tuples
    so callers can diff or commit them.
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    pairs: list[tuple[Path, Path]] = []

    generated = repo_root / "include" / "generated"
    for name in GENERATED_FILES:
        src = generated / name
        dst = dest_dir / name
        if src.exists():
            shutil.copyfile(src, dst)
            pairs.append((src, dst))

    artifacts = repo_root / "artifacts"
    for name in INTERMEDIATE_JSONS:
        src = artifacts / name
        if not src.exists():
            continue
        dst = dest_dir / name
        _filter_json_by_walker(src, walker, dst)
        pairs.append((src, dst))

    return pairs


@pytest.fixture(scope="module")
def pipeline_output(pipeline_workspace):
    """
    Run the pipeline once per test module and cache its outputs so the
    parametrized tests below don't each re-run the (slow) pipeline.
    """
    repo_root = pipeline_workspace["repo_root"]
    result = run_pipeline_clean(repo_root)
    if result.returncode != 0:
        pytest.fail(
            f"Pipeline failed (exit {result.returncode}).\n"
            f"STDOUT:\n{result.stdout}\n"
            f"STDERR:\n{result.stderr}"
        )
    cache = repo_root / "tests" / "compiler" / ".last_run_output"
    if cache.exists():
        shutil.rmtree(cache)
    cache.mkdir(parents=True)
    per_walker: dict[str, Path] = {}
    for walker in SHIPPED_WALKERS:
        per_walker[walker] = cache / walker
        _collect_outputs_for_walker(repo_root, walker, per_walker[walker])
    return per_walker


@pytest.mark.parametrize("walker", SHIPPED_WALKERS)
@pytest.mark.parametrize("filename", GENERATED_FILES + INTERMEDIATE_JSONS)
def test_generated_file_matches_golden(
    walker,
    filename,
    pipeline_output,
    golden_root,
    update_golden,
):
    actual = pipeline_output[walker] / filename
    if not actual.exists():
        pytest.skip(f"Pipeline did not produce {filename} for {walker}")
    golden_path = golden_root / walker / filename
    if update_golden:
        golden_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(actual, golden_path)
        return
    assert_file_matches_golden(actual, golden_path)
