"""Compiler-specific pytest fixtures."""

import difflib
from pathlib import Path
from typing import List

import pytest


# The five walkers currently shipped in include/app.cuh and exercised by
# run_templ_one.sh. These are the compiler's golden-test targets.
SHIPPED_WALKERS: List[str] = [
    "Node2vec",
    "Node2vec_weighted",
    "Metapath",
    "Metapath_weighted",
    "PPR_second",
]


# Files captured per walker. Intermediate JSONs are listed separately so
# diffs can attribute regressions to the analyzer vs. the codegen.
GENERATED_FILES = [
    "graph.cuh",
    "gpu_graph.cuh",
    "fill_dummy.cuh",
    "get_max_weight.cuh",
    "get_sum_weight.cuh",
    "walker_traits.cuh",
]

INTERMEDIATE_JSONS = [
    "walker_metadata.json",
    "llvm_analysis.json",
    "type_analysis.json",
]


@pytest.fixture(scope="session")
def golden_root(request) -> Path:
    """Root directory for committed golden snapshots."""
    return Path(request.config.rootdir) / "tests" / "compiler" / "golden"


def assert_file_matches_golden(actual_path: Path, golden_path: Path) -> None:
    """
    Assert that actual_path is byte-identical to golden_path. On mismatch,
    fail with a unified diff that identifies the exact change.
    """
    if not golden_path.exists():
        pytest.fail(
            f"Missing golden file: {golden_path}\n"
            f"Run `pytest tests/compiler/test_golden.py --update-golden` "
            f"to create it."
        )
    actual = actual_path.read_text()
    expected = golden_path.read_text()
    if actual == expected:
        return
    diff = "".join(
        difflib.unified_diff(
            expected.splitlines(keepends=True),
            actual.splitlines(keepends=True),
            fromfile=str(golden_path),
            tofile=str(actual_path),
        )
    )
    pytest.fail(
        f"Golden mismatch for {actual_path.name}:\n{diff}"
    )


def pytest_addoption(parser):
    """Register the --update-golden CLI flag for regenerating snapshots."""
    parser.addoption(
        "--update-golden",
        action="store_true",
        default=False,
        help="Overwrite committed golden snapshots with the current pipeline output.",
    )


@pytest.fixture(scope="session")
def update_golden(request) -> bool:
    """Whether this test session should update goldens instead of checking."""
    return bool(request.config.getoption("--update-golden"))
