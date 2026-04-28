"""
Bug #3 regression: graph mutation in update_weight must set ERVS_ONLY=1.

The current LLVM analyzer only checks for graph modifications in
get_weight / is_stop / Task::update. Phase 2 adds the same check for
update_weight.
"""

from pathlib import Path

import pytest

from tests.conftest import run_pipeline_clean
from tests.compiler.test_ervs_fallback import _parse_walker_traits


pytestmark = pytest.mark.compiler


@pytest.fixture(scope="module")
def pipeline_with_update_weight_mutation(pipeline_workspace):
    repo_root = pipeline_workspace["repo_root"]
    app_cuh = repo_root / "include" / "app.cuh"
    fixture = repo_root / "tests" / "compiler" / "fixtures" / "update_weight_mutation.cuh"

    original = app_cuh.read_text()
    fixture_text = fixture.read_text()
    app_cuh.write_text(
        original.rstrip()
        + "\n\n// === appended by test_update_weight_mutation.py ===\n"
        + fixture_text
    )

    try:
        result = run_pipeline_clean(repo_root)
        if result.returncode != 0:
            pytest.fail(
                f"Pipeline failed (exit {result.returncode}).\n"
                f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
            )
        yield _parse_walker_traits(
            repo_root / "include" / "generated" / "walker_traits.cuh"
        )
    finally:
        app_cuh.write_text(original)


def test_update_weight_mutation_triggers_ervs_only(pipeline_with_update_weight_mutation):
    traits = pipeline_with_update_weight_mutation
    assert "TestUpdateWeightMutation" in traits, (
        f"walker missing from trait map. Available: {sorted(traits.keys())}"
    )
    assert traits["TestUpdateWeightMutation"].get("ERVS_ONLY") == 1, (
        "update_weight mutates graph->adjwgt — compiler must set ERVS_ONLY=1. "
        f"Got traits = {traits['TestUpdateWeightMutation']}"
    )
