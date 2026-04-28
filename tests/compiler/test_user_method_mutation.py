"""
User-method generalization regression: graph mutation in any user-defined
method (not just one literally named "update_weight") must set ERVS_ONLY=1.

The metadata extractor records all user-defined methods on the walker
class; walker_dummy.cu calls each so its body is present in the LLVM IR;
the analyzer's bug #3 graph-mutation check then runs over every walker
method. This test pins that contract so a renamed mutating helper can't
silently slip through.
"""

from pathlib import Path

import pytest

from tests.conftest import run_pipeline_clean
from tests.compiler.test_ervs_fallback import _parse_walker_traits


pytestmark = pytest.mark.compiler


@pytest.fixture(scope="module")
def pipeline_with_user_method_mutation(pipeline_workspace):
    repo_root = pipeline_workspace["repo_root"]
    app_cuh = repo_root / "include" / "app.cuh"
    fixture = repo_root / "tests" / "compiler" / "fixtures" / "user_method_mutation.cuh"

    original = app_cuh.read_text()
    fixture_text = fixture.read_text()
    app_cuh.write_text(
        original.rstrip()
        + "\n\n// === appended by test_user_method_mutation.py ===\n"
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


def test_user_method_mutation_triggers_ervs_only(pipeline_with_user_method_mutation):
    traits = pipeline_with_user_method_mutation
    assert "TestUserMethodMutation" in traits, (
        f"walker missing from trait map. Available: {sorted(traits.keys())}"
    )
    assert traits["TestUserMethodMutation"].get("ERVS_ONLY") == 1, (
        "recompute_weights mutates graph->adjwgt — compiler must set "
        "ERVS_ONLY=1 for any user-defined method, not only update_weight. "
        f"Got traits = {traits['TestUserMethodMutation']}"
    )
