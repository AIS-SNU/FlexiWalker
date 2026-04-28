"""
Monotonicity-rewrite integration tests.

Appends synthetic walkers that exercise subtraction, division, and mixed
compositions, runs the full pipeline, and asserts the generated
get_max_weight.cuh contains the expected _MAX / _MIN suffixes per branch.
"""

import re

import pytest

from tests.conftest import run_pipeline_clean


pytestmark = pytest.mark.compiler


@pytest.fixture(scope="module")
def pipeline_with_monotonicity_walkers(pipeline_workspace):
    repo_root = pipeline_workspace["repo_root"]
    app_cuh = repo_root / "include" / "app.cuh"
    fixture = repo_root / "tests" / "compiler" / "fixtures" / "monotonicity_test_walkers.cuh"
    original = app_cuh.read_text()
    app_cuh.write_text(
        original.rstrip()
        + "\n\n// === appended by test_monotonicity.py ===\n"
        + fixture.read_text()
    )
    try:
        r = run_pipeline_clean(repo_root)
        if r.returncode != 0:
            pytest.fail(f"pipeline failed:\n{r.stdout}\n{r.stderr}")
        yield (repo_root / "include" / "generated" / "get_max_weight.cuh").read_text()
    finally:
        app_cuh.write_text(original)


def _method_body(text: str, class_name: str) -> str:
    m = re.search(
        rf"{class_name}::get_max_weight[^{{]*\{{(.*?)\n\}}",
        text, re.DOTALL,
    )
    assert m, f"Could not locate get_max_weight body for {class_name}"
    return m.group(1)


def test_subtraction_uses_min_on_right(pipeline_with_monotonicity_walkers):
    body = _method_body(pipeline_with_monotonicity_walkers, "TestSubtraction")
    assert "adjwgt_MAX" in body, (
        f"TestSubtraction: expected adjwgt_MAX in get_max_weight body.\n{body}"
    )
    assert "adjwgt_MIN" in body, (
        f"TestSubtraction: expected adjwgt_MIN on subtraction RHS.\n{body}"
    )


def test_division_flips_denominator_sign(pipeline_with_monotonicity_walkers):
    body = _method_body(pipeline_with_monotonicity_walkers, "TestDivision")
    assert "adjwgt_MAX" in body, f"TestDivision body:\n{body}"
    assert re.search(r"/\s*c\b", body), (
        f"TestDivision: expected literal `c` in denominator.\n{body}"
    )


def test_mixed_expression_assigns_correct_roles(pipeline_with_monotonicity_walkers):
    body = _method_body(pipeline_with_monotonicity_walkers, "TestMixed")
    assert "adjwgt_MAX" in body
    assert "adjwgt_MIN" in body, f"TestMixed body:\n{body}"
