"""
Regression tests for the eRVS_only (reservoir-only) fallback mechanism.

The 10 synthetic walkers in tests/compiler/fixtures/ervs_test_walkers.cuh
each exercise a specific condition the compiler's complexity analyzer is
expected to detect. These tests append the fixture to include/app.cuh,
run the pipeline, and assert the ERVS_ONLY constant in walker_traits.cuh
matches the expected value for each walker.

The 10 test cases and their expected ERVS_ONLY outcomes are codified in
the ERVS_CASES table below.
"""

import re
from pathlib import Path

import pytest

from tests.conftest import run_pipeline_clean


pytestmark = pytest.mark.compiler


# (class_name, expected_ervs_only) pairs. See §5.1.3 of the design spec.
ERVS_CASES: list[tuple[str, int]] = [
    ("TestRecursive", 1),        # recursive function call
    ("TestDeepNesting", 1),      # >= 5 levels of nested if
    ("TestComplexLoop", 1),      # data-dependent loop exit
    ("TestGraphModify", 1),      # graph mutation in get_weight
    ("TestSimple", 0),           # trivial — baseline
    ("TestNestedLoops", 0),      # 2-deep nested loops with fixed bounds — under MAX_NESTING_DEPTH=5, no fallback
    ("TestIndirectRecursion", 1),# mutual recursion across helpers
    ("TestFlatIfElse", 0),       # depth-1 if/else chain is fine
    ("TestSimpleLoop", 0),       # loop with fixed bounds is fine
    ("TestWarpSync", 0),         # __shfl_sync warns but does not fall back
    ("TestRoleConflict", 1),     # local used at both MAX and MIN — string-level suffix rewrite can't disambiguate
    ("TestPointerAlias", 1),     # bare graph->adjwgt outside subscript — can't tag suffix on aliased pointer
    ("TestHelperRhsFlip", 1),    # helper RHS contains BO_Sub — single statement can only carry one suffix
]


# Generic parser: extract every {class: {trait: value}} from walker_traits.cuh.
# Not hardcoded to ERVS_ONLY so it survives future trait additions (POSSIBLE_ZERO,
# UPDATE_FLAG, and any new flags added in Phase 2+).
_STRUCT_RE = re.compile(
    r"struct\s+WalkerTraits<(\w+)>\s*\{([^}]*)\}",
    re.DOTALL,
)
_TRAIT_RE = re.compile(
    r"static\s+constexpr\s+\w+\s+(\w+)\s*=\s*(-?\d+)"
)


def _parse_walker_traits(walker_traits_path: Path) -> dict[str, dict[str, int]]:
    """Parse walker_traits.cuh into {class_name: {trait_name: int_value}}."""
    text = walker_traits_path.read_text()
    result: dict[str, dict[str, int]] = {}
    for class_m in _STRUCT_RE.finditer(text):
        class_name, body = class_m.group(1), class_m.group(2)
        result[class_name] = {
            t.group(1): int(t.group(2)) for t in _TRAIT_RE.finditer(body)
        }
    return result


@pytest.fixture(scope="module")
def pipeline_with_ervs_walkers(pipeline_workspace):
    """
    Append the ervs_test_walkers.cuh fixture to include/app.cuh, run the
    full pipeline, and yield {traits, gpu_graph_text}. The
    pipeline_workspace fixture (from tests/conftest.py) restores
    include/generated and artifacts on teardown; we restore app.cuh
    ourselves.
    """
    repo_root = pipeline_workspace["repo_root"]
    app_cuh = repo_root / "include" / "app.cuh"
    fixture = repo_root / "tests" / "compiler" / "fixtures" / "ervs_test_walkers.cuh"

    original = app_cuh.read_text()
    fixture_text = fixture.read_text()

    new_app = (
        original.rstrip()
        + "\n\n// === appended by test_ervs_fallback.py ===\n"
        + fixture_text
    )
    app_cuh.write_text(new_app)

    try:
        result = run_pipeline_clean(repo_root)
        if result.returncode != 0:
            pytest.fail(
                f"Pipeline failed (exit {result.returncode}).\n"
                f"STDOUT:\n{result.stdout}\n"
                f"STDERR:\n{result.stderr}"
            )
        traits_path = repo_root / "include" / "generated" / "walker_traits.cuh"
        gpu_graph_path = repo_root / "include" / "generated" / "gpu_graph.cuh"
        yield {
            "traits": _parse_walker_traits(traits_path),
            "gpu_graph_text": gpu_graph_path.read_text(),
        }
    finally:
        app_cuh.write_text(original)


@pytest.mark.parametrize("class_name,expected", ERVS_CASES)
def test_ervs_only_flag(class_name, expected, pipeline_with_ervs_walkers):
    traits_map = pipeline_with_ervs_walkers["traits"]
    assert class_name in traits_map, (
        f"{class_name} was not found in generated walker_traits.cuh. "
        f"Available: {sorted(traits_map.keys())}"
    )
    traits = traits_map[class_name]
    assert "ERVS_ONLY" in traits, (
        f"{class_name} has no ERVS_ONLY trait. "
        f"Available traits: {sorted(traits.keys())}"
    )
    actual = traits["ERVS_ONLY"]
    assert actual == expected, (
        f"{class_name}: expected ERVS_ONLY={expected}, got {actual}. "
        f"If the compiler's behavior changed legitimately, update ERVS_CASES."
    )


def test_adjwgt_min_absent_when_no_min_role(pipeline_with_ervs_walkers):
    """
    Regression for Fix #3: gpu_graph.cuh must not allocate `adjwgt_MIN`
    when no walker reads it. The production walkers and every fixture
    walker in ervs_test_walkers.cuh use only `adjwgt_MAX` / `adjwgt_SUM`
    (the eRVS-forced cases never call get_max_weight at runtime, so any
    rewritten body suffix is irrelevant — but the body still gets
    emitted, so this test also exercises the role-propagation path).
    """
    text = pipeline_with_ervs_walkers["gpu_graph_text"]
    assert "adjwgt_MAX" in text, (
        "gpu_graph.cuh missing adjwgt_MAX; production walkers should require it."
    )
    assert "adjwgt_MIN" not in text, (
        "gpu_graph.cuh declares adjwgt_MIN even though no walker reads it. "
        "This is the regression Fix #3 addressed — see "
        "pipeline.stages.code_generator._compute_implicit_adjwgt_ops."
    )
