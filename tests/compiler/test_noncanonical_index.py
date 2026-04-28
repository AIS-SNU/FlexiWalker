"""
H2 regression: a walker whose get_weight uses a non-canonical index into a
preprocessed edge field must fall back to eRVS_only with a clear reason,
rather than silently emit wrong code via the textual rewrite in
ReturnVisitor::applyAdditionalReplacements.
"""

import json
import pytest

from tests.conftest import run_pipeline_clean
from tests.compiler.test_ervs_fallback import _parse_walker_traits


pytestmark = pytest.mark.compiler


@pytest.fixture(scope="module")
def pipeline_with_noncanonical_walker(pipeline_workspace):
    repo_root = pipeline_workspace["repo_root"]
    app_cuh = repo_root / "include" / "app.cuh"
    fixture = (
        repo_root / "tests" / "compiler" / "fixtures"
        / "noncanonical_index_walker.cuh"
    )
    original = app_cuh.read_text()
    app_cuh.write_text(
        original.rstrip()
        + "\n\n// === appended by test_noncanonical_index.py ===\n"
        + fixture.read_text()
    )
    try:
        r = run_pipeline_clean(repo_root)
        if r.returncode != 0:
            pytest.fail(f"pipeline failed:\n{r.stdout}\n{r.stderr}")
        yield {
            "repo_root": repo_root,
            "traits": _parse_walker_traits(
                repo_root / "include" / "generated" / "walker_traits.cuh"
            ),
            "type_analysis": json.loads(
                (repo_root / "artifacts" / "type_analysis.json").read_text()
            ),
        }
    finally:
        app_cuh.write_text(original)


def test_noncanonical_index_forces_ervs_only(pipeline_with_noncanonical_walker):
    traits = pipeline_with_noncanonical_walker["traits"]
    entry = traits.get("TestNonCanonicalIndex", {})
    assert entry.get("ERVS_ONLY") == 1, (
        "Non-canonical index must force eRVS_only; got "
        f"{entry}"
    )


def test_noncanonical_index_fallback_reason_mentions_field(
    pipeline_with_noncanonical_walker,
):
    type_analysis = pipeline_with_noncanonical_walker["type_analysis"]
    entry = type_analysis.get("TestNonCanonicalIndex", {})
    branches = entry.get("branches") or []
    assert branches, "no branches captured for TestNonCanonicalIndex"
    reasons = [b.get("fallback_reason", "") for b in branches]
    joined = " | ".join(reasons).lower()
    assert "non-canonical index" in joined, (
        f"fallback_reason should name the failure mode; got {reasons!r}"
    )
    assert "adjwgt" in joined, (
        f"fallback_reason should name the offending field; got {reasons!r}"
    )
