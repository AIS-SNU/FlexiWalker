"""
End-to-end test for bug #1: a walker using a newly declared per-edge field
(edge_timestamp) must get _MAX / _SUM pointers allocated automatically.
"""

import pytest

from tests.conftest import run_pipeline_clean


pytestmark = pytest.mark.compiler


@pytest.fixture
def pipeline_with_multifield(pipeline_workspace):
    repo_root = pipeline_workspace["repo_root"]
    app_cuh = repo_root / "include" / "app.cuh"
    cfg_live = repo_root / "config" / "graph_fields.config"
    walker_fixture = repo_root / "tests" / "compiler" / "fixtures" / "multifield_test_walker.cuh"
    cfg_fixture = repo_root / "tests" / "compiler" / "fixtures" / "multifield_graph_fields.config"

    original_app = app_cuh.read_text()
    original_cfg = cfg_live.read_text()

    app_cuh.write_text(
        original_app.rstrip()
        + "\n\n// === appended by test_multifield.py ===\n"
        + walker_fixture.read_text()
    )
    cfg_live.write_text(cfg_fixture.read_text())

    try:
        r = run_pipeline_clean(repo_root)
        if r.returncode != 0:
            pytest.fail(f"pipeline failed:\n{r.stdout}\n{r.stderr}")
        yield repo_root / "include" / "generated"
    finally:
        app_cuh.write_text(original_app)
        cfg_live.write_text(original_cfg)


def test_edge_timestamp_allocation_emitted(pipeline_with_multifield):
    gpu_graph = (pipeline_with_multifield / "gpu_graph.cuh").read_text()
    assert "edge_timestamp_MAX" in gpu_graph, (
        f"edge_timestamp_MAX pointer missing from gpu_graph.cuh\n{gpu_graph}"
    )
    assert "edge_timestamp_SUM" in gpu_graph, (
        f"edge_timestamp_SUM pointer missing from gpu_graph.cuh\n{gpu_graph}"
    )
    # preprocess=max,sum (no min) was declared, so _MIN should NOT appear
    # as a new pointer for edge_timestamp.
    assert "edge_timestamp_MIN" not in gpu_graph, (
        "edge_timestamp_MIN should NOT be allocated — config declared only max,sum"
    )


def test_get_max_weight_uses_edge_timestamp_max(pipeline_with_multifield):
    gm = (pipeline_with_multifield / "get_max_weight.cuh").read_text()
    assert "edge_timestamp_MAX" in gm, (
        f"get_max_weight.cuh must reference edge_timestamp_MAX.\n{gm}"
    )
