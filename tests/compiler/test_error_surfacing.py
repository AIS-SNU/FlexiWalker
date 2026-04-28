"""
Bug #5 regression: the code generator must raise PipelineError with an
actionable message when upstream JSON is missing, malformed, or when a
walker declared in walker_metadata has no corresponding analysis entry.
"""

import json
import shutil
from pathlib import Path

import pytest

from pipeline.base import PipelineError
from tests.conftest import run_pipeline_clean


pytestmark = pytest.mark.compiler


# Artifacts required for the staged copy; if any are missing we (re)run the
# pipeline once per test module rather than asserting partway through a test.
_REQUIRED_ARTIFACTS = (
    "walker_metadata.json",
    "type_analysis.json",
    "llvm_analysis.json",
    "graph_fields.json",
)


@pytest.fixture(scope="module")
def _populated_artifacts_dir(repo_root):
    """
    Ensure repo_root/artifacts/ has the files the error-surfacing tests need.
    Runs the pipeline once per module only if something is missing. Cost on
    a populated repo is zero; on a fresh checkout it's one pipeline run.
    """
    src = repo_root / "artifacts"
    missing = [f for f in _REQUIRED_ARTIFACTS if not (src / f).exists()]
    if missing:
        r = run_pipeline_clean(repo_root)
        if r.returncode != 0:
            pytest.fail(
                "Pipeline bootstrap for staged_artifacts failed:\n"
                + (r.stdout or "") + (r.stderr or "")
            )
        still_missing = [f for f in _REQUIRED_ARTIFACTS if not (src / f).exists()]
        if still_missing:
            pytest.fail(
                "Pipeline ran but artifacts still missing: " + ", ".join(still_missing)
            )
    return src


@pytest.fixture
def staged_artifacts(tmp_path, _populated_artifacts_dir):
    """
    Copy the current artifacts/ into a tmpdir so tests can mutate them
    without affecting the repo state. Depends on _populated_artifacts_dir
    so a fresh checkout auto-bootstraps.
    """
    dest = tmp_path / "artifacts"
    shutil.copytree(_populated_artifacts_dir, dest)
    return dest


def _invoke_code_generator_with_artifacts(artifacts_dir, tmp_path):
    """
    Import and invoke CodeGenerator against a staged artifacts dir.

    The code generator's validation runs during execute(), before any
    Jinja rendering. That's what we're exercising here.
    """
    from pipeline.stages.code_generator import CodeGenerator

    class FakeConfig:
        def __init__(self):
            self.root_dir = artifacts_dir.parent
            self.artifacts_dir = artifacts_dir
            self.generated_dir = tmp_path / "generated"
            self.generated_dir.mkdir(exist_ok=True)

        def get_artifact_path(self, name):
            return self.artifacts_dir / {
                "walker_metadata": "walker_metadata.json",
                "llvm_analysis": "llvm_analysis.json",
                "type_analysis": "type_analysis.json",
                "graph_fields": "graph_fields.json",
            }[name]

        def get_generated_path(self, name):
            return self.generated_dir / {
                "graph": "graph.cuh",
                "gpu_graph": "gpu_graph.cuh",
                "fill_dummy": "fill_dummy.cuh",
                "get_max_weight": "get_max_weight.cuh",
                "get_sum_weight": "get_sum_weight.cuh",
                "walker_traits": "walker_traits.cuh",
            }[name]

        def ensure_generated_dir(self):
            self.generated_dir.mkdir(exist_ok=True)

    stage = CodeGenerator(FakeConfig())
    stage.execute()


def test_missing_type_analysis_entry_raises(staged_artifacts, tmp_path):
    """Remove one walker from type_analysis.json; expect PipelineError."""
    type_path = staged_artifacts / "type_analysis.json"
    data = json.loads(type_path.read_text())
    meta = json.loads((staged_artifacts / "walker_metadata.json").read_text())
    walker_names = [k for k in meta.keys() if k != "headers"]
    assert walker_names, "walker_metadata.json unexpectedly empty"
    victim = walker_names[0]
    if victim in data:
        del data[victim]
    type_path.write_text(json.dumps(data, indent=2))

    with pytest.raises(PipelineError, match=f"no type-analysis data for '{victim}'"):
        _invoke_code_generator_with_artifacts(staged_artifacts, tmp_path)


def test_empty_branches_raises(staged_artifacts, tmp_path):
    """Force an empty branches list; expect PipelineError."""
    type_path = staged_artifacts / "type_analysis.json"
    data = json.loads(type_path.read_text())
    meta = json.loads((staged_artifacts / "walker_metadata.json").read_text())
    victim = next(iter(k for k in meta.keys() if k != "headers"))
    if victim in data:
        data[victim]["branches"] = []
    type_path.write_text(json.dumps(data, indent=2))

    with pytest.raises(PipelineError, match=f"no return branches for '{victim}'"):
        _invoke_code_generator_with_artifacts(staged_artifacts, tmp_path)


def test_malformed_json_raises(staged_artifacts, tmp_path):
    """Corrupt one JSON; expect PipelineError about parse failure."""
    type_path = staged_artifacts / "type_analysis.json"
    type_path.write_text("this is not valid json")
    with pytest.raises(PipelineError, match="failed to parse"):
        _invoke_code_generator_with_artifacts(staged_artifacts, tmp_path)


def test_missing_role_in_graph_fields_raises(repo_root):
    """
    A walker whose get_weight uses a field with role MIN must not compile
    if that field is declared with preprocess=max only — pipeline should
    raise PipelineError with a clear remediation message.
    """
    app_cuh = repo_root / "include" / "app.cuh"
    cfg = repo_root / "config" / "graph_fields.config"
    original_app = app_cuh.read_text()
    original_cfg = cfg.read_text()
    cfg.write_text(
        original_cfg
        + "\nedge_timestamp[TestMissingRole] = file:weight_t:edge:preprocess=max\n"
    )
    walker = """
class TestMissingRole : public WalkerMeta {
 public:
  TestMissingRole() {}
  LLVM_CTOR TestMissingRole(gpu_graph* _graph, int _max_depth)
      : WalkerMeta(_graph, _max_depth) {}
  __device__ weight_t get_weight(Task* task, int i) {
    return graph->adjwgt[task->neighbor_offset + i]
         - graph->edge_timestamp[task->neighbor_offset + i];
  }
  __device__ void fill_dummy(Task*, Task*, int, int);
  __device__ void fill_dummy(Task*, Task*, int, int, unsigned);
  __device__ weight_t get_max_weight(Task*, int);
  __device__ weight_t get_sum_weight(Task*, int);
  __device__ bool scan_thread(Task* t) { return t->degree > 0; }
};
"""
    app_cuh.write_text(
        original_app.rstrip()
        + "\n// === appended by test_missing_role ===\n"
        + walker
    )
    try:
        from tests.conftest import run_pipeline_clean
        r = run_pipeline_clean(repo_root)
        assert r.returncode != 0, (
            "Pipeline should have failed but exited 0.\nSTDOUT:\n" + r.stdout
        )
        combined = (r.stdout + r.stderr).lower()
        assert "edge_timestamp" in combined, (
            "PipelineError message must name the offending field.\n" + combined
        )
        assert "min" in combined, (
            "Error must mention the missing role (MIN).\n" + combined
        )
    finally:
        app_cuh.write_text(original_app)
        cfg.write_text(original_cfg)


def test_missing_sum_in_graph_fields_raises(repo_root):
    """
    H1 regression: get_sum_weight rewrites every MAX/MIN reference to _SUM,
    so any preprocessed edge field must also declare preprocess=sum. A field
    with preprocess=max,min but no sum must fail with an actionable error.
    """
    app_cuh = repo_root / "include" / "app.cuh"
    cfg = repo_root / "config" / "graph_fields.config"
    original_app = app_cuh.read_text()
    original_cfg = cfg.read_text()
    cfg.write_text(
        original_cfg
        + "\nedge_timestamp[TestMissingSum] = file:weight_t:edge:preprocess=max,min\n"
    )
    walker = """
class TestMissingSum : public WalkerMeta {
 public:
  TestMissingSum() {}
  LLVM_CTOR TestMissingSum(gpu_graph* _graph, int _max_depth)
      : WalkerMeta(_graph, _max_depth) {}
  __device__ weight_t get_weight(Task* task, int i) {
    return graph->adjwgt[task->neighbor_offset + i]
         - graph->edge_timestamp[task->neighbor_offset + i];
  }
  __device__ void fill_dummy(Task*, Task*, int, int);
  __device__ void fill_dummy(Task*, Task*, int, int, unsigned);
  __device__ weight_t get_max_weight(Task*, int);
  __device__ weight_t get_sum_weight(Task*, int);
  __device__ bool scan_thread(Task* t) { return t->degree > 0; }
};
"""
    app_cuh.write_text(
        original_app.rstrip()
        + "\n// === appended by test_missing_sum ===\n"
        + walker
    )
    try:
        from tests.conftest import run_pipeline_clean
        r = run_pipeline_clean(repo_root)
        assert r.returncode != 0, (
            "Pipeline should have failed but exited 0.\nSTDOUT:\n" + r.stdout
        )
        combined = (r.stdout + r.stderr).lower()
        assert "edge_timestamp" in combined, (
            "PipelineError must name the offending field.\n" + combined
        )
        assert "sum" in combined, (
            "PipelineError must mention the missing 'sum' preprocess op.\n"
            + combined
        )
    finally:
        app_cuh.write_text(original_app)
        cfg.write_text(original_cfg)
