"""
Tests the monotonicity analyzer's complexity budget:
  - MAX_ANALYSIS_DEPTH = 6: deeper expressions fall back to eRVS_only.
  - Unknown function in return: same fallback.
"""

import pytest

from tests.conftest import run_pipeline_clean
from tests.compiler.test_ervs_fallback import _parse_walker_traits


pytestmark = pytest.mark.compiler


# Depth-heavy walker with 7 nested arithmetic levels (over the 6 budget).
OVERBUDGET_WALKER = """
class TestOverBudgetDepth : public WalkerMeta {
 public:
  TestOverBudgetDepth() {}
  LLVM_CTOR TestOverBudgetDepth(gpu_graph* _graph, int _max_depth)
      : WalkerMeta(_graph, _max_depth) {}
  __device__ weight_t get_weight(Task* task, int i) {
    weight_t a = graph->adjwgt[task->neighbor_offset + i];
    // 7-level nested arithmetic: ((((((a + 1) - 2) * 3) / 4) + 5) - 6) * 7
    return ((((((a + 1.0f) - 2.0f) * 3.0f) / 4.0f) + 5.0f) - 6.0f) * 7.0f;
  }
  __device__ void fill_dummy(Task*, Task*, int, int);
  __device__ void fill_dummy(Task*, Task*, int, int, unsigned);
  __device__ weight_t get_max_weight(Task*, int);
  __device__ weight_t get_sum_weight(Task*, int);
  __device__ bool scan_thread(Task* t) { return t->degree > 0; }
};
"""

# Walker with an unknown function call in return.
UNKNOWN_FUNC_WALKER = """
__device__ weight_t my_mystery_fn(weight_t x) { return x * 2.0f; }

class TestUnknownFunction : public WalkerMeta {
 public:
  TestUnknownFunction() {}
  LLVM_CTOR TestUnknownFunction(gpu_graph* _graph, int _max_depth)
      : WalkerMeta(_graph, _max_depth) {}
  __device__ weight_t get_weight(Task* task, int i) {
    return my_mystery_fn(graph->adjwgt[task->neighbor_offset + i]);
  }
  __device__ void fill_dummy(Task*, Task*, int, int);
  __device__ void fill_dummy(Task*, Task*, int, int, unsigned);
  __device__ weight_t get_max_weight(Task*, int);
  __device__ weight_t get_sum_weight(Task*, int);
  __device__ bool scan_thread(Task* t) { return t->degree > 0; }
};
"""


@pytest.fixture
def pipeline_with(pipeline_workspace):
    """Factory fixture: append a walker snippet and run the pipeline."""
    repo_root = pipeline_workspace["repo_root"]
    app_cuh = repo_root / "include" / "app.cuh"
    original = app_cuh.read_text()

    def _run(snippet: str):
        app_cuh.write_text(
            original.rstrip()
            + "\n\n// === appended by test_budget.py ===\n"
            + snippet
        )
        r = run_pipeline_clean(repo_root)
        if r.returncode != 0:
            pytest.fail(f"pipeline failed:\n{r.stdout}\n{r.stderr}")
        return _parse_walker_traits(
            repo_root / "include" / "generated" / "walker_traits.cuh"
        )

    try:
        yield _run
    finally:
        app_cuh.write_text(original)


def test_depth_over_budget_falls_back(pipeline_with):
    traits = pipeline_with(OVERBUDGET_WALKER)
    assert traits.get("TestOverBudgetDepth", {}).get("ERVS_ONLY") == 1


def test_unknown_function_falls_back(pipeline_with):
    traits = pipeline_with(UNKNOWN_FUNC_WALKER)
    assert traits.get("TestUnknownFunction", {}).get("ERVS_ONLY") == 1
