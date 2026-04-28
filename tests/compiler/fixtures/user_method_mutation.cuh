// Synthetic walker: a user-defined helper (NOT named "update_weight")
// mutates the graph. The compiler must analyze every user-defined method
// on the walker class and force eRVS_only when any of them mutate the
// graph — not just methods with a hardcoded name. Regression test for the
// user-method generalization (audit Stage 2).
class TestUserMethodMutation : public WalkerMeta {
 public:
  TestUserMethodMutation() {}
  LLVM_CTOR TestUserMethodMutation(gpu_graph* _graph, int _max_depth)
      : WalkerMeta(_graph, _max_depth) {}

  __device__ weight_t get_weight(Task* task, int i) {
    return graph->adjwgt[task->neighbor_offset + i];
  }

  // Same mutation pattern as update_weight_mutation.cuh, but with a name
  // that does NOT match the legacy "update_weight" literal. The old logic
  // would silently skip this; the new logic must still detect it.
  __device__ void recompute_weights(Task* task, vtx_t selected_id) {
    graph->adjwgt[task->neighbor_offset + selected_id] *= 0.95;
    __threadfence();
  }

  __device__ void fill_dummy(Task* task, Task* dummy_task, int lid, int target_tid);
  __device__ void fill_dummy(Task* task, Task* dummy_task, int lid, int target_tid, unsigned mask);
  __device__ weight_t get_max_weight(Task* task, int i);
  __device__ weight_t get_sum_weight(Task* task, int i);

  __device__ bool scan_thread(Task* task) { return task->degree > 0; }
};
