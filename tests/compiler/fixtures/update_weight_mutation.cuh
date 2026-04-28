// Synthetic walker: update_weight mutates the graph, so the compiler must
// fall back to eRVS_only. Regression test for bug #3 (spec §3.3).
class TestUpdateWeightMutation : public WalkerMeta {
 public:
  TestUpdateWeightMutation() {}
  LLVM_CTOR TestUpdateWeightMutation(gpu_graph* _graph, int _max_depth)
      : WalkerMeta(_graph, _max_depth) {}

  __device__ weight_t get_weight(Task* task, int i) {
    return graph->adjwgt[task->neighbor_offset + i];
  }

  // The key line: update_weight mutates graph->adjwgt. Preprocessed
  // _MAX/_SUM aggregates would be invalidated — compiler must detect
  // this and force eRVS_only.
  __device__ void update_weight(Task* task, vtx_t selected_id) {
    graph->adjwgt[task->neighbor_offset + selected_id] *= 0.95;
    __threadfence();
  }

  __device__ void fill_dummy(Task* task, Task* dummy_task, int lid, int target_tid);
  __device__ void fill_dummy(Task* task, Task* dummy_task, int lid, int target_tid, unsigned mask);
  __device__ weight_t get_max_weight(Task* task, int i);
  __device__ weight_t get_sum_weight(Task* task, int i);

  __device__ bool scan_thread(Task* task) { return task->degree > 0; }
};
