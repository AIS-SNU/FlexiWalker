// Synthetic walker exercising a user-defined method with a const-reference
// parameter. The dummy stub synthesizes a call to every user method, so
// the metadata extractor must produce a well-formed argument for `const T&`
// — passing a literal "0" (the prior fallback) yields a malformed call and
// fails to compile. This regression pins that the dummy compiles cleanly.
class TestUserMethodConstRef : public WalkerMeta {
 public:
  TestUserMethodConstRef() {}
  LLVM_CTOR TestUserMethodConstRef(gpu_graph* _graph, int _max_depth)
      : WalkerMeta(_graph, _max_depth) {}

  __device__ weight_t get_weight(Task* task, int i) {
    return graph->adjwgt[task->neighbor_offset + i];
  }

  // const-ref to Task: must resolve to *task_ptr_<walker> in the dummy.
  // const-ref to int: must accept a literal (binds to const&).
  __device__ bool ref_helper(const Task& task, const int& depth) const {
    return task.length > depth;
  }

  __device__ void fill_dummy(Task* task, Task* dummy_task, int lid, int target_tid);
  __device__ void fill_dummy(Task* task, Task* dummy_task, int lid, int target_tid, unsigned mask);
  __device__ weight_t get_max_weight(Task* task, int i);
  __device__ weight_t get_sum_weight(Task* task, int i);

  __device__ bool scan_thread(Task* task) { return task->degree > 0; }
};
