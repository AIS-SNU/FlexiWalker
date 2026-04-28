// Synthetic walkers exercising the Phase 2 monotonicity AST walk.
// Each walker uses only adjwgt (default preprocess=max,min,sum) so no
// extra graph_fields.config entries are needed.

// Subtraction: `a - b` -> `a_MAX - b_MIN` at target=MAX.
class TestSubtraction : public WalkerMeta {
 public:
  TestSubtraction() {}
  LLVM_CTOR TestSubtraction(gpu_graph* _graph, int _max_depth)
      : WalkerMeta(_graph, _max_depth) {}
  __device__ weight_t get_weight(Task* task, int i) {
    return graph->adjwgt[task->neighbor_offset + i]
         - graph->adjwgt[task->current_vertex];
  }
  __device__ void fill_dummy(Task*, Task*, int, int);
  __device__ void fill_dummy(Task*, Task*, int, int, unsigned);
  __device__ weight_t get_max_weight(Task*, int);
  __device__ weight_t get_sum_weight(Task*, int);
  __device__ bool scan_thread(Task* t) { return t->degree > 0; }
};

// Division: `a / c` with c = constant hyperparameter.
class TestDivision : public WalkerMeta {
 public:
  float c;
  TestDivision() {}
  LLVM_CTOR TestDivision(gpu_graph* _graph, int _max_depth, float _c)
      : WalkerMeta(_graph, _max_depth), c(_c) {}
  __device__ weight_t get_weight(Task* task, int i) {
    return graph->adjwgt[task->neighbor_offset + i] / c;
  }
  __device__ void fill_dummy(Task*, Task*, int, int);
  __device__ void fill_dummy(Task*, Task*, int, int, unsigned);
  __device__ weight_t get_max_weight(Task*, int);
  __device__ weight_t get_sum_weight(Task*, int);
  __device__ bool scan_thread(Task* t) { return t->degree > 0; }
};

// Mixed: h * (a - b) — nested subtraction inside multiplication.
class TestMixed : public WalkerMeta {
 public:
  TestMixed() {}
  LLVM_CTOR TestMixed(gpu_graph* _graph, int _max_depth)
      : WalkerMeta(_graph, _max_depth) {}
  __device__ weight_t get_weight(Task* task, int i) {
    weight_t h = graph->adjwgt[task->current_vertex];
    return h * (graph->adjwgt[task->neighbor_offset + i]
              - graph->adjwgt[task->prev_vertex]);
  }
  __device__ void fill_dummy(Task*, Task*, int, int);
  __device__ void fill_dummy(Task*, Task*, int, int, unsigned);
  __device__ weight_t get_max_weight(Task*, int);
  __device__ weight_t get_sum_weight(Task*, int);
  __device__ bool scan_thread(Task* t) { return t->degree > 0; }
};
