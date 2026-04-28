// A walker that uses BOTH adjwgt and a custom edge_timestamp field.
// The compiler must allocate _MAX / _SUM for edge_timestamp based on the
// preprocess= token in the test-specific graph_fields.config.
class TestMultiField : public WalkerMeta {
 public:
  TestMultiField() {}
  LLVM_CTOR TestMultiField(gpu_graph* _graph, int _max_depth)
      : WalkerMeta(_graph, _max_depth) {}
  __device__ weight_t get_weight(Task* task, int i) {
    return graph->adjwgt[task->neighbor_offset + i]
         * graph->edge_timestamp[task->neighbor_offset + i];
  }
  __device__ void fill_dummy(Task*, Task*, int, int);
  __device__ void fill_dummy(Task*, Task*, int, int, unsigned);
  __device__ weight_t get_max_weight(Task*, int);
  __device__ weight_t get_sum_weight(Task*, int);
  __device__ bool scan_thread(Task* t) { return t->degree > 0; }
};
