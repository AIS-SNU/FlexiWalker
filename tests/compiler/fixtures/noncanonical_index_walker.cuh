// H2 regression: a walker whose get_weight subscripts a preprocessed edge
// field with a non-canonical index. The downstream textual rewrite in
// ReturnVisitor only understands `task->neighbor_offset [+ loop local]` and
// `task->prev_neighbor_offset [+ loop local]`; anything else (here: a
// constant-literal offset) would produce the wrong vertex ID after
// rewriting. The compiler must detect this via isCanonicalIndex and force
// eRVS_only rather than emit wrong code.
class TestNonCanonicalIndex : public WalkerMeta {
 public:
  TestNonCanonicalIndex() {}
  LLVM_CTOR TestNonCanonicalIndex(gpu_graph* _graph, int _max_depth)
      : WalkerMeta(_graph, _max_depth) {}
  __device__ weight_t get_weight(Task* task, int i) {
    // Index is `task->neighbor_offset + 3 * i` — not a canonical form the
    // textual rewrite can handle. Without the H2 guard, this would emit
    // adjwgt_MAX[task->current_vertex + 3 * i] after rewrite, which points
    // at the wrong memory.
    return graph->adjwgt[task->neighbor_offset + 3 * i];
  }
  __device__ void fill_dummy(Task*, Task*, int, int);
  __device__ void fill_dummy(Task*, Task*, int, int, unsigned);
  __device__ weight_t get_max_weight(Task*, int);
  __device__ weight_t get_sum_weight(Task*, int);
  __device__ bool scan_thread(Task* t) { return t->degree > 0; }
};
