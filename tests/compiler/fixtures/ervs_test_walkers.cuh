// ============================================================================
// Test cases for ERVS_ONLY detection
// ============================================================================

// Test 1: Recursive function call (should set ERVS_ONLY = 1)
class TestRecursive : public WalkerMeta {
 public:
  TestRecursive() {}
  LLVM_CTOR TestRecursive(gpu_graph* _graph, int _max_depth)
      : WalkerMeta(_graph, _max_depth) {}

  __device__ weight_t compute_recursive(int depth) {
    if (depth <= 0) return 1.0;
    return compute_recursive(depth - 1) * 2.0;  // Recursive call
  }

  __device__ weight_t get_weight(Task* task, int i) {
    return compute_recursive(5);  // Calls recursive function
  }

  __device__ void fill_dummy(Task* task, Task* dummy_task, int lid, int target_tid);
  __device__ void fill_dummy(Task* task, Task* dummy_task, int lid, int target_tid, unsigned mask);
  __device__ weight_t get_max_weight(Task* task, int i);
  __device__ weight_t get_sum_weight(Task* task, int i);

  __device__ bool scan_thread(Task* task) {
    return task->degree > 0;
  }
};

// Test 2: Deep nesting (should set ERVS_ONLY = 1)
class TestDeepNesting : public WalkerMeta {
 public:
  TestDeepNesting() {}
  LLVM_CTOR TestDeepNesting(gpu_graph* _graph, int _max_depth)
      : WalkerMeta(_graph, _max_depth) {}

  __device__ weight_t get_weight(Task* task, int i) {
    int value = 0;
    if (task->current_vertex > 0) {
      if (task->prev_vertex > 0) {
        if (task->length > 2) {
          if (graph->adjncy[task->neighbor_offset + i] > 100) {
            if (task->degree > 5) {
              if (i % 2 == 0) {
                value = 10;  // 6 levels deep
              } else {
                value = 5;
              }
            }
          }
        }
      }
    }
    return (weight_t)value;
  }

  __device__ void fill_dummy(Task* task, Task* dummy_task, int lid, int target_tid);
  __device__ void fill_dummy(Task* task, Task* dummy_task, int lid, int target_tid, unsigned mask);
  __device__ weight_t get_max_weight(Task* task, int i);
  __device__ weight_t get_sum_weight(Task* task, int i);

  __device__ bool scan_thread(Task* task) {
    return task->degree > 0;
  }
};

// Test 3: Loop with data-dependent exit (should set ERVS_ONLY = 1)
class TestComplexLoop : public WalkerMeta {
 public:
  TestComplexLoop() {}
  LLVM_CTOR TestComplexLoop(gpu_graph* _graph, int _max_depth)
      : WalkerMeta(_graph, _max_depth) {}

  __device__ weight_t get_weight(Task* task, int i) {
    weight_t sum = 0.0;
    int j = 0;
    // Loop with data-dependent exit (PHI node)
    while (j < task->degree) {
      weight_t w = graph->adjwgt[task->neighbor_offset + j];
      sum += w;
      if (sum > 10.0) break;  // Data-dependent break
      j++;
    }
    return sum;
  }

  __device__ void fill_dummy(Task* task, Task* dummy_task, int lid, int target_tid);
  __device__ void fill_dummy(Task* task, Task* dummy_task, int lid, int target_tid, unsigned mask);
  __device__ weight_t get_max_weight(Task* task, int i);
  __device__ weight_t get_sum_weight(Task* task, int i);

  __device__ bool scan_thread(Task* task) {
    return task->degree > 0;
  }
};

// Test 4: Graph modification (should set ERVS_ONLY = 1)
class TestGraphModify : public WalkerMeta {
 public:
  TestGraphModify() {}
  LLVM_CTOR TestGraphModify(gpu_graph* _graph, int _max_depth)
      : WalkerMeta(_graph, _max_depth) {}

  __device__ weight_t get_weight(Task* task, int i) {
    // Modify graph array
    graph->adjwgt[task->neighbor_offset + i] *= 0.9;  // Decay weight
    return graph->adjwgt[task->neighbor_offset + i];
  }

  __device__ void fill_dummy(Task* task, Task* dummy_task, int lid, int target_tid);
  __device__ void fill_dummy(Task* task, Task* dummy_task, int lid, int target_tid, unsigned mask);
  __device__ weight_t get_max_weight(Task* task, int i);
  __device__ weight_t get_sum_weight(Task* task, int i);

  __device__ bool scan_thread(Task* task) {
    return task->degree > 0;
  }
};

// Test 5: Simple walker (should NOT set ERVS_ONLY, should be 0)
class TestSimple : public WalkerMeta {
 public:
  TestSimple() {}
  LLVM_CTOR TestSimple(gpu_graph* _graph, int _max_depth)
      : WalkerMeta(_graph, _max_depth) {}

  __device__ weight_t get_weight(Task* task, int i) {
    return graph->adjwgt[task->neighbor_offset + i];
  }

  __device__ void fill_dummy(Task* task, Task* dummy_task, int lid, int target_tid);
  __device__ void fill_dummy(Task* task, Task* dummy_task, int lid, int target_tid, unsigned mask);
  __device__ weight_t get_max_weight(Task* task, int i);
  __device__ weight_t get_sum_weight(Task* task, int i);

  __device__ bool scan_thread(Task* task) {
    return task->degree > 0;
  }
};

// Test 6: Two-level nested loops with fixed bounds (should NOT set ERVS_ONLY; = 0).
// Only reaches nesting depth 2, below the compiler's MAX_NESTING_DEPTH=5 threshold.
// The original comment "should set ERVS_ONLY = 1" was aspirational; the compiler
// correctly treats bounded, shallow nesting as analyzable.
class TestNestedLoops : public WalkerMeta {
 public:
  TestNestedLoops() {}
  LLVM_CTOR TestNestedLoops(gpu_graph* _graph, int _max_depth)
      : WalkerMeta(_graph, _max_depth) {}

  __device__ weight_t get_weight(Task* task, int i) {
    weight_t sum = 0.0;
    // Nested loops
    for (int j = 0; j < task->degree; j++) {
      for (int k = 0; k < 5; k++) {
        sum += graph->adjwgt[task->neighbor_offset + j] * k;
      }
    }
    return sum;
  }

  __device__ void fill_dummy(Task* task, Task* dummy_task, int lid, int target_tid);
  __device__ void fill_dummy(Task* task, Task* dummy_task, int lid, int target_tid, unsigned mask);
  __device__ weight_t get_max_weight(Task* task, int i);
  __device__ weight_t get_sum_weight(Task* task, int i);

  __device__ bool scan_thread(Task* task) {
    return task->degree > 0;
  }
};

// Test 7: Indirect recursion (should set ERVS_ONLY = 1)
class TestIndirectRecursion : public WalkerMeta {
 public:
  TestIndirectRecursion() {}
  LLVM_CTOR TestIndirectRecursion(gpu_graph* _graph, int _max_depth)
      : WalkerMeta(_graph, _max_depth) {}

  __device__ weight_t helper_a(int depth);
  __device__ weight_t helper_b(int depth);

  __device__ weight_t get_weight(Task* task, int i) {
    return helper_a(3);
  }

  __device__ void fill_dummy(Task* task, Task* dummy_task, int lid, int target_tid);
  __device__ void fill_dummy(Task* task, Task* dummy_task, int lid, int target_tid, unsigned mask);
  __device__ weight_t get_max_weight(Task* task, int i);
  __device__ weight_t get_sum_weight(Task* task, int i);

  __device__ bool scan_thread(Task* task) {
    return task->degree > 0;
  }
};

__device__ weight_t TestIndirectRecursion::helper_a(int depth) {
  if (depth <= 0) return 1.0;
  return helper_b(depth - 1);  // Calls helper_b
}

__device__ weight_t TestIndirectRecursion::helper_b(int depth) {
  if (depth <= 0) return 1.0;
  return helper_a(depth - 1);  // Calls helper_a - indirect recursion!
}

// Test 8: Flat if-else chain (should NOT set ERVS_ONLY = 0)
class TestFlatIfElse : public WalkerMeta {
 public:
  TestFlatIfElse() {}
  LLVM_CTOR TestFlatIfElse(gpu_graph* _graph, int _max_depth)
      : WalkerMeta(_graph, _max_depth) {}

  __device__ weight_t get_weight(Task* task, int i) {
    weight_t w = graph->adjwgt[task->neighbor_offset + i];
    // Flat if-else-if chain (depth 1)
    if (w < 1.0) {
      return w * 0.5;
    } else if (w < 2.0) {
      return w * 0.7;
    } else if (w < 3.0) {
      return w * 0.9;
    } else if (w < 4.0) {
      return w * 1.1;
    } else if (w < 5.0) {
      return w * 1.3;
    } else {
      return w * 1.5;
    }
  }

  __device__ void fill_dummy(Task* task, Task* dummy_task, int lid, int target_tid);
  __device__ void fill_dummy(Task* task, Task* dummy_task, int lid, int target_tid, unsigned mask);
  __device__ weight_t get_max_weight(Task* task, int i);
  __device__ weight_t get_sum_weight(Task* task, int i);

  __device__ bool scan_thread(Task* task) {
    return task->degree > 0;
  }
};

// Test 9: Simple loop without data dependency (should NOT set ERVS_ONLY = 0)
class TestSimpleLoop : public WalkerMeta {
 public:
  TestSimpleLoop() {}
  LLVM_CTOR TestSimpleLoop(gpu_graph* _graph, int _max_depth)
      : WalkerMeta(_graph, _max_depth) {}

  __device__ weight_t get_weight(Task* task, int i) {
    weight_t sum = 0.0;
    // Simple loop with fixed bounds, no data-dependent exit
    for (int j = 0; j < task->degree; j++) {
      sum += graph->adjwgt[task->neighbor_offset + j];
    }
    return sum;
  }

  __device__ void fill_dummy(Task* task, Task* dummy_task, int lid, int target_tid);
  __device__ void fill_dummy(Task* task, Task* dummy_task, int lid, int target_tid, unsigned mask);
  __device__ weight_t get_max_weight(Task* task, int i);
  __device__ weight_t get_sum_weight(Task* task, int i);

  __device__ bool scan_thread(Task* task) {
    return task->degree > 0;
  }
};

// Test 10: Warp synchronization (should warn but NOT set ERVS_ONLY = 0)
class TestWarpSync : public WalkerMeta {
 public:
  TestWarpSync() {}
  LLVM_CTOR TestWarpSync(gpu_graph* _graph, int _max_depth)
      : WalkerMeta(_graph, _max_depth) {}

  __device__ weight_t get_weight(Task* task, int i) {
    weight_t w = graph->adjwgt[task->neighbor_offset + i];
    // Use warp shuffle - should trigger warning
    w = __shfl_sync(0xffffffff, w, 0);
    return w;
  }

  __device__ void fill_dummy(Task* task, Task* dummy_task, int lid, int target_tid);
  __device__ void fill_dummy(Task* task, Task* dummy_task, int lid, int target_tid, unsigned mask);
  __device__ weight_t get_max_weight(Task* task, int i);
  __device__ weight_t get_sum_weight(Task* task, int i);

  __device__ bool scan_thread(Task* task) {
    return task->degree > 0;
  }
};

// Test 11: Role conflict — helper local used at both MAX and MIN in return.
// `w + (1.0 - w)` makes `w` appear at MAX (LHS of +) and MIN (RHS of -).
// The string-level suffix rewrite in helper bodies can carry only one
// role per statement, so the rewriter must conservatively force eRVS.
class TestRoleConflict : public WalkerMeta {
 public:
  TestRoleConflict() {}
  LLVM_CTOR TestRoleConflict(gpu_graph* _graph, int _max_depth)
      : WalkerMeta(_graph, _max_depth) {}

  __device__ weight_t get_weight(Task* task, int i) {
    weight_t w = graph->adjwgt[task->neighbor_offset + i];
    return w + (1.0 - w);
  }

  __device__ void fill_dummy(Task* task, Task* dummy_task, int lid, int target_tid);
  __device__ void fill_dummy(Task* task, Task* dummy_task, int lid, int target_tid, unsigned mask);
  __device__ weight_t get_max_weight(Task* task, int i);
  __device__ weight_t get_sum_weight(Task* task, int i);

  __device__ bool scan_thread(Task* task) {
    return task->degree > 0;
  }
};

// Test 12: Pointer aliasing — bare reference to an indexed field outside
// any subscript. The pipeline cannot tag `_MAX` / `_MIN` on the aliased
// pointer, so it must fall back to eRVS rather than emit wrong code.
class TestPointerAlias : public WalkerMeta {
 public:
  TestPointerAlias() {}
  LLVM_CTOR TestPointerAlias(gpu_graph* _graph, int _max_depth)
      : WalkerMeta(_graph, _max_depth) {}

  __device__ weight_t get_weight(Task* task, int i) {
    weight_t* p = graph->adjwgt;
    return p[task->neighbor_offset + i];
  }

  __device__ void fill_dummy(Task* task, Task* dummy_task, int lid, int target_tid);
  __device__ void fill_dummy(Task* task, Task* dummy_task, int lid, int target_tid, unsigned mask);
  __device__ weight_t get_max_weight(Task* task, int i);
  __device__ weight_t get_sum_weight(Task* task, int i);

  __device__ bool scan_thread(Task* task) {
    return task->degree > 0;
  }
};

// Test 13: Role-flipping operator inside a helper RHS. Subtraction flips
// the role on its right operand, but a single helper statement can only
// carry one suffix. Force eRVS rather than emit a wrong suffix on one
// of the operands.
class TestHelperRhsFlip : public WalkerMeta {
 public:
  TestHelperRhsFlip() {}
  LLVM_CTOR TestHelperRhsFlip(gpu_graph* _graph, int _max_depth)
      : WalkerMeta(_graph, _max_depth) {}

  __device__ weight_t get_weight(Task* task, int i) {
    weight_t w = 1.0 - graph->adjwgt[task->neighbor_offset + i];
    return w;
  }

  __device__ void fill_dummy(Task* task, Task* dummy_task, int lid, int target_tid);
  __device__ void fill_dummy(Task* task, Task* dummy_task, int lid, int target_tid, unsigned mask);
  __device__ weight_t get_max_weight(Task* task, int i);
  __device__ weight_t get_sum_weight(Task* task, int i);

  __device__ bool scan_thread(Task* task) {
    return task->degree > 0;
  }
};
