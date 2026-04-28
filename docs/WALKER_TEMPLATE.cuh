/* ====================================================================
 * FlexiWalker Walker Template
 *
 * This is a template for creating new random walk workloads in FlexiWalker.
 * Copy this template and customize it for your algorithm.
 *
 * REQUIRED: constructor, get_weight()
 * OPTIONAL: is_stop() (defaults to len >= max_depth), update()
 * ====================================================================
 */

// ================================================================
// STEP 0 (OPTIONAL): Custom task type
// ================================================================
// The default `Task` carries current_vertex, prev_vertex, neighbor_offset,
// degree, and length. If your get_weight() needs history beyond that
// (e.g. the previous vertex's degree, two steps back, accumulated state),
// subclass `Task` in include/gpu_task.cuh and override update() to snapshot
// the extra fields before Task::update() overwrites them. See PPRSecondTask
// in include/gpu_task.cuh for a worked example, and the "Custom Task Types"
// section of docs/WALKER_API.md for the full recipe.
//
// If you add one, declare `using TaskType = MyCustomTask;` in the walker
// class below and replace every `Task*` in this template with `TaskType*`.

// TODO: Replace MyWalker with your walker name (e.g., MyAlgorithm, CustomRW, etc.)
class MyWalker : public WalkerMeta {
 public:
  // ================================================================
  // STEP 1: Define your walker's parameters (if needed)
  // ================================================================
  // Add any parameters your algorithm needs
  // Examples:
  //   float alpha;           // For probability-based algorithms
  //   int* schema;           // For pattern-based algorithms
  //   float p, q;            // For biased walks (like Node2Vec)

  // TODO: Add your parameters here (if any)

 public:
  // ================================================================
  // STEP 2: Implement constructors - REQUIRED
  // ================================================================

  // Default constructor (required, usually no changes needed)
  MyWalker() {}

  // Initialization constructor (required)
  // TODO: Add your parameters after _max_depth (if you added any in STEP 1)
  LLVM_CTOR MyWalker(gpu_graph* _graph, int _max_depth /* , your params */)
      : WalkerMeta(_graph, _max_depth) {
    // TODO: Initialize your parameters here
    // Example:
    //   this->alpha = _alpha;
    //   this->p = _p;
    //   this->q = _q;
  }

  // ================================================================
  // STEP 3: Implement get_weight() - REQUIRED
  // ================================================================
  // Computes the weight for sampling each neighbor edge
  //
  // Parameters:
  //   task - Current walk state (task->current_vertex, task->prev_vertex, etc.)
  //   i - Neighbor index (0 to task->degree-1)
  //
  // Return: Weight for this edge (0.0 = skip this edge)
  //
  __device__ weight_t get_weight(Task* task, int i) {
    // TODO: Implement your weight computation

    // Common patterns:

    // Pattern 1: Uniform random walk
    // return 1.0;

    // Pattern 2: Use graph edge weights
    // return graph->adjwgt[task->neighbor_offset + i];

    // Pattern 3: Conditional (e.g., MetaPath-style filtering)
    // return (condition) ? 1.0 : 0.0;

    // Pattern 4: Biased by previous vertex (Node2Vec-style)
    // if (task->prev_vertex == -1) return 1.0;  // First step
    // vtx_t next = graph->adjncy[task->neighbor_offset + i];
    // if (next == task->prev_vertex) return 1.0 / p;
    // if (!graph->check_connect(task->prev_vertex, next)) return 1.0 / q;
    // return 1.0;

    return graph->adjwgt[task->neighbor_offset + i];  // Default: use edge weights
  }

  // ================================================================
  // STEP 4 (OPTIONAL): Override is_stop() if needed
  // ================================================================
  // DEFAULT BEHAVIOR: Stops when len >= max_depth
  // Only implement this if you need custom termination (e.g., probabilistic)
  //
  // Uncomment and modify if needed:
  //
  // template <typename state_t>
  // __device__ bool is_stop(int len, state_t* state) {
  //   // Example: Probabilistic termination (PPR-style)
  //   float r = myrand_uniform(state);
  //   return (r <= termination_prob) || (len >= max_depth);
  // }

  // ================================================================
  // STEP 5 (OPTIONAL): Implement update() for custom state tracking
  // ================================================================
  // DEFAULT BEHAVIOR: Task::update() moves to next vertex
  // Only implement this if you need to modify graph or track extra state
  //
  // Use cases:
  //   - Reinforcement learning (modifying edge weights)
  //   - Tracking additional custom state
  //
  // Uncomment and modify if needed:
  //
  // __device__ void update(Task* task, vtx_t selected_id) {
  //   // Your custom logic (e.g., reinforce edge weight)
  //   atomicAdd(graph->adjwgt + task->neighbor_offset + selected_id, r);
  //   __threadfence();
  //
  //   // Then do standard update
  //   task->update(graph, selected_id);
  // }

  // ================================================================
  // AUTO-GENERATED METHODS - DO NOT IMPLEMENT
  // ================================================================
  // These are automatically generated by the compilation pipeline.
  // You only need to declare them (the actual implementation is generated):
  __device__ void fill_dummy(Task* task, Task* dummy_task, int lid, int target_tid);
  __device__ void fill_dummy(Task* task, Task* dummy_task, int lid, int target_tid, unsigned mask);
  __device__ weight_t get_max_weight(Task* task, int i);
  __device__ weight_t get_sum_weight(Task* task, int i);

  // Standard implementation that uses your get_weight():
  __device__ bool scan_thread(Task* task) {
    vtx_t size = task->degree;
    for (int i = 0; i < size; i++) {
      if (get_weight(task, i) > 0) return true;
    }
    return false;
  }
};

// ================================================================
// INTEGRATION: Adding your walker to FlexiWalker
// ================================================================
//
// After implementing your walker class above:
//
// 1. Add to include/app.cuh (before the #ifndef LLVM_ANALYSIS line)
//
// 2. Run the compilation pipeline (inside the toolchain container):
//    python3 run_pipeline.py --clean
//
// 3. Declare the flag in BOTH src/main.cu and src/walk.cu:
//
//    src/main.cu (in the DEFINE_bool block):
//        DEFINE_bool(MyWalker, false, "description of my walker");
//
//    src/walk.cu (top of file, next to the other DECLARE_bool lines):
//        DECLARE_bool(MyWalker);
//
//    If you have custom parameters, add matching DEFINE_* / DECLARE_* pairs:
//        // main.cu
//        DEFINE_double(myparam, 1.0, "description of my parameter");
//        // walk.cu (only if the dispatch chains reference it)
//        DECLARE_double(myparam);
//
// 4. Add a dispatch branch to ALL THREE if/else chains in src/walk.cu:
//
//    - profile_current_walker() uses profile_kernels<>():
//        } else if (FLAGS_MyWalker) {
//          MyWalker* walker = new MyWalker(graph_ptr, max_depth /* , FLAGS_myparam, ... */);
//          MyWalker* walker_ptr = get_device_ptr<MyWalker>(walker, 1);
//          ratio = profile_kernels<MyWalker>(walker_ptr);
//        }
//
//    - walk_batch() uses run_walker<>():
//        } else if (FLAGS_MyWalker) {
//          total_time = run_walker<MyWalker>(graph_ptr, start_points, result_pool,
//                                            batch_size, num_walkers, max_depth, block_num,
//                                            select_algo_cost_ratio
//                                            /* , FLAGS_myparam, ... */);
//        }
//
//    - walk_test() uses run_walker_test<>() (note: no batch_size):
//        } else if (FLAGS_MyWalker) {
//          total_time = run_walker_test<MyWalker>(graph_ptr, start_points_ptr, result_pool_ptr,
//                                                 block_num, num_walkers, max_depth,
//                                                 select_algo_cost_ratio
//                                                 /* , FLAGS_myparam, ... */);
//        }
//
//    All three sites are required. See WALKER_API.md §5 for worked examples
//    (no-param, scalar-param, and array-typed like Metapath).
//
// 5. Need an extra graph array (per-edge attribute, per-vertex score, ...)?
//    Declare it in config/graph_fields.config and point each dataset's config
//    at the binary. See the "Adding a Custom Graph Field" section in
//    WALKER_API.md and the binary file format notes in data/README.md.
//
// 6. Build and test:
//    cd build && make -j
//    ./bin/flowwalker --input ../data/wiki-Vote --MyWalker --n=100
//
// See WALKER_API.md for detailed documentation and examples.
