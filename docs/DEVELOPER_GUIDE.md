# FlexiWalker Developer Guide

Welcome to FlexiWalker! This guide helps you get started with developing new random walk algorithms.

## Documentation Index

- **[../README.md](../README.md)** — project overview, build, quick-start
- **[WALKER_API.md](WALKER_API.md)** — walker-class API reference + compiler contract
- **[WALKER_TEMPLATE.cuh](WALKER_TEMPLATE.cuh)** — copy-paste walker starter

## Quick Start: Add Your First Walker

### 1. Create Your Walker Class

Open `include/app.cuh` and add your walker (copy from `WALKER_TEMPLATE.cuh`):

```cuda
class MyWalker : public WalkerMeta {
 public:
  MyWalker() {}
  LLVM_CTOR MyWalker(gpu_graph* _graph, int _max_depth)
      : WalkerMeta(_graph, _max_depth) {}

  __device__ weight_t get_weight(Task* task, int i) {
    return graph->adjwgt[task->neighbor_offset + i];
  }
};
```

### 2. Run the Compilation Pipeline

Inside the toolchain container (see [Toolchain container](../README.md#toolchain-container-recommended)):

```bash
python3 run_pipeline.py --clean
```

This generates optimized code in `include/generated/`.

### 3. Add Command-Line Flag (Two Places)

In `src/main.cu` (around line 52), define the flag:

```cpp
DEFINE_bool(MyWalker, false, "my walker description");
```

In `src/walk.cu` (top of file, near the other `DECLARE_bool`s), declare it so the dispatch chains can reference it:

```cpp
DECLARE_bool(MyWalker);
```

Missing the `DECLARE_bool` produces a link error on the `FLAGS_MyWalker` references in the dispatch chains below.

### 4. Add Walker Instantiation to All Three Dispatch Chains

`src/walk.cu` contains **three** parallel if/else dispatch chains — `profile_current_walker()` (startup kernel profile), `walk_test()` (non-batch path), and `walk_batch()` (production path, what `--all`/`--autobatch` forces). All three need an else-if branch for your walker. See [WALKER_API.md §5](WALKER_API.md#5-add-instantiation-in-all-three-dispatch-chains-in-srcwalkcu) for the exact signatures of `profile_kernels<>()`, `run_walker<>()`, and `run_walker_test<>()`, and worked examples for no-param, scalar-param, and array-param (Metapath-style) walkers.

### 5. Build and Test

```bash
cd build && cmake .. && make -j
```

Then add `MyWalker` to the `WALKERS` array in `scripts/run_templ_one.sh` and run it on a prepared dataset:

```bash
scripts/run_templ_one.sh 0 com-youtube   # <gpu_id> <dataset_name>
```

**That's it!** See [WALKER_API.md](WALKER_API.md) for more examples and details.

## Key Concepts

### What You Implement

- **Constructor** - Initialize your walker with parameters
- **`get_weight()`** - Compute edge weights for sampling (**REQUIRED**)
- **`is_stop()`** - Custom termination logic (OPTIONAL, default: fixed-length)
- **`update()`** - Custom state updates (OPTIONAL, default: standard transition)

### What's Auto-Generated

The compilation pipeline generates:
- `fill_dummy()` - Task filling implementations
- `get_max_weight()` - Maximum weight computation
- `get_sum_weight()` - Sum weight computation

These are automatically created based on your `get_weight()` implementation.

## Example Walkers

For end-to-end examples (DeepWalk, PPR, Node2vec, Metapath, PPR-second, and their weighted variants), see the shipped walker classes in `include/app.cuh`.

## Available Data

### Task Structure
```cuda
task->walker_id          // Walker ID
task->current_vertex     // Current vertex
task->prev_vertex        // Previous vertex (-1 for first step)
task->neighbor_offset    // Offset into adjacency arrays
task->degree             // Number of neighbors
task->length             // Current walk length
```

### Graph Structure (via `graph` pointer)
```cuda
graph->xadj[v]                    // CSR row pointer
graph->adjncy[offset + i]         // Neighbor vertex
graph->adjwgt[offset + i]         // Edge weight
graph->edge_label[offset + i]     // Edge label (heterogeneous graphs)
graph->getDegree(v)               // Get vertex degree
graph->check_connect(u, v)        // Check if edge exists
```

### Need a new graph array?

If your walker needs graph data beyond the core CSR (`xadj`/`adjncy`/`adjwgt`) — for example an extra per-edge attribute or a per-vertex score — you declare it once in `config/graph_fields.config` and the pipeline wires it into `gpu_graph` for any walker that asks for it. See the [*Adding a Custom Graph Field*](WALKER_API.md#adding-a-custom-graph-field) section of WALKER_API.md for the three-step workflow (declare → point at binary → use), and [`data/README.md`](../data/README.md) for the binary file format your custom array must be written in.

## Troubleshooting

### Pipeline Errors
```bash
# Clean rebuild
python3 run_pipeline.py --clean --verbose
```

### Build Errors
```bash
# Make sure pipeline completed first
python3 run_pipeline.py --clean
cd build && make clean && make -j
```

### Runtime Issues
```bash
# Print walk results for debugging
./bin/flowwalker --input ../data/wiki-Vote --MyWalker --n=100 --printresult
```

## Getting Help

1. Check [WALKER_API.md](WALKER_API.md) for detailed API documentation
2. Look at existing walkers in `include/app.cuh` for examples
3. See [../README.md](../README.md) for general FlexiWalker information

## Contributing

When adding walkers for publication:
1. Follow the naming conventions in existing code
2. Document your algorithm in comments
3. Add command-line parameters with clear descriptions
4. Include example usage in your documentation
5. Test with multiple graph datasets

## License

See [../LICENSE](../LICENSE) for licensing information.
