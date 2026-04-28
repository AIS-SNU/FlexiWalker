#pragma once

#include "gpu_graph_base.cuh"
#include "generated/graph.cuh"

// Stub extension for struct gpu_graph for LLVM analysis
// This will be replaced by the full generated version in code_generator stage
class gpu_graph : public gpu_graph_base {
  public:
    ~gpu_graph() {}
    gpu_graph() : gpu_graph_base() {}
    explicit gpu_graph(graph* ginst) : gpu_graph_base(ginst) {}
};