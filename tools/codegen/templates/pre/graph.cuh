#pragma once

#include "graph_base.cuh"

// Stub graph class for LLVM analysis
// This will be replaced by the full generated version in code_generator stage
class graph : public graph_base {
  public:
    graph() : graph_base() {}
    graph(const char* xadj_file, const char* adjncy_file, const char* weight_file, const char* config_file = nullptr)
        : graph_base(xadj_file, adjncy_file, weight_file) {}
};
