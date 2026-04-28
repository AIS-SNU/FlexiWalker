#pragma once
#include <gflags/gflags.h>
#include <algorithm>
#include <iostream>
#include <limits>
#include "util.cuh"
#include "gpu_graph_base.cuh"
#include "generated/graph.cuh"

// Auto-generated unified struct extensions

DECLARE_bool(Deepwalk);
DECLARE_bool(Metapath);
DECLARE_bool(Metapath_weighted);
DECLARE_bool(Node2vec);
DECLARE_bool(Node2vec_weighted);
DECLARE_bool(PPR);
DECLARE_bool(PPR_second);

#ifndef LLVM_ANALYSIS

template <typename T>
__global__ void preprocess_kernel(
    const edge_t* __restrict__ xadj,       // size: vtx_num + 1
    const T* __restrict__ input,           // input weights
    T* __restrict__ output_max,            // output max per node (nullptr to skip)
    T* __restrict__ output_min,            // output min per node (nullptr to skip)
    T* __restrict__ output_sum,            // output sum per node (nullptr to skip)
    int vtx_num
) {
    int warpSize = 32;
    int global_thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    int warp_id = global_thread_id / warpSize;
    int lane_id = threadIdx.x % warpSize;

    int total_warps = (gridDim.x * blockDim.x) / warpSize;

    for (int node_idx = warp_id; node_idx < vtx_num; node_idx += total_warps) {
      edge_t start = xadj[node_idx];
      edge_t end = xadj[node_idx + 1];

      T max_val = -1;
      T min_val = std::numeric_limits<T>::max();
      T sum_val = 0;

      for (edge_t e = start + lane_id; e < end; e += warpSize) {
        max_val = max(max_val, input[e]);
        min_val = min(min_val, input[e]);
        sum_val += input[e];
      }

      // Warp-level reduction across lanes
      max_val = warpReduceMax(max_val);
      min_val = warpReduceMin(min_val);
      sum_val = warpReduce(sum_val);

      if (lane_id == 0) {
        if (output_max) output_max[node_idx] = max_val;
        if (output_min) output_min[node_idx] = min_val;
        if (output_sum) output_sum[node_idx] = sum_val;
      }
    }
}

#endif

// Extension for struct gpu_graph
class gpu_graph : public gpu_graph_base {
  public:
  weight_t * adjwgt_MAX = nullptr;
  weight_t * adjwgt_SUM = nullptr;
  int* edge_label = nullptr;

  public:
    ~gpu_graph() {
      if (adjwgt_MAX) cudaFree(adjwgt_MAX);
      if (adjwgt_SUM) cudaFree(adjwgt_SUM);
      if (edge_label) cudaFree(edge_label);
    }
    gpu_graph() : gpu_graph_base() {}
    explicit gpu_graph(graph* ginst) : gpu_graph_base(ginst) {
      int blockSize = 256;
      int numBlocks = 512;
      #ifndef LLVM_ANALYSIS
        if (FLAGS_umgraph) {
          CUDA_RT_CALL(cudaMallocManaged(&adjwgt_MAX, sizeof(weight_t ) * vtx_num));
        } else {
          CUDA_RT_CALL(cudaMalloc(&adjwgt_MAX, sizeof(weight_t ) * vtx_num));
        }
        if (FLAGS_umgraph) {
          CUDA_RT_CALL(cudaMallocManaged(&adjwgt_SUM, sizeof(weight_t ) * vtx_num));
        } else {
          CUDA_RT_CALL(cudaMalloc(&adjwgt_SUM, sizeof(weight_t ) * vtx_num));
        }

      // Additional fields from graph_fields.config
      // edge_label for Metapath, Metapath_weighted
      if (FLAGS_Metapath || FLAGS_Metapath_weighted) {
        // Copy from host graph
        if (ginst->edge_label != nullptr) {
          ll edge_label_sz = sizeof(int) * edge_num;
          if (FLAGS_umgraph) {
            CUDA_RT_CALL(cudaMallocManaged(&edge_label, edge_label_sz));
          } else {
            CUDA_RT_CALL(cudaMalloc(&edge_label, edge_label_sz));
          }
          CUDA_RT_CALL(cudaMemcpy(edge_label, ginst->edge_label, edge_label_sz, cudaMemcpyHostToDevice));
        }
      }

        if (FLAGS_Deepwalk || FLAGS_Metapath_weighted || FLAGS_Node2vec_weighted || FLAGS_PPR) {
          preprocess_kernel<weight_t ><<<numBlocks, blockSize>>>(
            xadj,
            adjwgt,
            adjwgt_MAX,
            nullptr,
            adjwgt_SUM,
            vtx_num);
          CUDA_RT_CALL(cudaDeviceSynchronize());
        }
      #endif
    }
};

