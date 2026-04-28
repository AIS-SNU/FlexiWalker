/* ====================================================================
 * Copyright (2024) Bytedance Ltd. and/or its affiliates
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * ====================================================================
 */
#pragma once
#include <gflags/gflags.h>

#include <algorithm>
#include <iostream>

#include "generated/graph.cuh"
#include "util.cuh"

DECLARE_bool(umgraph);

class gpu_graph_base {
 public:
  vtx_t vtx_num;               // vertex number
  edge_t edge_num;             // edge number
  vtx_t* adjncy;               // edge list
  weight_t* adjwgt = nullptr;  // edge weight
  edge_t* xadj;                // vertex list (the offset of edge, csr)

 public:
  ~gpu_graph_base() {
    if (xadj) cudaFree(xadj);
    if (adjncy) cudaFree(adjncy);
    if (adjwgt) cudaFree(adjwgt);
  }
  gpu_graph_base() {}
  explicit gpu_graph_base(graph* ginst) {
    vtx_num = ginst->vert_count;
    edge_num = ginst->edge_count;

    ll xadj_sz = sizeof(edge_t) * (ginst->vert_count + 1);
    ll edge_sz = sizeof(vtx_t) * ginst->edge_count;

    ll weight_sz = sizeof(weight_t) * ginst->edge_count;

    if (FLAGS_umgraph) {
      CUDA_RT_CALL(cudaMallocManaged(&xadj, xadj_sz));
      CUDA_RT_CALL(cudaMallocManaged(&adjncy, edge_sz));
      CUDA_RT_CALL(cudaMallocManaged(&adjwgt, weight_sz));
    } else {
      CUDA_RT_CALL(cudaMalloc(&xadj, xadj_sz));
      CUDA_RT_CALL(cudaMalloc(&adjncy, edge_sz));
      CUDA_RT_CALL(cudaMalloc(&adjwgt, weight_sz));
    }
    CUDA_RT_CALL(
        cudaMemcpy(xadj, ginst->xadj, xadj_sz, cudaMemcpyHostToDevice));
    CUDA_RT_CALL(
        cudaMemcpy(adjncy, ginst->adjncy, edge_sz, cudaMemcpyHostToDevice));
    CUDA_RT_CALL(
        cudaMemcpy(adjwgt, ginst->weight, weight_sz, cudaMemcpyHostToDevice));
  }

  __device__ vtx_t getDegree(vtx_t idx) { return xadj[idx + 1] - xadj[idx]; }
  __device__ void print_max_degree() {
    int gid = (blockDim.x * blockIdx.x) + threadIdx.x;
    if (gid == 0) {
      vtx_t max_degree = 0;
      vtx_t max_idx = 0;
      for (auto i = 0; i < vtx_num; i++) {
        vtx_t degree = xadj[i + 1] - xadj[i];
        if (degree > max_degree) {
          max_degree = degree;
          max_idx = i;
        }
      }
      printf("Max degree: %d, idx: %d\n", max_degree, max_idx);
    }
  }

  __device__ bool binarySearch(vtx_t* arr, vtx_t size, vtx_t target) {
    vtx_t l = 0;
    vtx_t r = size - 1;
    while (l <= r) {
      vtx_t m = l + (r - l) / 2;
      // Check if x is present at mid
      if (arr[m] == target) return true;
      // If x greater, ignore left half
      if (arr[m] < target) l = m + 1;
      // If x is smaller, ignore right half
      else
        r = m - 1;
    }
    // If we reach here, then element was not present
    return false;
  }
  __device__ bool check_connect(vtx_t src, vtx_t dst) {
    // check whether vertex src and dst are connected using binary search
    vtx_t src_degree = getDegree(src);

    return binarySearch(adjncy + xadj[src], src_degree, dst);
  }

  __device__ bool binarySearch(vtx_t* arr, vtx_t size, vtx_t target, edge_t* prev_offset) {
    vtx_t l = 0;
    vtx_t r = size - 1;
    while (l <= r) {
      vtx_t m = l + (r - l) / 2;
      // Check if x is present at mid
      if (arr[m] == target) {
        *prev_offset = m;
        return true;
      }
      // If x greater, ignore left half
      if (arr[m] < target) l = m + 1;
      // If x is smaller, ignore right half
      else
        r = m - 1;
    }
    // If we reach here, then element was not present
    return false;
  }
  __device__ bool check_connect(vtx_t src, vtx_t dst, edge_t* prev_offset) {
    // check whether vertex src and dst are connected using binary search
    vtx_t src_degree = getDegree(src);
    return binarySearch(adjncy + xadj[src], src_degree, dst, prev_offset);
  }
};