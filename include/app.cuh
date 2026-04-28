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

#include <cuda.h>
#include <curand.h>
#include <curand_kernel.h>
#include <gflags/gflags.h>
#include <stdio.h>
#include <stdlib.h>

#include <iostream>

#ifndef LLVM_ANALYSIS
#include <cub/cub.cuh>  // NOLINT
#endif

#ifdef LLVM_ANALYSIS
#define LLVM_CTOR __host__ __device__
#else
#define LLVM_CTOR
#endif

#include "gpu_task.cuh"
#include "generated/graph.cuh"
#include "myrand.cuh"
#include "util.cuh"

/*
Walker apps
*/
class WalkerMeta {
 public:
  using TaskType = Task; // Default Task type for all applications
  int max_depth;
  gpu_graph* graph;

 public:
  WalkerMeta() {}
  LLVM_CTOR WalkerMeta(gpu_graph* _graph, int _max_depth) {
    this->max_depth = _max_depth;
    this->graph = _graph;
  }
  // Default is_stop implementation: fixed-length walks
  // Override in subclass if you need custom termination logic
  template <typename state_t>
  __device__ bool is_stop(int len, state_t* state) {
    return len >= max_depth;
  }

  // Auto-generated methods (implemented in generated/*.cuh)
  // Note: get_weight, fill_dummy, get_max_weight, get_sum_weight, scan_thread
  // are declared in each walker subclass
};

class Deepwalk : public WalkerMeta {
 public:
  Deepwalk() {}
  LLVM_CTOR Deepwalk(gpu_graph* _graph, int _max_depth)
      : WalkerMeta(_graph, _max_depth) {}

  __device__ weight_t get_weight(Task* task, int i) {
    return graph->adjwgt[task->neighbor_offset + i];
  }

  // Uses default is_stop() from WalkerMeta: len >= max_depth

  // Auto-generated: fill_dummy, get_max_weight, get_sum_weight
  __device__ void fill_dummy(Task* task, Task* dummy_task, int lid, int target_tid);
  __device__ void fill_dummy(Task* task, Task* dummy_task, int lid, int target_tid, unsigned mask);
  __device__ weight_t get_max_weight(Task* task, int i);
  __device__ weight_t get_sum_weight(Task* task, int i);

  __device__ bool scan_thread(Task* task) {
    vtx_t size = task->degree;
    for (int i = 0; i < size; i++) {
      if (get_weight(task, i) > 0) return true;
    }
    return false;
  }
};

class PPR : public WalkerMeta {
 public:
  float tp;  // Termination probability

 public:
  PPR() {}
  LLVM_CTOR PPR(gpu_graph* _graph, int _max_depth, float _tp)
      : WalkerMeta(_graph, _max_depth) {
    this->tp = _tp;
  }

  __device__ weight_t get_weight(Task* task, int i) {
    return graph->adjwgt[task->neighbor_offset + i];
  }

  // Custom termination: probabilistic early stopping
  template <typename state_t>
  __device__ bool is_stop(int len, state_t* state) {
    float r = myrand_uniform(state);
    return (r <= tp) || (len >= max_depth);
  }

  // Auto-generated: fill_dummy, get_max_weight, get_sum_weight
  __device__ void fill_dummy(Task* task, Task* dummy_task, int lid, int target_tid);
  __device__ void fill_dummy(Task* task, Task* dummy_task, int lid, int target_tid, unsigned mask);
  __device__ weight_t get_max_weight(Task* task, int i);
  __device__ weight_t get_sum_weight(Task* task, int i);

  __device__ bool scan_thread(Task* task) {
    vtx_t size = task->degree;
    for (int i = 0; i < size; i++) {
      if (get_weight(task, i) > 0) return true;
    }
    return false;
  }
};

class Node2vec : public WalkerMeta {
 public:
  float p, q;  // Bias parameters: return (p) and in-out (q)

 public:
  Node2vec() {}
  LLVM_CTOR Node2vec(gpu_graph* _graph, int _max_depth, float _p, float _q)
      : WalkerMeta(_graph, _max_depth) {
    LOG("%s\n", __FUNCTION__);
    this->p = _p;
    this->q = _q;
  }

  __device__ weight_t get_weight(Task* task, int i) {
    if (task->prev_vertex == -1) {
      return 1.0;  // First step: uniform
    }

    vtx_t next = graph->adjncy[task->neighbor_offset + i];
    if (next == task->prev_vertex) {
      return 1.0 / p;  // Return to previous
    } else if (!graph->check_connect(task->prev_vertex, next)) {
      return 1.0 / q;  // Move away
    }
    return 1.0;  // Stay in neighborhood
  }

  // Uses default is_stop() from WalkerMeta: len >= max_depth
  // Auto-generated: fill_dummy, get_max_weight, get_sum_weight
  __device__ void fill_dummy(Task* task, Task* dummy_task, int lid, int target_tid);
  __device__ void fill_dummy(Task* task, Task* dummy_task, int lid, int target_tid, unsigned mask);
  __device__ weight_t get_max_weight(Task* task, int i);
  __device__ weight_t get_sum_weight(Task* task, int i);

  __device__ bool scan_thread(Task* task) {
    vtx_t size = task->degree;
    for (int i = 0; i < size; i++) {
      if (get_weight(task, i) > 0) return true;
    }
    return false;
  }
};

class Node2vec_weighted : public WalkerMeta {
 public:
  float p, q;  // Bias parameters: return (p) and in-out (q)

 public:
  Node2vec_weighted() {}
  LLVM_CTOR Node2vec_weighted(gpu_graph* _graph, int _max_depth, float _p, float _q)
      : WalkerMeta(_graph, _max_depth) {
    LOG("%s\n", __FUNCTION__);
    this->p = _p;
    this->q = _q;
  }

  __device__ weight_t get_weight(Task* task, int i) {
    weight_t w = graph->adjwgt[task->neighbor_offset + i];
    if (task->prev_vertex == -1) {
      return w;  // First step: use edge weight
    }

    vtx_t next = graph->adjncy[task->neighbor_offset + i];
    if (next == task->prev_vertex) {
      return w / p;  // Return to previous
    } else if (!graph->check_connect(task->prev_vertex, next)) {
      return w / q;  // Move away
    }
    return w;  // Stay in neighborhood
  }

  // Uses default is_stop() from WalkerMeta: len >= max_depth
  // Auto-generated: fill_dummy, get_max_weight, get_sum_weight
  __device__ void fill_dummy(Task* task, Task* dummy_task, int lid, int target_tid);
  __device__ void fill_dummy(Task* task, Task* dummy_task, int lid, int target_tid, unsigned mask);
  __device__ weight_t get_max_weight(Task* task, int i);
  __device__ weight_t get_sum_weight(Task* task, int i);

  __device__ bool scan_thread(Task* task) {
    vtx_t size = task->degree;
    for (int i = 0; i < size; i++) {
      if (get_weight(task, i) > 0) return true;
    }
    return false;
  }
};

class Metapath : public WalkerMeta {
 public:
  int* schema;      // Edge type pattern
  int schema_len;   // Pattern length

 public:
  Metapath() {}
  LLVM_CTOR Metapath(gpu_graph* _graph, int _max_depth, int* _schema, int _schema_len)
      : WalkerMeta(_graph, _max_depth) {
    LOG("%s\n", __FUNCTION__);
    this->schema = _schema;
    this->schema_len = _schema_len;
  }

  __device__ weight_t get_weight(Task* task, int i) {
    edge_t offset = task->neighbor_offset + i;
    int expected_label = schema[(task->length - 1) % schema_len];
    return graph->edge_label[offset] == expected_label ? 1.0 : 0.0;
  }

  // Uses default is_stop() from WalkerMeta: len >= max_depth
  // Auto-generated: fill_dummy, get_max_weight, get_sum_weight
  __device__ void fill_dummy(Task* task, Task* dummy_task, int lid, int target_tid);
  __device__ void fill_dummy(Task* task, Task* dummy_task, int lid, int target_tid, unsigned mask);
  __device__ weight_t get_max_weight(Task* task, int i);
  __device__ weight_t get_sum_weight(Task* task, int i);

  __device__ bool scan_thread(Task* task) {
    vtx_t size = task->degree;
    for (int i = 0; i < size; i++) {
      if (get_weight(task, i) > 0) return true;
    }
    return false;
  }
};

class Metapath_weighted : public WalkerMeta {
 public:
  int* schema;      // Edge type pattern
  int schema_len;   // Pattern length

 public:
  Metapath_weighted() {}
  LLVM_CTOR Metapath_weighted(gpu_graph* _graph, int _max_depth, int* _schema, int _schema_len)
      : WalkerMeta(_graph, _max_depth) {
    LOG("%s\n", __FUNCTION__);
    this->schema = _schema;
    this->schema_len = _schema_len;
  }

  __device__ weight_t get_weight(Task* task, int i) {
    edge_t offset = task->neighbor_offset + i;
    int expected_label = schema[(task->length - 1) % schema_len];
    weight_t w = graph->adjwgt[offset];
    return graph->edge_label[offset] == expected_label ? w : 0.0;
  }

  // Uses default is_stop() from WalkerMeta: len >= max_depth
  // Auto-generated: fill_dummy, get_max_weight, get_sum_weight
  __device__ void fill_dummy(Task* task, Task* dummy_task, int lid, int target_tid);
  __device__ void fill_dummy(Task* task, Task* dummy_task, int lid, int target_tid, unsigned mask);
  __device__ weight_t get_max_weight(Task* task, int i);
  __device__ weight_t get_sum_weight(Task* task, int i);

  __device__ bool scan_thread(Task* task) {
    vtx_t size = task->degree;
    for (int i = 0; i < size; i++) {
      if (get_weight(task, i) > 0) return true;
    }
    return false;
  }
};


class PPR_second : public WalkerMeta {
 public:
  float alpha;  // Mixing parameter
  using TaskType = PPRSecondTask;  // Uses custom task with prev_degree

 public:
  PPR_second() {}
  LLVM_CTOR PPR_second(gpu_graph* _graph, int _max_depth, float _alpha)
      : WalkerMeta(_graph, _max_depth) {
    LOG("%s\n", __FUNCTION__);
    this->alpha = _alpha;
  }

  __device__ weight_t get_weight(TaskType* task, int i) {
    if (task->prev_vertex == -1) {
      return (1.0 - alpha);
    }
    vtx_t next = graph->adjncy[task->neighbor_offset + i];
    vtx_t degree = task->degree;
    vtx_t prev_degree = task->prev_degree;
    vtx_t max_degree = max(degree, prev_degree);

    if (next == task->prev_vertex) {
      return (1.0 - alpha) / degree * max_degree;
    } else if (graph->check_connect(task->prev_vertex, next)) {
      return ((1.0 - alpha) / degree + alpha / prev_degree) * max_degree;
    } else {
      return (1.0 - alpha) / degree * max_degree;
    }
  }

  // Uses default is_stop() from WalkerMeta: len >= max_depth
  // Auto-generated: fill_dummy, get_max_weight, get_sum_weight
  __device__ void fill_dummy(TaskType* task, TaskType* dummy_task, int lid, int target_tid);
  __device__ void fill_dummy(TaskType* task, TaskType* dummy_task, int lid, int target_tid, unsigned mask);
  __device__ weight_t get_max_weight(TaskType* task, int i);
  __device__ weight_t get_sum_weight(TaskType* task, int i);

  __device__ bool scan_thread(TaskType* task) {
    vtx_t size = task->degree;
    for (int i = 0; i < size; i++) {
      if (get_weight(task, i) > 0) return true;
    }
    return false;
  }
};

#ifndef LLVM_ANALYSIS
#include "generated/fill_dummy.cuh"
#include "generated/get_max_weight.cuh"
#include "generated/get_sum_weight.cuh"
#endif