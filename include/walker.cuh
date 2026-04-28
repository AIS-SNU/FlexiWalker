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
#include <algorithm>
#include <random>
#include <vector>

#include "gpu_task.cuh"
#include "myrand.cuh"
#include "sampler.cuh"
#include "util.cuh"

// #define ONELOOP
#define TASK_NUM 64
#define THRESHOLD 1024

#define CHUNK_SIZE 1

/*
Walk functions
*/
double walk_test(vtx_t*& result_pool_ptr, gpu_graph* graph,
                 vtx_t* start_points,  // NOLINT
                 int max_depth, int num_walkers, weight_t select_algo_cost_ratio,
                 int* schema = NULL, int schema_len = 0);

double walk_batch(vtx_t*& result_pool, gpu_graph* graph,
                  vtx_t* start_points,  // NOLINT
                  int max_depth, int num_walkers, int batch_size, weight_t select_algo_cost_ratio,
                  int* schema = NULL, int schema_len = 0);

// Profiling function - profiles current walker and returns select_algo_cost_ratio
weight_t profile_current_walker(gpu_graph* graph_ptr, int max_depth,
                                int* schema, int schema_len);

/*
Walkers
*/

// Runtime selection of rejection vs reservior
template <typename walker_t, typename TaskType = typename walker_t::TaskType>
__device__ inline bool pick_res(walker_t* walker, TaskType* task, weight_t max_weight, const weight_t select_algo_cost_ratio) {
  return (select_algo_cost_ratio * max_weight > walker->get_sum_weight(task, 0));
}


template <typename walker_t, char update_flag, bool possible_zero, typename TaskType = typename walker_t::TaskType>
__global__ void walker_select_algo(walker_t* walker, vtx_t* start_points,
                              int* start_pointer, vtx_t* result_pool,
                              int walker_num, int* global_chunk_idx, const weight_t select_algo_cost_ratio) {

  int lid = threadIdx.x % WARP_SIZE;
  int gid = (blockDim.x * blockIdx.x) + threadIdx.x;

  gpu_graph* graph = walker->graph;
  int max_depth = walker->max_depth;

  vtx_t start_point, selected, dummy_selected;
  int start_idx, target_tid;
  weight_t max_weight;

  // initialize random
  __shared__ myrandStateArr state;
  myrand_init(1337, gid, 0, &state);

  TaskType task;
  TaskType dummy_task;

  if (update_flag == 0) max_weight = walker->get_max_weight(&task, 0);

  bool task_finished = true;
  bool use_res = false;
  bool continue_rjs = false;

  // initialize start_idx
  start_idx = atomicAdd(global_chunk_idx, 1);
  // create new tasks
  if (task_finished && start_idx < walker_num) {
    start_point = start_points[start_idx];
    result_pool[(u64)start_idx * max_depth] = start_point;
    task = TaskType(graph, start_point, start_idx);

    if constexpr (update_flag == 1) max_weight = walker->get_max_weight(&task, 0);

    task_finished = false;
  }
  __syncwarp();
  unsigned int mask = __ballot_sync(FULL_WARP_MASK, start_idx < walker_num); // mask for active threads

  // if update_flag == 0, requires only single max_weight update

  use_res = start_idx < walker_num ? pick_res<walker_t>(walker, &task, max_weight, select_algo_cost_ratio): false;
  unsigned int res_mask;
  int reject_counter = 0;
  vtx_t degree;

  // for (; chunk_idx < walker_num; ) {
  while (mask) { // mask represents active tasks
    __syncwarp();

    // check for reservior sampling cases
    res_mask = __ballot_sync(FULL_WARP_MASK, use_res) & mask; // mask for threads with reservior sampling

    // Warp-wise Reservior sampling
    while (res_mask) {
      target_tid = __ffs(res_mask) - 1;
      walker->fill_dummy(&task, &dummy_task, lid, target_tid);
      dummy_selected = sampler_warp_exp_jump_stable<walker_t, myrandStateArr>(walker, &dummy_task, &state);

      if (lid == target_tid) {
        selected = dummy_selected;
      }

      res_mask &= (res_mask - 1);
      __syncwarp();
    }

    if (!use_res) {
      // do rejection sampling for REJ_SYNC times
      if (!task_finished) {
        selected = -1;
        if constexpr (possible_zero) {
          // if possible_zero, do a semi-scan to check if a non-zero edge exists
          if (!continue_rjs) {
            continue_rjs = walker->scan_thread(&task) && max_weight > 0;
            degree = task.degree;
          }
          if (continue_rjs) {
            // do rejection sampling
            selected = sampler_rjs_sample_only_select_algo<walker_t, myrandStateArr>(walker, &task, &state, max_weight, degree, reject_counter, continue_rjs);
          }
        } else {
          // do rejection sampling
          if (!continue_rjs) {
            continue_rjs = max_weight > 0;
            degree = task.degree;
          }
          if (continue_rjs) selected = sampler_rjs_sample_only_select_algo<walker_t, myrandStateArr>(walker, &task, &state, max_weight, degree, reject_counter, continue_rjs);
        }
      }
    }


    // Thread-wise Sample
    // if (!use_res && !(task_finished)) { // while the current task is not finished
    if (!continue_rjs && !task_finished) {
      // finished traversal, so update task
      if (selected == -1) {
        task_finished = true;
      } else {
        selected = graph->adjncy[task.neighbor_offset + selected];
        result_pool[(u64)(task.walker_id) * max_depth + task.length] = selected;
        task.update(graph, selected);
        if (walker->is_stop(task.length, &state)) {
          task_finished = true;
        } else {
          task_finished = false;
          if constexpr (update_flag == 1) max_weight = walker->get_max_weight(&task, 0);
          use_res = pick_res<walker_t>(walker, &task, max_weight, select_algo_cost_ratio);
        }
      }
    }

    // finished a task (no chunk applied)
    if (task_finished && start_idx < walker_num) start_idx = atomicAdd(global_chunk_idx, 1);

    // create new tasks
    if (task_finished && start_idx < walker_num) {
      start_point = start_points[start_idx];
      result_pool[(u64)start_idx * max_depth] = start_point;
      task = TaskType(graph, start_point, start_idx);

      task_finished = false;
      if constexpr (update_flag == 1) max_weight = walker->get_max_weight(&task, 0);
      use_res = pick_res<walker_t>(walker, &task, max_weight, select_algo_cost_ratio);
    }

    __syncwarp();

    mask = __ballot_sync(FULL_WARP_MASK, start_idx < walker_num);

  }

}


// note that the pointers are already calculated when being fed
template <typename walker_t, typename state_t, typename TaskType = typename walker_t::TaskType> 
__device__ __forceinline__ void update_task(vtx_t selected, gpu_graph* graph, walker_t* walker, state_t* state, int max_depth,
                                            TaskType* task, bool* task_finished, int lid, vtx_t* result_pool) {
  if (lid == 0) {
    if (selected == -1) {
      // the current task is finished
      *task_finished = true;
    } else {
      selected = graph->adjncy[task->neighbor_offset + selected]; 
      result_pool[(u64)(task->walker_id) * max_depth + task->length] = selected; 
      task->update(graph, selected); 

      if (walker->is_stop(task->length, state)) {
        // the current task is finished
        *task_finished = true;
      } else {              
        *task_finished = false; 
      }
    }
  }

  __syncwarp();
  
}

// only requires algo and jump as template parameters
template <typename walker_t, typename TaskType = typename walker_t::TaskType>
__global__ void walker_ervs_only(walker_t* walker, vtx_t* start_points,
                              int* start_pointer, vtx_t* result_pool,
                              int walker_num, int* global_chunk_idx) {
                                      
  int lid = threadIdx.x % WARP_SIZE;
  int gid = (blockDim.x * blockIdx.x) + threadIdx.x;
  int wid = threadIdx.x / WARP_SIZE;

  gpu_graph* graph = walker->graph;
  int max_depth = walker->max_depth;

  vtx_t start_point, selected;
  int start_idx;

  // initialize random
  __shared__ myrandStateArr state;
  myrand_init(1337, gid, 0, &state);

  // initalize shared memory 
  extern __shared__ unsigned char shared_mem[];
  TaskType* task = (TaskType*)(shared_mem);
  bool* task_finished = (bool*)(task + (blockDim.x / WARP_SIZE));

  // initialize chunk_idx
  __syncwarp();
  if (lid == 0) {
    start_idx = atomicAdd(global_chunk_idx, CHUNK_SIZE);
  }
  start_idx = __shfl_sync(FULL_WARP_MASK, start_idx, 0);

  while(start_idx < walker_num) {
  
    if (lid == 0) {
      // create new task and put it in shared memory
      start_point = start_points[start_idx];
      result_pool[(u64)start_idx * max_depth] = start_point;
      task[wid] = TaskType(graph, start_point, start_idx);

      task_finished[wid] = false;
    } 
    __syncwarp();

    while(!(task_finished[wid])) { // while the current task is not finished
      selected = sampler_warp_exp_jump_stable<walker_t, myrandStateArr>(walker, &task[wid], &state);
      // finished traversal, so update task
      update_task<walker_t, myrandStateArr>(selected, graph, walker, &state, max_depth, task + wid, task_finished + wid, lid, result_pool);
    }

      // finished a task 
    __syncwarp();
    if (lid == 0) {
      start_idx = atomicAdd(global_chunk_idx, CHUNK_SIZE);
    }
    start_idx = __shfl_sync(FULL_WARP_MASK, start_idx, 0);

  }


}

/*
* Kernels for Profiling
*/

// Profile Kernel For Reservior Sampling
template <typename walker_t, typename TaskType = typename walker_t::TaskType>
__global__ void walker_async_chunk_profile(walker_t* walker, unsigned int* total_edges) {

  int lid = threadIdx.x % WARP_SIZE;
  int gid = (blockDim.x * blockIdx.x) + threadIdx.x;
  int wid = threadIdx.x / WARP_SIZE;
  int g_wid = gid / WARP_SIZE;
  int warp_per_kernel = (gridDim.x * blockDim.x) / WARP_SIZE;

  gpu_graph* graph = walker->graph;
  int max_depth = walker->max_depth;
  vtx_t walker_num = graph->vtx_num;
  vtx_t profile_num = walker_num / PROFILE_NODE_RATIO;
  int stride = (walker_num / profile_num)? (walker_num / profile_num): 1;

  vtx_t start_point, prev_point, selected;
  int start_idx;

  // initialize random
  __shared__ myrandStateArr state;
  myrand_init(1337, gid, 0, &state);

  // initalize shared memory
  extern __shared__ unsigned char shared_mem[];
  TaskType* task = (TaskType*)(shared_mem);
  bool* task_finished = (bool*)(task + (blockDim.x / WARP_SIZE));

  for (int i = g_wid; i < profile_num; i += warp_per_kernel) {
    if (i < walker_num) {
      prev_point = stride * i; // select an strided previous point
      start_point = graph->adjncy[graph->xadj[prev_point]]; // select the first neighbor as the starting point
      start_idx = i;

      assert(prev_point < walker_num);

      if (lid == 0) {
        task[wid] = TaskType(graph, prev_point, start_idx); // create new task and put it in shared memory
        task[wid].update(graph, start_point); // update it such that it isn't affected by the task being length 1
      }
      __syncwarp();

      // run actual kernel
      selected = sampler_warp_exp_jump_stable_profile<walker_t, myrandStateArr>(walker, &task[wid], &state);

      // finished a task
      __syncwarp();
      if (lid == 0) {
        atomicAdd(total_edges, PROFILE_EDGE_LIMIT);
      }
      __syncwarp();
    }
  }
}

// Profile Kernel For Rejection Sampling
template <typename walker_t, char update_flag, bool possible_zero, typename TaskType = typename walker_t::TaskType>
__global__ void walker_scattered_thread_profile(walker_t* walker, unsigned int* total_edges) {

  int lid = threadIdx.x % WARP_SIZE;
  int gid = (blockDim.x * blockIdx.x) + threadIdx.x;
  int g_wid = gid / WARP_SIZE;
  int warp_per_kernel = (gridDim.x * blockDim.x) / WARP_SIZE;

  gpu_graph* graph = walker->graph;
  vtx_t walker_num = graph->vtx_num;
  vtx_t profile_num = walker_num / PROFILE_NODE_RATIO * REJ_WORKLOAD_MULTIPLIER;
  int stride = (walker_num / profile_num)? (walker_num / profile_num): 1;

  vtx_t start_point, prev_point;
  int start_idx;
  unsigned int thread_total_edges = 0;

  // initialize random
  __shared__ myrandStateArr state;
  myrand_init(1337, gid, 0, &state);

  // initalize shared memory
  TaskType task;

  weight_t max_weight;
  // if update_flag == 0, requires only single max_weight update
  if constexpr (update_flag == 0) max_weight = walker->get_max_weight(&task, 0);

  // for (int i = gid; i < thread_per_kernel; i += thread_per_kernel) {
  for (int i = g_wid; i < profile_num; i += warp_per_kernel) {
    if (i < walker_num && lid == 0) {
      prev_point = stride * i; // select an strided previous point
      start_point = graph->adjncy[graph->xadj[prev_point]]; // select the first neighbor as the starting point
      start_idx = i;

      assert(prev_point < walker_num);

      task = TaskType(graph, prev_point, start_idx); // create new task and put it in shared memory
      task.update(graph, start_point); // update it such that it isn't affected by the task being length 1

      if constexpr (update_flag == 1) max_weight = walker->get_max_weight(&task, 0);

      if constexpr (possible_zero) {
        // if possible_zero, do a semi-scan to check if a non-zero edge exists
        if (walker->scan_thread(&task)) thread_total_edges = sampler_rjs_sample_only_profile<walker_t, myrandStateArr>(walker, &task, &state, max_weight);
        else thread_total_edges = 0;
      } else {
        thread_total_edges = sampler_rjs_sample_only_profile<walker_t, myrandStateArr>(walker, &task, &state, max_weight);
      }

      atomicAdd(total_edges, thread_total_edges);
    }
  }

}

