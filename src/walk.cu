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
#include <gflags/gflags.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include <fstream>
#include <iostream>
#include <map>
#include <random>

#include <cub/cub.cuh>  // NOLINT

#include "app.cuh"
#include "gpu_task.cuh"
#include "util.cuh"
#include "walker.cuh"

#include "generated/gpu_graph.cuh"
#include "generated/walker_traits.cuh"

DECLARE_double(tp);
DECLARE_double(p);
DECLARE_double(q);
DECLARE_double(alpha);
DECLARE_bool(Deepwalk);
DECLARE_bool(PPR);
DECLARE_bool(Node2vec);
DECLARE_bool(Node2vec_weighted);
DECLARE_bool(Metapath);
DECLARE_bool(Metapath_weighted);
DECLARE_bool(PPR_second);
DECLARE_bool(syn);
DECLARE_int32(block_multi);

// Profiling function for kernels
template <typename walker_t, typename TaskType = typename walker_t::TaskType>
weight_t profile_kernels(walker_t* walker_ptr) {

  if (WalkerTraits<walker_t>::ERVS_ONLY) {
    // Skip profiling step
    return -1.0;
  }

  int block_num = get_block_num(static_cast<int>(FLAGS_block_multi));
  
  weight_t select_algo_cost_ratio;

  // Profile
  unsigned int* reservior_total_edges = get_device_ptr<unsigned int>(1, 0);
  unsigned int* reject_total_edges = get_device_ptr<unsigned int>(1, 0);
  unsigned int reservior_total_edges_h = 0, reject_total_edges_h = 0;
  int profile_block = PROFILE_BLOCK;
  float reservior_total_time = 0.0f, reject_total_time = 0.0f;

  cudaEvent_t start, stop;

  CUDA_RT_CALL(cudaEventCreate(&start));
  CUDA_RT_CALL(cudaEventCreate(&stop));

  // Warmup for profiling
  cudaFree(0);  // Initializes the CUDA context (no-op that forces setup)
  warmup_kernel<<<1, 1>>>();
  cudaDeviceSynchronize();

  CUDA_RT_CALL(cudaEventRecord(start));
  walker_async_chunk_profile<walker_t><<<profile_block, BLOCK_SIZE, BLOCK_SIZE/WARP_SIZE*(sizeof(TaskType)+sizeof(bool))>>>
  (walker_ptr, reservior_total_edges);
  CUDA_RT_CALL(cudaEventRecord(stop));
  CUDA_RT_CALL(cudaEventSynchronize(stop));
  CUDA_RT_CALL(cudaEventElapsedTime(&reservior_total_time, start, stop));

  CUDA_RT_CALL(cudaEventRecord(start));
  walker_scattered_thread_profile<walker_t,
    WalkerTraits<walker_t>::UPDATE_FLAG,
    WalkerTraits<walker_t>::POSSIBLE_ZERO>
    <<<profile_block, BLOCK_SIZE, 0>>>
  (walker_ptr, reject_total_edges);
  CUDA_RT_CALL(cudaEventRecord(stop));
  CUDA_RT_CALL(cudaEventSynchronize(stop));
  CUDA_RT_CALL(cudaEventElapsedTime(&reject_total_time, start, stop));

  CUDA_RT_CALL(cudaEventDestroy(start));
  CUDA_RT_CALL(cudaEventDestroy(stop));

  CUDA_RT_CALL(cudaMemcpy(&reservior_total_edges_h, reservior_total_edges, sizeof(unsigned int), cudaMemcpyDeviceToHost));
  CUDA_RT_CALL(cudaMemcpy(&reject_total_edges_h, reject_total_edges, sizeof(unsigned int), cudaMemcpyDeviceToHost));

  select_algo_cost_ratio = (reject_total_time/reject_total_edges_h) / ((reservior_total_time / reservior_total_edges_h));

  printf("[PROFILE] Reservior: %f ms w. %u edges %f ms/edge \n", reservior_total_time, reservior_total_edges_h, (reservior_total_time) / reservior_total_edges_h );
  printf("[PROFILE] Rejection: %f ms w. %u edges %f ms/edge \n", reject_total_time, reject_total_edges_h, (reject_total_time) / reject_total_edges_h );
  printf("[PROFILE] Final select_algo cost ratio: %f\n", select_algo_cost_ratio);

  return select_algo_cost_ratio;
}

// Helper function to profile the current walker type
weight_t profile_current_walker(gpu_graph* graph_ptr, int max_depth,
                                int* schema, int schema_len) {
  weight_t ratio = 0;

  if (FLAGS_Deepwalk) {
    Deepwalk* walker = new Deepwalk(graph_ptr, max_depth);
    Deepwalk* walker_ptr = get_device_ptr(walker, 1);
    ratio = profile_kernels<Deepwalk>(walker_ptr);
  } else if (FLAGS_PPR) {
    PPR* walker = new PPR(graph_ptr, max_depth, FLAGS_tp);
    PPR* walker_ptr = get_device_ptr<PPR>(walker, 1);
    ratio = profile_kernels<PPR>(walker_ptr);
  } else if (FLAGS_Node2vec) {
    Node2vec* walker = new Node2vec(graph_ptr, max_depth, FLAGS_p, FLAGS_q);
    Node2vec* walker_ptr = get_device_ptr<Node2vec>(walker, 1);
    ratio = profile_kernels<Node2vec>(walker_ptr);
  } else if (FLAGS_Node2vec_weighted) {
    Node2vec_weighted* walker = new Node2vec_weighted(graph_ptr, max_depth, FLAGS_p, FLAGS_q);
    Node2vec_weighted* walker_ptr = get_device_ptr<Node2vec_weighted>(walker, 1);
    ratio = profile_kernels<Node2vec_weighted>(walker_ptr);
  } else if (FLAGS_Metapath) {
    int* schema_ptr = get_device_ptr<int>(schema, schema_len);
    Metapath* walker = new Metapath(graph_ptr, max_depth, schema_ptr, schema_len);
    Metapath* walker_ptr = get_device_ptr<Metapath>(walker, 1);
    ratio = profile_kernels<Metapath>(walker_ptr);
  } else if (FLAGS_Metapath_weighted) {
    int* schema_ptr = get_device_ptr<int>(schema, schema_len);
    Metapath_weighted* walker = new Metapath_weighted(graph_ptr, max_depth, schema_ptr, schema_len);
    Metapath_weighted* walker_ptr = get_device_ptr<Metapath_weighted>(walker, 1);
    ratio = profile_kernels<Metapath_weighted>(walker_ptr);
  } else if (FLAGS_PPR_second) {
    PPR_second* walker = new PPR_second(graph_ptr, max_depth, FLAGS_alpha);
    PPR_second* walker_ptr = get_device_ptr<PPR_second>(walker, 1);
    ratio = profile_kernels<PPR_second>(walker_ptr);
  }

  cudaDeviceSynchronize();
  return ratio;
}

__global__ void init_state(myrandStateArr* state) {
  //  init random number generator state
  int gid = (blockDim.x * blockIdx.x) + threadIdx.x;
  myrand_init(1337, gid, 0, state + blockIdx.x);
}

myrandStateArr* get_mystate(int num_blocks) {
  myrandStateArr* state_ptr;
  CUDA_RT_CALL(cudaMalloc(&state_ptr, num_blocks * sizeof(myrandStateArr)));
  init_state<<<num_blocks, BLOCK_SIZE>>>(state_ptr);
  return state_ptr;
}

template <typename walker_t, typename TaskType = typename walker_t::TaskType>
double timing(walker_t* walker_ptr, vtx_t* start_points_ptr,
              vtx_t* result_pool_ptr, int block_num, int num_walkers,
              int max_depth, weight_t select_algo_cost_ratio) {
  printf("========start timing\n");
  // Use the passed select_algo_cost_ratio (already computed once on GPU 0)
  if (select_algo_cost_ratio > -1.0) printf("[TIMING] Using select_algo cost ratio: %f\n", select_algo_cost_ratio);
  else printf("[TIMING] Running in eRVS_only mode.\n");

  double start_time, total_time;

  int* start_pointer = get_device_ptr<int>(1, 0);
  int* global_chunk_idx = get_device_ptr<int>(1, 0);
  start_time = wtime();
  if (WalkerTraits<walker_t>::ERVS_ONLY) {
    walker_ervs_only<walker_t>
    <<<block_num, BLOCK_SIZE, BLOCK_SIZE/WARP_SIZE*(sizeof(TaskType)+sizeof(bool))>>>
    (walker_ptr, start_points_ptr, start_pointer, result_pool_ptr,
      num_walkers, global_chunk_idx);
  } else {
    walker_select_algo<walker_t,
    WalkerTraits<walker_t>::UPDATE_FLAG,
    WalkerTraits<walker_t>::POSSIBLE_ZERO>
    <<<block_num, BLOCK_SIZE, BLOCK_SIZE/WARP_SIZE*sizeof(int)>>>
    (walker_ptr, start_points_ptr, start_pointer, result_pool_ptr,
      num_walkers, global_chunk_idx, select_algo_cost_ratio);
  }
  CUDA_RT_CALL(cudaDeviceSynchronize());
  total_time = wtime() - start_time;

  return total_time * 1000;
}

// #define TEST_RJS

template <typename walker_t, typename TaskType = typename walker_t::TaskType>
double timing_batch_async(walker_t* walker_ptr, vtx_t* start_points,
                          vtx_t* result_pool, int batch_size, int num_walkers,
                          int max_depth, int block_num, weight_t select_algo_cost_ratio) {
  cudaStream_t* streams = new cudaStream_t[2];
  for (int i = 0; i < 2; i++) {
    CUDA_RT_CALL(cudaStreamCreate(&streams[i]));
  }
  vtx_t* start_points_ptr = get_device_ptr<vtx_t>((u64)batch_size * 2, 0);
  vtx_t* result_pool_ptr =
      get_device_ptr<vtx_t>((u64)batch_size * max_depth * 2, -1);

  int* start_pointer = get_device_ptr<int>(2, 0);
  int* global_chunk_idx = get_device_ptr<int>(2, 0);

  // Use the passed select_algo_cost_ratio (already computed once on GPU 0)
  if (select_algo_cost_ratio > -1.0) printf("[TIMING_ASYNC] Using select_algo cost ratio: %f\n", select_algo_cost_ratio);
  else printf("[TIMING_ASYNC] Running in eRVS_only mode.\n");

  printf("========start timing (async)\n");
  double start_time, total_time;
  start_time = wtime();

  for (int i = 0; i < num_walkers; i += batch_size * 2) {
    int j = i + batch_size;
    int batch_num1 = min(batch_size, num_walkers - i);
    int batch_num2 = min(batch_size, num_walkers - j);

    CUDA_RT_CALL(cudaMemcpyAsync(start_points_ptr, start_points + i,
                                 (u64)sizeof(vtx_t) * batch_num1,
                                 cudaMemcpyHostToDevice, streams[0]));
    CUDA_RT_CALL(cudaMemsetAsync(result_pool_ptr, -1,
                                 (u64)sizeof(vtx_t) * batch_num1 * max_depth,
                                 streams[0]));
    CUDA_RT_CALL(cudaMemsetAsync(start_pointer, 0, sizeof(int), streams[0]));
    CUDA_RT_CALL(cudaMemsetAsync(global_chunk_idx, 0, sizeof(int), streams[0]));

    // FlexiWalker with select_algo
    if (WalkerTraits<walker_t>::ERVS_ONLY) {
      walker_ervs_only<walker_t>
      <<<block_num, BLOCK_SIZE, BLOCK_SIZE/WARP_SIZE*(sizeof(TaskType)+sizeof(bool)), streams[0]>>>
      (walker_ptr, start_points_ptr, start_pointer, result_pool_ptr,
        batch_num1, global_chunk_idx);
    } else {
      walker_select_algo<walker_t,
        WalkerTraits<walker_t>::UPDATE_FLAG,
        WalkerTraits<walker_t>::POSSIBLE_ZERO>
        <<<block_num, BLOCK_SIZE, BLOCK_SIZE/WARP_SIZE*sizeof(int), streams[0]>>>
        (walker_ptr, start_points_ptr, start_pointer, result_pool_ptr,
          batch_num1, global_chunk_idx, select_algo_cost_ratio);
    }

    CUDA_RT_CALL(cudaMemcpyAsync(result_pool + (u64)i * max_depth,
                                 result_pool_ptr,
                                 (u64)sizeof(vtx_t) * batch_num1 * max_depth,
                                 cudaMemcpyDeviceToHost, streams[0]));

    if (batch_num2 > 0) {
      vtx_t* b_start_points_ptr = start_points_ptr + batch_size;
      vtx_t* b_result_pool_ptr = result_pool_ptr + (u64)batch_size * max_depth;

      CUDA_RT_CALL(cudaMemcpyAsync(b_start_points_ptr, start_points + j,
                                   (u64)sizeof(vtx_t) * batch_num2,
                                   cudaMemcpyHostToDevice, streams[1]));
      CUDA_RT_CALL(cudaMemsetAsync(b_result_pool_ptr, -1,
                                   (u64)sizeof(vtx_t) * batch_num2 * max_depth,
                                   streams[1]));
      CUDA_RT_CALL(
          cudaMemsetAsync(start_pointer + 1, 0, sizeof(int), streams[1]));
      CUDA_RT_CALL(cudaMemsetAsync(global_chunk_idx + 1, 0, sizeof(int), streams[1]));

      // FlexiWalker with select_algo
      if (WalkerTraits<walker_t>::ERVS_ONLY) {
        walker_ervs_only<walker_t>
          <<<block_num, BLOCK_SIZE, BLOCK_SIZE/WARP_SIZE*(sizeof(TaskType)+sizeof(bool)), streams[1]>>>
          (walker_ptr, b_start_points_ptr, start_pointer + 1, b_result_pool_ptr,
            batch_num2, global_chunk_idx + 1);
      } else {
        walker_select_algo<walker_t,
          WalkerTraits<walker_t>::UPDATE_FLAG,
          WalkerTraits<walker_t>::POSSIBLE_ZERO>
          <<<block_num, BLOCK_SIZE, BLOCK_SIZE/WARP_SIZE*sizeof(int), streams[1]>>>
          (walker_ptr, b_start_points_ptr, start_pointer + 1, b_result_pool_ptr,
            batch_num2, global_chunk_idx + 1, select_algo_cost_ratio);
      }
      CUDA_RT_CALL(cudaMemcpyAsync(result_pool + (u64)j * max_depth,
                                   b_result_pool_ptr,
                                   (u64)sizeof(vtx_t) * batch_num2 * max_depth,
                                   cudaMemcpyDeviceToHost, streams[1]));
    }
  }
  for (int i = 0; i < 2; i++) {
    CUDA_RT_CALL(cudaStreamSynchronize(streams[i]));
  }
  total_time = wtime() - start_time;

  for (int i = 0; i < 2; i++) {
    CUDA_RT_CALL(cudaStreamDestroy(streams[i]));
  }

  return total_time * 1000;
}

template <typename walker_t, typename TaskType = typename walker_t::TaskType>
double timing_batch_sync(walker_t* walker_ptr, vtx_t* start_points,
                         vtx_t* result_pool, int batch_size, int num_walkers,
                         int max_depth, int block_num, weight_t select_algo_cost_ratio) {
  // Use the passed select_algo_cost_ratio (already computed once on GPU 0)
  if (select_algo_cost_ratio > -1.0) printf("[TIMING_SYNC] Using select_algo cost ratio: %f\n", select_algo_cost_ratio);
  else printf("[TIMING_SYNC] Running in eRVS_only mode.\n");

  int stream_num = (num_walkers % batch_size == 0)
                       ? num_walkers / batch_size
                       : num_walkers / batch_size + 1;
  vtx_t* start_points_ptr = get_device_ptr<vtx_t>(batch_size, 0);
  vtx_t* result_pool_ptr =
      get_device_ptr<vtx_t>((u64)batch_size * max_depth, -1);

  int* start_pointer = get_device_ptr<int>(1, 0);
  int* global_chunk_idx = get_device_ptr<int>(1, 0);

  printf("========start timing (sync)\n");
  double start_time, compute_time, total_time, transfer_s, transfer_time;

  compute_time = 0;
  transfer_time = 0;

  total_time = wtime();
  for (int i = 0; i < stream_num; i++) {
    int batch_num = min(batch_size, num_walkers - i * batch_size);
    CUDA_RT_CALL(cudaMemcpy(start_points_ptr,
                            start_points + (u64)i * batch_size,
                            sizeof(vtx_t) * batch_num, cudaMemcpyHostToDevice));
    CUDA_RT_CALL(cudaMemset(result_pool_ptr, -1,
                            (u64)sizeof(vtx_t) * batch_num * max_depth));
    CUDA_RT_CALL(cudaMemset(start_pointer, 0, sizeof(int)));
    CUDA_RT_CALL(cudaMemset(global_chunk_idx, 0, sizeof(int)));

    start_time = wtime();
    // FlexiWalker with select_algo
    if (WalkerTraits<walker_t>::ERVS_ONLY) {
      walker_ervs_only<walker_t>
        <<<block_num, BLOCK_SIZE, BLOCK_SIZE/WARP_SIZE*(sizeof(TaskType)+sizeof(bool))>>>
        (walker_ptr, start_points_ptr, start_pointer, result_pool_ptr,
          batch_num, global_chunk_idx);
    } else {
      walker_select_algo<walker_t,
        WalkerTraits<walker_t>::UPDATE_FLAG,
        WalkerTraits<walker_t>::POSSIBLE_ZERO>
        <<<block_num, BLOCK_SIZE, BLOCK_SIZE/WARP_SIZE*sizeof(int)>>>
        (walker_ptr, start_points_ptr, start_pointer, result_pool_ptr,
          batch_num, global_chunk_idx, select_algo_cost_ratio);
    }

    CUDA_RT_CALL(cudaDeviceSynchronize());
    compute_time += wtime() - start_time;

    transfer_s = wtime();
    CUDA_RT_CALL(cudaMemcpy(
        result_pool + (u64)i * batch_size * max_depth, result_pool_ptr,
        (u64)sizeof(vtx_t) * batch_num * max_depth, cudaMemcpyDeviceToHost));
    transfer_time += wtime() - transfer_s;
  }
  CUDA_RT_CALL(cudaDeviceSynchronize());
  total_time = wtime() - total_time;

  printf("compute time=%.6f ms, transfer time=%.6f ms\n", compute_time * 1000,
         transfer_time * 1000);
  return total_time * 1000;
}

// ================================================================
// Template helpers for walker instantiation
// ================================================================
// These templates eliminate boilerplate when adding new walkers.
// Defined here after all timing functions they depend on.

// For batch mode (walk_batch)
template<typename WalkerType, typename... Args>
double run_walker(gpu_graph* graph_ptr, vtx_t* start_points, vtx_t* result_pool,
                  int batch_size, int num_walkers, int max_depth, int block_num,
                  weight_t select_algo_cost_ratio, Args... args) {
  WalkerType* walker = new WalkerType(graph_ptr, max_depth, args...);
  WalkerType* walker_ptr = get_device_ptr<WalkerType>(walker, 1);

  if (FLAGS_syn)
    return timing_batch_sync(walker_ptr, start_points, result_pool, batch_size,
                            num_walkers, max_depth, block_num, select_algo_cost_ratio);
  else
    return timing_batch_async(walker_ptr, start_points, result_pool, batch_size,
                             num_walkers, max_depth, block_num, select_algo_cost_ratio);
}

// For test mode (walk_test)
template<typename WalkerType, typename... Args>
double run_walker_test(gpu_graph* graph_ptr, vtx_t* start_points_ptr, vtx_t* result_pool_ptr,
                       int block_num, int num_walkers, int max_depth,
                       weight_t select_algo_cost_ratio, Args... args) {
  WalkerType* walker = new WalkerType(graph_ptr, max_depth, args...);
  WalkerType* walker_ptr = get_device_ptr<WalkerType>(walker, 1);
  return timing<WalkerType>(walker_ptr, start_points_ptr, result_pool_ptr,
                           block_num, num_walkers, max_depth, select_algo_cost_ratio);
}

double walk_test(vtx_t*& result_pool_ptr, gpu_graph* graph,
                 vtx_t* start_points,  // NOLINT
                 int max_depth, int num_walkers, weight_t select_algo_cost_ratio,
                 int* schema,
                 int schema_len) {
  LOG("%s\n", __FUNCTION__);
  double total_time;

  gpu_graph* graph_ptr = get_device_ptr<gpu_graph>(graph, 1);
  vtx_t* start_points_ptr = get_device_ptr<vtx_t>(start_points, num_walkers);
  result_pool_ptr = get_device_ptr<vtx_t>((u64)num_walkers * max_depth, -1);

  // int sm_count = get_block_num(1);
  // int block_num = get_block_num(num_walkers);
  int block_num = get_block_num() * 2;

  // Use template helper for clean walker instantiation
  if (FLAGS_Deepwalk) {
    total_time = run_walker_test<Deepwalk>(graph_ptr, start_points_ptr, result_pool_ptr,
                                           block_num, num_walkers, max_depth,
                                           select_algo_cost_ratio);
  } else if (FLAGS_PPR) {
    total_time = run_walker_test<PPR>(graph_ptr, start_points_ptr, result_pool_ptr,
                                      block_num, num_walkers, max_depth,
                                      select_algo_cost_ratio, FLAGS_tp);
  } else if (FLAGS_Node2vec) {
    total_time = run_walker_test<Node2vec>(graph_ptr, start_points_ptr, result_pool_ptr,
                                           block_num, num_walkers, max_depth,
                                           select_algo_cost_ratio, FLAGS_p, FLAGS_q);
  } else if (FLAGS_Node2vec_weighted) {
    total_time = run_walker_test<Node2vec_weighted>(graph_ptr, start_points_ptr, result_pool_ptr,
                                                    block_num, num_walkers, max_depth,
                                                    select_algo_cost_ratio, FLAGS_p, FLAGS_q);
  } else if (FLAGS_Metapath) {
    int* schema_ptr = get_device_ptr<int>(schema, schema_len);
    total_time = run_walker_test<Metapath>(graph_ptr, start_points_ptr, result_pool_ptr,
                                           block_num, num_walkers, max_depth,
                                           select_algo_cost_ratio, schema_ptr, schema_len);
  } else if (FLAGS_Metapath_weighted) {
    int* schema_ptr = get_device_ptr<int>(schema, schema_len);
    total_time = run_walker_test<Metapath_weighted>(graph_ptr, start_points_ptr, result_pool_ptr,
                                                    block_num, num_walkers, max_depth,
                                                    select_algo_cost_ratio, schema_ptr, schema_len);
  } else if (FLAGS_PPR_second) {
    total_time = run_walker_test<PPR_second>(graph_ptr, start_points_ptr, result_pool_ptr,
                                             block_num, num_walkers, max_depth,
                                             select_algo_cost_ratio, FLAGS_alpha);
  } else {
    printf("Please choose a walk mode\n");
    exit(0);
  }
  CUDA_RT_CALL(cudaDeviceSynchronize());

  LOG("grid:%d,block:%d,sampling time:%.6f ms\n", block_num, BLOCK_SIZE,
      total_time);

  return total_time;
}

double walk_batch(vtx_t*& result_pool, gpu_graph* graph,
                  vtx_t* start_points,  // NOLINT
                  int max_depth, int num_walkers, int batch_size, weight_t select_algo_cost_ratio,
                  int* schema, int schema_len) {
  LOG("%s\n", __FUNCTION__);
  double total_time;

  gpu_graph* graph_ptr = get_device_ptr<gpu_graph>(graph, 1);

  // vtx_t *result_pool;
  cudaMallocHost(&result_pool, (u64)num_walkers * max_depth * sizeof(vtx_t));

  // int sm_count = get_block_num(1);
  // int block_num = get_block_num(min(batch_size, num_walkers));
  int block_num = get_block_num(static_cast<int>(FLAGS_block_multi));

  // Use template helper for clean walker instantiation
  if (FLAGS_Deepwalk) {
    total_time = run_walker<Deepwalk>(graph_ptr, start_points, result_pool,
                                      batch_size, num_walkers, max_depth, block_num,
                                      select_algo_cost_ratio);
  } else if (FLAGS_PPR) {
    total_time = run_walker<PPR>(graph_ptr, start_points, result_pool,
                                 batch_size, num_walkers, max_depth, block_num,
                                 select_algo_cost_ratio, FLAGS_tp);
  } else if (FLAGS_Node2vec) {
    total_time = run_walker<Node2vec>(graph_ptr, start_points, result_pool,
                                      batch_size, num_walkers, max_depth, block_num,
                                      select_algo_cost_ratio, FLAGS_p, FLAGS_q);
  } else if (FLAGS_Node2vec_weighted) {
    total_time = run_walker<Node2vec_weighted>(graph_ptr, start_points, result_pool,
                                               batch_size, num_walkers, max_depth, block_num,
                                               select_algo_cost_ratio, FLAGS_p, FLAGS_q);
  } else if (FLAGS_Metapath) {
    int* schema_ptr = get_device_ptr<int>(schema, schema_len);
    total_time = run_walker<Metapath>(graph_ptr, start_points, result_pool,
                                      batch_size, num_walkers, max_depth, block_num,
                                      select_algo_cost_ratio, schema_ptr, schema_len);
  } else if (FLAGS_Metapath_weighted) {
    int* schema_ptr = get_device_ptr<int>(schema, schema_len);
    total_time = run_walker<Metapath_weighted>(graph_ptr, start_points, result_pool,
                                               batch_size, num_walkers, max_depth, block_num,
                                               select_algo_cost_ratio, schema_ptr, schema_len);
  } else if (FLAGS_PPR_second) {
    total_time = run_walker<PPR_second>(graph_ptr, start_points, result_pool,
                                        batch_size, num_walkers, max_depth, block_num,
                                        select_algo_cost_ratio, FLAGS_alpha);
  } else {
    printf("Please choose a walk mode\n");
    exit(0);
  }

  CUDA_RT_CALL(cudaDeviceSynchronize());

  LOG("grid:%d,block:%d,sampling time:%.6f ms\n", block_num, BLOCK_SIZE,
      total_time);

  return total_time;
}
