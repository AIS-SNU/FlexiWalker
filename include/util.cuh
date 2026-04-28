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

#include <assert.h>
#include <cuda.h>
#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

#include <algorithm>
#include <cstdint>
#include <iostream>
#include <set>
#include <tuple>
#include <utility>

using u64 = unsigned long long;  // NOLINT
// CUDA API uses 'unsigned long long'
using ll = int64_t;
using uint = unsigned int;
using vtx_t = int;
using edge_t = unsigned int;
using weight_t = float;

#define TID (threadIdx.x + blockIdx.x * blockDim.x)
#define LTID (threadIdx.x)
#define BID (blockIdx.x)
#define LID (threadIdx.x % 32)
#define WID (threadIdx.x / 32)
#define GWID (TID / 32)
#define MIN(x, y) ((x < y) ? x : y)
#define MAX(x, y) ((x > y) ? x : y)

#define WARP_SIZE 32
#define BLOCK_SIZE 512
#define WARP_PER_BLK (BLOCK_SIZE / 32)
#define FULL_WARP_MASK 0xffffffff

// SY
#define SMALL_BUFFER_SIZE 32
#define LARGE_BUFFER_SIZE 32
#define BUFFER_NUM 10
#define TASK_ELEM 6
#define SUBWARP_SIZE 4
#define UPDATE 32
#define PRERAND 2
#define REJ_SYNC 8
#define PROFILE_BLOCK block_num
#define PROFILE_NODE_RATIO 256
#define PROFILE_EDGE_LIMIT 8192
#define PROFILE_EDGE_LIMIT_REJ 256
#define REJ_WORKLOAD_MULTIPLIER 1

#define CUDA_RT_CALL(call)                                               \
  {                                                                      \
    cudaError_t cudaStatus = call;                                       \
    if (cudaSuccess != cudaStatus) {                                     \
      fprintf(stderr,                                                    \
              "%s:%d ERROR: CUDA RT call \"%s\" failed "                 \
              "with "                                                    \
              "%s (%d).\n",                                              \
              __FILE__, __LINE__, #call, cudaGetErrorString(cudaStatus), \
              cudaStatus);                                               \
      exit(cudaStatus);                                                  \
    }                                                                    \
  }

#define H_ERR(ans) \
  { gpuAssert((ans), __FILE__, __LINE__); }

namespace print {
template <typename... Args>
__host__ __device__ __forceinline__ void myprintf(const char* file, int line,
                                                  const char* __format,
                                                  Args... args) {
#if defined(__CUDA_ARCH__)
  // if (LID == 0)
  {
    printf("%s:%d GPU: ", file, line);
    printf(__format, args...);
  }
#else
  printf("%s:%d HOST: ", file, line);
  printf(__format, args...);
#endif
}

}  // namespace print
#define LOG(...) print::myprintf(__FILE__, __LINE__, __VA_ARGS__)

inline double wtime() {
  double time[2];
  struct timeval time1;
  gettimeofday(&time1, NULL);

  time[0] = time1.tv_sec;
  time[1] = time1.tv_usec;

  return time[0] + time[1] * 1.0e-6;
}

#define FULL_WARP_MASK 0xffffffff

template <typename T>
__inline__ __device__ T warpReduce(T val) {
  // T val_shuffled;
  for (int offset = 16; offset > 0; offset /= 2)
    val += __shfl_down_sync(FULL_WARP_MASK, val, offset);
  return val;
}
template <typename T>
__inline__ __device__ T warpReduceMax(T val) {
  for (int mask = WARP_SIZE / 2; mask > 0; mask >>= 1)
    val = max(val, __shfl_xor_sync(FULL_WARP_MASK, val, mask, WARP_SIZE));
  return val;
}
template <typename T>
__inline__ __device__ T warpReduceMin(T val) {
  for (int mask = WARP_SIZE / 2; mask > 0; mask >>= 1)
    val = min(val, __shfl_xor_sync(FULL_WARP_MASK, val, mask, WARP_SIZE));
  return val;
}

/*
// subwarp versions
template <typename T>
__inline__ __device__ T warpReduceMax(T val, unsigned subwarp_mask) {
  for (int mask = SUBWARP_SIZE / 2; mask > 0; mask >>= 1)
    val = max(val, __shfl_xor_sync(subwarp_mask, val, mask, SUBWARP_SIZE));
  return val;
}
template <typename T>
__inline__ __device__ T warpReduceMin(T val, unsigned subwarp_mask) {
  for (int mask = SUBWARP_SIZE / 2; mask > 0; mask >>= 1)
    val = min(val, __shfl_xor_sync(subwarp_mask, val, mask, SUBWARP_SIZE));
  return val;
}
*/

static __global__ void warmup_kernel() {}

inline size_t get_avail_mem() {
  int device;
  cudaGetDevice(&device);
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, device);
  size_t avail;
  size_t total;
  cudaMemGetInfo(&avail, &total);
  printf(
      "Amount of total memory: %g GB, avail memory: %g GB, take up: %g GB, "
      "%g MB, %g KB\n",
      total / (1024.0 * 1024.0 * 1024.0), avail / (1024.0 * 1024.0 * 1024.0),
      (total - avail) / (1024.0 * 1024.0 * 1024.0),
      (total - avail) / (1024.0 * 1024.0), (total - avail) / (1024.0));
  return avail;
}

inline int get_clk() {
  int device;
  int peak_clk = 1;
  cudaGetDevice(&device);
  cudaDeviceGetAttribute(&peak_clk, cudaDevAttrClockRate, device);
  return peak_clk;
}

inline int get_block_num(int multiplier = 2) {
  int device;
  cudaDeviceProp prop;
  cudaGetDevice(&device);
  cudaGetDeviceProperties(&prop, device);
  int n_sm = prop.multiProcessorCount;
  int b_per_sm = prop.maxThreadsPerMultiProcessor / BLOCK_SIZE;
  return n_sm * b_per_sm * multiplier;
}

template <typename T>
T* get_device_ptr(T* host_ptr, u64 n);

template <typename T>
T* get_device_ptr(u64 n, int init_val);

// Template implementations (must be in header for templates)
template <typename T>
inline T* get_device_ptr(T* host_ptr, u64 n) {
  T* device_ptr;
  CUDA_RT_CALL(cudaMalloc(&device_ptr, (u64)sizeof(T) * n));
  CUDA_RT_CALL(
      cudaMemcpy(device_ptr, host_ptr, (u64)sizeof(T) * n, cudaMemcpyDefault));
  return device_ptr;
}

template <typename T>
inline T* get_device_ptr(u64 n, int init_val) {
  T* device_ptr;
  CUDA_RT_CALL(cudaMalloc(&device_ptr, (u64)sizeof(T) * n));
  CUDA_RT_CALL(cudaMemset(device_ptr, init_val, (u64)sizeof(T) * n));
  return device_ptr;
}
 