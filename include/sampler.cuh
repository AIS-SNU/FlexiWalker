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

#include <cub/cub.cuh>  // NOLINT

#include "app.cuh"
#include "gpu_task.cuh"
#include "myrand.cuh"

template <typename walker_t, typename state_t, typename TaskType = typename walker_t::TaskType>
__device__ inline vtx_t sampler_warp_exp_jump_stable(walker_t* walker, TaskType* task,
                                     state_t* state) {
  vtx_t size = task->degree;
  int lid = threadIdx.x % WARP_SIZE;
  vtx_t padded_size = ((size + WARP_SIZE - 1) / WARP_SIZE) * WARP_SIZE;

  ll selected_id = -1;
  ll global_selected_id = -1;

  weight_t global_min_key = FLT_MAX;
  int thresh_update_counter = 0;
  weight_t min_key = FLT_MAX; 
  weight_t temp_key = FLT_MAX;
  weight_t w;

  weight_t thresh = 0;

  // initialize global thresh 
  int i = lid;
  while (i < padded_size) {
    if (i < size) {
      w = walker->get_weight(task, i);
      if (w > 0) {
        selected_id = i + (ll)size * lid;
        min_key = -(logf(myrand_uniform(state)) / w);
      }
    }
    global_min_key = warpReduceMin(min_key);
    i += WARP_SIZE;
    if (global_min_key != FLT_MAX) {
      global_selected_id = warpReduceMax(((ll)(global_min_key==min_key))*selected_id);
      thresh = -logf(myrand_uniform(state))/global_min_key;
      break;
    }
  }
  // temporarily assume that the threshold will be computed in the first 32 elements
  // if (global_max_key == 0) assert(false); 

  for ( ; i < padded_size; i += WARP_SIZE) {
    if (i < size) {
      w = walker->get_weight(task, i);

      if (w > 0) {
        thresh -= w;
        if (thresh <= 0) {
          temp_key = exp((-global_min_key)*w);
          temp_key = temp_key + (myrand_uniform(state))*(1-temp_key);
          thresh = -logf(myrand_uniform(state))/global_min_key; 
          temp_key = -logf(temp_key)/w;
          if (temp_key < min_key) {
            min_key = temp_key;
            selected_id = i + (ll)size * lid;
          }
          
        }
      }

    }

    thresh_update_counter++;

    if (thresh_update_counter == UPDATE) {
      __syncwarp();
      temp_key = warpReduceMin(min_key);
      if (temp_key < global_min_key) {
        global_min_key = temp_key;
        global_selected_id = warpReduceMax(((ll)(temp_key==min_key))*selected_id);
      }
      thresh = -logf(myrand_uniform(state))/global_min_key;
      // reset  
      min_key = FLT_MAX;
      thresh_update_counter = 0;
      __syncwarp();
    }
  }

  // Output final edge
  __syncwarp();
  temp_key = warpReduceMin(min_key);
  if (temp_key < global_min_key) {
    global_min_key = temp_key;
    global_selected_id = warpReduceMax(((ll)(temp_key==min_key))*selected_id);
  }
  
  if (global_min_key == FLT_MAX) return -1;

  // selected_id = warpReduceMax(((int)(temp_key==max_key))*selected_id);
  return global_selected_id % size;
}



template <typename walker_t, typename state_t, typename TaskType = typename walker_t::TaskType>
__device__ __forceinline__ vtx_t sampler_rjs_sample_only_select_algo(walker_t* walker, TaskType* task,
                                          state_t* state, weight_t max_weight, vtx_t size, int& reject_counter, bool& continue_rjs) {
  vtx_t selected = -1;
  weight_t y;

  do {
    selected = myrand(state) % size;
    y = myrand_uniform(state) * max_weight;
    reject_counter++;
    if (reject_counter > REJ_SYNC) {
      continue_rjs = true;
      reject_counter = 0;
      return selected;
    }
  } while (y > walker->get_weight(task, selected));

  continue_rjs = false;

  return selected;
}

template <typename walker_t, typename state_t, typename TaskType = typename walker_t::TaskType>
__device__ inline unsigned int sampler_rjs_sample_only_profile(walker_t* walker, TaskType* task,
                                          state_t* state, weight_t max_weight) {
  vtx_t size = task->degree;
  vtx_t selected = -1;
  weight_t y;
  bool original_skip; 
  unsigned int total_edges = 0;

  if (max_weight > 0) {
    do {
      selected = myrand(state) % size;
      y = myrand_uniform(state) * max_weight;
      total_edges++;
      original_skip = y > walker->get_weight(task, selected);
    } while (total_edges < PROFILE_EDGE_LIMIT_REJ);
  }

  return total_edges;
}

template <typename walker_t, typename state_t, typename TaskType = typename walker_t::TaskType>
__device__ inline vtx_t sampler_warp_exp_jump_stable_profile(walker_t* walker, TaskType* task,
                                     state_t* state) {
  vtx_t size = walker->graph->edge_num;
  int lid = threadIdx.x % WARP_SIZE;
  vtx_t padded_size = PROFILE_EDGE_LIMIT;//((size + WARP_SIZE - 1) / WARP_SIZE) * WARP_SIZE);

  ll selected_id = -1;
  ll global_selected_id = -1;

  weight_t global_min_key = FLT_MAX;
  int thresh_update_counter = 0;
  weight_t min_key = FLT_MAX; 
  weight_t temp_key = FLT_MAX;
  weight_t w;

  weight_t thresh = 0;

  // initialize global thresh 
  int i = lid;
  while (i < padded_size) {
    if (i < size) {
      w = walker->get_weight(task, i);
      if (w > 0) {
        selected_id = i + (ll)size * lid;
        min_key = -(logf(myrand_uniform(state)) / w);
      }
    }
    global_min_key = warpReduceMin(min_key);
    i += WARP_SIZE;
    if (global_min_key != FLT_MAX) {
      global_selected_id = warpReduceMax(((ll)(global_min_key==min_key))*selected_id);
      thresh = -logf(myrand_uniform(state))/global_min_key;
      break;
    }
  }
  // temporarily assume that the threshold will be computed in the first 32 elements
  // if (global_max_key == 0) assert(false); 

  for ( ; i < padded_size; i += WARP_SIZE) {
    if (i < size) {
      w = walker->get_weight(task, i);

      if (w > 0) {
        thresh -= w;
        if (thresh <= 0) {
          temp_key = exp((-global_min_key)*w);
          temp_key = temp_key + (myrand_uniform(state))*(1-temp_key);
          thresh = -logf(myrand_uniform(state))/global_min_key; 
          temp_key = -logf(temp_key)/w;
          if (temp_key < min_key) {
            min_key = temp_key;
            selected_id = i + (ll)size * lid;
          }
          
        }
      }

    }

    thresh_update_counter++;

    if (thresh_update_counter == UPDATE) {
      __syncwarp();
      temp_key = warpReduceMin(min_key);
      if (temp_key < global_min_key) {
        global_min_key = temp_key;
        global_selected_id = warpReduceMax(((ll)(temp_key==min_key))*selected_id);
      }
      thresh = -logf(myrand_uniform(state))/global_min_key;
      // reset  
      min_key = FLT_MAX;
      thresh_update_counter = 0;
      __syncwarp();
    }
  }

  // Output final edge
  __syncwarp();
  temp_key = warpReduceMin(min_key);
  if (temp_key < global_min_key) {
    global_min_key = temp_key;
    global_selected_id = warpReduceMax(((ll)(temp_key==min_key))*selected_id);
  }
  
  if (global_min_key == FLT_MAX) return -1;

  // selected_id = warpReduceMax(((int)(temp_key==max_key))*selected_id);
  return global_selected_id % size;
}


