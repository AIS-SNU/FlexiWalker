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
#include <cuda_runtime.h>
#include <gflags/gflags.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include <fstream>
#include <iostream>
#include <map>
#include <string>
#include <vector>
#include <sstream>

#include <cub/cub.cuh>  // NOLINT

#include "app.cuh"
#include "generated/gpu_graph.cuh"
#include "generated/graph.cuh"
#include "graph_config.cuh"
#include "util.cuh"
#include "walker.cuh"

using namespace std;  // NOLINT

DEFINE_string(input, "../data/wiki-Vote", "input dataset");
DEFINE_string(b_file, "../res/bucket.csv", "degree bucket file");
DEFINE_string(config, "../config/graphs/wiki-Vote.config", "graph config file (YAML) with additional field paths");

DEFINE_int32(n, 4000, "sample size");
DEFINE_bool(seq, false, "start from 0 to n-1");
DEFINE_bool(all, false, "start from all nodes");
DEFINE_int32(d, 80, "depth");

DEFINE_bool(umgraph, false, "enable unified memory to store graph.");

DEFINE_bool(Deepwalk, false, "deepwalk");
DEFINE_bool(Node2vec, false, "node2vec");
DEFINE_bool(Node2vec_weighted, false, "node2vec weighted");
DEFINE_bool(PPR, false, "ppr");
DEFINE_bool(Metapath, false, "metapath");
DEFINE_bool(Metapath_weighted, false, "metapath weighted");
DEFINE_bool(PPR_second, false, "second-order pagerank");

DEFINE_int32(block_multi, 2, "the multiplier for the number of blocks launched");

DEFINE_double(p, 2.0, "hyper-parameter p for node2vec");
DEFINE_double(q, 0.5, "hyper-parameter q for node2vec");
DEFINE_double(tp, 0.2, "terminate probabiility");
DEFINE_int32(schemalen, 5, "number of labels");
DEFINE_string(schema, "0,1,2,3,4", "metapath schema");
DEFINE_double(alpha, 0.2, "hyper-parameter alpha for second-order pagerank");
DEFINE_int32(GPU_count, 1, "number of GPUs to use. default is a single gpu.");

DEFINE_bool(printresult, false, "printresult");
DEFINE_bool(printworkload, false, "printworkload");
DEFINE_bool(save_degree, false, "save degree distribution");

DEFINE_bool(autobatch, false, "use adaptive batch size");
DEFINE_bool(batch, false, "use batch mode");
DEFINE_bool(syn, false, "enable synchronized batch");
DEFINE_uint32(batchsize, 10000000, "batch size");
DEFINE_int32(
    headroom, 512,
    "GPU memory headroom for other data structures while using autobatch(MB)");

DEFINE_bool(lognorm, false, "lognorm edge weight");

DEFINE_bool(sanitycheck, false, "sanity check of GPU random walk results");
DEFINE_int32(ompt, 8, "number of omp threads to use during sanity check");

void calculate_workload(int num_walkers, int max_depth, vtx_t* result_pool_ptr,
                        graph* ginst, double total_s, bool is_host = false) {
  vtx_t* result = result_pool_ptr;
  if (!is_host) {
    result = reinterpret_cast<vtx_t*>(
        malloc((u64)sizeof(vtx_t) * num_walkers * max_depth));
    CUDA_RT_CALL(cudaMemcpy(result, result_pool_ptr,
                            (u64)sizeof(vtx_t) * num_walkers * max_depth,
                            cudaMemcpyDeviceToHost));
  }

  u64 sampled_v = 0;
  u64 sampled_e = 0;
  for (int i = 0; i < num_walkers; i++) {
    vtx_t vtx_pre = -1;
    for (int j = 0; j < max_depth; j++) {
      u64 res_offset = (u64)i * max_depth + j;
      vtx_t vtx = result[res_offset];
      if (vtx == -1) {
        break;
      }
      if (j > 0) sampled_v++;
      if (vtx_pre != -1)
        sampled_e += ginst->xadj[vtx_pre + 1] - ginst->xadj[vtx_pre];
      vtx_pre = vtx;
    }
  }
  printf("Total sampled vertex: %llu\n", sampled_v);
  printf("Total sampled edge: %llu,throughput: %f edges per s\n", sampled_e,
         sampled_e / total_s);
}

void calculate_workload(int num_walkers, int max_depth, vtx_t* result_pool_ptr,
  graph* ginst, double total_s, u64* sampled_v_ptr, u64* sampled_e_ptr, bool is_host = false) {
  vtx_t* result = result_pool_ptr;
  if (!is_host) {
  result = reinterpret_cast<vtx_t*>(
  malloc((u64)sizeof(vtx_t) * num_walkers * max_depth));
  CUDA_RT_CALL(cudaMemcpy(result, result_pool_ptr,
        (u64)sizeof(vtx_t) * num_walkers * max_depth,
        cudaMemcpyDeviceToHost));
  }

  u64 sampled_v = 0;
  u64 sampled_e = 0;
  for (int i = 0; i < num_walkers; i++) {
  vtx_t vtx_pre = -1;
  for (int j = 0; j < max_depth; j++) {
  u64 res_offset = (u64)i * max_depth + j;
  vtx_t vtx = result[res_offset];
  if (vtx == -1) {
  break;
  }
  if (j > 0) sampled_v++;
  if (vtx_pre != -1)
  sampled_e += ginst->xadj[vtx_pre + 1] - ginst->xadj[vtx_pre];
  vtx_pre = vtx;
  }
  }
  *sampled_v_ptr = sampled_v;
  *sampled_e_ptr = sampled_e;
}



void degree_bucket(int num_walkers, int max_depth, vtx_t* result_pool_ptr,
                   graph* ginst, bool is_host = false) {
  vtx_t* result = result_pool_ptr;
  if (!is_host) {
    result = reinterpret_cast<vtx_t*>(
        malloc((u64)sizeof(vtx_t) * num_walkers * max_depth));
    CUDA_RT_CALL(cudaMemcpy(result, result_pool_ptr,
                            (u64)sizeof(vtx_t) * num_walkers * max_depth,
                            cudaMemcpyDeviceToHost));
  }

  std::map<edge_t, vtx_t> degree_mp;
  for (int i = 0; i < num_walkers; i++) {
    vtx_t vtx_pre = -1;
    for (int j = 0; j < max_depth; j++) {
      u64 res_offset = (u64)i * max_depth + j;
      vtx_t vtx = result[res_offset];
      if (vtx == -1) {
        break;
      }

      if (vtx_pre != -1) {
        edge_t degree = ginst->xadj[vtx_pre + 1] - ginst->xadj[vtx_pre];
        degree_mp[degree]++;
      }
      vtx_pre = vtx;
    }
  }
  ofstream degree_bucket_ofs(FLAGS_b_file);
  for (auto i = degree_mp.begin(); i != degree_mp.end(); i++) {
    degree_bucket_ofs << i->first << "," << i->second << endl;
  }
  degree_bucket_ofs.close();
}

void print_res(int num_walkers, int max_depth, vtx_t* result_pool,
               bool is_host = false) {
  vtx_t* result = result_pool;
  if (!is_host) {
    result = reinterpret_cast<vtx_t*>(
        malloc((u64)sizeof(vtx_t) * num_walkers * max_depth));
    CUDA_RT_CALL(cudaMemcpy(result, result_pool,
                            (u64)sizeof(vtx_t) * num_walkers * max_depth,
                            cudaMemcpyDeviceToHost));
  }

  u64 sampled = 0;
  for (int i = 0; i < num_walkers; i++) {
    printf("----------------------\nWalker %d:\n", i);
    int len = max_depth;
    for (int j = 0; j < max_depth; j++) {
      u64 res_offset = i * max_depth + j;
      if (static_cast<int>(result[res_offset]) == -1) {
        len = j;
        break;
      }
      if (j > 0) sampled++;
      printf("%d\t", result[res_offset]);
    }
    printf("\nTotal length=%d\n", len);
  }
  printf("Total sampled: %llu\n", sampled);
}

void print_res_metapath(int num_walkers, int max_depth, vtx_t* result_pool,
                        graph* graph, bool is_host = false) {
  vtx_t* result = result_pool;
  if (!is_host) {
    result = reinterpret_cast<vtx_t*>(
        malloc((u64)sizeof(vtx_t) * num_walkers * max_depth));
    CUDA_RT_CALL(cudaMemcpy(result, result_pool,
                            (u64)sizeof(vtx_t) * num_walkers * max_depth,
                            cudaMemcpyDeviceToHost));
  }

  u64 sampled = 0;
  for (u64 i = 0; i < num_walkers; i++) {
    printf("----------------------\nWalker %llu:\n", i);
    int len = max_depth;
    int pre_vtx = -1;

    for (int j = 0; j < max_depth; j++) {
      u64 res_offset = i * max_depth + j;
      vtx_t vtx = result[res_offset];
      if (vtx == -1) {
        len = j;
        break;
      }
      if (pre_vtx != -1) {
        edge_t pre_offset = graph->xadj[pre_vtx];
        vtx_t pre_degree = graph->xadj[pre_vtx + 1] - pre_offset;
        for (vtx_t k = 0; k < pre_degree; k++) {
          if (graph->adjncy[pre_offset + k] == vtx) {
            printf("[%d]\t", graph->edge_label[pre_offset + k]);
            break;
          }
        }
      }
      if (j > 0) sampled++;
      pre_vtx = vtx;
      printf("%d\t", vtx);
    }
    printf("\nTotal length=%d\n", len);
  }
  printf("Total sampled: %llu\n", sampled);
}

int* get_metapath(int* schema_len) {
  vector<int> v_schema;
  *schema_len = 0;
  while (FLAGS_schema.find(",") != string::npos) {
    string tmp = FLAGS_schema.substr(0, FLAGS_schema.find(","));
    v_schema.push_back(stoi(tmp));
    FLAGS_schema = FLAGS_schema.substr(FLAGS_schema.find(",") + 1);
  }
  v_schema.push_back(stoi(FLAGS_schema));
  *schema_len = v_schema.size();
  FLAGS_schemalen = *schema_len;
  int* schema = reinterpret_cast<int*>(malloc(*schema_len * sizeof(int)));
  for (int i = 0; i < *schema_len; i++) {
    schema[i] = v_schema[i];
  }
  return schema;
}

void adjust_flags(uint query_num) {
  if (FLAGS_Metapath || FLAGS_Metapath_weighted) {
    FLAGS_d = FLAGS_schemalen;
  }
  if (FLAGS_all) {
    FLAGS_seq = true;
    FLAGS_batch = true;
    FLAGS_autobatch = true;
  }
  if (FLAGS_autobatch) {
    FLAGS_batch = true;

    size_t avail = get_avail_mem();

    int bs = min((avail - FLAGS_headroom * 1024 * 1024) / (4 * (FLAGS_d + 2)),
                 (size_t)INT_MAX);

    assert(bs > 0);
    if (FLAGS_syn == false) bs /= 2;
    FLAGS_batchsize = min(bs, query_num);
    printf("batch size=%u\n", FLAGS_batchsize);
  }
}
vtx_t* get_startpoints(graph* ginst, int sample_size) {
  vtx_t* start_points;
  CUDA_RT_CALL(cudaMallocHost(&start_points, sample_size * sizeof(vtx_t)));

  if (FLAGS_PPR) {
    vtx_t idx = ginst->get_maxdegree_offset();
    for (int i = 0; i < sample_size; i++) {
      start_points[i] = idx;
    }
  } else {
    vtx_t num_node = ginst->vert_count;
    unsigned int local_seed = time(NULL);

    for (int i = 0; i < sample_size; i++) {
      if (FLAGS_seq)
        start_points[i] = i;
      else
        start_points[i] = rand_r(&local_seed) % num_node;
    }
  }
  return start_points;
}

// Extracts a subset of indices from a pre-shuffled array
vtx_t* get_startpoints_from_array(const vector<vtx_t>& shuffled_indices,
  int sample_size, int start_idx) {
  // Allocate pinned memory for start points
  vtx_t* start_points;
  CUDA_RT_CALL(cudaMallocHost(&start_points, sample_size * sizeof(vtx_t)));

  // Extract the subset based on the specified range
  for (int i = 0; i < sample_size; i++) {
  start_points[i] = shuffled_indices[start_idx + i];
  }

  return start_points;
}

void sanity_check(int num_walkers, int max_depth, vtx_t* result_pool, graph* ginst,
               bool is_host = false) {
  vtx_t* result = result_pool;
  if (!is_host) {
    result = reinterpret_cast<vtx_t*>(
        malloc((u64)sizeof(vtx_t) * num_walkers * max_depth));
    CUDA_RT_CALL(cudaMemcpy(result, result_pool,
                            (u64)sizeof(vtx_t) * num_walkers * max_depth,
                            cudaMemcpyDeviceToHost));
  }

  edge_t* xadj = ginst->xadj;
  vtx_t* adjncy = ginst->adjncy;
  bool all_right = true;
  long long unsigned int total_edge_searched = 0;
  vector<vtx_t> wrong_walker_ids;

  // local values to accelerate with omp

  u64 sampled = 0;

  omp_set_num_threads(static_cast<int>(FLAGS_ompt));

  #pragma omp parallel reduction(+:total_edge_searched, sampled)
  {
    bool local_all_right = true;  // Thread-local flag for correctness
    
    #pragma omp for
    for (int i = 0; i < num_walkers; i++) {
      bool right_walker = true;
      for (int j = 0; j < max_depth; j++) {
        u64 res_offset = i * max_depth + j;
        if (static_cast<int>(result[res_offset]) == -1) {
          break;
        }
        if (j > 0) sampled++;

        // check sanity (if the sampled node is actually connected)
        if (j > 0) {
          vtx_t prior_vtx_id = result[res_offset-1];
          vtx_t curr_vtx_id = result[res_offset];
          edge_t prior_vtx_ptr = xadj[prior_vtx_id];
          edge_t prior_vtx_edge_count = xadj[prior_vtx_id+1] - xadj[prior_vtx_id];
          // assert(0 <= prior_vtx_edge_count);
          bool right_edge = false;
          for (vtx_t k=0; k < prior_vtx_edge_count; k++) {
            if (adjncy[prior_vtx_ptr+k] == curr_vtx_id) {
              right_edge = true;
              break;
            }
          }
          right_walker = right_walker && right_edge;
          total_edge_searched += prior_vtx_edge_count;
        }

      }
      if (!right_walker) {
        local_all_right = false;
        #pragma omp critical
        {
          wrong_walker_ids.push_back(i);
        }
      }
    }

    #pragma omp critical
    {
      all_right = all_right && local_all_right;
    }
    
  }
  printf("Total sampled: %llu\n", sampled);
  printf("Total edges searched: %llu\n", total_edge_searched);
  if (all_right) {
    printf("Sanity check SUCCESS!\n");
  } else {
    printf("Sanity check FAILED!\n");
    for (size_t i = 0; i < wrong_walker_ids.size(); ++i) {
        cout << wrong_walker_ids[i] << " ";
    }
    cout << endl;
  }
}


int main(int argc, char* argv[]) {
  #ifdef _OPENMP
    printf("[OK] OpenMP is enabled. Version: %d\n", _OPENMP);
  #else
      printf("[WARN] OpenMP is NOT enabled. Check compilation flags.\n");
  #endif

  printf(
      "\n-------------------------------------------------------------------"
      "--------------------------------------\n");
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  string adj_file, edge_file, weight_file;
  const char* config_file = FLAGS_config.empty() ? nullptr : FLAGS_config.c_str();

  // If config file is provided, read base graph paths from it
  if (config_file != nullptr) {
    GraphConfig config(config_file);
    if (config.is_loaded()) {
      adj_file = config.get_path("xadj");
      edge_file = config.get_path("adjncy");
      weight_file = config.get_path("adjwgt");

      if (adj_file.empty() || edge_file.empty() || weight_file.empty()) {
        std::cerr << "Error: Config file missing required fields (xadj, adjncy, adjwgt)" << std::endl;
        return 1;
      }
      std::cout << "Using graph files from config:" << std::endl;
      std::cout << "  xadj: " << adj_file << std::endl;
      std::cout << "  adjncy: " << edge_file << std::endl;
      std::cout << "  adjwgt: " << weight_file << std::endl;
    } else {
      std::cerr << "Error: Failed to load config file: " << FLAGS_config << std::endl;
      return 1;
    }
  } else {
    // Fallback to constructing paths from --input flag
    adj_file = FLAGS_input + "_xadj.bin";
    edge_file = FLAGS_input + "_edge.bin";
    weight_file = FLAGS_input + "_weight.bin";
    std::cout << "Using graph files from --input flag:" << std::endl;
    std::cout << "  xadj: " << adj_file << std::endl;
    std::cout << "  adjncy: " << edge_file << std::endl;
    std::cout << "  adjwgt: " << weight_file << std::endl;
  }

  graph* ginst;

  // Create graph - config file will be reloaded by graph constructor for additional fields
  ginst = new graph(adj_file.c_str(), edge_file.c_str(), weight_file.c_str(), config_file);

  // LOG("Read graph \n");
  printf("Read Graph!\n");
  ginst->print_max_degree();
  // ginst->print_degree();
  // ginst->print_weight();

  gpu_graph* ggraph;
  vector<gpu_graph*> ggraph_list;
  
  if (FLAGS_GPU_count == 1) ggraph = new gpu_graph(ginst);
  else {
    for (int g = 0; g < FLAGS_GPU_count; g++) {
      CUDA_RT_CALL(cudaSetDevice(g));
      ggraph_list.push_back(new gpu_graph(ginst));
    }

    for (int g = 0; g < FLAGS_GPU_count; g++) {
      CUDA_RT_CALL(cudaSetDevice(g));
      CUDA_RT_CALL(cudaDeviceSynchronize());
    }
    
  }

  int schema_len;
  int* schema;
  if (FLAGS_Metapath || FLAGS_Metapath_weighted) {
    schema = get_metapath(&schema_len);
    printf("schema len=%d\n", schema_len);
    for (int i = 0; i < schema_len; i++) {
      printf("%d ", schema[i]);
    }
    printf("\n");
  }

  int sample_size = FLAGS_n;

  if (FLAGS_all) sample_size = ginst->vert_count;
  adjust_flags(sample_size);
  int depth = FLAGS_d + 1;

  // Profile once on GPU 0 for select_algo_cost_ratio (always run, no flag check)
  weight_t select_algo_cost_ratio = 0;
  CUDA_RT_CALL(cudaSetDevice(0));
  gpu_graph* graph_ptr_for_profile;
  if (FLAGS_GPU_count == 1) {
    graph_ptr_for_profile = get_device_ptr<gpu_graph>(ggraph, 1);
  } else {
    graph_ptr_for_profile = get_device_ptr<gpu_graph>(ggraph_list[0], 1);
  }
  select_algo_cost_ratio = profile_current_walker(graph_ptr_for_profile, depth, schema, schema_len);
  CUDA_RT_CALL(cudaDeviceSynchronize());
  if (select_algo_cost_ratio > -1.0) printf("[MAIN] Profiled select_algo_cost_ratio on GPU 0: %f\n", select_algo_cost_ratio);
  else printf("[MAIN] Running in eRVS_only mode.\n");

  vtx_t* start_points;
  vector<vtx_t*> start_points_list;
  vector<int> sample_size_list;
  vector<double> sample_time_list(FLAGS_GPU_count, 0.0);
  vector<u64> sampled_v_list(FLAGS_GPU_count, 0);
  vector<u64> sampled_e_list(FLAGS_GPU_count, 0);
  if (FLAGS_GPU_count == 1) start_points = get_startpoints(ginst, sample_size);
  else {
    // Multi-GPU scenario
    int base_sample_size = sample_size / FLAGS_GPU_count;  // Base samples per GPU
    int remainder = sample_size % FLAGS_GPU_count;         // Remaining samples to distribute
    int start_idx = 0;

    // Generate and shuffle indices globally (only done once)
    std::vector<vtx_t> global_shuffled_indices(ginst->vert_count);
    for (vtx_t i = 0; i < ginst->vert_count; ++i) {
        global_shuffled_indices[i] = i;
    }

    // Shuffle with a robust random generator
    std::mt19937 rng(7524); // rng(static_cast<unsigned int>(time(NULL)));
    std::shuffle(global_shuffled_indices.begin(), global_shuffled_indices.end(), rng);

    
    for (int g = 0; g < FLAGS_GPU_count; g++) {
      int g_sample_size = base_sample_size + (g < remainder ? 1 : 0);
      
      sample_size_list.push_back(g_sample_size);

      start_points_list.push_back(get_startpoints_from_array(global_shuffled_indices, g_sample_size, start_idx));
      
      start_idx += g_sample_size;
    }
  
  }

  // Helper function to process walk results
  auto process_walk_results = [&](vtx_t* result_pool, int sample_size, double sample_time,
                                   u64* sampled_v_ptr = nullptr, u64* sampled_e_ptr = nullptr) {
    bool is_host = FLAGS_batch;

    if (FLAGS_printresult) {
      if (FLAGS_Metapath || FLAGS_Metapath_weighted) {
        print_res_metapath(sample_size, depth, result_pool, ginst, is_host);
      } else {
        print_res(sample_size, depth, result_pool, is_host);
      }
    }

    if (FLAGS_printworkload) {
      if (sampled_v_ptr && sampled_e_ptr) {
        calculate_workload(sample_size, depth, result_pool, ginst, sample_time,
                          sampled_v_ptr, sampled_e_ptr, is_host);
      } else {
        calculate_workload(sample_size, depth, result_pool, ginst, sample_time, is_host);
      }
    }

    if (FLAGS_save_degree) {
      degree_bucket(sample_size, depth, result_pool, ginst, is_host);
    }

    if (FLAGS_sanitycheck) {
      sanity_check(sample_size, depth, result_pool, ginst, is_host);
    }
  };

  if (FLAGS_GPU_count == 1) {
    vtx_t* result_pool = NULL;
    double sample_time = 0;
    if (FLAGS_batch) {
      sample_time = walk_batch(result_pool, ggraph, start_points, depth,
                              sample_size, FLAGS_batchsize, select_algo_cost_ratio, schema, schema_len);
    } else {
      sample_time = walk_test(result_pool, ggraph, start_points, depth,
                              sample_size, select_algo_cost_ratio, schema, schema_len);
    }

    process_walk_results(result_pool, sample_size, sample_time);
  } else {

    // Multi-GPU
    #pragma omp parallel for num_threads(FLAGS_GPU_count)
    for (int g = 0; g < FLAGS_GPU_count; g++) {

      printf("GPU %d being processed by thread %d out of %d threads\n",
        g, omp_get_thread_num(), omp_get_num_threads());
      // Set the CUDA device for this thread
      CUDA_RT_CALL(cudaSetDevice(g));

      // Thread-specific variables
      gpu_graph* ggraph = ggraph_list[g];
      vtx_t* start_points = start_points_list[g];
      int sample_size = sample_size_list[g];
      vtx_t* result_pool = NULL;

      if (FLAGS_batch) {
        sample_time_list[g] = walk_batch(result_pool, ggraph, start_points, depth,
                                sample_size, FLAGS_batchsize, select_algo_cost_ratio, schema, schema_len);
      } else {
        sample_time_list[g] = walk_test(result_pool, ggraph, start_points, depth,
                                sample_size, select_algo_cost_ratio, schema, schema_len);
      }

      process_walk_results(result_pool, sample_size, sample_time_list[g],
                          &(sampled_v_list[g]), &(sampled_e_list[g]));
    }
  }

  // If multi-GPU, print the final results 
  if (FLAGS_GPU_count != 1) {
    u64 total_sampled_v = 0;
    u64 total_sampled_e = 0;
    double max_sample_time = 0.0;

    printf("\nPer-GPU Results:\n");
    printf("GPU\tSampled Vertices\tSampled Edges\tTime (s)\n");
    for (int g = 0; g < FLAGS_GPU_count; g++) {
        printf("%d\t%llu\t%llu\t%f\n",
              g, sampled_v_list[g], sampled_e_list[g], sample_time_list[g]);
        total_sampled_v += sampled_v_list[g];
        total_sampled_e += sampled_e_list[g];
        if (sample_time_list[g] > max_sample_time) {
            max_sample_time = sample_time_list[g];
        }
    }

    printf("\nTotal Results Across All GPUs:\n");
    printf("Total Sampled Vertices: %llu\n", total_sampled_v);
    printf("Total Sampled Edges: %llu\n", total_sampled_e);
    printf("Maximum Time Taken by Any GPU: %f s\n", max_sample_time);
    printf("Overall Throughput: %f edges/s\n", total_sampled_e / max_sample_time);
  }

  // Cleanup

  delete ginst;
  if (FLAGS_GPU_count == 1 ) delete ggraph;
  else {
    for (auto graph_ptr : ggraph_list) {
      delete graph_ptr;
    }
  }

  if (FLAGS_GPU_count == 1 ) cudaFreeHost(start_points);
  else {
    for (auto start_ptr : start_points_list) {
      cudaFreeHost(start_ptr);
    }
  }

  return 0;
}
