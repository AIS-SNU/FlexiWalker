#pragma once
#include <gflags/gflags.h>
#include <algorithm>
#include <iostream>
#include "util.cuh"
#include "graph_base.cuh"
#include "graph_config.cuh"

// Auto-generated unified struct extensions
DECLARE_bool(Deepwalk);
DECLARE_bool(Metapath);
DECLARE_bool(Metapath_weighted);
DECLARE_bool(Node2vec);
DECLARE_bool(Node2vec_weighted);
DECLARE_bool(PPR);
DECLARE_bool(PPR_second);

// Extension for struct graph
class graph : public graph_base {
 public:
  int* edge_label = nullptr;

 public:
  ~graph() {
    if (edge_label) delete[] edge_label;
  }

  graph() : graph_base() {}

  explicit graph(const char* xadj_file, const char* adjncy_file,
                 const char* weight_file, const char* config_file = nullptr)
      : graph_base(xadj_file, adjncy_file, weight_file) {

    // Load runtime config for additional fields
    GraphConfig config;
    if (config_file != nullptr) {
      config.load(config_file);
    }
    // edge_label for Metapath, Metapath_weighted
    if (FLAGS_Metapath || FLAGS_Metapath_weighted) {
      // Load from file
      std::string field_path = config.get_path("edge_label");
      if (field_path.empty()) {
        std::cerr << "Error: edge_label path not configured for Metapath/Metapath_weighted" << std::endl;
        std::cerr << "Please add 'edge_label = /path/to/file' to your graph config" << std::endl;
        exit(1);
      }

      FILE* file = fopen(field_path.c_str(), "rb");
      if (file != NULL) {
        size_t array_size = edge_count;
        int* tmp_edge_label = NULL;
        if (posix_memalign(reinterpret_cast<void**>(&tmp_edge_label), getpagesize(),
                           sizeof(int) * array_size))
          perror("posix_memalign");

        size_t ret = fread(tmp_edge_label, sizeof(int), array_size, file);
        assert(ret == array_size);
        fclose(file);

        edge_label = reinterpret_cast<int*>(tmp_edge_label);
        std::cout << "Loaded edge_label from " << field_path << std::endl;
      } else {
        std::cerr << "Error: Cannot open edge_label file: " << field_path << std::endl;
        exit(1);
      }
    }
  }
};