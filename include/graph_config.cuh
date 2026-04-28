#pragma once
#include <fstream>
#include <string>
#include <map>
#include <iostream>
#include <sstream>
#include <vector>

// Simple config file parser for graph configuration
// Reads graph-specific file paths at runtime (no recompilation needed)
//
// Format: field_name = /path/to/file
//
// Examples:
//   xadj = /data/wiki-Vote/wiki-Vote.xadj
//   adjncy = /data/wiki-Vote/wiki-Vote.adjncy
//   edge_label = /data/wiki-Vote/wiki-Vote.label

class GraphConfig {
 private:
  std::map<std::string, std::string> field_paths;
  bool loaded;

  // Trim whitespace from string
  static std::string trim(const std::string& str) {
    size_t start = str.find_first_not_of(" \t\r\n");
    size_t end = str.find_last_not_of(" \t\r\n");
    if (start == std::string::npos) return "";
    return str.substr(start, end - start + 1);
  }

 public:
  GraphConfig() : loaded(false) {}

  explicit GraphConfig(const std::string& config_file) : loaded(false) {
    load(config_file);
  }

  bool load(const std::string& config_file) {
    std::ifstream file(config_file);
    if (!file.is_open()) {
      std::cerr << "Warning: Failed to open graph config file '" << config_file << "'" << std::endl;
      std::cerr << "Using command-line --input path as fallback." << std::endl;
      return false;
    }

    std::string line;
    int line_num = 0;
    while (std::getline(file, line)) {
      line_num++;
      line = trim(line);

      // Skip empty lines and comments
      if (line.empty() || line[0] == '#') {
        continue;
      }

      // Parse "field_name = path"
      size_t eq_pos = line.find('=');
      if (eq_pos == std::string::npos) {
        std::cerr << "Warning: Invalid config line " << line_num << ": " << line << std::endl;
        continue;
      }

      std::string field_name = trim(line.substr(0, eq_pos));
      std::string path = trim(line.substr(eq_pos + 1));

      if (field_name.empty() || path.empty()) {
        std::cerr << "Warning: Invalid config line " << line_num << ": " << line << std::endl;
        continue;
      }

      field_paths[field_name] = path;
    }

    file.close();
    loaded = true;
    return true;
  }

  bool has_field(const std::string& field_name) const {
    return field_paths.find(field_name) != field_paths.end();
  }

  std::string get_path(const std::string& field_name) const {
    auto it = field_paths.find(field_name);
    if (it != field_paths.end()) {
      return it->second;
    }
    return "";
  }

  bool is_loaded() const { return loaded; }
};
