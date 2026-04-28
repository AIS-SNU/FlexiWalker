#include "JsonEmitter.hpp"
#include "util.hpp"

#include <fstream>
#include <filesystem>
#include <nlohmann/json.hpp>
#include <vector>
#include <string>

JsonEmitter::JsonEmitter(
    type_analysis::FullAnalysisResult &resultRef,
    const std::map<std::string, std::map<std::string, std::vector<std::string>>> &fieldMapRef, 
    const std::map<std::string, std::map<std::string, std::vector<std::string>>> &typeMapRef,
    const std::string &outputPathStr)
    : results(resultRef), fieldMap(fieldMapRef), typeMap(typeMapRef), outputPath(outputPathStr) {}

JsonEmitter::~JsonEmitter() {
  using json = nlohmann::json;
  json j;

  // Output Necessary Headers
  j["headers"] = results.globalHeaders;

  // Emit accessed field names per class
  for (const auto &[walkerClass, structMap] : fieldMap) {
    for (const auto &[structName, fields] : structMap) {
      j[walkerClass]["accessedFieldNames"][structName] = fields;
    }
  }

  // Emit accessed field types per class
  for (const auto &[walkerClass, structMap] : typeMap) {
    for (const auto &[structName, types] : structMap) {
      j[walkerClass]["accessedFieldTypes"][structName] = types;
    }
  }

  for (const auto& [cls, methodInfo] : results.classes) {
    // Emit return branches
    for (const auto& br : methodInfo.branches) {
      json branchJson = {
        {"body", br.body},
        {"return_expr", br.return_expr},
        {"force_eRVS_only", br.forceERVSOnly},
        {"fallback_reason", br.fallbackReason},
      };
      branchJson["field_roles"] = json::array();
      for (const auto &fr : br.fieldRoles) {
        branchJson["field_roles"].push_back({
          {"field", fr.field},
          {"role",  fr.role}
        });
      }
      j[cls]["branches"].push_back(branchJson);
    }

    // Emit task_fields used in the entire get_weight method
    j[cls]["task_fields"] = std::vector<std::string>(
      methodInfo.taskFields.begin(),
      methodInfo.taskFields.end()
    );
  }

  std::filesystem::path output_path(outputPath);
  std::filesystem::create_directories(output_path.parent_path());  // Create the directory if it doesn't exist
  
  std::ofstream out(output_path);
  out << j.dump(2);
}
