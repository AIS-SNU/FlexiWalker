#pragma once

#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "util.hpp"
#include <map>
#include <vector>
#include <string>
#include <set>

using namespace clang;
using namespace clang::ast_matchers;

namespace type_analysis {

/**
 * @brief Collects struct field information from AST matching
 * 
 * This class analyzes struct definitions to extract field names and types,
 * particularly focusing on fields that are accessed by walker classes.
 */
class StructFieldCollector : public MatchFinder::MatchCallback {
public:
    /// Map: ClassName -> StructName -> FieldNames
    std::map<std::string, std::map<std::string, std::vector<std::string>>> fieldMap;
    
    /// Map: ClassName -> StructName -> FieldIndices (for targeted analysis)
    std::map<std::string, std::map<std::string, std::set<unsigned>>> targetFieldsByStruct;
    
    /// Map: ClassName -> StructName -> FieldTypes
    std::map<std::string, std::map<std::string, std::vector<std::string>>> typeMap;

    /**
     * @brief Sets target field indices for focused analysis
     * 
     * @param targets Map of target field indices organized by class and struct
     */
    inline void setTargetIndices(const std::map<std::string, std::map<std::string, std::set<unsigned>>> &targets) {
        targetFieldsByStruct = targets;
    }
    
    /**
     * @brief Called when AST matcher finds a matching node
     * 
     * @param Result The matching result containing AST nodes
     */
    void run(const MatchFinder::MatchResult &Result) override;
};

} // namespace type_analysis