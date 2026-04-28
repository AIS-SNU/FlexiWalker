#pragma once
#include "llvm/Support/raw_ostream.h"

#include <string>
#include <vector>
#include <set>
#include <map>

namespace type_analysis {

/// Configuration constants for the type analyzer
class AnalyzerConfig {
public:
    static constexpr bool DEBUG_ENABLED = true;
    static constexpr int JSON_INDENT = 2;
    
    // Default filenames
    static constexpr const char* DEFAULT_CONFIG_FILE = "walker_analysis.json";
    static constexpr const char* DEFAULT_OUTPUT_FILE = "walker_return.json";
};

/// (field_name, role) pair annotating a return branch's dependencies.
/// role ∈ {"MAX", "MIN"} per the monotonicity analysis (spec §5.2.1.2).
struct FieldRole {
    std::string field;
    std::string role;
};

/// Structure representing a return branch in walker analysis
struct ReturnBranch {
    std::vector<std::string> body;        ///< Code body of the branch
    std::string return_expr;              ///< Return expression
    std::vector<FieldRole> fieldRoles;    ///< Per-field role annotations (Phase 2)
    bool forceERVSOnly = false;           ///< Rewriter hit a non-analyzable pattern
    std::string fallbackReason;           ///< Diagnostic for forceERVSOnly
};

/// Analysis result for a single walker class
struct ClassAnalysisResult {
    std::vector<ReturnBranch> branches;   ///< All return branches found
    std::set<std::string> taskFields;     ///< Task fields accessed
};

/// Complete analysis result for all walker classes
struct FullAnalysisResult {
    std::map<std::string, ClassAnalysisResult> classes;  ///< Per-class results
    std::vector<std::string> globalHeaders;              ///< Global header includes
};

/**
 * @brief Prints field mapping information for debugging
 * 
 * @param fieldMap Map of class -> struct -> fields for debugging output
 */
inline void printFieldMap(const std::map<std::string, std::map<std::string, std::vector<std::string>>> &fieldMap) {
    llvm::errs() << "=== Field Map Contents ===\n";
    for (const auto &[outerKey, innerMap] : fieldMap) {
        llvm::errs() << "Class: " << outerKey << "\n";
        for (const auto &[structName, fields] : innerMap) {
            llvm::errs() << "  Struct: " << structName << "\n";
            for (const auto &field : fields) {
                llvm::errs() << "    - " << field << "\n";
            }
        }
    }
    llvm::errs() << "==========================\n";
}

} // namespace type_analysis

// Backward compatibility macros
extern bool DEBUG;