#pragma once

#include "llvm/Demangle/Demangle.h"
#include "llvm/Support/raw_ostream.h"
#include <nlohmann/json.hpp>

#include <string>
#include <regex>
#include <algorithm>
#include <vector>
#include <map>
#include <set>
#include <fstream>

namespace llvm_analysis {

/// Configuration constants for the LLVM analyzer
class AnalyzerConfig {
public:
    static constexpr bool DEBUG_ENABLED = true;
    static constexpr bool PRINT_ENABLED = true;
    
    // Target method names to analyze
    static constexpr const char* GET_WEIGHT_METHOD = "get_weight";
    static constexpr const char* IS_STOP_METHOD = "is_stop";
    static constexpr const char* UPDATE_METHOD = "update";
    
    // JSON output formatting
    static constexpr int JSON_INDENT = 2;
};

/// Flags indicating the type of return value analysis
enum class ReturnFlag : int {
    INVALID = -1,
    CONST = 0,           ///< Constant value
    APP_MEMBER = 1,      ///< Application member access
    TASK_MEMBER = 2,     ///< Task member access
    INDEX_MEMBER = 3,    ///< Index-based member access
    DYNAMIC_MEMBER = 4   ///< Dynamic member access
};

/// Flags indicating the type of update pattern
enum class UpdateFlag : int {
    SINGLE = 0,         ///< Update max at the start of the kernel
    SINGLE_PER_ITER = 1,///< Update max at the start of the iteration
    SCAN_PER_ITER = 2   ///< Needs scan per each iteration
};

} // namespace llvm_analysis

// Legacy compatibility - keep old enum names for now to avoid breaking existing code
using ReturnFlag = llvm_analysis::ReturnFlag;
using UpdateFlag = llvm_analysis::UpdateFlag;
constexpr auto INVALID = ReturnFlag::INVALID;
constexpr auto CONST = ReturnFlag::CONST;
constexpr auto APP_MEMBER = ReturnFlag::APP_MEMBER;
constexpr auto TASK_MEMBER = ReturnFlag::TASK_MEMBER;
constexpr auto INDEX_MEMBER = ReturnFlag::INDEX_MEMBER;
constexpr auto DYNAMIC_MEMBER = ReturnFlag::DYNAMIC_MEMBER;
constexpr auto SINGLE = UpdateFlag::SINGLE;
constexpr auto SINGLE_PER_ITER = UpdateFlag::SINGLE_PER_ITER;
constexpr auto SCAN_PER_ITER = UpdateFlag::SCAN_PER_ITER;

// For backward compatibility with DEBUG/PRINT macros
#define DEBUG llvm_analysis::AnalyzerConfig::DEBUG_ENABLED
#define PRINT llvm_analysis::AnalyzerConfig::PRINT_ENABLED

namespace llvm_analysis {

/**
 * @brief Checks if a function name matches a target method name.
 * 
 * Demangles C++ function names and extracts class and method information
 * for walker analysis.
 * 
 * @param name The mangled function name
 * @param target_name The target method name to match
 * @param className Output parameter for extracted class name
 * @param macroName Output parameter for uppercased class name
 * @return true if the function matches the target name
 */
inline bool funcNameCheck(const std::string& name, const std::string& target_name,
                         std::string& className, std::string& macroName) {
    // Early rejection for performance
    if (name.find(target_name) == std::string::npos) {
        return false;
    }

    // Demangle the function name
    std::string demangled = llvm::demangle(name);

    // Match ClassName::target_name immediately followed by "(" — the
    // \( anchor prevents prefix-bleed (e.g. target "update" matching
    // "Foo::update_weight" because update is a prefix of update_weight).
    std::string pattern = R"((\w+)::)" + target_name + R"(\()";
    std::regex re(pattern);
    std::smatch match;

    if (!std::regex_search(demangled, match, re)) {
        return false;
    }

    // Extract class name and create macro version
    className = match[1].str();
    macroName = className;
    std::transform(macroName.begin(), macroName.end(), macroName.begin(), ::toupper);

    return true;
}

/**
 * @brief Extracts ClassName and MethodName from a (possibly mangled)
 * function name without filtering on a specific method name.
 *
 * Used by the user-method analysis path: any method on a walker class
 * whose name isn't already handled by a dedicated branch is treated as
 * a user-defined helper and put through the same mutation/sync checks
 * as update_weight used to be.
 *
 * @return true iff the demangled name has the form `ClassName::MethodName(`
 */
inline bool extractClassAndMethod(const std::string& name,
                                  std::string& className,
                                  std::string& methodName,
                                  std::string& macroName) {
    std::string demangled = llvm::demangle(name);
    static const std::regex re(R"((\w+)::(\w+)\()");
    std::smatch match;
    if (!std::regex_search(demangled, match, re)) {
        return false;
    }
    className = match[1].str();
    methodName = match[2].str();
    macroName = className;
    std::transform(macroName.begin(), macroName.end(), macroName.begin(), ::toupper);
    return true;
}

/**
 * @brief Checks if a function name corresponds to an atomic operation.
 * 
 * @param calleeName The name of the called function
 * @return true if the function is an atomic operation
 */
inline bool isAtomicFunctionName(const std::string& calleeName) {
    static const std::vector<std::string> atomicFunctions = {
        "atomicAdd", "atomicSub", "atomicCAS", "atomicExch",
        "atomicMax", "atomicMin", "atomicAnd", "atomicOr",
        "atomicXor", "atomicInc", "atomicDec"
    };

    return std::any_of(atomicFunctions.begin(), atomicFunctions.end(),
                      [&calleeName](const std::string& atomicFunc) {
                          return calleeName.find(atomicFunc) != std::string::npos;
                      });
}

/**
 * @brief Converts return flags to corresponding update flags.
 * 
 * Maps the type of return value analysis to the appropriate update pattern
 * for optimization purposes.
 * 
 * @param walkerReturnFlagMap Map of class names to return flags
 * @param walkerUpdateFlagMap Output map of class names to update flags
 */
inline void convertReturnFlagsToUpdateFlags(
    const std::map<std::string, ReturnFlag>& walkerReturnFlagMap,
    std::map<std::string, UpdateFlag>& walkerUpdateFlagMap) {
    
    for (const auto& [className, retFlag] : walkerReturnFlagMap) {
        UpdateFlag updateFlag;

        switch (retFlag) {
            case ReturnFlag::CONST:
            case ReturnFlag::APP_MEMBER:
                updateFlag = UpdateFlag::SINGLE;
                break;

            case ReturnFlag::TASK_MEMBER:
            case ReturnFlag::INDEX_MEMBER:
                updateFlag = UpdateFlag::SINGLE_PER_ITER;
                break;

            case ReturnFlag::DYNAMIC_MEMBER:
                updateFlag = UpdateFlag::SCAN_PER_ITER;
                break;

            default:
                // Conservative fallback - assume worst case
                updateFlag = UpdateFlag::SCAN_PER_ITER;
                break;
        }

        walkerUpdateFlagMap[className] = updateFlag;
    }
}

} // namespace llvm_analysis

//=============================================================================
// JSON Output Functions
//=============================================================================

using json = nlohmann::json;

/**
 * @brief Writes comprehensive walker analysis results to JSON file
 * 
 * Creates a JSON file containing all analysis results including return flags,
 * update flags, field access patterns, and external variable usage.
 * 
 * @param filename Output JSON filename
 * @param walkerReturnFlagMap Map of return analysis results
 * @param walkerUpdateFlagMap Map of update pattern results
 * @param updatedFields Map of fields updated by each walker
 * @param accessedByIndex Map of fields accessed by index
 * @param possibleZeroMap Map indicating zero return possibilities
 * @param externalLocalsMap Map of external local variables used
 */

inline void writeWalkerAnalysisJSON(
    const std::string& filename,
    const std::map<std::string, ReturnFlag>& walkerReturnFlagMap,
    const std::map<std::string, UpdateFlag>& walkerUpdateFlagMap,
    const std::map<std::string, std::set<std::pair<std::string, uint64_t>>>& updatedFields,
    const std::map<std::string, std::set<std::pair<std::string, uint64_t>>>& accessedByIndex,
    const std::map<std::string, bool>& possibleZeroMap,
    const std::map<std::string, std::set<std::string>>& externalLocalsMap,
    const std::map<std::string, bool>& eRVS_onlyMap)
{
    json j;

    for (const auto& [className, retFlag] : walkerReturnFlagMap) {
        json classJson;
        classJson["returnFlag"] = static_cast<int>(retFlag);
        
        classJson["externallyDefinedLocals"] = json::array();
        if (externalLocalsMap.count(className)) {
            for (const auto& localName : externalLocalsMap.at(className)) {
                classJson["externallyDefinedLocals"].push_back(localName);
            }
        }

        // Add update flag if available
        if (walkerUpdateFlagMap.count(className)) {
            classJson["updateFlag"] = static_cast<int>(walkerUpdateFlagMap.at(className));
        } else {
            classJson["updateFlag"] = -1; // fallback
        }

        if (possibleZeroMap.count(className)) {
            classJson["possibleZero"] = static_cast<int>(possibleZeroMap.at(className));
        } else {
            classJson["possibleZero"] = -1; // fallback;
        }

        // Add eRVS_only flag
        if (eRVS_onlyMap.count(className)) {
            classJson["eRVS_only"] = eRVS_onlyMap.at(className);
        } else {
            classJson["eRVS_only"] = false;  // default: no graph modification
        }
    
        // Safe access to updatedFields
        classJson["updatedFields"] = json::array();
        if (updatedFields.count(className)) {
            for (const auto& [structName, fieldIdx] : updatedFields.at(className)) {
                json entry = json::array({structName, fieldIdx});
                classJson["updatedFields"].push_back(entry);
            }
        }
    
        // Safe access to accessedByIndex
        classJson["accessedByIndex"] = json::array();
        if (accessedByIndex.count(className)) {
            for (const auto& [structName, fieldIdx] : accessedByIndex.at(className)) {
                json entry = json::array({structName, fieldIdx});
                classJson["accessedByIndex"].push_back(entry);
            }
        }
    
        j[className] = classJson;
    }

    std::ofstream out(filename);
    if (!out.is_open()) {
        llvm::errs() << "Error opening file: " << filename << " for writing.\n";
        return;
    }

    out << j.dump(llvm_analysis::AnalyzerConfig::JSON_INDENT) << "\n";
    out.close();
}

//=============================================================================
// Debug and Utility Functions
//=============================================================================

/**
 * @brief Prints a map of class names to field sets for debugging
 * 
 * @param mapSet The map to print
 */
inline void printMapSet(const std::map<std::string, std::set<std::pair<std::string, uint64_t>>> &mapSet) {
    for (const auto& [className, fieldSet] : mapSet) {
        llvm::errs() << "Class: " << className << "\n";
        for (const auto& [structName, fieldIdx] : fieldSet) {
            llvm::errs() << "  Indexed field: " << structName << "[" << fieldIdx << "]\n";
        }
    }
}

/**
 * @brief Prints updated fields information for debugging
 */

/**
 * @brief Prints updated fields information for debugging
 */
inline void printUpdatedFields(const std::map<std::string, std::set<std::pair<std::string, uint64_t>>> &updatedFields) {
    llvm::errs() << "Updated Fields:\n";
    printMapSet(updatedFields);
}

/**
 * @brief Prints fields accessed by index for debugging
 */
inline void printAccessedByIndex(const std::map<std::string, std::set<std::pair<std::string, uint64_t>>> &accessedByIndex) {
    llvm::errs() << "Accessed By Index:\n";
    printMapSet(accessedByIndex);
}

/**
 * @brief Loads walker metadata and builds task-to-walker mapping
 *
 * Reads walker_metadata.json and creates a mapping from task types
 * to the list of walkers that use each task type.
 *
 * @return Map from task class name to vector of walker names
 */
/**
 * @brief Extracts the set of walker class names from walker_metadata.json.
 *
 * Used to filter the per-function loop in the analyzer: only methods on
 * classes that are actually walkers should go through the user-method
 * mutation analysis path. Without this filter, helper functions on
 * non-walker classes (or anything that happens to demangle as
 * `Foo::bar(`) would be analyzed too.
 */
inline std::set<std::string> loadWalkerNames(
    const std::map<std::string, std::vector<std::string>>& taskToWalkersMap) {
    std::set<std::string> walkerNames;
    for (const auto& [_, walkers] : taskToWalkersMap) {
        for (const auto& w : walkers) {
            walkerNames.insert(w);
        }
    }
    return walkerNames;
}

inline std::map<std::string, std::vector<std::string>> loadTaskToWalkerMapping() {
    std::map<std::string, std::vector<std::string>> taskToWalkersMap;

    // Try multiple paths to find walker_metadata.json
    std::vector<std::string> searchPaths = {
        "artifacts/walker_metadata.json",           // From project root
        "../artifacts/walker_metadata.json",        // One level up
        "../../artifacts/walker_metadata.json",     // Two levels up
        "../../../artifacts/walker_metadata.json"   // Three levels up
    };

    std::ifstream metadataFile;
    std::string foundPath;
    for (const auto &path : searchPaths) {
        metadataFile.open(path);
        if (metadataFile.is_open()) {
            foundPath = path;
            break;
        }
    }

    if (!metadataFile.is_open()) {
        llvm::errs() << "Warning: Could not open walker_metadata.json (tried multiple paths)\n";
        return taskToWalkersMap;
    }

    try {
        json metadata;
        metadataFile >> metadata;

        // Build mapping: task_type -> list of walkers using it
        for (auto &[walkerName, walkerData] : metadata.items()) {
            if (walkerData.contains("task_type")) {
                std::string taskType = walkerData["task_type"];
                taskToWalkersMap[taskType].push_back(walkerName);
            }
        }
    } catch (const std::exception &e) {
        llvm::errs() << "Error parsing walker_metadata.json: " << e.what() << "\n";
    }

    return taskToWalkersMap;
}

/**
 * @brief Converts ReturnFlag enum to human-readable string
 *
 * @param flag The return flag to convert
 * @return String representation of the flag
 */
inline std::string returnFlagToString(ReturnFlag flag) {
    switch (flag) {
        case CONST: return "CONST";
        case APP_MEMBER: return "APP_MEMBER";
        case TASK_MEMBER: return "TASK_MEMBER";
        case INDEX_MEMBER: return "INDEX_MEMBER";
        case DYNAMIC_MEMBER: return "DYNAMIC_MEMBER";
        case INVALID: return "INVALID";
        default: return "UNKNOWN";
    }
}

/**
 * @brief Prints walker return flags for debugging
 * 
 * @param walkerReturnFlagMap Map of class names to return flags
 */
inline void printWalkerFlags(const std::map<std::string, ReturnFlag> &walkerReturnFlagMap) {
    llvm::errs() << "Walker Return Flags:\n";
    for (const auto &entry : walkerReturnFlagMap) {
        llvm::errs() << "  " << entry.first << " → " << returnFlagToString(entry.second)
                     << " (" << static_cast<int>(entry.second) << ")\n";
    }
}