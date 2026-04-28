#include "llvm/IR/Module.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/PassManager.h"
#include "llvm/Passes/PassPlugin.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/CommandLine.h"

#include <fstream>
#include <string>
#include <unordered_set>
#include <queue>
#include <map>

#include "AnalyzeReturn.hpp"
#include "AnalyzeUpdate.hpp"
#include "AnalyzeSynchronization.hpp"
#include "AnalyzeComplexity.hpp"
#include "util.hpp"

using namespace llvm;
using namespace llvm_analysis;

namespace {

/// Command-line option for JSON output path
static cl::opt<std::string> OutputJSONPath(
    "walker-json",
    cl::desc("Path to output walker analysis JSON file"),
    cl::value_desc("filename"),
    cl::init("walker_analysis.json"));

/// Command-line option for synchronization warnings output. Empty
/// value (the default) skips the file write — warnings still go to
/// stderr. The pipeline sets this to the artifacts directory so
/// multiple invocations don't clobber each other's CWD.
static cl::opt<std::string> SyncWarningsPath(
    "walker-sync-warnings",
    cl::desc("Path to output synchronization warnings (empty = stderr only)"),
    cl::value_desc("filename"),
    cl::init(""));

} // anonymous namespace

/**
 * @brief LLVM Pass for analyzing walker class methods.
 *
 * Dispatches per-function: get_weight / is_stop / Task::update receive
 * their dedicated analyses (return flags, sync, graph mutation,
 * complexity); any other method on a class in the walker set is treated
 * as a user-defined helper and runs through analyzeUpdate +
 * graph-mutation + sync. Emits walker_analysis.json with per-walker
 * trait flags consumed by the CodeGenerator stage.
 */
struct AdjwgtDetectorPass : PassInfoMixin<AdjwgtDetectorPass> {
    PreservedAnalyses run(Module &M, ModuleAnalysisManager &) {
        // Analysis result containers
        AnalysisResults results;

        // Load walker metadata to build task-to-walker mapping
        std::map<std::string, std::vector<std::string>> taskToWalkersMap = loadTaskToWalkerMapping();

        // Walker class name set — used to filter the user-method
        // analysis path so only methods on actual walker classes are
        // checked for graph mutation / sync / field updates.
        std::set<std::string> walkerNames = loadWalkerNames(taskToWalkersMap);

        if (AnalyzerConfig::DEBUG_ENABLED) {
            llvm::errs() << "[Main] Task-to-Walker mapping:\n";
            for (const auto &[taskName, walkers] : taskToWalkersMap) {
                llvm::errs() << "  " << taskName << " -> [";
                for (size_t i = 0; i < walkers.size(); ++i) {
                    llvm::errs() << walkers[i];
                    if (i < walkers.size() - 1) llvm::errs() << ", ";
                }
                llvm::errs() << "]\n";
            }
        }

        // Process all functions in the module.
        //
        // Per-function dispatch: extract (className, methodName) once,
        // then route by methodName. Standard-API methods get their
        // dedicated branches; any other method on a class in the walker
        // set is treated as a user-defined helper and runs through the
        // same mutation / sync / field-update checks update_weight used
        // to get exclusively. This generalizes the bug #3 graph-mutation
        // detection past the legacy "update_weight" literal.
        std::map<std::string, bool> taskModifiesGraph;  // Track which Task classes modify graph
        for (Function &F : M) {
            std::string className, methodName, macroName;
            std::string functionName = F.getName().str();

            if (!llvm_analysis::extractClassAndMethod(functionName, className, methodName, macroName)) {
                if (AnalyzerConfig::DEBUG_ENABLED) {
                    llvm::errs() << "[Main] skipping unparsed function: " << functionName << "\n";
                }
                continue;
            }

            // Skip constructors (methodName == className) and destructors
            // (methodName starts with ~). Both do legitimate `this`
            // member initialization / cleanup that would muddy the
            // user-method analysis (e.g. `this->graph = _graph` would
            // otherwise be flagged as a graph mutation). The metadata
            // extractor already excludes ctors/dtors from the dummy stub.
            if (methodName == className) {
                continue;
            }
            if (!methodName.empty() && methodName[0] == '~') {
                continue;
            }

            if (methodName == AnalyzerConfig::GET_WEIGHT_METHOD) {
                llvm_analysis::analyzeReturn(F, results.walkerReturnFlagMap, results.updatedFields[className],
                            results.accessedByIndex[className], results.possibleZeroMap,
                            results.externalLocalsMap[className], className, macroName);

                bool graphModified = false;
                llvm_analysis::analyzeGraphModification(F, graphModified);
                if (graphModified) {
                    results.eRVS_onlyMap[className] = true;
                }

                bool isComplex = false;
                llvm_analysis::analyzeComplexity(F, isComplex, className);
                if (isComplex) {
                    results.eRVS_onlyMap[className] = true;
                }

                llvm_analysis::analyzeSynchronization(F, results.detectedSyncs, className, "get_weight");
            } else if (methodName == AnalyzerConfig::IS_STOP_METHOD) {
                bool graphModified = false;
                llvm_analysis::analyzeGraphModification(F, graphModified);
                if (graphModified) {
                    results.eRVS_onlyMap[className] = true;
                }

                llvm_analysis::analyzeSynchronization(F, results.detectedSyncs, className, "is_stop");
            } else if (methodName == AnalyzerConfig::UPDATE_METHOD) {
                // Task::update — graph modification here propagates to
                // every walker that uses this Task type (handled below
                // via taskModifiesGraph + taskToWalkersMap).
                bool graphModified = false;
                llvm_analysis::analyzeGraphModification(F, graphModified);
                taskModifiesGraph[className] = graphModified;

                llvm_analysis::analyzeSynchronization(F, results.detectedSyncs, className, "update");
            } else if (walkerNames.count(className)) {
                // Defensive name-based skip for standard-API and
                // codegen-generated methods. The metadata extractor
                // already drops these from the dummy stub, so under
                // normal flow their bodies don't reach IR — but a
                // user could provide inline definitions, which would
                // survive. Treating them as user helpers would feed
                // their field-access patterns into the analyzer and
                // perturb downstream rewrites (e.g. MonotonicityRewriter
                // adding a spurious _MAX/_MIN suffix).
                if (methodName == "scan_thread" ||
                    methodName == "fill_dummy" ||
                    methodName == "get_max_weight" ||
                    methodName == "get_sum_weight") {
                    continue;
                }

                // Any other method on a walker class — treat as a
                // user-defined helper. This is the generalized bug #3
                // path: graph mutation in any user-coded method (not
                // just one literally named "update_weight") forces
                // eRVS_only.
                llvm_analysis::analyzeUpdate(F, results.updatedFields[className]);

                bool graphModified = false;
                llvm_analysis::analyzeGraphModification(F, graphModified);
                if (graphModified) {
                    results.eRVS_onlyMap[className] = true;
                }

                llvm_analysis::analyzeSynchronization(F, results.detectedSyncs, className, methodName);
            }
        }

        // Propagate Task graph modifications to walkers using those tasks
        if (AnalyzerConfig::DEBUG_ENABLED) {
            llvm::errs() << "[Main] Task graph modification detection:\n";
            for (const auto &[taskClassName, modifiesGraph] : taskModifiesGraph) {
                llvm::errs() << "  " << taskClassName << " modifies graph: " << (modifiesGraph ? "YES" : "NO") << "\n";
            }
        }

        for (const auto &[taskClassName, modifiesGraph] : taskModifiesGraph) {
            if (modifiesGraph && taskToWalkersMap.count(taskClassName)) {
                if (AnalyzerConfig::DEBUG_ENABLED) {
                    llvm::errs() << "[Main] Propagating eRVS_only from task " << taskClassName << " to walkers: ";
                }
                // Mark all walkers that use this task
                for (const auto &walkerName : taskToWalkersMap[taskClassName]) {
                    results.eRVS_onlyMap[walkerName] = true;
                    if (AnalyzerConfig::DEBUG_ENABLED) {
                        llvm::errs() << walkerName << " ";
                    }
                }
                if (AnalyzerConfig::DEBUG_ENABLED) {
                    llvm::errs() << "\n";
                }
            }
        }

        // Print debug information if enabled
        if (AnalyzerConfig::PRINT_ENABLED) {
            printAnalysisResults(results);
        }

        // Print synchronization warnings (always, regardless of PRINT_ENABLED)
        llvm_analysis::printSynchronizationWarnings(results.detectedSyncs,
                                                    SyncWarningsPath.getValue());

        // Convert return flags to update flags
        llvm_analysis::convertReturnFlagsToUpdateFlags(results.walkerReturnFlagMap, results.walkerUpdateFlagMap);

        // Write analysis results to JSON
        if (!writeAnalysisJSON(results)) {
            llvm::errs() << "Error: Failed to write analysis results to JSON\n";
        }

        return PreservedAnalyses::all();
    }

private:
    /**
     * @brief Container for all analysis results.
     */
    struct AnalysisResults {
        std::map<std::string, ReturnFlag> walkerReturnFlagMap;
        std::map<std::string, UpdateFlag> walkerUpdateFlagMap;
        std::map<std::string, std::set<std::pair<std::string, uint64_t>>> updatedFields;
        std::map<std::string, std::set<std::pair<std::string, uint64_t>>> accessedByIndex;
        std::map<std::string, bool> possibleZeroMap;
        std::map<std::string, std::set<std::string>> externalLocalsMap;
        std::vector<SyncPrimitiveInfo> detectedSyncs;
        std::map<std::string, bool> eRVS_onlyMap;  // Per-walker flag for graph modification
    };

    /**
     * @brief Prints analysis results for debugging.
     */
    void printAnalysisResults(const AnalysisResults& results) {
        printUpdatedFields(results.updatedFields);
        printWalkerFlags(results.walkerReturnFlagMap);
        printAccessedByIndex(results.accessedByIndex);
    }

    /**
     * @brief Writes analysis results to JSON file.
     */
    bool writeAnalysisJSON(const AnalysisResults& results) {
        try {
            writeWalkerAnalysisJSON(
                OutputJSONPath.getValue(),
                results.walkerReturnFlagMap,
                results.walkerUpdateFlagMap,
                results.updatedFields,
                results.accessedByIndex,
                results.possibleZeroMap,
                results.externalLocalsMap,
                results.eRVS_onlyMap);
            return true;
        } catch (const std::exception& e) {
            llvm::errs() << "JSON write error: " << e.what() << "\n";
            return false;
        }
    }
};
/**
 * @brief Plugin registration for the LLVM pass manager.
 */
extern "C" LLVM_ATTRIBUTE_WEAK ::llvm::PassPluginLibraryInfo llvmGetPassPluginInfo() {
    return {
        LLVM_PLUGIN_API_VERSION, 
        "AdjwgtDetector", 
        LLVM_VERSION_STRING,
        [](PassBuilder &PB) {
            PB.registerPipelineParsingCallback(
                [](StringRef Name, ModulePassManager &MPM,
                   ArrayRef<PassBuilder::PipelineElement>) {
                    if (Name == "adjwgt-detector") {
                        MPM.addPass(AdjwgtDetectorPass());
                        return true;
                    }
                    return false;
                });
        }
    };
}
