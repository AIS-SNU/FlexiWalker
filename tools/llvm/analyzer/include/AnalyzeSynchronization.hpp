#pragma once

#include "llvm/IR/Function.h"
#include "llvm/IR/Instructions.h"
#include "llvm/Support/raw_ostream.h"

#include <vector>
#include <string>
#include <set>

namespace llvm_analysis {

/**
 * @brief Information about a detected synchronization primitive
 */
struct SyncPrimitiveInfo {
    std::string functionName;   ///< Name of the function containing the sync primitive
    std::string primitiveName;  ///< Name of the synchronization primitive detected
    std::string className;      ///< Walker class name
    unsigned lineNumber;        ///< Line number (if available)
};

/**
 * @brief Analyzes a function for inter-thread communication and synchronization primitives
 *
 * This function scans for potentially dangerous operations in user-written walker code:
 * - Warp intrinsics: __ballot_sync, __shfl_*, __any_sync, __all_sync, etc.
 * - Synchronization: __syncwarp, __syncthreads, __threadfence*, etc.
 * - Atomic operations: atomicAdd, atomicCAS, etc.
 *
 * These operations can cause deadlocks or stalls when used incorrectly in walker code
 * because different threads may take different control flow paths.
 *
 * @param F The function to analyze
 * @param detectedSyncs Output vector of detected synchronization primitives
 * @param className Name of the walker class
 * @param methodName Name of the method being analyzed (get_weight, is_stop, update)
 * @return true if any synchronization primitives were found
 */
bool analyzeSynchronization(llvm::Function &F,
                           std::vector<SyncPrimitiveInfo> &detectedSyncs,
                           const std::string &className,
                           const std::string &methodName);

/**
 * @brief Prints warnings for detected synchronization primitives
 *
 * @param detectedSyncs Vector of detected synchronization primitives
 * @param outputPath Path to write the warnings file (empty = skip file write)
 */
void printSynchronizationWarnings(const std::vector<SyncPrimitiveInfo> &detectedSyncs,
                                  const std::string &outputPath = "");

} // namespace llvm_analysis
