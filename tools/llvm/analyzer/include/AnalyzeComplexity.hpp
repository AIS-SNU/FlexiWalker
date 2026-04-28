#pragma once

#include "llvm/IR/Function.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/CFG.h"
#include "llvm/Support/raw_ostream.h"

#include <set>
#include <string>

namespace llvm_analysis {

/**
 * @brief Configuration for complexity thresholds
 */
struct ComplexityThresholds {
    static constexpr unsigned MAX_NESTING_DEPTH = 5;  ///< Maximum allowed nesting depth
    static constexpr unsigned MAX_LOOP_DEPTH = 3;     ///< Maximum allowed loop nesting
};

/**
 * @brief Analyzes code complexity in get_weight() to detect patterns requiring eRVS
 *
 * Detects complex control flow patterns that make automatic code generation
 * for get_max_weight() and get_sum_weight() difficult or error-prone:
 * - Recursive function calls
 * - Loops with data-dependent exits (break/continue based on runtime values)
 * - Deeply nested structures (loops, conditionals)
 *
 * When detected, sets eRVS_only flag to use more conservative sampling.
 *
 * @param F The get_weight function to analyze
 * @param isComplex Output flag indicating if complex patterns were found
 * @param className Name of the walker class
 * @return true if complex patterns were detected
 */
bool analyzeComplexity(llvm::Function &F,
                      bool &isComplex,
                      const std::string &className);

/**
 * @brief Checks if a function contains recursive calls
 *
 * @param F The function to check
 * @param visited Set of already visited functions (for indirect recursion)
 * @return true if recursion is detected
 */
bool hasRecursiveCalls(llvm::Function &F, std::set<llvm::Function*> &visited);

/**
 * @brief Checks if control flow has data-dependent exits
 *
 * Detects loops/branches where exit depends on runtime values,
 * making static analysis difficult.
 *
 * @param F The function to analyze
 * @return true if data-dependent exits are found
 */
bool hasDataDependentExits(llvm::Function &F);

/**
 * @brief Computes the nesting depth of control flow structures
 *
 * @param F The function to analyze
 * @return Maximum nesting depth found
 */
unsigned computeNestingDepth(llvm::Function &F);

} // namespace llvm_analysis
