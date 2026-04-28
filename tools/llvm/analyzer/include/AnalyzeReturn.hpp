#pragma once

#include "llvm/IR/Instructions.h"
#include "llvm/IR/Function.h"

#include <string>
#include <unordered_set>
#include <map>

#include "util.hpp"

using llvm::Value;
using llvm::Function;
using llvm::AllocaInst;

namespace llvm_analysis {

/**
 * @brief Analyzes the return value patterns of walker get_weight methods
 * 
 * This function examines the control flow and data dependencies in a function
 * to determine what type of value is being returned (constant, member access, etc.)
 * 
 * @param F The function to analyze
 * @param walkerFlagMap Output map storing return flags for each walker class
 * @param updatedFields Set of fields that are updated during analysis
 * @param accessedByIndex Set of fields accessed by index
 * @param possibleZero Map indicating if zero return values are possible
 * @param externalLocals Set of externally defined local variables
 * @param className Name of the walker class being analyzed
 * @param macroName Macro context for the analysis
 * @return true if analysis completed successfully, false otherwise
 */
bool analyzeReturn(Function &F, std::map<std::string, ReturnFlag> &walkerFlagMap, 
                  std::set<std::pair<std::string, uint64_t>> &updatedFields, 
                  std::set<std::pair<std::string, uint64_t>> &accessedByIndex, 
                  std::map<std::string, bool> &possibleZero, 
                  std::set<std::string> &externalLocals, 
                  std::string className, std::string macroName);

/**
 * @brief Recursively traverses values to determine return patterns
 * 
 * This is the core analysis function that recursively examines LLVM values
 * to classify the type of return pattern being used.
 * 
 * @param val The LLVM value to analyze
 * @param visited Set of already visited values to prevent cycles
 * @param updatedFields Set of fields that are updated
 * @param accessedByIndex Set of fields accessed by index
 * @param possibleZero Reference to track if zero values are possible
 * @param ssaToVarName Mapping from SSA values to variable names
 * @param externallyDefinedLocals Set of externally defined allocas
 * @param externalLocals Set of external local variable names
 * @param className Name of the walker class
 * @return ReturnFlag indicating the pattern type
 */
ReturnFlag funcTraverser(Value* val, std::unordered_set<Value*>& visited, 
                        std::set<std::pair<std::string, uint64_t>> &updatedFields, 
                        std::set<std::pair<std::string, uint64_t>> &accessedByIndex, 
                        bool &possibleZero, std::map<Value*, std::string> &ssaToVarName, 
                        std::set<AllocaInst*> &externallyDefinedLocals, 
                        std::set<std::string> &externalLocals, std::string className);

} // namespace llvm_analysis