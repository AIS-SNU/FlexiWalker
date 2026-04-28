#pragma once

#include "llvm/IR/Function.h"
#include "llvm/IR/Instructions.h"

#include <string>
#include <set>

#include "util.hpp"

using llvm::Value;
using llvm::Function;
using llvm::GetElementPtrInst;

namespace llvm_analysis {

/**
 * @brief Analyzes update patterns in walker update_weight methods
 * 
 * This function examines store instructions and atomic operations to determine
 * which struct fields are being updated during the walker execution.
 * 
 * @param F The function to analyze
 * @param updatedFields Output set of (struct_name, field_index) pairs that are updated
 * @return true if any updates were found, false otherwise
 */
bool analyzeUpdate(Function &F, std::set<std::pair<std::string, uint64_t>> &updatedFields);

/**
 * @brief Traces a pointer back to its originating struct GEP instruction
 *
 * This function follows pointer chains through loads, stores, and casts to find
 * the GetElementPtrInst that accesses a struct field.
 *
 * @param ptr The pointer value to trace
 * @return GetElementPtrInst* pointing to struct field access, or nullptr if not found
 */
GetElementPtrInst* traceToStructGEP(Value* ptr);

/**
 * @brief Analyzes whether user code modifies gpu_graph fields
 *
 * Detects stores to gpu_graph fields like adjwgt, edge_label, etc.
 * When detected, eRVS_only should be set to true because dynamic array
 * modifications require extended RVS (eRVS) sampling.
 *
 * @param F The function to analyze
 * @param graphFieldModified Output flag indicating if gpu_graph fields are modified
 * @return true if gpu_graph field writes were detected
 */
bool analyzeGraphModification(Function &F, bool &graphFieldModified);

} // namespace llvm_analysis