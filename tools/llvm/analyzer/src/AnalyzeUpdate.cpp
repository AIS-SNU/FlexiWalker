#include "llvm/IR/Module.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/PassManager.h"
#include "llvm/Passes/PassPlugin.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Support/raw_ostream.h"

#include <fstream>
#include <string>
#include <unordered_set>
#include <queue>
#include <map>

#include "AnalyzeUpdate.hpp"
#include "util.hpp"

using namespace llvm;

namespace llvm_analysis {

GetElementPtrInst* traceToStructGEP(Value* ptr) {
    std::unordered_set<Value*> visited;
    if (DEBUG) errs() << "startGEPTrace: " << *ptr << "\n";

    while (ptr) {
        if (!visited.insert(ptr).second) break; // avoid cycles
        ptr = ptr->stripPointerCasts();

        if (DEBUG) errs() << "startGEPTrace while: " << *ptr << "\n";

        if (auto *gep = dyn_cast<GetElementPtrInst>(ptr)) {
            Type *srcTy = gep->getSourceElementType();
            if (isa<StructType>(srcTy)) {
                if (DEBUG) errs() << "startGEPTrace while: success" << "\n";
                return gep;  // found struct field access
            }
            ptr = gep->getPointerOperand();
        } else if (auto *load = dyn_cast<LoadInst>(ptr)) {
            ptr = load->getPointerOperand();
        } else if (auto *alloca = dyn_cast<AllocaInst>(ptr)) {
            // Alloca may be stored to from multiple paths (loop carries,
            // conditional reassignments). Try every store's value as a
            // candidate source pointer; return the first that traces to
            // a struct GEP. Picking only the first store would miss the
            // actual data flow when the relevant assignment is later in
            // textual order.
            for (User *U : alloca->users()) {
                if (auto *store = dyn_cast<StoreInst>(U)) {
                    if (store->getPointerOperand() == alloca) {
                        if (auto *gep = traceToStructGEP(store->getValueOperand())) {
                            return gep;
                        }
                    }
                }
            }
            return nullptr;
        } else if (auto *store = dyn_cast<StoreInst>(ptr)) {
            ptr = store->getValueOperand(); // if this is part of pointer-to-pointer forwarding
        } else if (auto *inst = dyn_cast<Instruction>(ptr)) {
            if (inst->getNumOperands() > 0)
                ptr = inst->getOperand(0);  // fallback: chase first operand
            else
                break;
        } else {
            break;
        }
    }
    return nullptr;
}

bool analyzeUpdate(Function &F, std::set<std::pair<std::string, uint64_t>> &updatedFields) {
    bool updated = false;

    for (Instruction &I : instructions(F)) {
        Value *ptr = nullptr;

        if (auto *store = dyn_cast<StoreInst>(&I)) {
            ptr = store->getPointerOperand();
            if (DEBUG) errs() << "analyzeUpdate: StoreInst: " << *store << "\n";
        } else if (auto *call = dyn_cast<CallInst>(&I)) {
            Function *callee = call->getCalledFunction();
            if (!callee) continue;
            
            if (DEBUG) errs() << "analyzeUpdate: CallInst: " << *call << "\n";

            std::string calleeName = callee->getName().str();
            if (llvm_analysis::isAtomicFunctionName(calleeName)) {
                if (call->arg_size() >= 1) {
                    ptr = call->getArgOperand(0);
                    if (DEBUG) errs() << "analyzeUpdate: atomicInst: " << *ptr << "\n";
                }
            }
        }

        if (!ptr) continue;

        if (GetElementPtrInst *gep = traceToStructGEP(ptr)) {
            if (DEBUG) errs() << "analyzeUpdate: tracetoGEP: " << *gep << "\n";
            Type *sourceTy = gep->getSourceElementType();
            if (StructType *structTy = dyn_cast<StructType>(sourceTy)) {
                if (structTy->hasName()) {
                    StringRef structName = structTy->getName(); // e.g., %class.gpu_graph
                    if (gep->getNumIndices() >= 2) {
                        auto idxIt = gep->idx_begin();
                        ++idxIt; // skip leading 0 index
                        if (ConstantInt *ci = dyn_cast<ConstantInt>(*idxIt)) {
                            uint64_t fieldIndex = ci->getZExtValue();
                            updatedFields.insert({structName.str(), fieldIndex});
                            updated = true;
                        }
                    }
                }
            }
        }
    }

    return updated;
}

/**
 * @brief Recursively traces a value back to check if it originates from gpu_graph
 *
 * Handles: loads, GEP, bitcasts, pointer arithmetic (add/sub), phi nodes
 */
static bool tracesToGraphField(Value *val, std::set<Value*> &visited) {
    if (!val || visited.count(val)) return false;
    visited.insert(val);

    // Check if it's a load from a gpu_graph field. Match the canonical
    // struct names — `class.gpu_graph` and its base `class.gpu_graph_base`
    // (which actually owns adjwgt/adjncy/xadj) plus the C-linkage struct
    // spellings. A naked substring match on "gpu_graph" would also catch
    // unrelated types whose mangled name happens to include the string
    // (nested types, captures), so the set is enumerated explicitly.
    if (auto *loadInst = llvm::dyn_cast<llvm::LoadInst>(val)) {
        Value *loadPtr = loadInst->getPointerOperand();
        if (GetElementPtrInst *gep = traceToStructGEP(loadPtr)) {
            llvm::Type *srcTy = gep->getSourceElementType();
            if (llvm::StructType *structTy = llvm::dyn_cast<llvm::StructType>(srcTy)) {
                if (structTy->hasName()) {
                    llvm::StringRef name = structTy->getName();
                    if (name == "class.gpu_graph" ||
                        name == "class.gpu_graph_base" ||
                        name == "struct.gpu_graph" ||
                        name == "struct.gpu_graph_base") {
                        return true;
                    }
                }
            }
        }
        // Recursively check what we're loading from
        return tracesToGraphField(loadPtr, visited);
    }

    // Check GEP instructions (array indexing or pointer arithmetic)
    if (auto *gep = llvm::dyn_cast<llvm::GetElementPtrInst>(val)) {
        return tracesToGraphField(gep->getPointerOperand(), visited);
    }

    // Check pointer arithmetic (add, sub with pointers)
    if (auto *binOp = llvm::dyn_cast<llvm::BinaryOperator>(val)) {
        if (binOp->getOpcode() == llvm::Instruction::Add ||
            binOp->getOpcode() == llvm::Instruction::Sub) {
            return tracesToGraphField(binOp->getOperand(0), visited) ||
                   tracesToGraphField(binOp->getOperand(1), visited);
        }
    }

    // Check bitcasts and other casts
    if (auto *castInst = llvm::dyn_cast<llvm::CastInst>(val)) {
        return tracesToGraphField(castInst->getOperand(0), visited);
    }

    // Check PHI nodes (from loops, conditionals)
    if (auto *phi = llvm::dyn_cast<llvm::PHINode>(val)) {
        for (unsigned i = 0; i < phi->getNumIncomingValues(); ++i) {
            if (tracesToGraphField(phi->getIncomingValue(i), visited)) {
                return true;
            }
        }
    }

    return false;
}

bool analyzeGraphModification(Function &F, bool &graphFieldModified) {
    graphFieldModified = false;

    // Iterate through all instructions
    for (auto &I : llvm::instructions(F)) {
        // Check for store instructions
        if (auto *storeInst = llvm::dyn_cast<llvm::StoreInst>(&I)) {
            Value *ptr = storeInst->getPointerOperand();
            std::set<Value*> visited;

            // Trace the pointer to see if it originates from gpu_graph
            if (tracesToGraphField(ptr, visited)) {
                graphFieldModified = true;

                if (AnalyzerConfig::DEBUG_ENABLED) {
                    llvm::errs() << "[analyzeGraphModification] Detected store to gpu_graph-derived pointer"
                               << " in function " << F.getName() << "\n";
                }
                return true;
            }
        }

        // Native LLVM atomics (atomicrmw, cmpxchg). The CallInst branch
        // below catches CUDA atomicAdd/atomicCAS/etc. that lower as
        // intrinsic calls, but recent CUDA / NVPTX targets sometimes
        // lower atomics directly to atomicrmw/cmpxchg IR — those would
        // otherwise be missed and let a graph-mutating walker slip past
        // the eRVS_only gate.
        if (auto *atomicRMW = llvm::dyn_cast<llvm::AtomicRMWInst>(&I)) {
            std::set<Value*> visited;
            if (tracesToGraphField(atomicRMW->getPointerOperand(), visited)) {
                graphFieldModified = true;
                if (AnalyzerConfig::DEBUG_ENABLED) {
                    llvm::errs() << "[analyzeGraphModification] Detected atomicrmw on gpu_graph-derived pointer"
                               << " in function " << F.getName() << "\n";
                }
                return true;
            }
        }
        if (auto *cmpXchg = llvm::dyn_cast<llvm::AtomicCmpXchgInst>(&I)) {
            std::set<Value*> visited;
            if (tracesToGraphField(cmpXchg->getPointerOperand(), visited)) {
                graphFieldModified = true;
                if (AnalyzerConfig::DEBUG_ENABLED) {
                    llvm::errs() << "[analyzeGraphModification] Detected cmpxchg on gpu_graph-derived pointer"
                               << " in function " << F.getName() << "\n";
                }
                return true;
            }
        }

        // Check for atomic operations
        if (auto *callInst = llvm::dyn_cast<llvm::CallInst>(&I)) {
            llvm::Function *calledFunc = callInst->getCalledFunction();
            if (calledFunc) {
                std::string calledName = calledFunc->getName().str();

                // Check if it's an atomic operation
                if (isAtomicFunctionName(calledName) && callInst->arg_size() >= 1) {
                    Value *atomicPtr = callInst->getArgOperand(0);
                    std::set<Value*> visited;

                    // Trace to see if atomic target is from gpu_graph
                    if (tracesToGraphField(atomicPtr, visited)) {
                        graphFieldModified = true;

                        if (AnalyzerConfig::DEBUG_ENABLED) {
                            llvm::errs() << "[analyzeGraphModification] Detected atomic op ("
                                       << calledName << ") on gpu_graph-derived pointer"
                                       << " in function " << F.getName() << "\n";
                        }
                        return true;
                    }
                }
            }
        }
    }

    return graphFieldModified;
}

} // namespace llvm_analysis