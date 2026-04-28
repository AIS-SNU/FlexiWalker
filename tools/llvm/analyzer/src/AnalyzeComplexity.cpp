#include "AnalyzeComplexity.hpp"
#include "util.hpp"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Dominators.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/ADT/SmallVector.h"

#include <algorithm>
#include <queue>

namespace llvm_analysis {

bool hasRecursiveCalls(llvm::Function &F, std::set<llvm::Function*> &visited) {
    if (visited.count(&F)) {
        // Found a cycle - this is recursion
        return true;
    }

    visited.insert(&F);

    // Check all call instructions in this function
    for (auto &I : llvm::instructions(F)) {
        if (auto *callInst = llvm::dyn_cast<llvm::CallInst>(&I)) {
            llvm::Function *calledFunc = callInst->getCalledFunction();

            if (!calledFunc) continue;  // Indirect call, skip

            // Direct recursion
            if (calledFunc == &F) {
                return true;
            }

            // Indirect recursion - check called function
            if (hasRecursiveCalls(*calledFunc, visited)) {
                return true;
            }
        }
    }

    visited.erase(&F);
    return false;
}

// Walks `condition` backwards through cmps/binops/casts and reports
// whether the value was produced by a PHI (loop-carried dependency) or
// a Load (memory-dependent). For early-exit conditions either is taken
// as evidence that the loop bound isn't a simple induction variable.
static bool conditionIsDataDependent(llvm::Value *condition) {
    std::set<llvm::Value*> visited;
    std::queue<llvm::Value*> worklist;
    worklist.push(condition);

    while (!worklist.empty()) {
        llvm::Value *val = worklist.front();
        worklist.pop();

        if (!val || !visited.insert(val).second) continue;

        if (llvm::isa<llvm::PHINode>(val) || llvm::isa<llvm::LoadInst>(val)) {
            return true;
        }

        if (auto *cmp = llvm::dyn_cast<llvm::CmpInst>(val)) {
            worklist.push(cmp->getOperand(0));
            worklist.push(cmp->getOperand(1));
            continue;
        }
        if (auto *binOp = llvm::dyn_cast<llvm::BinaryOperator>(val)) {
            worklist.push(binOp->getOperand(0));
            worklist.push(binOp->getOperand(1));
            continue;
        }
        if (auto *cast = llvm::dyn_cast<llvm::CastInst>(val)) {
            worklist.push(cast->getOperand(0));
            continue;
        }
    }
    return false;
}

bool hasDataDependentExits(llvm::Function &F) {
    // Detect loops whose exits aren't simple counter checks. The
    // previous implementation re-derived loop structure from
    // function-iteration order — that only works because clang typically
    // lays blocks out in topological-ish order, and breaks for any IR
    // with non-trivial layout (loop rotation, switch lowering, inlined
    // helpers). LoopInfo gives us the canonical CFG analysis: proper
    // header / latch / exiting-block identification using dominance.
    llvm::DominatorTree DT(F);
    llvm::LoopInfo LI(DT);

    if (LI.empty()) {
        return false;  // No loops
    }

    unsigned complexExits = 0;
    for (llvm::Loop *loop : LI.getLoopsInPreorder()) {
        llvm::SmallVector<llvm::BasicBlock*, 4> exitingBlocks;
        loop->getExitingBlocks(exitingBlocks);

        for (llvm::BasicBlock *exitingBB : exitingBlocks) {
            // Skip the loop header — that's the natural loop condition,
            // not an early exit. (Anything else is a `break`, an early
            // `return`, or a data-dependent guard inside the body.)
            if (exitingBB == loop->getHeader()) {
                continue;
            }

            auto *term = exitingBB->getTerminator();
            auto *br = llvm::dyn_cast<llvm::BranchInst>(term);
            if (!br || !br->isConditional()) continue;

            if (conditionIsDataDependent(br->getCondition())) {
                complexExits++;
            }
        }
    }

    return complexExits > 0;
}

unsigned computeNestingDepth(llvm::Function &F) {
    // Compute actual nesting depth using dominator tree analysis
    // A flat if-else-if chain has depth 1, nested ifs increase depth

    llvm::DominatorTree DT(F);

    // For each basic block, compute its nesting level based on dominator depth
    // and number of conditional branches on the path to entry
    unsigned maxDepth = 0;

    for (auto &BB : F) {
        // Count conditional branches on the dominator path to entry
        unsigned branchCount = 0;
        llvm::BasicBlock *current = &BB;

        while (current != nullptr) {
            // Check if current block ends with a conditional branch
            auto *term = current->getTerminator();
            if (term && llvm::isa<llvm::BranchInst>(term)) {
                auto *br = llvm::cast<llvm::BranchInst>(term);
                if (br->isConditional()) {
                    branchCount++;
                }
            }

            // Move up the dominator tree
            llvm::DomTreeNode *node = DT.getNode(current);
            if (!node || !node->getIDom()) {
                break;
            }
            current = node->getIDom()->getBlock();
        }

        maxDepth = std::max(maxDepth, branchCount);
    }

    return maxDepth;
}

bool analyzeComplexity(llvm::Function &F, bool &isComplex, const std::string &className) {
    isComplex = false;

    // 1. Check for recursive calls
    std::set<llvm::Function*> visited;
    if (hasRecursiveCalls(F, visited)) {
        isComplex = true;
        if (AnalyzerConfig::DEBUG_ENABLED) {
            llvm::errs() << "[AnalyzeComplexity] " << className << "::get_weight() "
                        << "contains recursive calls\n";
        }
        return true;
    }

    // 2. Check for data-dependent loop exits
    if (hasDataDependentExits(F)) {
        isComplex = true;
        if (AnalyzerConfig::DEBUG_ENABLED) {
            llvm::errs() << "[AnalyzeComplexity] " << className << "::get_weight() "
                        << "contains data-dependent exits\n";
        }
        return true;
    }

    // 3. Check for deep nesting
    unsigned nestingDepth = computeNestingDepth(F);
    if (nestingDepth > ComplexityThresholds::MAX_NESTING_DEPTH) {
        isComplex = true;
        if (AnalyzerConfig::DEBUG_ENABLED) {
            llvm::errs() << "[AnalyzeComplexity] " << className << "::get_weight() "
                        << "has excessive nesting depth: " << nestingDepth
                        << " (max allowed: " << ComplexityThresholds::MAX_NESTING_DEPTH << ")\n";
        }
        return true;
    }

    return false;
}

} // namespace llvm_analysis
