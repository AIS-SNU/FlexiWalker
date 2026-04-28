#include "AnalyzeSynchronization.hpp"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <fstream>

namespace llvm_analysis {

// List of dangerous synchronization primitives
static const std::vector<std::string> WARP_INTRINSICS = {
    "__ballot_sync",
    "__shfl_sync",
    "__shfl_up_sync",
    "__shfl_down_sync",
    "__shfl_xor_sync",
    "__any_sync",
    "__all_sync",
    "__uni_sync",
    "__activemask",
    "__match_any_sync",
    "__match_all_sync"
};

static const std::vector<std::string> SYNC_PRIMITIVES = {
    "__syncwarp",
    "__syncthreads",
    "__syncthreads_count",
    "__syncthreads_and",
    "__syncthreads_or",
    "__threadfence",
    "__threadfence_block",
    "__threadfence_system"
};

static const std::vector<std::string> ATOMIC_PRIMITIVES = {
    "atomicAdd",
    "atomicSub",
    "atomicExch",
    "atomicMin",
    "atomicMax",
    "atomicInc",
    "atomicDec",
    "atomicCAS",
    "atomicAnd",
    "atomicOr",
    "atomicXor"
};

/**
 * @brief Checks if `funcName` references one of the dangerous primitives.
 *
 * Matching uses a longest-prefix-first scan over the union of all
 * primitive lists: a substring search would mis-attribute
 * "__syncthreads_count" as "__syncthreads" because the shorter name is
 * also a substring. By sorting candidates by length descending we report
 * the longest applicable primitive name. We still allow substring
 * matching (rather than equality) because device function names appear
 * mangled in IR — the primitive name may be embedded in a longer symbol.
 */
static bool isDangerousPrimitive(const std::string &funcName, std::string &primitiveName) {
    static const std::vector<std::string> ALL_PRIMITIVES = []() {
        std::vector<std::string> all;
        all.insert(all.end(), WARP_INTRINSICS.begin(), WARP_INTRINSICS.end());
        all.insert(all.end(), SYNC_PRIMITIVES.begin(), SYNC_PRIMITIVES.end());
        all.insert(all.end(), ATOMIC_PRIMITIVES.begin(), ATOMIC_PRIMITIVES.end());
        std::sort(all.begin(), all.end(),
                  [](const std::string &a, const std::string &b) {
                      return a.size() > b.size();
                  });
        return all;
    }();

    for (const auto &name : ALL_PRIMITIVES) {
        if (funcName.find(name) != std::string::npos) {
            primitiveName = name;
            return true;
        }
    }

    return false;
}

bool analyzeSynchronization(llvm::Function &F,
                           std::vector<SyncPrimitiveInfo> &detectedSyncs,
                           const std::string &className,
                           const std::string &methodName) {
    bool foundAny = false;

    auto recordNative = [&](const llvm::Instruction *inst, const char *label) {
        SyncPrimitiveInfo info;
        info.functionName = methodName;
        info.primitiveName = label;
        info.className = className;
        info.lineNumber = 0;
        if (const llvm::DebugLoc &debugLoc = inst->getDebugLoc()) {
            info.lineNumber = debugLoc.getLine();
        }
        detectedSyncs.push_back(info);
        foundAny = true;
    };

    // Iterate through all instructions in the function
    for (auto &I : llvm::instructions(F)) {
        // Native LLVM atomics (atomicrmw, cmpxchg). CUDA atomic builtins
        // sometimes lower to these directly rather than to named calls,
        // so the CallInst branch below would miss them.
        if (auto *atomicRMW = llvm::dyn_cast<llvm::AtomicRMWInst>(&I)) {
            recordNative(atomicRMW, "atomicrmw");
            continue;
        }
        if (auto *cmpXchg = llvm::dyn_cast<llvm::AtomicCmpXchgInst>(&I)) {
            recordNative(cmpXchg, "cmpxchg");
            continue;
        }

        // Check if this is a call instruction
        if (auto *callInst = llvm::dyn_cast<llvm::CallInst>(&I)) {
            llvm::Function *calledFunc = callInst->getCalledFunction();

            if (!calledFunc) {
                // Indirect call - check the called value's name
                if (auto *calledValue = callInst->getCalledOperand()) {
                    std::string calledName = calledValue->getName().str();
                    std::string primitiveName;

                    if (isDangerousPrimitive(calledName, primitiveName)) {
                        SyncPrimitiveInfo info;
                        info.functionName = methodName;
                        info.primitiveName = primitiveName;
                        info.className = className;
                        info.lineNumber = 0; // LLVM IR may not have source line info

                        // Try to get debug location if available
                        if (const llvm::DebugLoc &debugLoc = callInst->getDebugLoc()) {
                            info.lineNumber = debugLoc.getLine();
                        }

                        detectedSyncs.push_back(info);
                        foundAny = true;
                    }
                }
            } else {
                // Direct call - check the function name
                std::string calledName = calledFunc->getName().str();
                std::string primitiveName;

                if (isDangerousPrimitive(calledName, primitiveName)) {
                    SyncPrimitiveInfo info;
                    info.functionName = methodName;
                    info.primitiveName = primitiveName;
                    info.className = className;
                    info.lineNumber = 0;

                    // Try to get debug location if available
                    if (const llvm::DebugLoc &debugLoc = callInst->getDebugLoc()) {
                        info.lineNumber = debugLoc.getLine();
                    }

                    detectedSyncs.push_back(info);
                    foundAny = true;
                }
            }
        }
    }

    return foundAny;
}

void printSynchronizationWarnings(const std::vector<SyncPrimitiveInfo> &detectedSyncs,
                                  const std::string &outputPath) {
    if (detectedSyncs.empty()) {
        return;
    }

    // Prepare warning message
    std::string warningMessage;
    llvm::raw_string_ostream ss(warningMessage);

    ss << "\n";
    ss << "================================================================================\n";
    ss << "WARNING: Inter-thread Communication Detected in User Code\n";
    ss << "================================================================================\n";
    ss << "\n";
    ss << "The following synchronization primitives were detected in user-written walker\n";
    ss << "methods. These operations can cause DEADLOCKS or STALLS because different\n";
    ss << "threads may take different control flow paths during random walk execution.\n";
    ss << "\n";

    for (const auto &info : detectedSyncs) {
        ss << "  [" << info.className << "::" << info.functionName << "] ";
        ss << "Found: " << info.primitiveName;

        if (info.lineNumber > 0) {
            ss << " (line " << info.lineNumber << ")";
        }

        ss << "\n";
    }

    ss << "\n";
    ss << "Recommendation:\n";
    ss << "  - Avoid using warp intrinsics (__ballot_sync, __shfl_*, etc.)\n";
    ss << "  - Avoid synchronization primitives (__syncwarp, __syncthreads, etc.)\n";
    ss << "  - Use atomic operations only when absolutely necessary\n";
    ss << "  - Ensure all threads in a warp take the same code path\n";
    ss << "\n";
    ss << "================================================================================\n";
    ss << "\n";

    ss.flush();

    // Print to stderr
    llvm::errs() << warningMessage;

    // Write to file when an explicit path is provided. Empty path means
    // "stderr only" — historically this wrote to a fixed
    // `synchronization_warnings.txt` in the process CWD, which collided
    // when multiple pipeline invocations shared a directory.
    if (outputPath.empty()) {
        return;
    }
    std::ofstream outFile(outputPath);
    if (outFile.is_open()) {
        outFile << warningMessage;
        outFile.close();
        llvm::errs() << "Synchronization warnings written to: " << outputPath << "\n";
    } else {
        llvm::errs() << "Error: Could not open " << outputPath << " for writing\n";
    }
}

} // namespace llvm_analysis
