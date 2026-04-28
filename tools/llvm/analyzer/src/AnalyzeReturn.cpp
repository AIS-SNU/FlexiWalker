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

#include "AnalyzeReturn.hpp"

using namespace llvm;

namespace llvm_analysis {

// Forward declarations for cleaner organization
static ReturnFlag handleConstant(Constant* constant, bool& possibleZero);
static ReturnFlag handleAlloca(AllocaInst* alloca, std::unordered_set<Value*>& visited, 
                              std::set<std::pair<std::string, uint64_t>>& updatedFields, 
                              std::set<std::pair<std::string, uint64_t>>& accessedByIndex, 
                              bool& possibleZero, std::map<Value*, std::string>& ssaToVarName, 
                              std::set<AllocaInst*>& externallyDefinedLocals, 
                              std::set<std::string>& externalLocals, std::string className);
static ReturnFlag handleSelect(SelectInst* sel, std::unordered_set<Value*>& visited, 
                              std::set<std::pair<std::string, uint64_t>>& updatedFields, 
                              std::set<std::pair<std::string, uint64_t>>& accessedByIndex, 
                              bool& possibleZero, std::map<Value*, std::string>& ssaToVarName, 
                              std::set<AllocaInst*>& externallyDefinedLocals, 
                              std::set<std::string>& externalLocals, std::string className);
static ReturnFlag handleGEP(GetElementPtrInst* gep, std::unordered_set<Value*>& visited, 
                            std::set<std::pair<std::string, uint64_t>>& updatedFields, 
                            std::set<std::pair<std::string, uint64_t>>& accessedByIndex, 
                            bool& possibleZero, std::map<Value*, std::string>& ssaToVarName, 
                            std::set<AllocaInst*>& externallyDefinedLocals, 
                            std::set<std::string>& externalLocals, std::string className);
static ReturnFlag handleLoad(LoadInst* load, std::unordered_set<Value*>& visited, 
                            std::set<std::pair<std::string, uint64_t>>& updatedFields, 
                            std::set<std::pair<std::string, uint64_t>>& accessedByIndex, 
                            bool& possibleZero, std::map<Value*, std::string>& ssaToVarName, 
                            std::set<AllocaInst*>& externallyDefinedLocals, 
                            std::set<std::string>& externalLocals, std::string className);
static ReturnFlag handleArgument(Argument* arg);
static ReturnFlag handlePHI(PHINode* phi, std::unordered_set<Value*>& visited, 
                            std::set<std::pair<std::string, uint64_t>>& updatedFields, 
                            std::set<std::pair<std::string, uint64_t>>& accessedByIndex, 
                            bool& possibleZero, std::map<Value*, std::string>& ssaToVarName, 
                            std::set<AllocaInst*>& externallyDefinedLocals, 
                            std::set<std::string>& externalLocals, std::string className);
static ReturnFlag handleBinaryOp(BinaryOperator* binOp, std::unordered_set<Value*>& visited, 
                                 std::set<std::pair<std::string, uint64_t>>& updatedFields, 
                                 std::set<std::pair<std::string, uint64_t>>& accessedByIndex, 
                                 bool& possibleZero, std::map<Value*, std::string>& ssaToVarName, 
                                 std::set<AllocaInst*>& externallyDefinedLocals, 
                                 std::set<std::string>& externalLocals, std::string className);
static ReturnFlag handleCast(CastInst* cast, std::unordered_set<Value*>& visited, 
                             std::set<std::pair<std::string, uint64_t>>& updatedFields, 
                             std::set<std::pair<std::string, uint64_t>>& accessedByIndex, 
                             bool& possibleZero, std::map<Value*, std::string>& ssaToVarName, 
                             std::set<AllocaInst*>& externallyDefinedLocals, 
                             std::set<std::string>& externalLocals, std::string className);
static ReturnFlag handleInstruction(Instruction* inst, std::unordered_set<Value*>& visited, 
                                   std::set<std::pair<std::string, uint64_t>>& updatedFields, 
                                   std::set<std::pair<std::string, uint64_t>>& accessedByIndex, 
                                   bool& possibleZero, std::map<Value*, std::string>& ssaToVarName, 
                                   std::set<AllocaInst*>& externallyDefinedLocals, 
                                   std::set<std::string>& externalLocals, std::string className);

// Main traversal function - now much cleaner
ReturnFlag funcTraverser(Value* val, 
    std::unordered_set<Value*>& visited, 
    std::set<std::pair<std::string, uint64_t>> &updatedFields, 
    std::set<std::pair<std::string, uint64_t>> &accessedByIndex, 
    bool &possibleZero, 
    std::map<Value*, std::string> &ssaToVarName, 
    std::set<AllocaInst*> &externallyDefinedLocals, 
    std::set<std::string> &externalLocals,
    std::string className) {
    
    // Check if already visited
    if (!visited.insert(val).second) return INVALID;

    if (DEBUG) errs() << *val << "\n";

    // Dispatch to specialized handlers based on value type
    if (auto* constant = dyn_cast<Constant>(val)) {
        return handleConstant(constant, possibleZero);
    }
    if (auto* alloca = dyn_cast<AllocaInst>(val)) {
        return handleAlloca(alloca, visited, updatedFields, accessedByIndex, possibleZero, 
                           ssaToVarName, externallyDefinedLocals, externalLocals, className);
    }
    if (auto* select = dyn_cast<SelectInst>(val)) {
        return handleSelect(select, visited, updatedFields, accessedByIndex, possibleZero, 
                           ssaToVarName, externallyDefinedLocals, externalLocals, className);
    }
    if (auto* gep = dyn_cast<GetElementPtrInst>(val)) {
        return handleGEP(gep, visited, updatedFields, accessedByIndex, possibleZero, 
                        ssaToVarName, externallyDefinedLocals, externalLocals, className);
    }
    if (auto* load = dyn_cast<LoadInst>(val)) {
        return handleLoad(load, visited, updatedFields, accessedByIndex, possibleZero, 
                         ssaToVarName, externallyDefinedLocals, externalLocals, className);
    }
    if (auto* arg = dyn_cast<Argument>(val)) {
        return handleArgument(arg);
    }
    if (auto* phi = dyn_cast<PHINode>(val)) {
        return handlePHI(phi, visited, updatedFields, accessedByIndex, possibleZero, 
                        ssaToVarName, externallyDefinedLocals, externalLocals, className);
    }
    if (auto* binOp = dyn_cast<BinaryOperator>(val)) {
        return handleBinaryOp(binOp, visited, updatedFields, accessedByIndex, possibleZero, 
                             ssaToVarName, externallyDefinedLocals, externalLocals, className);
    }
    if (auto* cast = dyn_cast<CastInst>(val)) {
        return handleCast(cast, visited, updatedFields, accessedByIndex, possibleZero, 
                         ssaToVarName, externallyDefinedLocals, externalLocals, className);
    }
    if (auto* inst = dyn_cast<Instruction>(val)) {
        return handleInstruction(inst, visited, updatedFields, accessedByIndex, possibleZero, 
                                ssaToVarName, externallyDefinedLocals, externalLocals, className);
    }

    // Some Other Case Not Currently Considered
    return INVALID;
}

// Specialized handler implementations
static ReturnFlag handleConstant(Constant* constant, bool& possibleZero) {
    if (auto* ci = dyn_cast<ConstantInt>(constant)) {
        if (ci->isZero()) possibleZero = true;
    } else if (auto* cf = dyn_cast<ConstantFP>(constant)) {
        if (cf->isZero()) possibleZero = true;
    }
    return CONST;
}

static ReturnFlag handleAlloca(AllocaInst* alloca, std::unordered_set<Value*>& visited, 
                              std::set<std::pair<std::string, uint64_t>>& updatedFields, 
                              std::set<std::pair<std::string, uint64_t>>& accessedByIndex, 
                              bool& possibleZero, std::map<Value*, std::string>& ssaToVarName, 
                              std::set<AllocaInst*>& externallyDefinedLocals, 
                              std::set<std::string>& externalLocals, std::string className) {
    bool allConst = true;
    ReturnFlag worst = INVALID;

    for (User* user : alloca->users()) {
        if (StoreInst* store = dyn_cast<StoreInst>(user)) {
            Value* storedVal = store->getValueOperand();
            ReturnFlag f = funcTraverser(storedVal, visited, updatedFields, accessedByIndex, 
                                       possibleZero, ssaToVarName, externallyDefinedLocals, 
                                       externalLocals, className);

            if (f > worst) worst = f;
            if (f != CONST) allConst = false;
        }
    }

    if (!allConst) return worst;
    return CONST;
}

static ReturnFlag handleSelect(SelectInst* sel, std::unordered_set<Value*>& visited, 
                              std::set<std::pair<std::string, uint64_t>>& updatedFields, 
                              std::set<std::pair<std::string, uint64_t>>& accessedByIndex, 
                              bool& possibleZero, std::map<Value*, std::string>& ssaToVarName, 
                              std::set<AllocaInst*>& externallyDefinedLocals, 
                              std::set<std::string>& externalLocals, std::string className) {
    // Only care about true/false values — not the condition!
    Value* trueVal = sel->getTrueValue();
    Value* falseVal = sel->getFalseValue();

    ReturnFlag tf = funcTraverser(trueVal, visited, updatedFields, accessedByIndex, 
                                 possibleZero, ssaToVarName, externallyDefinedLocals, 
                                 externalLocals, className);
    ReturnFlag ff = funcTraverser(falseVal, visited, updatedFields, accessedByIndex, 
                                 possibleZero, ssaToVarName, externallyDefinedLocals, 
                                 externalLocals, className);

    return std::max(tf, ff); // conservatively take the worst
}

static ReturnFlag handleGEP(GetElementPtrInst* gep, std::unordered_set<Value*>& visited, 
                            std::set<std::pair<std::string, uint64_t>>& updatedFields, 
                            std::set<std::pair<std::string, uint64_t>>& accessedByIndex, 
                            bool& possibleZero, std::map<Value*, std::string>& ssaToVarName, 
                            std::set<AllocaInst*>& externallyDefinedLocals, 
                            std::set<std::string>& externalLocals, std::string className) {
    Value* base = gep->getPointerOperand()->stripPointerCasts();
    ReturnFlag worst = funcTraverser(base, visited, updatedFields, accessedByIndex, 
                                    possibleZero, ssaToVarName, externallyDefinedLocals, 
                                    externalLocals, className);

    Type* sourceTy = gep->getSourceElementType();

    if (StructType* structTy = dyn_cast<StructType>(sourceTy)) {
        if (structTy->hasName()) {
            StringRef tyName = structTy->getName();  // e.g., %class.Node2vec
            std::string structName = tyName.str();
            if (DEBUG) errs() << "GEP is accessing type: " << tyName << "\n";

            // 🔍 Check for updated fields
            if (updatedFields.size() > 0 && gep->getNumIndices() >= 2) {
                auto idxIt = gep->idx_begin();
                ++idxIt; // skip the 0
                if (ConstantInt* CI = dyn_cast<ConstantInt>(*idxIt)) {
                    uint64_t fieldIdx = CI->getZExtValue();
                    if (DEBUG) errs() << "updatedFields.idxIt: " << fieldIdx << "\n";
                    if (updatedFields.count({structName, fieldIdx})) {
                        if (DEBUG) errs() << "GEP accessing updated field: " << structName << "[" << fieldIdx << "]\n";
                        return DYNAMIC_MEMBER;
                    }
                }
            }

            // If the struct is the walker class
            if (tyName.contains(className)) {
                return std::max(APP_MEMBER, worst);
            }
        }
    }

    // Detect if THIS GEP has any non-constant index — that's the robust signal
    // that the access is indexed (e.g., `float[i]` on a loaded field pointer).
    // Relying on `worst == INDEX_MEMBER` is unreliable for multi-field walkers:
    // the shared `visited` set causes the second field's index chain to short-circuit
    // to INVALID, which loses the INDEX_MEMBER flag and drops the capture.
    bool hasNonConstIdx = false;
    for (auto idx = gep->idx_begin(); idx != gep->idx_end(); ++idx) {
        // Skip constant integer indices from affecting possibleZero
        if (isa<ConstantInt>(idx)) {
            if (DEBUG) errs() << "Skipping constant GEP index: " << *idx << "\n";
            continue;
        }

        hasNonConstIdx = true;
        ReturnFlag idxFlag = funcTraverser(*idx, visited, updatedFields, accessedByIndex,
                                          possibleZero, ssaToVarName, externallyDefinedLocals,
                                          externalLocals, className);
        if (DEBUG) errs() << "GEP index: " << **idx << " → flag: " << static_cast<int>(idxFlag) << "\n";
        worst = std::max(worst, idxFlag);
    }

    if (hasNonConstIdx) {
        if (DEBUG) errs() << "GEP has non-constant index; tracing to struct field\n";

        // Follow pointer chain to find struct GEP that produced this pointer
        Value* gepBase = gep->getPointerOperand()->stripPointerCasts();

        // Step through a Load if necessary
        if (LoadInst* li = dyn_cast<LoadInst>(gepBase)) {
            gepBase = li->getPointerOperand()->stripPointerCasts();
        }

        // Now try to identify the struct and field index
        if (GetElementPtrInst* baseGEP = dyn_cast<GetElementPtrInst>(gepBase)) {
            Type* baseTy = baseGEP->getSourceElementType();
            if (StructType* structTy = dyn_cast<StructType>(baseTy)) {
                if (structTy->hasName() && baseGEP->getNumIndices() >= 2) {
                    std::string structName = structTy->getName().str();

                    auto idxIt = baseGEP->idx_begin();
                    ++idxIt; // skip leading 0
                    if (ConstantInt* CI = dyn_cast<ConstantInt>(*idxIt)) {
                        uint64_t fieldIdx = CI->getZExtValue();
                        accessedByIndex.insert({structName, fieldIdx});

                        if (DEBUG) {
                            errs() << "Indexed access → " << structName
                                   << "[" << fieldIdx << "]\n";
                        }
                    }
                }
            }
        }
    }

    return worst;
}

static ReturnFlag handleLoad(LoadInst* load, std::unordered_set<Value*>& visited, 
                            std::set<std::pair<std::string, uint64_t>>& updatedFields, 
                            std::set<std::pair<std::string, uint64_t>>& accessedByIndex, 
                            bool& possibleZero, std::map<Value*, std::string>& ssaToVarName, 
                            std::set<AllocaInst*>& externallyDefinedLocals, 
                            std::set<std::string>& externalLocals, std::string className) {
    Value* ptr = load->getPointerOperand()->stripPointerCasts();

    if (auto* alloca = dyn_cast<AllocaInst>(ptr)) {
        if (ssaToVarName.count(alloca) && externallyDefinedLocals.count(alloca)) {
            if (DEBUG) errs() << "✅ Externally defined variable used: " << ssaToVarName[alloca] << "\n";
            externalLocals.insert(ssaToVarName[alloca]);
        }
    }

    return funcTraverser(ptr, visited, updatedFields, accessedByIndex, possibleZero, 
                        ssaToVarName, externallyDefinedLocals, externalLocals, className);
}

static ReturnFlag handleArgument(Argument* arg) {
    unsigned argIndex = 0;
    for (auto& a : arg->getParent()->args()) {
        if (&a == arg) break;
        argIndex++;
    }

    if (DEBUG) errs() << "Function argument index: " << argIndex << "\n";

    if (argIndex == 1) return TASK_MEMBER;     // task
    if (argIndex == 2) return INDEX_MEMBER;    // i
    return INVALID;
}

static ReturnFlag handlePHI(PHINode* phi, std::unordered_set<Value*>& visited, 
                            std::set<std::pair<std::string, uint64_t>>& updatedFields, 
                            std::set<std::pair<std::string, uint64_t>>& accessedByIndex, 
                            bool& possibleZero, std::map<Value*, std::string>& ssaToVarName, 
                            std::set<AllocaInst*>& externallyDefinedLocals, 
                            std::set<std::string>& externalLocals, std::string className) {
    if (DEBUG) errs() << "Analyzing PHI node with " << phi->getNumIncomingValues() << " incoming values\n";
    
    ReturnFlag worst = INVALID;
    
    // Analyze all incoming values from different basic blocks
    for (unsigned i = 0; i < phi->getNumIncomingValues(); ++i) {
        Value* incomingVal = phi->getIncomingValue(i);
        ReturnFlag flag = funcTraverser(incomingVal, visited, updatedFields, accessedByIndex, 
                                       possibleZero, ssaToVarName, externallyDefinedLocals, 
                                       externalLocals, className);
        worst = std::max(worst, flag);
    }
    
    return worst; // Take the most dynamic among all incoming values
}

static ReturnFlag handleBinaryOp(BinaryOperator* binOp, std::unordered_set<Value*>& visited, 
                                 std::set<std::pair<std::string, uint64_t>>& updatedFields, 
                                 std::set<std::pair<std::string, uint64_t>>& accessedByIndex, 
                                 bool& possibleZero, std::map<Value*, std::string>& ssaToVarName, 
                                 std::set<AllocaInst*>& externallyDefinedLocals, 
                                 std::set<std::string>& externalLocals, std::string className) {
    if (DEBUG) errs() << "Analyzing binary operation: " << binOp->getOpcodeName() << "\n";
    
    Value* lhs = binOp->getOperand(0);
    Value* rhs = binOp->getOperand(1);
    
    ReturnFlag lhsFlag = funcTraverser(lhs, visited, updatedFields, accessedByIndex, 
                                      possibleZero, ssaToVarName, externallyDefinedLocals, 
                                      externalLocals, className);
    ReturnFlag rhsFlag = funcTraverser(rhs, visited, updatedFields, accessedByIndex, 
                                      possibleZero, ssaToVarName, externallyDefinedLocals, 
                                      externalLocals, className);
    
    // Binary operations inherit the most dynamic operand
    return std::max(lhsFlag, rhsFlag);
}

static ReturnFlag handleCast(CastInst* cast, std::unordered_set<Value*>& visited, 
                             std::set<std::pair<std::string, uint64_t>>& updatedFields, 
                             std::set<std::pair<std::string, uint64_t>>& accessedByIndex, 
                             bool& possibleZero, std::map<Value*, std::string>& ssaToVarName, 
                             std::set<AllocaInst*>& externallyDefinedLocals, 
                             std::set<std::string>& externalLocals, std::string className) {
    if (DEBUG) errs() << "Analyzing cast instruction: " << cast->getOpcodeName() << "\n";
    
    Value* operand = cast->getOperand(0);
    
    // Cast instructions preserve the dependency characteristics of their operand
    return funcTraverser(operand, visited, updatedFields, accessedByIndex, 
                        possibleZero, ssaToVarName, externallyDefinedLocals, 
                        externalLocals, className);
}

static ReturnFlag handleInstruction(Instruction* inst, std::unordered_set<Value*>& visited, 
                                   std::set<std::pair<std::string, uint64_t>>& updatedFields, 
                                   std::set<std::pair<std::string, uint64_t>>& accessedByIndex, 
                                   bool& possibleZero, std::map<Value*, std::string>& ssaToVarName, 
                                   std::set<AllocaInst*>& externallyDefinedLocals, 
                                   std::set<std::string>& externalLocals, std::string className) {
    ReturnFlag worst = INVALID;

    for (Use& op : inst->operands()) {
        ReturnFlag f = funcTraverser(op.get(), visited, updatedFields, accessedByIndex, 
                                    possibleZero, ssaToVarName, externallyDefinedLocals, 
                                    externalLocals, className);
        if (f > worst) worst = f;
    }

    return worst;
}

// Entry wrapper
bool analyzeReturn(Function &F, std::map<std::string, ReturnFlag> &walkerFlagMap, std::set<std::pair<std::string, uint64_t>> &updatedFields, std::set<std::pair<std::string, uint64_t>> &accessedByIndex, std::map<std::string, bool> &possibleZeroMap, std::set<std::string> &externalLocals, std::string className, std::string macroName) {
    ReturnFlag flag = INVALID;
    bool hasReturn = false;
    bool possibleZero = false;

    std::map<Value*, std::string> ssaToVarName;

    for (Instruction &I : instructions(F)) {
        if (auto* dbgDeclare = dyn_cast<DbgDeclareInst>(&I)) {
            Value* var = dbgDeclare->getAddress();
            auto* localVar = dbgDeclare->getVariable();
            if (localVar && isa<DILocalVariable>(localVar)) {
                ssaToVarName[var] = localVar->getName().str();
            }
        } else if (auto* dbgValue = dyn_cast<DbgValueInst>(&I)) {
            Value* var = dbgValue->getValue();
            auto* localVar = dbgValue->getVariable();
            if (localVar && isa<DILocalVariable>(localVar)) {
                ssaToVarName[var] = localVar->getName().str();
            }
        }
    }

    std::set<AllocaInst*> externallyDefinedLocals;

    for (Instruction &I : instructions(F)) {
        if (auto* call = dyn_cast<CallInst>(&I)) {
            for (unsigned i = 0; i < call->arg_size(); ++i) {
                Value* arg = call->getArgOperand(i);
                Value* stripped = arg->stripPointerCasts();
    
                if (auto* alloca = dyn_cast<AllocaInst>(stripped)) {
                    externallyDefinedLocals.insert(alloca);
                }
            }
        }
    }

    for (Instruction &I : instructions(F)) {
        if (auto* ret = dyn_cast<ReturnInst>(&I)) {
            std::unordered_set<Value*> visited;
            hasReturn = true;
            Value* retVal = ret->getReturnValue();
            ReturnFlag retFlag = funcTraverser(retVal, visited, updatedFields, accessedByIndex, possibleZero, ssaToVarName, externallyDefinedLocals, externalLocals, className);

            flag = std::max(flag, retFlag); 
        }
    }

    possibleZeroMap[className] = possibleZero;

    if (hasReturn) {
        walkerFlagMap[className] = flag;
        return true;
    } else return false;
}

} // namespace llvm_analysis
