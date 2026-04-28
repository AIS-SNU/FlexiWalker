#ifndef VAR_PRUNER_H
#define VAR_PRUNER_H

#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/AST/AST.h"
#include "clang/Rewrite/Core/Rewriter.h"
#include <string>

extern bool DEBUG;

class VarPruner : public clang::RecursiveASTVisitor<VarPruner> {
public:
    clang::Rewriter &R;
    std::string varToRemove;

    VarPruner(clang::Rewriter &R, std::string var);

    bool VisitBinaryOperator(clang::BinaryOperator *BO);

    // Handle compound assignment operators (+=, -=, etc.)
    bool VisitCompoundAssignOperator(clang::CompoundAssignOperator *CAO);

    // Handle unary increment/decrement
    bool VisitUnaryOperator(clang::UnaryOperator *UO);
};

#endif // VAR_PRUNER_H
