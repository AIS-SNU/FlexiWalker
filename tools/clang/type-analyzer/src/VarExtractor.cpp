#include "VarExtractor.hpp"
#include "clang/AST/Decl.h"

using namespace clang;

VarExtractor::VarExtractor(const ASTContext &ctx, const CXXRecordDecl *cls)
    : Ctx(ctx), CurrentClass(cls) {}

// Handle implicit access to class fields (like just `p`)
bool VarExtractor::VisitDeclRefExpr(DeclRefExpr *DRE) {
    if (const auto *VD = dyn_cast<VarDecl>(DRE->getDecl())) {
        // Skip class fields and function parameters
        if (isa<FieldDecl>(VD) || isa<ParmVarDecl>(VD)) {
            return true;
        }

        vars.insert(VD->getNameAsString());
    }
    return true;
}

bool VarExtractor::VisitMemberExpr(MemberExpr *ME) {
    const Expr *base = ME->getBase()->IgnoreImpCasts()->IgnoreParens();
    if (const auto *baseDRE = dyn_cast<DeclRefExpr>(base)) {
        std::string baseName = baseDRE->getNameInfo().getAsString();
        if (baseName == "task") {
            std::string field = ME->getMemberNameInfo().getAsString();
            task_fields.insert(field); // track task->field
        }
    }
    return true;
}

// Handle array subscript expressions (e.g., arr[i])
bool VarExtractor::VisitArraySubscriptExpr(ArraySubscriptExpr *ASE) {
    // Visit both base and index expressions
    return true; // Let default traversal handle sub-expressions
}

// Handle call expressions (function calls)
bool VarExtractor::VisitCallExpr(CallExpr *CE) {
    // Let default traversal handle arguments
    return true;
}

// Handle conditional expressions (ternary operator)
bool VarExtractor::VisitConditionalOperator(ConditionalOperator *CO) {
    // Let default traversal handle condition, true, and false expressions
    return true;
}

// Handle cast expressions
bool VarExtractor::VisitCastExpr(CastExpr *CE) {
    // Let default traversal handle sub-expression
    return true;
}

std::string VarExtractor::reconstructMemberExpr(const Expr *E) {
    // Recurse through nested member/decl references to build full expression string
    std::string result;
    const Expr *curr = E;

    while (curr) {
        curr = curr->IgnoreImpCasts()->IgnoreParens();

        if (const auto *ME = dyn_cast<MemberExpr>(curr)) {
            std::string member = ME->getMemberNameInfo().getAsString();
            result = "->" + member + result;
            curr = ME->getBase();
        } else if (const auto *DRE = dyn_cast<DeclRefExpr>(curr)) {
            result = DRE->getNameInfo().getAsString() + result;
            break;
        } else {
            break;
        }
    }

    return result;
}
