#ifndef VAR_EXTRACTOR_H
#define VAR_EXTRACTOR_H

#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/AST/AST.h"
#include <set>
#include <string>

class VarExtractor : public clang::RecursiveASTVisitor<VarExtractor> {
public:
    std::set<std::string> vars;
    std::set<std::string> task_fields;
    const clang::ASTContext &Ctx;
    const clang::CXXRecordDecl *CurrentClass;

    VarExtractor(const clang::ASTContext &ctx, const clang::CXXRecordDecl *cls);

    // Handle implicit access to class fields (like just `p`)
    bool VisitDeclRefExpr(clang::DeclRefExpr *DRE);

    bool VisitMemberExpr(clang::MemberExpr *ME);

    // Handle array subscript expressions (e.g., arr[i])
    bool VisitArraySubscriptExpr(clang::ArraySubscriptExpr *ASE);

    // Handle call expressions (function calls)
    bool VisitCallExpr(clang::CallExpr *CE);

    // Handle conditional expressions (ternary operator)
    bool VisitConditionalOperator(clang::ConditionalOperator *CO);

    // Handle cast expressions
    bool VisitCastExpr(clang::CastExpr *CE);

private:
    std::string reconstructMemberExpr(const clang::Expr *E);
};

#endif // VAR_EXTRACTOR_H
