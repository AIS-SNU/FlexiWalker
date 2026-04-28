#ifndef RETURN_VISITOR_H
#define RETURN_VISITOR_H

#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/AST/AST.h"
#include "clang/Analysis/CFG.h"
#include "clang/Rewrite/Core/Rewriter.h"
#include "MonotonicityRewriter.hpp"
#include "util.hpp"
#include <map>
#include <set>
#include <string>
#include <vector>
#include <memory>

extern bool DEBUG;

class ReturnVisitor : public clang::RecursiveASTVisitor<ReturnVisitor> {
public:
    const clang::ASTContext &Ctx;
    const clang::CXXMethodDecl *Method;
    const clang::CXXRecordDecl *CurrentClass;
    type_analysis::FullAnalysisResult &resultMap;
    std::string className;

    const std::map<std::string, std::set<std::string>> &externalLocalsMap;
    const std::map<std::string, std::map<std::string, std::vector<std::string>>> &fieldMap;
    std::vector<const clang::CompoundStmt*> compoundStack;

    std::unique_ptr<clang::CFG> Cfg;

    ReturnVisitor(const clang::ASTContext &Ctx, const clang::CXXMethodDecl *Method, 
                 type_analysis::FullAnalysisResult &Result, 
                 const std::map<std::string, std::set<std::string>> &Map, 
                 const std::map<std::string, std::map<std::string, std::vector<std::string>>> &fMap);

    bool TraverseCompoundStmt(clang::CompoundStmt *CS);

    std::string getSourceForExpr(const clang::Stmt* stmt, const clang::ASTContext& ctx);

    /// Result of body-line collection. relevantLines holds the rewritten
    /// helper statements; forceERVSOnly / fallbackReason are populated when
    /// the role-aware rewrite hits a pattern we can't safely express as a
    /// single suffix (role conflicts, pointer aliasing, role-flipping ops
    /// inside helper RHSes).
    struct BodyRewriteResult {
        std::vector<std::string> relevantLines;
        bool forceERVSOnly = false;
        std::string fallbackReason;
    };

    BodyRewriteResult collectRelevantLines(
        const std::set<std::string> &initialVars,
        const std::map<std::string, type_analysis::Role> &localRoles,
        const std::unique_ptr<clang::CFG> &Cfg,
        const clang::ASTContext &Ctx,
        const clang::CXXRecordDecl *CurrentClass
    );

    bool VisitReturnStmt(clang::ReturnStmt *ret);

    // CollectionState is exposed publicly only so the file-local
    // forceBodyERVS helper in ReturnVisitor.cpp can take it by reference.
    // Treat as an implementation detail.
    struct CollectionState {
        std::set<std::string> remainingVars;
        std::set<std::string> seenVars;
        std::set<std::string> seenLines;
        std::vector<std::string> relevantLines;
        /// Per-local roles propagated from the return expression. A helper
        /// statement assigning to `w` should rewrite its RHS at this role.
        const std::map<std::string, type_analysis::Role> *localRoles = nullptr;
        bool forceERVSOnly = false;
        std::string fallbackReason;
        const clang::ASTContext &Ctx;
        const clang::CXXRecordDecl *CurrentClass;
    };

private:
    void traverseCFGBlocks(const std::unique_ptr<clang::CFG> &Cfg, CollectionState &state, clang::Rewriter &rewriter);

    void processStatement(const clang::Stmt *stmt, CollectionState &state, clang::Rewriter &rewriter);

    void processDeclaration(const std::string &lhsVar, const clang::Expr *rhsExpr, const clang::Stmt *fullStmt, 
                           CollectionState &state, clang::Rewriter &rewriter);

    void processAssignment(const std::string &lhsVar, const clang::Expr *rhsExpr, const clang::Stmt *fullStmt, 
                          CollectionState &state, clang::Rewriter &rewriter);

    void processCompoundAssignment(const std::string &lhsVar, const clang::Expr *rhsExpr, const clang::Stmt *fullStmt, 
                                  CollectionState &state, clang::Rewriter &rewriter);

    void applyVariablePruning(const clang::Stmt *fullStmt, clang::Rewriter &rewriter);

    std::string generateModifiedLine(const clang::Stmt *fullStmt, const clang::ASTContext &Ctx, clang::Rewriter &rewriter, type_analysis::Role role);

    std::string applyFieldReplacements(const std::string &input, type_analysis::Role role);

    /// Inspect the helper-statement RHS for role-flipping operators
    /// (BO_Sub, BO_Div, UO_Minus). The string-level field-suffix rewrite
    /// can only carry one role per statement, so any flip inside a helper
    /// would silently produce a wrong suffix on one operand. Force eRVS
    /// in that case rather than emit incorrect code.
    bool helperRhsHasRoleFlip(const clang::Expr *rhs);

    /// Detect non-subscripted access to indexed graph fields in a helper
    /// RHS (the pointer-aliasing pattern). MonotonicityRewriter already
    /// catches this for return expressions; here we want the same
    /// guarantee for helper bodies.
    bool helperRhsHasPointerAlias(const clang::Expr *rhs, std::string &offendingField);

    std::string applyAdditionalReplacements(const std::string &input);

    std::string replaceWholeWords(const std::string &input, const std::string &from, const std::string &to);

    std::string replaceSpecialPattern(const std::string &input, const std::string &pattern, const std::string &replacement);

    std::string getPrunedLine(const clang::Stmt *fullStmt, const clang::ASTContext &Ctx, clang::Rewriter &rewriter);

    void updateVariableDependencies(const clang::Expr *rhsExpr, const std::string &lhsVar, CollectionState &state);
};

#endif // RETURN_VISITOR_H
