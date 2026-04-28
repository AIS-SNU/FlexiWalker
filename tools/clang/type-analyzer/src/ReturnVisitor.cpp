#include "ReturnVisitor.hpp"
#include "MonotonicityRewriter.hpp"
#include "VarExtractor.hpp"
#include "VarPruner.hpp"
#include "clang/Lex/Lexer.h"
#include "clang/Basic/SourceManager.h"
#include "llvm/Support/raw_ostream.h"
#include <sstream>

using namespace clang;

ReturnVisitor::ReturnVisitor(const ASTContext &Ctx, const CXXMethodDecl *Method, 
                           type_analysis::FullAnalysisResult &Result, 
                           const std::map<std::string, std::set<std::string>> &Map, 
                           const std::map<std::string, std::map<std::string, std::vector<std::string>>> &fMap)
    : Ctx(Ctx), Method(Method), CurrentClass(Method->getParent()), resultMap(Result), 
      externalLocalsMap(Map), fieldMap(fMap) {
    if (Method->hasBody()) {
        CFG::BuildOptions options;
        options.AddImplicitDtors = true;
        Cfg = CFG::buildCFG(Method, Method->getBody(), const_cast<ASTContext*>(&Ctx), options);
    }

    className = Method->getParent()->getNameAsString();
}

bool ReturnVisitor::TraverseCompoundStmt(CompoundStmt *CS) {
    compoundStack.push_back(CS);
    bool ret = RecursiveASTVisitor::TraverseCompoundStmt(CS);
    compoundStack.pop_back();
    return ret;
}

std::string ReturnVisitor::getSourceForExpr(const Stmt* stmt, const ASTContext& ctx) {
    const SourceManager &SM = ctx.getSourceManager();
    LangOptions LangOpts;
    SourceLocation StartLoc = stmt->getBeginLoc();
    SourceLocation EndLoc = Lexer::getLocForEndOfToken(stmt->getEndLoc(), 0, SM, LangOpts);
    return std::string(SM.getCharacterData(StartLoc),
                       SM.getCharacterData(EndLoc) - SM.getCharacterData(StartLoc));
}

ReturnVisitor::BodyRewriteResult ReturnVisitor::collectRelevantLines(
    const std::set<std::string> &initialVars,
    const std::map<std::string, type_analysis::Role> &localRoles,
    const std::unique_ptr<clang::CFG> &Cfg,
    const ASTContext &Ctx,
    const CXXRecordDecl *CurrentClass
) {
    if (!Cfg) return {};

    CollectionState state{initialVars, {}, {}, {}, &localRoles, false, "", Ctx, CurrentClass};
    Rewriter rewriter;
    rewriter.setSourceMgr(const_cast<SourceManager&>(Ctx.getSourceManager()), Ctx.getLangOpts());

    traverseCFGBlocks(Cfg, state, rewriter);
    BodyRewriteResult br;
    br.relevantLines = std::move(state.relevantLines);
    br.forceERVSOnly = state.forceERVSOnly;
    br.fallbackReason = std::move(state.fallbackReason);
    return br;
}

void ReturnVisitor::traverseCFGBlocks(const std::unique_ptr<clang::CFG> &Cfg, CollectionState &state, Rewriter &rewriter) {
    for (auto blockIt = Cfg->rbegin(); blockIt != Cfg->rend(); ++blockIt) {
        const CFGBlock *block = *blockIt;
        
        for (auto elemIt = block->rbegin(); elemIt != block->rend(); ++elemIt) {
            if (auto optCFGStmt = elemIt->getAs<CFGStmt>()) {
                const Stmt *stmt = optCFGStmt->getStmt();
                processStatement(stmt, state, rewriter);
            }
        }
    }
}

void ReturnVisitor::processStatement(const Stmt *stmt, CollectionState &state, Rewriter &rewriter) {
    // Case 1: Compound assignment operators (+=, -=, *=, /=) - Check this FIRST
    if (const auto *compoundAssign = dyn_cast<CompoundAssignOperator>(stmt)) {
        if (const auto *lhs = dyn_cast<DeclRefExpr>(compoundAssign->getLHS())) {
            // For compound assignments, don't remove the variable from remainingVars
            // because we still need to find its original declaration
            processCompoundAssignment(lhs->getNameInfo().getAsString(), compoundAssign->getRHS(), stmt, state, rewriter);
        }
    }
    // Case 2: Binary assignment (e.g., x = y + z)
    else if (const auto *binOp = dyn_cast<BinaryOperator>(stmt)) {
        if (binOp->isAssignmentOp()) {
            if (const auto *lhs = dyn_cast<DeclRefExpr>(binOp->getLHS())) {
                processAssignment(lhs->getNameInfo().getAsString(), binOp->getRHS(), stmt, state, rewriter);
            }
        }
    }
    // Case 3: DeclStmt with initializer (e.g., int x = ...)
    else if (const auto *declStmt = dyn_cast<DeclStmt>(stmt)) {
        for (const Decl *D : declStmt->decls()) {
            if (const auto *varDecl = dyn_cast<VarDecl>(D)) {
                if (const Expr *init = varDecl->getInit()) {
                    processDeclaration(varDecl->getNameAsString(), init, stmt, state, rewriter);
                }
            }
        }
    }
    // Case 4: Unary increment/decrement (++x, x++, --x, x--)
    else if (const auto *unaryOp = dyn_cast<UnaryOperator>(stmt)) {
        if (unaryOp->isIncrementDecrementOp()) {
            if (const auto *subExpr = dyn_cast<DeclRefExpr>(unaryOp->getSubExpr())) {
                // For increment/decrement, the variable depends on itself
                std::string varName = subExpr->getNameInfo().getAsString();
                if (state.remainingVars.count(varName)) {
                    processAssignment(varName, subExpr, stmt, state, rewriter);
                }
            }
        }
    }
    // Case 5: Call expressions that might modify variables (e.g., function calls with side effects)
    else if (const auto *callExpr = dyn_cast<CallExpr>(stmt)) {
        // For now, we conservatively assume call expressions might affect any variable
        // This could be refined to analyze specific function calls
    }
}

// Look up the role at which this helper local appears in the return
// expression. Locals not referenced in the return expression default to
// MAX (matching the previous blanket behavior — though such statements
// shouldn't normally be collected by the CFG walk in the first place).
static type_analysis::Role roleForLocal(
    const std::map<std::string, type_analysis::Role> *localRoles,
    const std::string &name) {
    if (!localRoles) return type_analysis::Role::MAX;
    auto it = localRoles->find(name);
    return it == localRoles->end() ? type_analysis::Role::MAX : it->second;
}

// Mark the body collection as eRVS-only and stash the first reason. We
// keep walking so debug logs still show what else was problematic, but
// downstream codegen will use eRVS_only and treat the body as dead.
static void forceBodyERVS(ReturnVisitor::CollectionState &state,
                          const std::string &reason) {
    if (!state.forceERVSOnly) {
        state.forceERVSOnly = true;
        state.fallbackReason = reason;
    }
}

void ReturnVisitor::processDeclaration(const std::string &lhsVar, const Expr *rhsExpr, const Stmt *fullStmt,
                                     CollectionState &state, Rewriter &rewriter) {
    // For declarations, we should capture it if:
    // 1. The declared variable is in remainingVars (needed by later code), OR
    // 2. The RHS expression contains variables in remainingVars

    VarExtractor rhsExtractor(state.Ctx, state.CurrentClass);
    rhsExtractor.TraverseStmt(const_cast<Expr*>(rhsExpr));

    bool isRelevant = false;

    // Check if this variable is needed (is in remainingVars)
    if (state.remainingVars.count(lhsVar)) {
        isRelevant = true;
    }

    // Check if any RHS variables are in remainingVars
    for (const std::string &v : rhsExtractor.vars) {
        if (state.remainingVars.count(v)) {
            isRelevant = true;
            break;
        }
    }

    if (!isRelevant) {
        return;
    }

    std::string line = getSourceForExpr(fullStmt, state.Ctx);
    if (!state.seenLines.insert(line).second) return;

    // Sanity-check the RHS for patterns the string-level rewrite can't
    // express correctly. Both bail out via forceERVS rather than emit
    // wrong code.
    if (helperRhsHasRoleFlip(rhsExpr)) {
        forceBodyERVS(state,
            "helper local '" + lhsVar + "' RHS contains role-flipping operator "
            "(subtraction, division, or unary minus); cannot tag suffix "
            "without per-operand AST rewrite");
    }
    std::string offendingField;
    if (helperRhsHasPointerAlias(rhsExpr, offendingField)) {
        forceBodyERVS(state,
            "non-subscripted access to indexed field '" + offendingField +
            "' in helper statement (pointer aliasing)");
    }

    // Apply variable pruning
    applyVariablePruning(fullStmt, rewriter);

    type_analysis::Role role = roleForLocal(state.localRoles, lhsVar);
    std::string modifiedLine = generateModifiedLine(fullStmt, state.Ctx, rewriter, role);
    state.relevantLines.insert(state.relevantLines.begin(), modifiedLine);

    // Update variable dependencies for any RHS variables
    for (const std::string &v : rhsExtractor.vars) {
        if (!state.seenVars.count(v)) {
            state.remainingVars.insert(v);
        }
    }

    // Mark this variable as seen and remove from remaining
    state.seenVars.insert(lhsVar);
    state.remainingVars.erase(lhsVar);
}

void ReturnVisitor::processAssignment(const std::string &lhsVar, const Expr *rhsExpr, const Stmt *fullStmt,
                                    CollectionState &state, Rewriter &rewriter) {
    if (!state.remainingVars.count(lhsVar)) return;

    std::string line = getSourceForExpr(fullStmt, state.Ctx);
    if (!state.seenLines.insert(line).second) return;

    if (helperRhsHasRoleFlip(rhsExpr)) {
        forceBodyERVS(state,
            "helper local '" + lhsVar + "' RHS contains role-flipping operator "
            "(subtraction, division, or unary minus); cannot tag suffix "
            "without per-operand AST rewrite");
    }
    std::string offendingField;
    if (helperRhsHasPointerAlias(rhsExpr, offendingField)) {
        forceBodyERVS(state,
            "non-subscripted access to indexed field '" + offendingField +
            "' in helper statement (pointer aliasing)");
    }

    // Apply variable pruning
    applyVariablePruning(fullStmt, rewriter);

    type_analysis::Role role = roleForLocal(state.localRoles, lhsVar);
    std::string modifiedLine = generateModifiedLine(fullStmt, state.Ctx, rewriter, role);
    state.relevantLines.insert(state.relevantLines.begin(), modifiedLine);

    // Extract RHS variable dependencies
    updateVariableDependencies(rhsExpr, lhsVar, state);
}

void ReturnVisitor::processCompoundAssignment(const std::string &lhsVar, const Expr *rhsExpr, const Stmt *fullStmt,
                                            CollectionState &state, Rewriter &rewriter) {
    if (!state.remainingVars.count(lhsVar)) return;

    std::string line = getSourceForExpr(fullStmt, state.Ctx);
    if (!state.seenLines.insert(line).second) return;

    if (helperRhsHasRoleFlip(rhsExpr)) {
        forceBodyERVS(state,
            "helper local '" + lhsVar + "' RHS contains role-flipping operator "
            "(subtraction, division, or unary minus); cannot tag suffix "
            "without per-operand AST rewrite");
    }
    std::string offendingField;
    if (helperRhsHasPointerAlias(rhsExpr, offendingField)) {
        forceBodyERVS(state,
            "non-subscripted access to indexed field '" + offendingField +
            "' in helper statement (pointer aliasing)");
    }

    // Apply variable pruning
    applyVariablePruning(fullStmt, rewriter);

    type_analysis::Role role = roleForLocal(state.localRoles, lhsVar);
    std::string modifiedLine = generateModifiedLine(fullStmt, state.Ctx, rewriter, role);
    state.relevantLines.insert(state.relevantLines.begin(), modifiedLine);

    // For compound assignments, extract RHS dependencies but DON'T remove lhsVar from remainingVars
    // because we still need to find its original declaration
    VarExtractor rhsExtractor(state.Ctx, state.CurrentClass);
    rhsExtractor.TraverseStmt(const_cast<Expr*>(rhsExpr));

    for (const std::string &v : rhsExtractor.vars) {
        if (!state.seenVars.count(v)) {
            state.remainingVars.insert(v);
        }
    }

    // DON'T remove lhsVar from remainingVars for compound assignments
    // state.seenVars.insert(lhsVar);  // Don't mark as seen either
    // state.remainingVars.erase(lhsVar);  // Don't remove from remaining
}

void ReturnVisitor::applyVariablePruning(const Stmt *fullStmt, Rewriter &rewriter) {
    auto it = externalLocalsMap.find(className);
    if (it != externalLocalsMap.end()) {
        for (const auto& var : it->second) {
            VarPruner pruner(rewriter, var);
            pruner.TraverseStmt(const_cast<Stmt*>(fullStmt));
        }
    }
}

std::string ReturnVisitor::generateModifiedLine(const Stmt *fullStmt, const ASTContext &Ctx, Rewriter &rewriter, type_analysis::Role role) {
    SourceLocation start = fullStmt->getBeginLoc();
    SourceLocation end = Lexer::getLocForEndOfToken(fullStmt->getEndLoc(), 0, Ctx.getSourceManager(), Ctx.getLangOpts());
    CharSourceRange stmtRange = CharSourceRange::getCharRange(start, end);
    std::string prunedLine = rewriter.getRewrittenText(stmtRange);

    std::string modifiedLine = applyFieldReplacements(prunedLine, role);
    return applyAdditionalReplacements(modifiedLine);
}

std::string ReturnVisitor::applyFieldReplacements(const std::string &input, type_analysis::Role role) {
    std::string result = input;
    std::string suffix = type_analysis::MonotonicityRewriter::suffixFor(role);
    auto fieldIt = fieldMap.find(className);
    if (fieldIt != fieldMap.end()) {
        const auto& structFields = fieldIt->second;
        for (const auto& [structName, fields] : structFields) {
            for (const auto& field : fields) {
                result = replaceWholeWords(result, field, field + suffix);
            }
        }
    }
    return result;
}

bool ReturnVisitor::helperRhsHasRoleFlip(const clang::Expr *rhs) {
    if (!rhs) return false;
    const clang::Expr *e = rhs->IgnoreImpCasts()->IgnoreParens();
    if (const auto *uo = clang::dyn_cast<clang::UnaryOperator>(e)) {
        if (uo->getOpcode() == clang::UO_Minus) return true;
    }
    if (const auto *bo = clang::dyn_cast<clang::BinaryOperator>(e)) {
        if (bo->getOpcode() == clang::BO_Sub || bo->getOpcode() == clang::BO_Div) {
            return true;
        }
    }
    // Recurse over operand subtrees so nested flips are caught.
    for (const clang::Stmt *child : e->children()) {
        if (const auto *childExpr = clang::dyn_cast_or_null<clang::Expr>(child)) {
            if (helperRhsHasRoleFlip(childExpr)) return true;
        }
    }
    return false;
}

bool ReturnVisitor::helperRhsHasPointerAlias(const clang::Expr *rhs, std::string &offendingField) {
    if (!rhs) return false;
    const clang::Expr *e = rhs->IgnoreImpCasts()->IgnoreParens();
    // Bare member access to an indexed field is the pointer-aliasing
    // pattern we're guarding against. ArraySubscriptExpr children are
    // handled by their own subscript path, so we only flag MemberExprs
    // that aren't directly the base of a subscript.
    if (const auto *me = clang::dyn_cast<clang::MemberExpr>(e)) {
        std::string name = me->getMemberNameInfo().getAsString();
        // Build a temporary rewriter purely to reuse isIndexedField. We
        // could split that helper out into util.hpp, but keeping the
        // construction here colocates the gate with its rationale.
        type_analysis::MonotonicityRewriter probe(
            Ctx, fieldMap, className, /*maxDepth=*/0, /*maxBranches=*/0, {});
        if (probe.isIndexedFieldName(name)) {
            offendingField = name;
            return true;
        }
        return false;
    }
    if (clang::isa<clang::ArraySubscriptExpr>(e)) {
        // Indexed access is fine — it's the canonical shape. Recurse
        // into the index sub-expression in case *it* aliases a pointer.
        const auto *ase = clang::cast<clang::ArraySubscriptExpr>(e);
        return helperRhsHasPointerAlias(ase->getIdx(), offendingField);
    }
    for (const clang::Stmt *child : e->children()) {
        if (const auto *childExpr = clang::dyn_cast_or_null<clang::Expr>(child)) {
            if (helperRhsHasPointerAlias(childExpr, offendingField)) return true;
        }
    }
    return false;
}

std::string ReturnVisitor::applyAdditionalReplacements(const std::string &input) {
    std::string result = input;
    result = replaceWholeWords(result, "prev_neighbor_offset", "prev_vertex");
    result = replaceSpecialPattern(result, "neighbor_offset", "current_vertex");
    return result;
}

std::string ReturnVisitor::replaceWholeWords(const std::string &input, const std::string &from, const std::string &to) {
    std::string result = input;
    size_t pos = 0;
    while ((pos = result.find(from, pos)) != std::string::npos) {
        if ((pos == 0 || !isalnum(result[pos - 1])) &&
            (pos + from.size() == result.size() || !isalnum(result[pos + from.size()]))) {
            result.replace(pos, from.size(), to);
            pos += to.size();
        } else {
            pos += from.size();
        }
    }
    return result;
}

std::string ReturnVisitor::replaceSpecialPattern(const std::string &input, const std::string &pattern, const std::string &replacement) {
    std::string result = input;
    size_t pos = 0;
    while ((pos = result.find(pattern, pos)) != std::string::npos) {
        // Avoid re-replacing part of "prev_neighbor_offset"
        if (pos >= 5 && result.substr(pos - 5, 20) == "prev_neighbor_offset") {
            pos += pattern.size();
        } else {
            result.replace(pos, pattern.size(), replacement);
            pos += replacement.size();
        }
    }
    return result;
}

std::string ReturnVisitor::getPrunedLine(const Stmt *fullStmt, const ASTContext &Ctx, Rewriter &rewriter) {
    SourceLocation start = fullStmt->getBeginLoc();
    SourceLocation end = Lexer::getLocForEndOfToken(fullStmt->getEndLoc(), 0, Ctx.getSourceManager(), Ctx.getLangOpts());
    CharSourceRange stmtRange = CharSourceRange::getCharRange(start, end);
    return rewriter.getRewrittenText(stmtRange);
}

void ReturnVisitor::updateVariableDependencies(const Expr *rhsExpr, const std::string &lhsVar, CollectionState &state) {
    VarExtractor rhsExtractor(state.Ctx, state.CurrentClass);
    rhsExtractor.TraverseStmt(const_cast<Expr*>(rhsExpr));

    for (const std::string &v : rhsExtractor.vars) {
        if (!state.seenVars.count(v)) {
            state.remainingVars.insert(v);
        }
    }

    state.seenVars.insert(lhsVar);
    state.remainingVars.erase(lhsVar);
}

bool ReturnVisitor::VisitReturnStmt(ReturnStmt *ret) {
    const Expr *retExpr = ret->getRetValue();
    if (!retExpr) return true;
    
    // If a return expression
    const SourceManager &SM = Ctx.getSourceManager();
    std::string className = Method->getParent()->getNameAsString();
    std::vector<const Expr*> retExprVec;

    // ✅ Check if this is a ternary expression
    const Expr* strippedExpr = retExpr->IgnoreImpCasts()->IgnoreParens();
    if (const auto *condOp = dyn_cast<ConditionalOperator>(strippedExpr)) {
        retExprVec.push_back(condOp->getTrueExpr());
        retExprVec.push_back(condOp->getFalseExpr());
    } else {
        // ✅ Otherwise treat as a normal return expression
        retExprVec.push_back(retExpr);
    }

    for (auto expr: retExprVec) {
        std::string exprText = getSourceForExpr(expr, Ctx);
        VarExtractor extractor(Ctx, CurrentClass);
        extractor.TraverseStmt(const_cast<Expr*>(expr));
        
        Rewriter rewriter;
        rewriter.setSourceMgr(const_cast<SourceManager&>(Ctx.getSourceManager()), Ctx.getLangOpts());
        auto it = externalLocalsMap.find(className);
        if (it != externalLocalsMap.end()) {
            for (const auto& var : it->second) {
                VarPruner pruner(rewriter, var);
                pruner.TraverseStmt(const_cast<Expr*>(expr));
            }
        }

        SourceLocation start = expr->getBeginLoc();
        SourceLocation end = Lexer::getLocForEndOfToken(expr->getEndLoc(), 0, Ctx.getSourceManager(), Ctx.getLangOpts());
        CharSourceRange range = CharSourceRange::getCharRange(start, end);
        std::string prunedExpr = rewriter.getRewrittenText(range);

        // Phase 2: use AST-based MonotonicityRewriter on the return expression.
        // Each outer ternary arm has already been split into its own `expr`,
        // so the rewriter sees a single-branch expression and propagates roles.
        std::set<std::string> extLocals;
        auto elIt = externalLocalsMap.find(className);
        if (elIt != externalLocalsMap.end()) extLocals = elIt->second;
        type_analysis::MonotonicityRewriter monoRewriter(
            Ctx, fieldMap, className,
            /*maxDepth=*/6, /*maxBranches=*/8,
            extLocals);
        auto rr = monoRewriter.rewrite(expr);

        // Conflict escalation: a helper local used at both MAX and MIN in
        // the return expression can only carry one suffix in body lines, so
        // route the walker through eRVS_only. Reported on the first
        // offender; debug log shows them all.
        if (!rr.forceERVSOnly && !rr.conflictingLocals.empty()) {
            rr.forceERVSOnly = true;
            rr.fallbackReason = "helper local '" + *rr.conflictingLocals.begin() +
                                "' used at conflicting roles in return expression";
        }

        // Collect body lines using the role map from the rewriter so each
        // helper statement gets the right _MAX / _MIN suffix on its
        // accessed fields. The body collector also escalates to eRVS on
        // role-flipping helpers and pointer-aliased indexed fields.
        BodyRewriteResult bodyResult = collectRelevantLines(
            extractor.vars, rr.localRoles, Cfg, Ctx, CurrentClass);
        if (!rr.forceERVSOnly && bodyResult.forceERVSOnly) {
            rr.forceERVSOnly = true;
            rr.fallbackReason = bodyResult.fallbackReason;
        }

        // On forceERVSOnly the walker runs via walker_ervs_only, which never
        // calls get_max_weight / get_sum_weight, so the body is dead code.
        // Emit the original pruned expression verbatim rather than a blind
        // _MAX rewrite that would misrepresent subtraction/division roles.
        std::string modifiedExpr = rr.forceERVSOnly ? prunedExpr : rr.expression;
        modifiedExpr = applyAdditionalReplacements(modifiedExpr);

        type_analysis::ReturnBranch branch;
        branch.body = bodyResult.relevantLines;
        branch.return_expr = modifiedExpr;
        for (const auto &[field, role] : rr.fieldRoles) {
            branch.fieldRoles.push_back(type_analysis::FieldRole{
                field,
                role == type_analysis::Role::MAX ? "MAX" : "MIN"
            });
        }
        branch.forceERVSOnly = rr.forceERVSOnly;
        branch.fallbackReason = rr.fallbackReason;
        resultMap.classes[className].branches.push_back(branch);

        if (DEBUG) {
            llvm::errs() << "Expression: " << exprText << "\n";
            llvm::errs() << "Pruned expr: " << prunedExpr << "\n";
            llvm::errs() << "Modified expr: " << modifiedExpr << "\n";
            for (const std::string &var : extractor.vars) {
                llvm::outs() << "  - uses var: " << var << "\n";
            }

            llvm::outs() << "Relevant instructions:\n";
            for (const std::string &line : bodyResult.relevantLines) {
                llvm::outs() << "  > " << line << "\n";
            }
        }
    }

    return true;
}
