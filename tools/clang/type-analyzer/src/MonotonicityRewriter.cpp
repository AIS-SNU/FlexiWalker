#include "MonotonicityRewriter.hpp"

#include "clang/AST/OperationKinds.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Lex/Lexer.h"
#include <sstream>

using namespace clang;
using namespace type_analysis;

namespace {

// Precedence ranks (lower = binds tighter). Mirrors C++ enough to decide when
// paren emission is required; unused ops are handled via the forceERVSOnly path.
constexpr int PREC_PRIMARY = 0;  // literals, identifiers, member/subscript, calls, parens
constexpr int PREC_UNARY   = 3;
constexpr int PREC_MULDIV  = 5;
constexpr int PREC_ADDSUB  = 6;
constexpr int PREC_TOPLEVEL = 99;

std::string sourceText(const Expr *e, const ASTContext &Ctx) {
    SourceLocation start = e->getBeginLoc();
    SourceLocation end = Lexer::getLocForEndOfToken(
        e->getEndLoc(), 0, Ctx.getSourceManager(), Ctx.getLangOpts());
    return std::string(
        Ctx.getSourceManager().getCharacterData(start),
        Ctx.getSourceManager().getCharacterData(end) -
            Ctx.getSourceManager().getCharacterData(start));
}

bool isKnownMonotone(const std::string &name) {
    // Whitelist of unary, monotonically-increasing-on-nonneg functions.
    return name == "exp" || name == "log" || name == "sqrt";
}

}  // namespace

MonotonicityRewriter::MonotonicityRewriter(
    const ASTContext &Ctx,
    const std::map<std::string, std::map<std::string, std::vector<std::string>>> &fMap,
    const std::string &cls,
    unsigned dDepth,
    unsigned dBranches,
    const std::set<std::string> &extLocals)
    : Ctx(Ctx), fieldMap(fMap), className(cls),
      maxDepth(dDepth), maxBranches(dBranches),
      externalLocals(extLocals), branchCount(0) {}

bool MonotonicityRewriter::isIndexedField(const std::string &name) const {
    // "Indexed" is a property of the graph schema, not of any single
    // walker. The current walker may pointer-alias the field (skipping
    // the LLVM analyzer's accessedByIndex registration), so first check
    // its own entry, then fall back to any sibling walker in this
    // compile unit. This makes pointer-aliasing detection in helper
    // bodies work for walkers that never subscript the field.
    auto fieldNameAppears = [&](const std::map<std::string, std::vector<std::string>> &m) {
        for (const auto &[structName, fields] : m) {
            for (const auto &f : fields) {
                if (f == name) return true;
            }
        }
        return false;
    };

    auto fIt = fieldMap.find(className);
    if (fIt != fieldMap.end() && fieldNameAppears(fIt->second)) return true;

    for (const auto &[walkerName, structFields] : fieldMap) {
        if (walkerName == className) continue;
        if (fieldNameAppears(structFields)) return true;
    }
    return false;
}

bool MonotonicityRewriter::isPrunedVar(const Expr *e) const {
    if (!e) return false;
    const Expr *inner = e->IgnoreImpCasts()->IgnoreParens();
    if (const auto *dre = dyn_cast<DeclRefExpr>(inner)) {
        return externalLocals.count(dre->getNameInfo().getAsString()) > 0;
    }
    return false;
}

bool MonotonicityRewriter::isCanonicalIndex(const Expr *idx) const {
    if (!idx) return false;
    const Expr *e = idx->IgnoreImpCasts()->IgnoreParens();

    // Member names that produce a valid vertex-ID index after
    // applyAdditionalReplacements runs:
    //   - `neighbor_offset` / `prev_neighbor_offset` get textually rewritten
    //     to `current_vertex` / `prev_vertex`.
    //   - `current_vertex` / `prev_vertex` are already vertex IDs and are
    //     not touched by the rewrite.
    auto isCanonicalMember = [](const Expr *x) {
        const Expr *inner = x->IgnoreImpCasts()->IgnoreParens();
        const auto *me = dyn_cast<MemberExpr>(inner);
        if (!me) return false;
        std::string name = me->getMemberNameInfo().getAsString();
        return name == "neighbor_offset" ||
               name == "prev_neighbor_offset" ||
               name == "current_vertex" ||
               name == "prev_vertex";
    };

    // Bare canonical member.
    if (isCanonicalMember(e)) return true;

    // `member + pruned_local` or `pruned_local + member` — the pruned local
    // is elided downstream, so both shapes reduce to the bare member.
    if (const auto *bo = dyn_cast<BinaryOperator>(e)) {
        if (bo->getOpcode() == BO_Add) {
            if (isCanonicalMember(bo->getLHS()) && isPrunedVar(bo->getRHS())) return true;
            if (isCanonicalMember(bo->getRHS()) && isPrunedVar(bo->getLHS())) return true;
        }
    }
    return false;
}

RewriteResult MonotonicityRewriter::rewrite(const Expr *expr, Role rootRole) {
    RewriteResult result;
    result.forceERVSOnly = false;
    branchCount = 0;
    result.expression = walk(expr, rootRole, 0, PREC_TOPLEVEL, result);
    return result;
}

std::string MonotonicityRewriter::walk(
    const Expr *expr, Role role, unsigned depth, int parentPrec,
    RewriteResult &result) {
    if (result.forceERVSOnly) return "";  // short-circuit

    if (depth > maxDepth) {
        result.forceERVSOnly = true;
        result.fallbackReason = "expression depth exceeds MAX_ANALYSIS_DEPTH";
        return "";
    }

    if (!expr) return "";

    const Expr *e = expr->IgnoreImpCasts();

    // Parenthesized sub-expression: preserve source-level parens verbatim so
    // goldens stay byte-identical with the previous text-based rewriter.
    if (const auto *pe = dyn_cast<ParenExpr>(e)) {
        std::string inner = walk(pe->getSubExpr(), role, depth + 1,
                                 PREC_TOPLEVEL, result);
        if (result.forceERVSOnly) return "";
        return "(" + inner + ")";
    }

    // Numeric / boolean literal.
    if (isa<IntegerLiteral>(e) || isa<FloatingLiteral>(e) ||
        isa<CharacterLiteral>(e) || isa<CXXBoolLiteralExpr>(e)) {
        return sourceText(e, Ctx);
    }

    // Identifier reference — hyperparameters, locals.
    if (const auto *dre = dyn_cast<DeclRefExpr>(e)) {
        std::string name = dre->getNameInfo().getAsString();
        // Record the role this local was used at. Hyperparameters and
        // pruned externals are uninteresting (they're not redeclared in
        // helper bodies), but checking the decl kind here is more work
        // than it saves — bookkeeping a few extra map entries is cheap.
        if (!externalLocals.count(name)) {
            auto [it, inserted] = result.localRoles.emplace(name, role);
            if (!inserted && it->second != role) {
                result.conflictingLocals.insert(name);
            }
        }
        return sourceText(dre, Ctx);
    }

    // Array subscript — the primary suffix site.
    if (const auto *ase = dyn_cast<ArraySubscriptExpr>(e)) {
        std::string baseText = sourceText(ase->getBase(), Ctx);
        // Index is inside [...]; treat the bracket as a fresh top-level context
        // so child operators don't self-wrap in parens.
        std::string idxText = walk(ase->getIdx(), role, depth + 1,
                                   PREC_TOPLEVEL, result);
        if (result.forceERVSOnly) return "";

        // Extract trailing identifier for role lookup (e.g. "graph->adjwgt" -> "adjwgt").
        std::string fieldName = baseText;
        auto arrowPos = fieldName.rfind("->");
        auto dotPos = fieldName.rfind('.');
        size_t lastSep = 0;
        if (arrowPos != std::string::npos) lastSep = std::max(lastSep, arrowPos + 2);
        if (dotPos != std::string::npos) lastSep = std::max(lastSep, dotPos + 1);
        fieldName = fieldName.substr(lastSep);

        if (isIndexedField(fieldName)) {
            // The downstream textual rewrite in ReturnVisitor converts
            // `neighbor_offset` / `prev_neighbor_offset` tokens to
            // `current_vertex` / `prev_vertex`. That rewrite is only correct
            // when the index AST is a canonical form; a non-canonical index
            // would silently produce the wrong vertex ID. Force eRVS-only
            // rather than emit wrong code.
            if (!isCanonicalIndex(ase->getIdx())) {
                result.forceERVSOnly = true;
                result.fallbackReason =
                    "non-canonical index for preprocessed field '" + fieldName +
                    "' (expected task->neighbor_offset [+ loop local] or "
                    "task->prev_neighbor_offset [+ loop local])";
                return "";
            }
            result.fieldRoles.insert({fieldName, role});
            std::string prefix = baseText.substr(0, lastSep);
            return prefix + fieldName + suffix(role) + "[" + idxText + "]";
        }
        return baseText + "[" + idxText + "]";
    }

    // Unary operators.
    if (const auto *uo = dyn_cast<UnaryOperator>(e)) {
        switch (uo->getOpcode()) {
            case UO_Minus: {
                std::string sub = walk(uo->getSubExpr(), flip(role),
                                       depth + 1, PREC_UNARY, result);
                if (result.forceERVSOnly) return "";
                std::string text = "-" + sub;
                return (PREC_UNARY > parentPrec) ? "(" + text + ")" : text;
            }
            case UO_Plus:
                return walk(uo->getSubExpr(), role, depth + 1,
                            parentPrec, result);
            default:
                result.forceERVSOnly = true;
                result.fallbackReason = "non-arithmetic unary operator in return expression";
                return "";
        }
    }

    // Binary operators.
    if (const auto *bo = dyn_cast<BinaryOperator>(e)) {
        // External-local pruning: mirror VarPruner's text-level elision for
        // single-variable operands like `i` or `prev_offset`. When one side is
        // a pruned DeclRef, the binary op semantically reduces to the other
        // side, inheriting the parent's role.
        if (!externalLocals.empty()) {
            if (isPrunedVar(bo->getRHS())) {
                return walk(bo->getLHS(), role, depth + 1, parentPrec, result);
            }
            if (isPrunedVar(bo->getLHS())) {
                return walk(bo->getRHS(), role, depth + 1, parentPrec, result);
            }
        }

        Role leftRole = role;
        Role rightRole = role;
        std::string op;
        int myPrec;
        switch (bo->getOpcode()) {
            case BO_Add: op = " + "; myPrec = PREC_ADDSUB; break;
            case BO_Sub: op = " - "; rightRole = flip(role); myPrec = PREC_ADDSUB; break;
            case BO_Mul: op = " * "; myPrec = PREC_MULDIV; break;
            case BO_Div: op = " / "; rightRole = flip(role); myPrec = PREC_MULDIV; break;
            default:
                result.forceERVSOnly = true;
                result.fallbackReason = "non-arithmetic binary operator in return expression";
                return "";
        }
        std::string lhs = walk(bo->getLHS(), leftRole, depth + 1, myPrec, result);
        std::string rhs = walk(bo->getRHS(), rightRole, depth + 1, myPrec, result);
        if (result.forceERVSOnly) return "";
        std::string text = lhs + op + rhs;
        return (myPrec > parentPrec) ? "(" + text + ")" : text;
    }

    // Ternary / conditional — counts against branch budget.
    if (const auto *cop = dyn_cast<ConditionalOperator>(e)) {
        if (++branchCount > maxBranches) {
            result.forceERVSOnly = true;
            result.fallbackReason = "branch count exceeds MAX_ANALYSIS_BRANCHES";
            return "";
        }
        std::string t = walk(cop->getTrueExpr(), role, depth + 1,
                             PREC_TOPLEVEL, result);
        std::string f = walk(cop->getFalseExpr(), role, depth + 1,
                             PREC_TOPLEVEL, result);
        if (result.forceERVSOnly) return "";
        std::string fn = (role == Role::MAX) ? "max" : "min";
        return fn + "(" + t + ", " + f + ")";
    }

    // Single-arg monotone function call.
    if (const auto *call = dyn_cast<CallExpr>(e)) {
        const Expr *calleeExpr = call->getCallee()->IgnoreImpCasts();
        if (const auto *callee = dyn_cast<DeclRefExpr>(calleeExpr)) {
            std::string fname = callee->getNameInfo().getAsString();
            if (isKnownMonotone(fname) && call->getNumArgs() == 1) {
                std::string arg = walk(call->getArg(0), role, depth + 1,
                                       PREC_TOPLEVEL, result);
                if (result.forceERVSOnly) return "";
                return fname + "(" + arg + ")";
            }
        }
        result.forceERVSOnly = true;
        result.fallbackReason = "unknown function call in return expression";
        return "";
    }

    // Member access without subscript (e.g., task->degree) — emit verbatim,
    // unless the member is a graph-indexed field. A bare `graph->adjwgt`
    // outside an array-subscript context means the user is taking the raw
    // pointer (e.g. `auto p = graph->adjwgt; ... p[i]`); we have no way to
    // tag the suffix on the aliased pointer, so force eRVS_only.
    if (const auto *me = dyn_cast<MemberExpr>(e)) {
        std::string memberName = me->getMemberNameInfo().getAsString();
        if (isIndexedField(memberName)) {
            result.forceERVSOnly = true;
            result.fallbackReason =
                "non-subscripted access to indexed field '" + memberName +
                "' (pointer aliasing)";
            return "";
        }
        return sourceText(me, Ctx);
    }

    // Anything else: conservative fallback.
    result.forceERVSOnly = true;
    result.fallbackReason = "unsupported AST node in return expression";
    return "";
}
