#ifndef MONOTONICITY_REWRITER_HPP
#define MONOTONICITY_REWRITER_HPP

#include "clang/AST/AST.h"
#include "clang/AST/ASTContext.h"
#include <map>
#include <set>
#include <string>
#include <vector>

namespace type_analysis {

/// Per-variable contribution role to the return value's upper bound.
enum class Role { MAX, MIN };

/// Outcome of a single-branch rewrite.
struct RewriteResult {
    /// Rewritten expression with _MAX / _MIN suffixes applied to indexed fields.
    std::string expression;
    /// Set of (field_name, Role) pairs referenced by this branch. Used by
    /// the pipeline to validate preprocessing declarations in graph_fields.config.
    std::set<std::pair<std::string, Role>> fieldRoles;
    /// First role observed for each local DeclRef in the return expression.
    /// Used by ReturnVisitor to choose the right suffix when emitting helper
    /// body statements that *define* these locals.
    std::map<std::string, Role> localRoles;
    /// Locals that appeared at both MAX and MIN in the return expression.
    /// A single helper statement can only carry one suffix, so a conflict
    /// forces the walker into eRVS_only.
    std::set<std::string> conflictingLocals;
    /// True if the walk hit a fallback condition (unknown function, non-
    /// arithmetic operator, or complexity budget exceeded).
    bool forceERVSOnly;
    /// Human-readable reason for the fallback, if any. Empty when forceERVSOnly is false.
    std::string fallbackReason;
};

/// Recursively rewrites a return-expression AST under a propagated target-sign,
/// emitting a string with per-reference _MAX / _MIN suffixes and the set of
/// (field, role) pairs it depends on. Implements the monotonicity analysis
/// described in spec §5.2.1.2.
class MonotonicityRewriter {
public:
    MonotonicityRewriter(
        const clang::ASTContext &Ctx,
        const std::map<std::string, std::map<std::string, std::vector<std::string>>> &fieldMap,
        const std::string &className,
        unsigned maxDepth,
        unsigned maxBranches,
        const std::set<std::string> &externalLocals = {}
    );

    /// Rewrite a single return-expression AST under the given root role
    /// (defaults to MAX, matching the return-value semantics).
    RewriteResult rewrite(const clang::Expr *expr, Role rootRole = Role::MAX);

    /// Whether `name` denotes an edge field declared with array semantics
    /// (i.e. indexed by vertex/edge ID). Exposed so callers (ReturnVisitor)
    /// can detect pointer-aliasing patterns that would bypass the rewriter.
    bool isIndexedFieldName(const std::string &name) const { return isIndexedField(name); }
    static std::string suffixFor(Role r) { return suffix(r); }

private:
    const clang::ASTContext &Ctx;
    const std::map<std::string, std::map<std::string, std::vector<std::string>>> &fieldMap;
    std::string className;
    unsigned maxDepth;
    unsigned maxBranches;
    std::set<std::string> externalLocals;
    unsigned branchCount;

    bool isIndexedField(const std::string &name) const;
    bool isPrunedVar(const clang::Expr *e) const;

    /// True when `idx` is a shape the downstream textual rewrite can safely
    /// convert (bare `task->neighbor_offset` / `task->prev_neighbor_offset`,
    /// optionally added to a pruned local). Anything else would collapse to
    /// the wrong vertex ID after `applyAdditionalReplacements`.
    bool isCanonicalIndex(const clang::Expr *idx) const;

    /// Walk returning the emitted text. `parentPrec` is the precedence of the
    /// enclosing operator; walk() adds parentheses only when its own precedence
    /// is higher (weaker binding) than the parent's.
    std::string walk(const clang::Expr *expr, Role role, unsigned depth,
                     int parentPrec, RewriteResult &result);

    static Role flip(Role r) { return r == Role::MAX ? Role::MIN : Role::MAX; }

    static std::string suffix(Role r) { return r == Role::MAX ? "_MAX" : "_MIN"; }
};

}  // namespace type_analysis

#endif  // MONOTONICITY_REWRITER_HPP
