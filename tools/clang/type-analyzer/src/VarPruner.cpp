#include "VarPruner.hpp"
#include "clang/Lex/Lexer.h"
#include "llvm/Support/raw_ostream.h"

using namespace clang;

VarPruner::VarPruner(Rewriter &R, std::string var) : R(R), varToRemove(std::move(var)) {}

bool VarPruner::VisitBinaryOperator(BinaryOperator *BO) {
    if (!BO->getSourceRange().isValid() || BO->getExprLoc().isMacroID())
        return true;

    Expr *LHS = BO->getLHS()->IgnoreImpCasts();
    Expr *RHS = BO->getRHS()->IgnoreImpCasts();

    auto isTargetVar = [&](Expr *E) {
        if (auto *DRE = dyn_cast<DeclRefExpr>(E)) {
            return DRE->getNameInfo().getAsString() == varToRemove;
        }
        return false;
    };

    const SourceManager &SM = R.getSourceMgr();
    LangOptions LangOpts = R.getLangOpts();
    CharSourceRange range = CharSourceRange::getTokenRange(BO->getSourceRange());

    if (isTargetVar(LHS)) {
        std::string rhsText = Lexer::getSourceText(
            CharSourceRange::getTokenRange(RHS->getSourceRange()), SM, LangOpts
        ).str();

        CharSourceRange safeRange = CharSourceRange::getTokenRange(BO->getSourceRange());

        if (DEBUG)
            llvm::errs() << "[PRUNING] Replacing `" 
                         << Lexer::getSourceText(safeRange, SM, LangOpts) 
                         << "` with `" << rhsText << "`\n";

        R.ReplaceText(safeRange, rhsText);
    }
    else if (isTargetVar(RHS)) {
        std::string lhsText = Lexer::getSourceText(
            CharSourceRange::getTokenRange(LHS->getSourceRange()), SM, LangOpts
        ).str();

        CharSourceRange safeRange = CharSourceRange::getTokenRange(BO->getSourceRange());

        if (DEBUG)
            llvm::errs() << "[PRUNING] Replacing `" 
                         << Lexer::getSourceText(safeRange, SM, LangOpts) 
                         << "` with `" << lhsText << "`\n";

        R.ReplaceText(safeRange, lhsText);
    }

    return true;
}

// Handle compound assignment operators (+=, -=, etc.)
bool VarPruner::VisitCompoundAssignOperator(CompoundAssignOperator *CAO) {
    if (!CAO->getSourceRange().isValid() || CAO->getExprLoc().isMacroID())
        return true;

    Expr *LHS = CAO->getLHS()->IgnoreImpCasts();

    if (auto *DRE = dyn_cast<DeclRefExpr>(LHS)) {
        if (DRE->getNameInfo().getAsString() == varToRemove) {
            const SourceManager &SM = R.getSourceMgr();
            LangOptions LangOpts = R.getLangOpts();

            std::string rhsText = Lexer::getSourceText(
                CharSourceRange::getTokenRange(CAO->getRHS()->getSourceRange()), SM, LangOpts
            ).str();

            CharSourceRange safeRange = CharSourceRange::getTokenRange(CAO->getSourceRange());

            if (DEBUG)
                llvm::errs() << "[PRUNING] Replacing compound assignment `" 
                             << Lexer::getSourceText(safeRange, SM, LangOpts) 
                             << "` with `" << rhsText << "`\n";

            R.ReplaceText(safeRange, rhsText);
        }
    }

    return true;
}

// Handle unary increment/decrement
bool VarPruner::VisitUnaryOperator(UnaryOperator *UO) {
    if (!UO->getSourceRange().isValid() || UO->getExprLoc().isMacroID())
        return true;

    if (UO->isIncrementDecrementOp()) {
        if (auto *DRE = dyn_cast<DeclRefExpr>(UO->getSubExpr())) {
            if (DRE->getNameInfo().getAsString() == varToRemove) {
                const SourceManager &SM = R.getSourceMgr();
                LangOptions LangOpts = R.getLangOpts();
                CharSourceRange safeRange = CharSourceRange::getTokenRange(UO->getSourceRange());

                if (DEBUG)
                    llvm::errs() << "[PRUNING] Removing increment/decrement `" 
                                 << Lexer::getSourceText(safeRange, SM, LangOpts) << "`\n";

                // For increment/decrement, we might want to replace with a constant or remove entirely
                R.ReplaceText(safeRange, "0"); // or some other appropriate replacement
            }
        }
    }

    return true;
}
