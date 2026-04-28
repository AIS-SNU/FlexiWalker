#include "GetWeightHandler.hpp"
#include "util.hpp"
#include "VarExtractor.hpp"
#include "VarPruner.hpp"
#include "ReturnVisitor.hpp"

#include "clang/AST/AST.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/Lex/Lexer.h"
#include "llvm/Support/raw_ostream.h"
#include "clang/Analysis/CFG.h"             // clang::CFG and CFG::buildCFG
#include "clang/AST/ASTContext.h"           // ASTContext
#include "clang/AST/Stmt.h"                 // ReturnStmt, CompoundStmt, etc.
#include "clang/Basic/SourceManager.h"      // SourceManager, SourceLocation
#include "llvm/ADT/StringRef.h"
#include "clang/Tooling/Tooling.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Rewrite/Core/Rewriter.h"
#include "clang/AST/Expr.h"  

#include <set>
#include <vector>
#include <sstream>

using namespace clang;

bool DEBUG = type_analysis::AnalyzerConfig::DEBUG_ENABLED;


  
void extractGlobalHeaders(const clang::SourceManager &SM, std::vector<std::string> &headerLines) {
  clang::FileID mainFileID = SM.getMainFileID();
  llvm::StringRef fileContent = SM.getBufferData(mainFileID);

  llvm::SmallVector<llvm::StringRef, 64> lines;
  fileContent.split(lines, '\n');

  for (auto line : lines) {
    line = line.trim();
    if (line.startswith("#include") ||
      line.startswith("#if") ||
      line.startswith("#else") ||
      line.startswith("#endif") ||
      line.startswith("#define") ||
      line.startswith("#pragma")) {
      headerLines.push_back(line.str());
    }
  }
}

  
void GetWeightHandler::run(const ast_matchers::MatchFinder::MatchResult &Result) {
  const auto *Method = Result.Nodes.getNodeAs<CXXMethodDecl>("method");
  if (!Method) {
    if (DEBUG) llvm::errs() << "[ERROR] No method node found in match result\n";
    return;
  }
  
  if (!Method->hasBody()) {
    if (DEBUG) llvm::errs() << "[WARNING] Method has no body: " 
                           << Method->getQualifiedNameAsString() << "\n";
    return;
  }
  
  if (!Method->getParent()) {
    if (DEBUG) llvm::errs() << "[ERROR] Method has no parent class: " 
                           << Method->getQualifiedNameAsString() << "\n";
    return;
  }

  // Validate method signature (get_weight should return numeric type)
  const QualType returnType = Method->getReturnType();
  if (!returnType->isArithmeticType()) {
    if (DEBUG) llvm::errs() << "[WARNING] get_weight method returns non-arithmetic type: " 
                           << returnType.getAsString() << " in class "
                           << Method->getParent()->getNameAsString() << "\n";
  }

  // Print method name and class for debug
  if (DEBUG) llvm::outs() << "Matched get_weight() in class: " 
               << Method->getParent()->getNameAsString() << "\n";

  // Extract headers once, globally
  if (resultMap.globalHeaders.empty()) {
    extractGlobalHeaders(Result.Context->getSourceManager(), resultMap.globalHeaders);
  }
  
  extractReturns(Method, *Result.Context);
}

void GetWeightHandler::extractReturns(const CXXMethodDecl* Method, const ASTContext& Context) {
  const Stmt* Body = Method->getBody();
  if (!Body) return;

  // Extract return-related info
  ReturnVisitor visitor(Context, Method, resultMap, externalLocalsMap, fieldMap);
  visitor.TraverseStmt(const_cast<Stmt*>(Body));

  // Extract task class fields used in the function
  VarExtractor extractor(Context, Method->getParent());
  extractor.TraverseStmt(const_cast<Stmt*>(Body));

  std::string className = Method->getParent()->getNameAsString();
  resultMap.classes[className].taskFields = extractor.task_fields;
  
}