#pragma once
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/AST/ASTContext.h"
#include <map>
#include <set>
#include <string>
#include <vector>

#include "util.hpp"

/**
 * @brief Handler for analyzing get_weight method implementations
 * 
 * This class processes AST nodes to extract return patterns and variable
 * usage from walker get_weight methods for optimization analysis.
 */
class GetWeightHandler : public clang::ast_matchers::MatchFinder::MatchCallback {
public:
    /// Complete analysis results for all walker classes
    type_analysis::FullAnalysisResult resultMap;
    
    /// Map of external local variables per walker class
    std::map<std::string, std::set<std::string>> externalLocalsMap;
    
    /// Map of struct field information per walker class
    std::map<std::string, std::map<std::string, std::vector<std::string>>> fieldMap;

    /**
     * @brief Called when AST matcher finds a matching get_weight method
     * 
     * @param Result The matching result containing AST nodes
     */
    void run(const clang::ast_matchers::MatchFinder::MatchResult &Result) override;
    
    /**
     * @brief Sets external local variables mapping
     * 
     * @param map Map of class names to external variable sets
     */
    void setExternalLocals(const std::map<std::string, std::set<std::string>> &map) {
        externalLocalsMap = map;
    }
    
    /**
     * @brief Sets struct field mapping information
     * 
     * @param map Map of struct field information
     */
    void setStructFieldMap(const std::map<std::string, std::map<std::string, std::vector<std::string>>> &map) {
        fieldMap = map;
    }

private:
    /**
     * @brief Extracts return statements and expressions from method body
     * 
     * @param Method The method declaration to analyze
     * @param Context AST context for analysis
     */
    void extractReturns(const clang::CXXMethodDecl* Method, const clang::ASTContext& Context);
};