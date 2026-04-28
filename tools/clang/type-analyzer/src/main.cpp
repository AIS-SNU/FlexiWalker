#include "clang/Tooling/CommonOptionsParser.h"
#include "clang/Frontend/FrontendActions.h"
#include "clang/Tooling/Tooling.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "llvm/Support/CommandLine.h"
#include <fstream> 
#include <nlohmann/json.hpp>

#include "GetWeightHandler.hpp"
#include "StructFieldCollector.hpp"
#include "JsonEmitter.hpp"

using namespace clang::tooling;
using namespace clang::ast_matchers;
using namespace llvm;
using json = nlohmann::json;

namespace {

/// CLI category and options
static cl::OptionCategory ToolCategory("type-analyzer options");

static llvm::cl::opt<std::string> ConfigPath(
    "config",
    llvm::cl::desc("Path to the JSON config file"),
    llvm::cl::value_desc("filename"),
    llvm::cl::init(""),
    llvm::cl::cat(ToolCategory));

static cl::opt<std::string> OutputPath(
    "o",
    cl::desc("Path to output JSON file"),
    cl::value_desc("filename"),
    cl::init(type_analysis::AnalyzerConfig::DEFAULT_OUTPUT_FILE),
    cl::cat(ToolCategory));

} // anonymous namespace

/**
 * @brief Main entry point for the type analyzer tool
 * 
 * This tool analyzes walker classes to extract type information and generate
 * optimized code patterns based on field access analysis.
 */
int main(int argc, const char **argv) {
    auto ExpectedParser = CommonOptionsParser::create(argc, argv, ToolCategory);
    if (!ExpectedParser) {
        llvm::errs() << "Error parsing command-line arguments:\n";
        llvm::logAllUnhandledErrors(ExpectedParser.takeError(), llvm::errs(), "");
        return 1;
    }

    CommonOptionsParser &OptionsParser = ExpectedParser.get();
    ClangTool Tool(OptionsParser.getCompilations(), OptionsParser.getSourcePathList());

    // Data structures for analysis results
    std::map<std::string, std::set<std::string>> externalLocalsMap;
    std::map<std::string, std::map<std::string, std::set<unsigned>>> accessedByIndexMap;

    // === Load configuration file if provided ===
    if (!ConfigPath.empty()) {
        std::ifstream input(ConfigPath);
        if (!input) {
            llvm::errs() << "Error: Could not open config file: " << ConfigPath << "\n";
            return 1;
        }

        json j;
        try {
            input >> j;
        } catch (const std::exception &e) {
            llvm::errs() << "Error parsing JSON config: " << e.what() << "\n";
            return 1;
        }

        // Process configuration data
        for (auto &[className, info] : j.items()) {
            if (info.contains("accessedByIndex")) {
                for (const auto &entry : info["accessedByIndex"]) {
                    std::string structName = entry[0];
                    unsigned fieldIdx = entry[1];
                    accessedByIndexMap[className][structName].insert(fieldIdx);
                }
            }

            // TODO: Consider removing this fallback behavior
            if (info.contains("updatedFields")) {
                for (const auto &entry : info["updatedFields"]) {
                    std::string structName = entry[0].get<std::string>();
                    unsigned fieldIdx = entry[1];
                    accessedByIndexMap[className][structName].insert(fieldIdx);
                }
            }

            if (info.contains("externallyDefinedLocals")) {
                for (const auto &var : info["externallyDefinedLocals"]) {
                    externalLocalsMap[className].insert(var.get<std::string>());
                }
            }
            
            // Insert "i" as default external variable
            externalLocalsMap[className].insert("i");
        }
    }

    // === First Pass: Collect Struct Fields ===
    MatchFinder StructFinder;
    type_analysis::StructFieldCollector StructCollector;
    StructCollector.setTargetIndices(accessedByIndexMap);

    StructFinder.addMatcher(
        cxxRecordDecl(isDefinition()).bind("record"),
        &StructCollector
    );

    // Run first pass to collect struct information
    int result = Tool.run(newFrontendActionFactory(&StructFinder).get());
    if (result != 0) {
        llvm::errs() << "Error during struct field collection phase\n";
        return result;
    }

    // === Second Pass: Analyze get_weight methods ===
    MatchFinder Finder;
    GetWeightHandler Handler;

    Handler.setExternalLocals(externalLocalsMap);
    Handler.setStructFieldMap(StructCollector.fieldMap);

    if (DEBUG) {
        type_analysis::printFieldMap(StructCollector.fieldMap);
    }

    Finder.addMatcher(
        cxxMethodDecl(hasName("get_weight"), isDefinition()).bind("method"),
        &Handler
    );

    // Initialize JSON emitter for output
    JsonEmitter emitter(Handler.resultMap, StructCollector.fieldMap, 
                       StructCollector.typeMap, OutputPath);

    // Run second pass to analyze methods
    result = Tool.run(newFrontendActionFactory(&Finder).get());
    
    if (result == 0) {
        llvm::outs() << "Type analysis completed successfully.\n";
        llvm::outs() << "Output written to: " << OutputPath << "\n";
    } else {
        llvm::errs() << "Error during get_weight analysis phase\n";
    }

    return result;
}
