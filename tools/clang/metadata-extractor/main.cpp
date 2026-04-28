#include "WalkerInfoExtractor.hpp"
#include <fstream>
#include "clang/Tooling/CommonOptionsParser.h"
#include "clang/Tooling/Tooling.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Error.h"

using namespace clang;
using namespace clang::tooling;
using namespace llvm;

namespace {

/// CLI category and options
static cl::OptionCategory WalkerCategory("walker-tool options");

static cl::opt<std::string> OutputFile(
    "o",
    cl::desc("Specify output JSON filename"),
    cl::value_desc("filename"),
    cl::init("walker_meta.json"),
    cl::cat(WalkerCategory));

/// Default JSON indentation for readable output
constexpr int JSON_INDENT = 2;

} // anonymous namespace

/**
 * @brief AST Consumer for processing walker metadata.
 */
class WalkerConsumer : public clang::ASTConsumer {
public:
    explicit WalkerConsumer(ASTContext *Context) : Extractor(Context) {
        if (!Context) {
            llvm::errs() << "Error: Null ASTContext provided to WalkerConsumer\n";
        }
    }

    void HandleTranslationUnit(ASTContext &Context) override {
        // Traverse the AST to extract metadata
        Extractor.TraverseDecl(Context.getTranslationUnitDecl());

        // Write results to output file
        if (!writeOutputFile()) {
            llvm::errs() << "Error: Failed to write output file\n";
        }
    }

private:
    walker_metadata::WalkerInfoExtractor Extractor;

    /**
     * @brief Writes the extracted metadata to the output file.
     * @return true if successful, false otherwise
     */
    bool writeOutputFile() {
        std::ofstream out(OutputFile.getValue());
        if (!out) {
            llvm::errs() << "Failed to open output file: " << OutputFile.getValue() << "\n";
            return false;
        }

        try {
            out << Extractor.getOutput().dump(JSON_INDENT);
            return true;
        } catch (const std::exception &e) {
            llvm::errs() << "Error writing JSON output: " << e.what() << "\n";
            return false;
        }
    }
};

/**
 * @brief Frontend action for walker metadata extraction.
 */
class WalkerAction : public clang::ASTFrontendAction {
public:
    std::unique_ptr<clang::ASTConsumer> CreateASTConsumer(
        clang::CompilerInstance &CI,
        llvm::StringRef) override {
        return std::make_unique<WalkerConsumer>(&CI.getASTContext());
    }
};

/**
 * @brief Main entry point for the walker metadata extraction tool.
 */
int main(int argc, const char **argv) {
    auto ExpectedParser = CommonOptionsParser::create(argc, argv, WalkerCategory);
    if (!ExpectedParser) {
        llvm::errs() << "Error parsing command-line arguments:\n";
        llvm::logAllUnhandledErrors(ExpectedParser.takeError(), llvm::errs(), "");
        return 1;
    }

    CommonOptionsParser &OptionsParser = ExpectedParser.get();

    ClangTool Tool(OptionsParser.getCompilations(), OptionsParser.getSourcePathList());
    
    int result = Tool.run(newFrontendActionFactory<WalkerAction>().get());
    
    if (result == 0) {
        llvm::outs() << "Walker metadata extraction completed successfully.\n";
        llvm::outs() << "Output written to: " << OutputFile.getValue() << "\n";
    } else {
        llvm::errs() << "Walker metadata extraction failed with code: " << result << "\n";
    }
    
    return result;
}
