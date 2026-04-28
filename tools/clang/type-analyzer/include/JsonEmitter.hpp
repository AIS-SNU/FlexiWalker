#pragma once
#include <string>
#include <map>
#include <vector>

#include "util.hpp"

/**
 * @brief Handles JSON output generation for type analysis results
 * 
 * This class manages the serialization of analysis results to JSON format,
 * automatically writing output when the object is destroyed (RAII pattern).
 */
class JsonEmitter {
public:
    /**
     * @brief Constructs JSON emitter with analysis results
     * 
     * @param resultRef Reference to full analysis results
     * @param fieldMapRef Reference to field mapping information
     * @param typeMapRef Reference to type mapping information
     * @param outputPathStr Path for JSON output file
     */
    JsonEmitter(type_analysis::FullAnalysisResult &resultRef,
                const std::map<std::string, std::map<std::string, std::vector<std::string>>> &fieldMapRef,
                const std::map<std::string, std::map<std::string, std::vector<std::string>>> &typeMapRef,
                const std::string &outputPathStr);

    /**
     * @brief Destructor - automatically writes JSON output
     */
    ~JsonEmitter();

private:
    /// Reference to analysis results
    type_analysis::FullAnalysisResult &results;
    
    /// Reference to field mapping data
    const std::map<std::string, std::map<std::string, std::vector<std::string>>> &fieldMap;
    
    /// Reference to type mapping data
    const std::map<std::string, std::map<std::string, std::vector<std::string>>> &typeMap;
    
    /// Output file path
    std::string outputPath;
};