#include "llvm/Support/raw_ostream.h"

#include "StructFieldCollector.hpp"
#include "util.hpp"

namespace type_analysis {

void StructFieldCollector::run(const MatchFinder::MatchResult &Result) {
    const auto *Record = Result.Nodes.getNodeAs<CXXRecordDecl>("record");
    if (!Record) {
        if (DEBUG) llvm::errs() << "[ERROR] No record node found in match result\n";
        return;
    }
    
    if (!Record->isThisDeclarationADefinition()) {
        if (DEBUG) llvm::errs() << "[INFO] Skipping forward declaration: " 
                               << Record->getQualifiedNameAsString() << "\n";
        return;
    }

    std::string originalName = Record->getQualifiedNameAsString();
    std::string structName = "class." + Record->getQualifiedNameAsString(); // e.g., "class.gpu_graph"

    // Validate that the record has fields
    if (Record->field_empty()) {
        if (DEBUG) llvm::errs() << "[WARNING] Record has no fields: " << originalName << "\n";
        return;
    }

    // Count total fields for validation
    unsigned totalFields = std::distance(Record->field_begin(), Record->field_end());
    // LLVM lays out embedded base-class subobjects BEFORE declared fields, so the
    // LLVM field index for the Nth declared field equals N + numBases. Mirror that
    // here so inherited-field accesses on derived structs resolve correctly.
    unsigned numBases = std::distance(Record->bases_begin(), Record->bases_end());

    for (const auto &walkerEntry : targetFieldsByStruct) {
        const std::string &walkerClass = walkerEntry.first;
        const auto &structFieldMap = walkerEntry.second;

        auto it = structFieldMap.find(structName);
        if (it == structFieldMap.end()) continue; // this walker doesn't access this struct

        const auto &fieldIndices = it->second;

        // Validate field indices are within bounds
        for (unsigned idx : fieldIndices) {
            if (idx >= numBases + totalFields) {
                if (DEBUG) llvm::errs() << "[ERROR] Field index " << idx
                                       << " out of bounds for struct " << structName
                                       << " (has " << numBases << " bases + "
                                       << totalFields << " fields)\n";
                continue;
            }
        }

        std::vector<std::string> selectedFields;
        std::vector<std::string> fieldTypes;
        unsigned idx = numBases;

        for (const auto *field : Record->fields()) {
            if (fieldIndices.count(idx)) {
                std::string fieldName = field->getNameAsString();
                std::string fieldType = field->getType().getAsString();

                // Validate field name is not empty
                if (fieldName.empty()) {
                    if (DEBUG) llvm::errs() << "[WARNING] Field at index " << idx 
                                           << " has empty name in struct " << structName << "\n";
                    fieldName = "unnamed_field_" + std::to_string(idx);
                }

                selectedFields.push_back(fieldName);
                fieldTypes.push_back(fieldType);

                if (DEBUG) {
                    llvm::errs() << "[StructFieldCollector] "
                             << "Walker class: " << walkerClass
                             << ", Struct: " << structName
                             << ", Field Index: " << idx
                             << ", Field Name: " << fieldName 
                             << ", Field Type: " << fieldType << "\n";
                }
            }
            ++idx;
        }

        // Only store non-empty results
        if (!selectedFields.empty()) {
            fieldMap[walkerClass][originalName] = std::move(selectedFields);
            typeMap[walkerClass][structName] = std::move(fieldTypes);
        }
    }
}

} // namespace type_analysis