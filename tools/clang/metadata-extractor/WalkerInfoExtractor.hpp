#pragma once

#include "clang/AST/AST.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Tooling/Tooling.h"
#include "llvm/Support/raw_ostream.h"
#include "nlohmann/json.hpp"

#include <string>
#include <vector>

using namespace clang;
using json = nlohmann::json;

namespace walker_metadata {

/// Constants for metadata extraction
struct MetadataConstants {
    // Base class names to detect
    static constexpr const char* WALKER_META_CLASS = "WalkerMeta";

    // Default dummy values for different types
    static constexpr const char* DEFAULT_FLOAT_VALUE = "0.5";
    static constexpr const char* DEFAULT_INT_VALUE = "0";
    static constexpr const char* DEFAULT_GRAPH_VALUE = "g";
    static constexpr const char* DEFAULT_POINTER_VALUE = "nullptr";
    static constexpr const char* DEFAULT_FALLBACK_VALUE = "0";

    // Graph record type names (matched against canonical record decl name)
    static constexpr const char* GRAPH_RECORD_NAME = "graph";
    static constexpr const char* GPU_GRAPH_RECORD_NAME = "gpu_graph";
};

} // namespace walker_metadata

namespace walker_metadata {

/**
 * @brief Extracts metadata information from Walker classes.
 * 
 * This class traverses the AST to find classes that inherit from WalkerMeta
 * and extracts their constructor arguments and methods for code generation.
 */
class WalkerInfoExtractor : public RecursiveASTVisitor<WalkerInfoExtractor> {
public:
    explicit WalkerInfoExtractor(ASTContext *Context) : Context(Context) {
        if (!Context) {
            llvm::errs() << "Error: Null ASTContext provided to WalkerInfoExtractor\n";
        }
    }

    /**
     * @brief Visits CXX record declarations to extract walker metadata.
     * @param Declaration The class declaration to analyze
     * @return true to continue AST traversal
     */
    bool VisitCXXRecordDecl(CXXRecordDecl *Declaration) {
        // Early validation checks
        if (!Declaration || !Declaration->hasDefinition() || 
            !Declaration->isThisDeclarationADefinition()) {
            return true;
        }

        if (!Declaration->getIdentifier()) {
            return true;
        }

        std::string className = Declaration->getNameAsString();
        if (!inheritsFromWalkerMeta(Declaration)) {
            return true;
        }

        json &entry = output[className];

        // Extract constructor information
        if (!extractConstructorInfo(Declaration, entry)) {
            llvm::errs() << "Warning: Failed to extract constructor info for "
                        << className << "\n";
        }

        // Extract TaskType information
        extractTaskType(Declaration, entry);

        // Record every user-defined method on the walker class (other
        // than the standard-API methods get_weight / is_stop, which are
        // already called explicitly in walker_dummy.j2). The codegen
        // template emits one call per entry, so the LLVM analyzer sees
        // each body in IR — that's how the bug #3 graph-mutation check
        // generalizes to any user-coded helper, not just one literally
        // named "update_weight".
        extractUserMethods(Declaration, entry);

        return true;
    }

    /**
     * @brief Gets the extracted metadata as JSON.
     * @return Const reference to the JSON output
     */
    const json &getOutput() const { return output; }

private:
    ASTContext *Context;
    json output;

    /**
     * @brief Checks if a class inherits from WalkerMeta.
     * @param decl The class declaration to check
     * @return true if the class inherits from WalkerMeta
     */
    bool inheritsFromWalkerMeta(const CXXRecordDecl *decl) {
        if (!decl) return false;
        
        for (const auto &base : decl->bases()) {
            const auto *baseType = base.getType()->getAsCXXRecordDecl();
            if (baseType && baseType->getNameAsString() == MetadataConstants::WALKER_META_CLASS) {
                return true;
            }
        }
        return false;
    }

    /**
     * @brief Extracts constructor information from a class declaration.
     * @param decl The class declaration
     * @param entry JSON entry to populate
     * @return true if extraction was successful
     */
    bool extractConstructorInfo(const CXXRecordDecl *decl, json &entry) {
        if (!decl) return false;

        for (auto ctor : decl->ctors()) {
            if (ctor->isImplicit()) {
                continue;
            }

            std::vector<std::string> args;
            for (const auto *param : ctor->parameters()) {
                if (!param) continue;
                args.push_back(getDummyValue(param->getType()));
            }

            // Use the first non-empty constructor we find
            if (!args.empty()) {
                entry["args"] = args;
                return true;
            }
        }

        // If no suitable constructor found, add empty args
        entry["args"] = json::array();
        return false;
    }

    /**
     * @brief Generates a dummy literal/initializer for a constructor parameter.
     *
     * Uses Clang's type API (not substring matching on type spellings) so
     * widths and qualifiers are handled correctly:
     *   - graph* / gpu_graph* resolve to "g" (the dummy_kernel param), so the
     *     synthesized walker_dummy.cu produces real loads-through-pointer for
     *     LLVM IR analysis instead of null-deref UB.
     *   - all other pointer types resolve to nullptr (fixes int*, T* for any T).
     *   - any integer width (int/int64_t/size_t/bool/...) resolves to 0.
     *   - any floating type (float/double) resolves to 0.5 (nonzero default
     *     avoids accidental constant folding through divides at higher -O).
     *   - reference types: graph&/gpu_graph& dereference to "*g"; primitive
     *     `const T&` binds to the matching literal (0 / 0.5). Non-const
     *     references to primitives still fall through — no production walker
     *     uses that signature, and supporting it would need a scratch local
     *     in walker_dummy.j2.
     */
    std::string getDummyValue(QualType qt) {
        qt = qt.getCanonicalType();

        if (qt->isReferenceType()) {
            QualType pointee = qt->getPointeeType().getCanonicalType();
            if (const auto *recordType = pointee->getAs<RecordType>()) {
                if (const auto *recordDecl = recordType->getDecl()) {
                    const std::string name = recordDecl->getNameAsString();
                    if (name == MetadataConstants::GRAPH_RECORD_NAME ||
                        name == MetadataConstants::GPU_GRAPH_RECORD_NAME) {
                        return std::string("*") + MetadataConstants::DEFAULT_GRAPH_VALUE;
                    }
                }
            }
            // Primitive const-ref: literals bind to const T&.
            if (pointee->isIntegerType()) return MetadataConstants::DEFAULT_INT_VALUE;
            if (pointee->isFloatingType()) return MetadataConstants::DEFAULT_FLOAT_VALUE;
            return MetadataConstants::DEFAULT_FALLBACK_VALUE;
        }
        if (qt->isPointerType()) {
            QualType pointee = qt->getPointeeType().getCanonicalType();
            if (const auto *recordType = pointee->getAs<RecordType>()) {
                if (const auto *recordDecl = recordType->getDecl()) {
                    const std::string name = recordDecl->getNameAsString();
                    if (name == MetadataConstants::GRAPH_RECORD_NAME ||
                        name == MetadataConstants::GPU_GRAPH_RECORD_NAME) {
                        return MetadataConstants::DEFAULT_GRAPH_VALUE;
                    }
                }
            }
            return MetadataConstants::DEFAULT_POINTER_VALUE;
        }
        if (qt->isIntegerType()) {
            return MetadataConstants::DEFAULT_INT_VALUE;
        }
        if (qt->isFloatingType()) {
            return MetadataConstants::DEFAULT_FLOAT_VALUE;
        }
        return MetadataConstants::DEFAULT_FALLBACK_VALUE;
    }

    /**
     * @brief Generates a dummy argument for a user-method parameter.
     *
     * Like getDummyValue(), but additionally resolves Task* / TaskType*
     * to "task_ptr_<walkerName>" — the per-walker dummy task pointer
     * declared by walker_dummy.j2 — so calls to user methods produce
     * real loads-through-pointer instead of null-deref UB.
     *
     * Reference parameters: dereference the matching pointer dummy when
     * the referent is graph/gpu_graph or Task/TaskType; for primitive
     * `const T&`, the literal binds directly. Non-const references to
     * primitives are not supported (would need a scratch local in
     * walker_dummy.j2 — no production walker hits this).
     */
    std::string getMethodArgValue(QualType qt,
                                  const std::string &walkerName,
                                  const std::string &taskTypeName) {
        qt = qt.getCanonicalType();

        if (qt->isReferenceType()) {
            QualType pointee = qt->getPointeeType().getCanonicalType();
            if (const auto *recordType = pointee->getAs<RecordType>()) {
                if (const auto *recordDecl = recordType->getDecl()) {
                    const std::string name = recordDecl->getNameAsString();
                    if (name == MetadataConstants::GRAPH_RECORD_NAME ||
                        name == MetadataConstants::GPU_GRAPH_RECORD_NAME) {
                        return std::string("*") + MetadataConstants::DEFAULT_GRAPH_VALUE;
                    }
                    if (name == taskTypeName || name == "Task") {
                        return "*task_ptr_" + walkerName;
                    }
                }
            }
            if (pointee->isIntegerType()) return MetadataConstants::DEFAULT_INT_VALUE;
            if (pointee->isFloatingType()) return MetadataConstants::DEFAULT_FLOAT_VALUE;
            return MetadataConstants::DEFAULT_FALLBACK_VALUE;
        }
        if (qt->isPointerType()) {
            QualType pointee = qt->getPointeeType().getCanonicalType();
            if (const auto *recordType = pointee->getAs<RecordType>()) {
                if (const auto *recordDecl = recordType->getDecl()) {
                    const std::string name = recordDecl->getNameAsString();
                    if (name == MetadataConstants::GRAPH_RECORD_NAME ||
                        name == MetadataConstants::GPU_GRAPH_RECORD_NAME) {
                        return MetadataConstants::DEFAULT_GRAPH_VALUE;
                    }
                    // Match either the resolved TaskType (e.g. "PPRSecondTask")
                    // or the bare base class name "Task" — users may write
                    // either spelling in their method signatures.
                    if (name == taskTypeName || name == "Task") {
                        return "task_ptr_" + walkerName;
                    }
                }
            }
            return MetadataConstants::DEFAULT_POINTER_VALUE;
        }
        if (qt->isIntegerType()) {
            return MetadataConstants::DEFAULT_INT_VALUE;
        }
        if (qt->isFloatingType()) {
            return MetadataConstants::DEFAULT_FLOAT_VALUE;
        }
        return MetadataConstants::DEFAULT_FALLBACK_VALUE;
    }

    /**
     * @brief Records all user-defined methods on the walker class.
     *
     * Each entry is `{name, args}`. Skipped:
     *   - implicit methods (compiler-generated)
     *   - constructors / destructors (handled by extractConstructorInfo)
     *   - declaration-only methods (no body — calling them from the
     *     dummy stub would fail at link time, and the analyzer cannot
     *     see a body it does not have)
     *   - standard-API and codegen-generated methods by name:
     *       get_weight, is_stop      — called explicitly by walker_dummy.j2
     *       scan_thread              — framework dispatch, not user logic;
     *                                  including it perturbs IR field-access
     *                                  patterns that drive MonotonicityRewriter
     *       fill_dummy / get_max_weight / get_sum_weight
     *                                — bodies are emitted by codegen .cuh
     *                                  files, not authored by the user
     *
     * Name-based exclusion (rather than hasBody-only) is intentional: a
     * user could write inline definitions for the codegen-generated
     * methods, which would survive the hasBody filter and leak into the
     * dummy stub.
     *
     * Emitted methods are called from the synthesized dummy_kernel so
     * their bodies become part of the LLVM IR. Downstream, the LLVM
     * analyzer applies the bug #3 graph-mutation check to every walker
     * method — generalizing the check past the legacy "update_weight"
     * literal to any user-coded helper.
     */
    void extractUserMethods(const CXXRecordDecl *decl, json &entry) {
        json methods = json::array();
        if (!decl) {
            entry["user_methods"] = methods;
            return;
        }

        const std::string walkerName = decl->getNameAsString();
        const std::string taskTypeName = entry.value("task_type", std::string("Task"));

        for (const auto *m : decl->methods()) {
            if (!m || m->isImplicit()) continue;
            if (isa<CXXConstructorDecl>(m) || isa<CXXDestructorDecl>(m)) continue;
            if (!m->hasBody()) continue;

            const std::string methodName = m->getNameAsString();
            if (methodName == "get_weight" || methodName == "is_stop" ||
                methodName == "scan_thread" || methodName == "fill_dummy" ||
                methodName == "get_max_weight" || methodName == "get_sum_weight") continue;

            json mj;
            mj["name"] = methodName;
            json args = json::array();
            for (const auto *param : m->parameters()) {
                if (!param) continue;
                args.push_back(getMethodArgValue(param->getType(), walkerName, taskTypeName));
            }
            mj["args"] = args;
            methods.push_back(mj);
        }
        entry["user_methods"] = methods;
    }

    /**
     * @brief Extracts TaskType typedef/using declaration from walker class.
     *
     * Uses Clang name lookup (which walks the inheritance chain) instead of
     * iterating the class's direct members, so a walker that inherits its
     * TaskType from WalkerMeta resolves to the actual inherited type rather
     * than silently falling through to the hardcoded "Task" default.
     */
    void extractTaskType(const CXXRecordDecl *decl, json &entry) {
        if (!decl) return;

        // Default TaskType is "Task" - only used if lookup finds nothing,
        // which should be impossible since WalkerMeta itself defines one.
        std::string taskTypeName = "Task";

        DeclarationName name(&Context->Idents.get("TaskType"));
        auto results = decl->lookup(name);
        for (auto *found : results) {
            QualType underlyingType;
            if (auto *typeAlias = dyn_cast<TypeAliasDecl>(found)) {
                underlyingType = typeAlias->getUnderlyingType();
            } else if (auto *typedefDecl = dyn_cast<TypedefNameDecl>(found)) {
                underlyingType = typedefDecl->getUnderlyingType();
            } else {
                continue;
            }

            if (const auto *recordType = underlyingType->getAs<RecordType>()) {
                if (const auto *recordDecl = recordType->getDecl()) {
                    taskTypeName = recordDecl->getNameAsString();
                }
            }
            break;
        }

        entry["task_type"] = taskTypeName;
    }
};

} // namespace walker_metadata
