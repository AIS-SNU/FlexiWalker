cmake_minimum_required(VERSION 3.15)
project(WalkerMetaExtractor)

set(PROJECT_NAME pre-tool)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Force LLVM 17 directory discovery
set(LLVM_DIR "/usr/lib/llvm-17/lib/cmake/llvm")
set(Clang_DIR "/usr/lib/llvm-17/lib/cmake/clang")

# LLVM Setup
find_package(LLVM REQUIRED CONFIG)
message(STATUS "Found LLVM ${LLVM_PACKAGE_VERSION}")
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")

# Set Clang (included via LLVM)
find_package(Clang REQUIRED CONFIG)
message(STATUS "Found Clang in: ${Clang_DIR}")

include_directories(SYSTEM
  ${LLVM_INCLUDE_DIRS}
  ${CLANG_INCLUDE_DIRS}
)

link_directories(${LLVM_LIBRARY_DIRS})
add_definitions(${LLVM_DEFINITIONS})
add_definitions(${CLANG_DEFINITIONS})

# Disable RTTI to match LLVM/Clang build
add_compile_options(-fno-rtti)

# Optional: Add nlohmann/json (if installed via package manager)
find_package(nlohmann_json REQUIRED)

# Your sources
add_executable(${PROJECT_NAME}
  main.cpp
  WalkerInfoExtractor.hpp
)

# Link required Clang/LLVM components
target_link_libraries(${PROJECT_NAME}
  PRIVATE
  clangTooling
  clangFrontend
  clangAST
  clangBasic
  clangLex
  clangDriver
  clangParse
  clangSema
  clangAnalysis
  clangSerialization
  clangEdit
  clangFormat
  nlohmann_json::nlohmann_json
)
