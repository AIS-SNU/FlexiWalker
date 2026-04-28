cmake_minimum_required(VERSION 3.15)

set(BUILD_SHARED_LIBS ON)

project(AdjwgtDetector LANGUAGES C CXX)

# Detect LLVM
find_package(LLVM REQUIRED CONFIG)

# Use LLVM's CMake helpers
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")
include(HandleLLVMOptions)
include(AddLLVM)

message(STATUS "Found LLVM ${LLVM_PACKAGE_VERSION}")
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")

add_definitions(-DLLVM_DISABLE_ABI_BREAKING_CHECKS_ENFORCING=1)

set(LLVM_PASS_NAME AdjwgtDetector)

file(GLOB SOURCES
  src/*.cpp
)

add_library(${LLVM_PASS_NAME} MODULE ${SOURCES})

target_include_directories(${LLVM_PASS_NAME} 
  PRIVATE 
    ${LLVM_INCLUDE_DIRS}
    ${CMAKE_SOURCE_DIR}/include
)

target_compile_definitions(${LLVM_PASS_NAME}
  PRIVATE LLVM_USE_NEW_PASS_MANAGER
)

set_target_properties(${LLVM_PASS_NAME} PROPERTIES
  POSITION_INDEPENDENT_CODE ON
  LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
  PREFIX ""
  SUFFIX ".so"
)
