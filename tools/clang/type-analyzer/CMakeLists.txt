cmake_minimum_required(VERSION 3.15)
project(GetWeightASTTool)

set(PROJECT_NAME "ast-tool")

# Force LLVM 17 directory discovery
set(LLVM_DIR "/usr/lib/llvm-17/lib/cmake/llvm")
set(Clang_DIR "/usr/lib/llvm-17/lib/cmake/clang")

# Find LLVM/Clang
find_package(LLVM REQUIRED CONFIG)
find_package(Clang REQUIRED CONFIG)

message(STATUS "Found LLVM ${LLVM_PACKAGE_VERSION}")

# Include LLVM and Clang headers
include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${CLANG_INCLUDE_DIRS})
add_definitions(${LLVM_DEFINITIONS})
add_definitions(-DSOURCE_DIR="${CMAKE_SOURCE_DIR}")

# Add your include and src directories
include_directories(${CMAKE_SOURCE_DIR}/include)

# Glob source files (alternatively list them explicitly)
file(GLOB_RECURSE AST_TOOL_SOURCES
  "${CMAKE_SOURCE_DIR}/src/*.cpp"
)

# Create the executable
add_executable(${PROJECT_NAME} ${AST_TOOL_SOURCES})

# Link against required Clang libs
target_link_libraries(${PROJECT_NAME}
  clangTooling
  clangBasic
  clangASTMatchers
  clangAST
  clangFrontend
)
