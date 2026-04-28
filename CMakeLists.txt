cmake_minimum_required(VERSION 3.15)

project(flowwalker VERSION 1.0 LANGUAGES CXX CUDA)

set(EXECUTABLE_OUTPUT_PATH ${PROJECT_BINARY_DIR}/bin)
set(CUDA_PATH "/usr/local/cuda")
set(HEAD_DIR ${PROJECT_SOURCE_DIR}/include)



add_executable(${PROJECT_NAME} ${PROJECT_SOURCE_DIR}/src/main.cu ${PROJECT_SOURCE_DIR}/src/walk.cu)


include_directories(${HEAD_DIR})

# set gpu architecture
set_property(TARGET ${PROJECT_NAME} PROPERTY CUDA_ARCHITECTURES "70;75;80;86")

# Link gflags
target_link_libraries(${PROJECT_NAME} -lgflags)


# enable compute-sanitizer
target_compile_options(${PROJECT_NAME} PUBLIC -lineinfo)

target_compile_options(${PROJECT_NAME} PUBLIC -O3)

# target_compile_options(${PROJECT_NAME} PUBLIC -maxrregcount=32)


# Enable dynamic parallelism
set_target_properties(${PROJECT_NAME} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_compile_options(${PROJECT_NAME} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
        --relocatable-device-code=true
        --expt-relaxed-constexpr
        >)
target_link_libraries(${PROJECT_NAME} -lcudadevrt)


# Link curand
target_link_libraries(${PROJECT_NAME} -lcurand)

# Link OpenMP
find_package(OpenMP REQUIRED)
target_link_libraries(${PROJECT_NAME} OpenMP::OpenMP_CXX)

target_compile_options(${PROJECT_NAME} PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-fopenmp>)
target_compile_options(${PROJECT_NAME} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-fopenmp>)