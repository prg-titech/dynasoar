cmake_minimum_required(VERSION 3.2 FATAL_ERROR)
project(soa-alloc)

# Set up CUDA.
find_package(CUDA 9.0 REQUIRED)
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}; -std=c++11)
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}; --expt-extended-lambda)
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}; -gencode arch=compute_50,code=sm_50)
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}; -gencode arch=compute_61,code=sm_61)
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}; -maxrregcount=64)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Debug)
endif()

set(CMAKE_CXX_FLAGS          "-Wall -Wextra")
set(CMAKE_CXX_FLAGS_DEBUG    "-g -O0")
set(CMAKE_CXX_FLAGS_RELEASE  "-O3 -DNDEBUG")

#add_subdirectory(lib/googletest)
#enable_testing()

add_subdirectory(example)
