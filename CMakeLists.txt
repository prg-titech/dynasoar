cmake_minimum_required(VERSION 3.2 FATAL_ERROR)
project(soa-alloc)

# Set up CUDA.
find_package(CUDA 9.0 REQUIRED)
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}; -std=c++11 -lineinfo)
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

include_directories(lib/cub)

# Choose an allocator.
#set(CMAKE_REQUIRED_DEFINITIONS -DALLOCATOR)

if(ALLOCATOR STREQUAL "MallocMC")
  message(STATUS "Building with MallocMC")
  include_directories(${CMAKE_SOURCE_DIR} ${CMAKE_SOURCE_DIR}/example/configuration/mallocmc/mallocMC ${CMAKE_SOURCE_DIR}/example/configuration/mallocmc)
elseif(ALLOCATOR STREQUAL "SoaAlloc")
  message(STATUS "Building with SoaAlloc")
  include_directories(${CMAKE_SOURCE_DIR} ${CMAKE_SOURCE_DIR}/example/configuration/soa_alloc)
elseif(ALLOCATOR STREQUAL "Cuda")
  message(STATUS "Building with standard CUDA allocator")
  include_directories(${CMAKE_SOURCE_DIR} ${CMAKE_SOURCE_DIR}/example/configuration/cuda_alloc)
else()
  message(FATAL_ERROR "Set allocator with -DALLOCATOR.")
endif()

add_subdirectory(example)
