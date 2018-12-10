cmake_minimum_required(VERSION 3.2 FATAL_ERROR)
project(soa-alloc)

# Set up CUDA
find_package(CUDA 9.0 REQUIRED)
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}; -std=c++11 --expt-extended-lambda -gencode arch=compute_50,code=sm_50 -gencode arch=compute_61,code=sm_61)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Debug)
endif()

set(CMAKE_CXX_FLAGS "-Wall -Wextra")
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG -Wno-unused-parameter")

#add_subdirectory(lib/googletest)
#enable_testing()

add_subdirectory(example)
