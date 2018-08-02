/*
 *  Copyright (c) 2014, Faculty of Informatics, Masaryk University
 *  All rights reserved.
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions are met:
 *      * Redistributions of source code must retain the above copyright
 *        notice, this list of conditions and the following disclaimer.
 *      * Redistributions in binary form must reproduce the above copyright
 *        notice, this list of conditions and the following disclaimer in the
 *        documentation and/or other materials provided with the distribution.
 *      * Neither the name of NVIDIA Corporation nor the
 *        names of its contributors may be used to endorse or promote products
 *        derived from this software without specific prior written permission.
 *
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 *  ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 *  WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 *  DISCLAIMED. IN NO EVENT SHALL <COPYRIGHT HOLDER> BE LIABLE FOR ANY
 *  DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 *  (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 *  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 *  ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 *  (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 *  SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 *  Authors:
 *  Marek Vinkler <xvinkl@fi.muni.cz>
 */

/*
 *  Register Efficient Memory Allocator for GPUs
 *  Marek Vinkler, Vlastimil Havran
 *  Proc. High-Performance Graphics 2014
 *
 *  Allocation and deallocation routines for these allocators:
 *  CudaMalloc       - mallocCudaMalloc, freeCudaMalloc
 *  AtomicMalloc     - mallocAtomicMalloc
 *  AWMalloc         - mallocAtomicWrapMalloc
 *  CMalloc          - mallocCircularMalloc, freeCircularMalloc
 *  CFMalloc         - mallocCircularFusedMalloc, freeCircularFusedMalloc
 *  CMMalloc         - mallocCircularMultiMalloc, freeCircularMultiMalloc
 *  CFMMalloc        - mallocCircularFusedMultiMalloc, freeCircularFusedMultiMalloc
 *  ScatterAlloc     - mallocScatterAlloc, freeScatterAlloc
 *  FDGMalloc        - mallocFDGMalloc, freeFDGMalloc
 *  Version: 1.0
 */

#define _M_X64

#pragma once
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#ifdef __CUDACC__
#include <stdio.h>

typedef unsigned int uint;
typedef unsigned long long int ullint;
#endif

// Enum for various locking states
enum AllocatorLockType {AllocatorLockType_Free = 0, AllocatorLockType_Set};

// A structure holding information about dynamic memory heap
struct AllocInfo
{
	unsigned int heapSize;
	unsigned int payload;
	double maxFrag;
	double chunkRatio;
};

//------------------------------------------------------------------------
// Debugging tests
//------------------------------------------------------------------------

#define CHECK_OUT_OF_MEMORY // For unknown reason cannot be used together with WRITE_ALL_TEST

//------------------------------------------------------------------------
// CircularMalloc
//------------------------------------------------------------------------

#ifndef CIRCULAR_MALLOC_DOUBLY_LINKED
#define CIRCULAR_MALLOC_HEADER_SIZE (2*sizeof(unsigned int))
#define CIRCULAR_MALLOC_NEXT_OFS sizeof(unsigned int)
#else
#define CIRCULAR_MALLOC_HEADER_SIZE (4*sizeof(unsigned int))
#define CIRCULAR_MALLOC_PREV_OFS sizeof(unsigned int)
#define CIRCULAR_MALLOC_NEXT_OFS (2*sizeof(unsigned int))
#endif
//#define CIRCULAR_MALLOC_CHECK_INTERNAL_FRAGMENTATION
//#define CIRCULAR_MALLOC_CHECK_EXTERNAL_FRAGMENTATION


#ifdef __CUDACC__
//------------------------------------------------------------------------
// ScatterAlloc - Downloadable from: http://www.icg.tugraz.at/project/mvp
//------------------------------------------------------------------------



//------------------------------------------------------------------------
// FDGMalloc - Downloadable from: http://www.gris.informatik.tu-darmstadt.de/projects/fdgmalloc/
//------------------------------------------------------------------------
#include "fdg/FDGMalloc.cuh"
#include "fdg/FDGMalloc.cu"

//------------------------------------------------------------------------

// Heap data
__device__ char* g_heapBase; // The base pointer to the heap
__device__ uint g_heapOffset; // Current location in the heap
__device__ uint* g_heapMultiOffset; // Current location in the heap for each multiprocessor
__device__ uint g_numSM; // Number of SMs on the device
__device__ uint g_heapLock; // Lock for updating the heap

__constant__ AllocInfo c_alloc;

#ifdef CIRCULAR_MALLOC_CHECK_INTERNAL_FRAGMENTATION
__device__ float* g_interFragSum; // A float for every thread
#endif

__device__ __forceinline__ void* mallocCudaMalloc(uint allocSize);
__device__ __forceinline__ void freeCudaMalloc(void* ptr);

__device__ __forceinline__ void* mallocAtomicMalloc(uint allocSize);

__device__ __forceinline__ void* mallocAtomicWrapMalloc(uint allocSize);

__device__ __forceinline__ void* mallocCircularMalloc(uint allocSize);
__device__ __forceinline__ void freeCircularMalloc(void* ptr);

__device__ __forceinline__ void* mallocCircularFusedMalloc(uint allocSize);
__device__ __forceinline__ void freeCircularFusedMalloc(void* ptr);

__device__ __forceinline__ void* mallocCircularMultiMalloc(uint allocSize);
__device__ __forceinline__ void freeCircularMultiMalloc(void* ptr);

__device__ __forceinline__ void* mallocCircularFusedMultiMalloc(uint allocSize);
__device__ __forceinline__ void freeCircularFusedMultiMalloc(void* ptr);


__device__ __forceinline__ void* mallocFDGMalloc(FDG::Warp* warp, uint allocSize);
__device__ __forceinline__ void freeFDGMalloc(FDG::Warp* warp);


extern "C" __global__ void CircularMallocPrepare1(uint numChunks);
extern "C" __global__ void CircularMallocPrepare3(uint numChunks, uint rootChunk);

extern "C" __global__ void CircularFusedMallocPrepare1(uint numChunks);
extern "C" __global__ void CircularFusedMallocPrepare3(uint numChunks, uint rootChunk);

extern "C" __global__ void CircularMultiMallocPrepare1(uint numChunks);
extern "C" __global__ void CircularMultiMallocPrepare3(uint numChunks, uint rootChunk);

extern "C" __global__ void CircularFusedMultiMallocPrepare1(uint numChunks);
extern "C" __global__ void CircularFusedMultiMallocPrepare3(uint numChunks, uint rootChunk);

#endif

//------------------------------------------------------------------------