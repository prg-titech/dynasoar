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

#pragma once
#define WARP_SIZE 32
#define STR_VALUE(arg) #arg
#define STR(arg) STR_VALUE(arg)

/**
 * \brief A structure holding information about dynamic memory heap and test setting.
 */
struct AllocTestInfo
{
	float allocScale;			//!< Allocation scale, range of allocation sizes.
	unsigned int testIters;		//!< Number of test iterations.
	float pAlloc;				//!< Allocation probability.
	float pFree;				//!< Dealocation probability.
};

#define ALIGN 16
#define COALESCE_WARP

//------------------------------------------------------------------------
// Debugging tests
//------------------------------------------------------------------------

// Test whether volatile save followed by volatile load returns the same value
//#define WRITE_READ_TEST
//#define WRITE_ALL_TEST
//#define READ_ALL_TEST
// Save allocated pointers into an auxiliary array and test the memory in a subsequent kernel
//#define SAVE_PTR_TEST

//------------------------------------------------------------------------
// CircularMalloc
//------------------------------------------------------------------------

#define CIRCULAR_MALLOC_CHECK_DEADLOCK
//#define CIRCULAR_MALLOC_GLOBAL_HEAP_LOCK // Use a single global lock for the heap
//#define CIRCULAR_MALLOC_DOUBLY_LINKED
#define CIRCULAR_MALLOC_PRELOCK
#define CIRCULAR_MALLOC_CONNECT_CHUNKS
// How to write data into global memory
// 0 - direct memory access
// 1 - through inline PTX caching qualifier
// 2 - through atomic operation
#define CIRCULAR_MALLOC_MEM_ACCESS_TYPE 2
#define MEM_ACCESS_TYPE CIRCULAR_MALLOC_MEM_ACCESS_TYPE // Macro in warp_common.cu
#define CIRCULAR_MALLOC_WAIT_COUNT 1000000

#ifndef CIRCULAR_MALLOC_DOUBLY_LINKED
#define CIRCULAR_MALLOC_HEADER_SIZE (2*sizeof(unsigned int))
#define CIRCULAR_MALLOC_NEXT_OFS sizeof(unsigned int)
#else
#define CIRCULAR_MALLOC_HEADER_SIZE (4*sizeof(unsigned int))
#define CIRCULAR_MALLOC_PREV_OFS sizeof(unsigned int)
#define CIRCULAR_MALLOC_NEXT_OFS (2*sizeof(unsigned int))
#endif

//------------------------------------------------------------------------
// ScatterAlloc
//------------------------------------------------------------------------

#define SCATTER_ALLOC_PAGESIZE 4096
#define SCATTER_ALLOC_ACCESSBLOCKS 8
#define SCATTER_ALLOC_REGIONSIZE 16
#define SCATTER_ALLOC_WASTEFACTOR 2
#define SCATTER_ALLOC_COALESCING 1
#define SCATTER_ALLOC_RESETPAGES 1

// Include the allocators modified by the above macros
#include "../kernels/gpualloc.h"

#ifdef __CUDACC__
// Test arrays
__device__ int** g_ptrArray;	//!< Array holding the allocated pointer for all threads
__device__ float* g_random;		//!< Array holding random numbers for all threads and iterations
__device__ int g_iter;			//!< The iteration count
__device__ int g_kernelFail;	//!< Non-zero if the run failed

__constant__ AllocTestInfo c_env;
#endif

//------------------------------------------------------------------------