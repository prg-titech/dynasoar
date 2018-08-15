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
    Test of various dynamic allocation methods.
*/

#include "AllocTest.hpp"
#include "../kernels/gpualloc.cu"

//------------------------------------------------------------------------
//-----------------------Internal test functions--------------------------
//------------------------------------------------------------------------

__device__ __forceinline__ void readWriteTest(int* ptr, int tid)
{
	*(volatile int*)ptr = tid;
	int tidR = *(volatile int*)ptr;

	if(tid != tidR)
	{
		printf("Allocation followed by read and write error inside kernel for tid %d.\n", tid);
		g_kernelFail = 1;
	}
}

//------------------------------------------------------------------------

__device__ __forceinline__ void writeAllTest(int* ptr, uint allocSize, int tid, uint laneid)
{
	volatile int* wrt = ptr;
	uint intSize = allocSize / sizeof(int);
	for(uint ofs = laneid; ofs < intSize; ofs += WARP_SIZE)
	{
		*(wrt + ofs) = tid;
	}
}

//------------------------------------------------------------------------

__device__ __forceinline__ void readAllTest(int* ptr, uint allocSize, int tid, uint laneid)
{
	volatile int* wrt = ptr;
	uint intSize = allocSize / sizeof(int);
	for(uint ofs = laneid; ofs < intSize; ofs += WARP_SIZE)
	{
		int tidR = *(wrt + ofs);
		if(tid != tidR)
		{
			printf("Allocation of size allocSize %u followed by writes of tid %d and reads of tid %d.\n", allocSize, tid, tidR);
			g_kernelFail = 1;
		}
	}
}

//------------------------------------------------------------------------

__device__ __forceinline__ uint allocSize(uint iter)
{
	// Use variable allocation sizes?
	if(c_env.allocScale != 0.f)
	{
		int tid = blockIdx.x * blockDim.x + threadIdx.x;
		int wid = tid / WARP_SIZE;
		// Choose allocation size based on the random number and allocScale
		// Random number in range [0,1]
		float random = g_random[wid*c_env.testIters + iter];
		uint rndAlloc = (c_env.allocScale * random) * c_alloc.payload;
		rndAlloc = max(rndAlloc, 4); // Prevent allocations smaller than a single word
		//printf("%f\t%f\t%f\t%d\n", (c_env.allocScale * random) * c_alloc.payload, c_env.allocScale, random, c_alloc.payload);
		return rndAlloc;
	}
	else
	{
		return c_alloc.payload;
	}
}

//------------------------------------------------------------------------
//-------------------------------Kernels----------------------------------
//------------------------------------------------------------------------


//------------------------------------------------------------------------
// CudaMalloc - Allocator using Cuda malloc
//------------------------------------------------------------------------

extern "C" __global__ void CudaMalloc_Basic(void)
{
	int* dyn = (int*)mallocCudaMalloc(16);
	freeCudaMalloc(dyn);
}

//------------------------------------------------------------------------

extern "C" __global__ void CudaMalloc_AllocDealloc(void)
{
	uint laneid = GPUTools::laneid();
#if defined(WRITE_READ_TEST) || defined(WRITE_ALL_TEST) || defined(SAVE_PTR_TEST)
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
#endif
#ifdef WRITE_ALL_TEST
	int wid = threadIdx.x / WARP_SIZE;
	__shared__ int* shrd[32];
#endif

	// Only the first thread allocates some memory. In real application coallescing may be used instead
	int* dyn = NULL;
	uint allocated = allocSize(g_iter);
	if(laneid == 0)
	{
		dyn = (int*)mallocCudaMalloc(allocated);
#ifdef WRITE_ALL_TEST
		shrd[wid] = dyn;
#endif
	}

#ifdef WRITE_ALL_TEST
	//__syncthreads();
	dyn = shrd[wid];
#endif

#ifdef CHECK_OUT_OF_MEMORY
	if(dyn == NULL)
	{
		if(laneid == 0)
		{
			printf("CudaMalloc: Out of memory!\n");
			g_kernelFail = 1;
		}
		return;
	}
#endif

#ifdef WRITE_ALL_TEST
	writeAllTest(dyn, allocated, tid, laneid);
	//__threadfence();
#ifdef READ_ALL_TEST
	readAllTest(dyn, allocated, tid, laneid);
#endif
#endif

	if(laneid == 0)
	{
#ifdef WRITE_READ_TEST
		readWriteTest(dyn, tid);
#endif

#ifdef SAVE_PTR_TEST
		*(volatile int*)dyn = tid;
		g_ptrArray[tid / WARP_SIZE] = dyn;
#else
		freeCudaMalloc(dyn);
#endif
	}
}

//------------------------------------------------------------------------

extern "C" __global__ void CudaMalloc_AllocCycleDealloc(void)
{
	uint laneid = GPUTools::laneid();
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int wid = tid / WARP_SIZE;

	// Only the first thread allocates some memory. In real application coallescing may be used instead
	if(laneid == 0)
	{
		for(int i = 0; i < c_env.testIters; i++)
		{
			int* dyn = (int*)mallocCudaMalloc(allocSize(i));
			g_ptrArray[wid*c_env.testIters + i] = dyn;
#ifdef CHECK_OUT_OF_MEMORY
			if(dyn == NULL)
			{
				printf("CudaMalloc: Out of memory!\n");
				g_kernelFail = 1;
				return;
			}
#endif
#if defined(WRITE_READ_TEST) || defined(SAVE_PTR_TEST)
			// Write value
			*(volatile int*)dyn = tid + i;
#endif
		}

		for(int i = 0; i < c_env.testIters; i++)
		{
			int* dyn = g_ptrArray[wid*c_env.testIters + i];
#ifdef WRITE_READ_TEST
			// Read value
			int tidR = *(volatile int*)dyn;
			if(tid + i != tidR)
			{
				printf("Allocation followed by read and write error inside kernel for tid %d.\n", tid);
				g_kernelFail = 1;
			}
#endif

#ifndef SAVE_PTR_TEST
			freeCudaMalloc(dyn);
#endif
		}
	}
}

//------------------------------------------------------------------------

extern "C" __global__ void CudaMalloc_Probability(void)
{
	uint laneid = GPUTools::laneid();
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int wid = tid / WARP_SIZE;

	// Get random number for this iteration
	float random = g_random[wid*c_env.testIters + g_iter];

	// Only the first thread allocates some memory. In real application coallescing may be used instead
	if(laneid == 0)
	{
		int* dyn = g_ptrArray[wid]; // Read pointer from memory

		if(dyn == NULL && random > c_env.pAlloc) // If not already allocated and random test succeeds
		{
			dyn = (int*)mallocCudaMalloc(allocSize(c_env.testIters-g_iter));
#ifdef CHECK_OUT_OF_MEMORY
			if(dyn == NULL)
			{
				printf("CudaMalloc: Out of memory!\n");
				g_kernelFail = 1;
				return;
			}
#endif

#ifdef WRITE_READ_TEST
			readWriteTest(dyn, tid);
#endif
		}
		else if(dyn != NULL && random > c_env.pFree) // If allocated and random test succeeds
		{
			freeCudaMalloc(dyn);
			dyn = NULL;
		}

		g_ptrArray[wid] = dyn;
	}
}

//------------------------------------------------------------------------
// AtomicMalloc - Allocator using a single atomic operation
//------------------------------------------------------------------------

extern "C" __global__ void AtomicMalloc_Basic(void)
{
	mallocAtomicMalloc(16);
}

//------------------------------------------------------------------------

extern "C" __global__ void AtomicMalloc_AllocDealloc(void)
{
	uint laneid = GPUTools::laneid();
#if defined(WRITE_READ_TEST) || defined(WRITE_ALL_TEST) || defined(SAVE_PTR_TEST)
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
#endif
#ifdef WRITE_ALL_TEST
	int wid = threadIdx.x / WARP_SIZE;
	__shared__ int* shrd[32];
#endif

	// Only the first thread allocates some memory. In real application coallescing may be used instead
	int* dyn = NULL;
	uint allocated = allocSize(g_iter);
	if(laneid == 0)
	{
		dyn = (int*)mallocAtomicMalloc(allocated);
#ifdef WRITE_ALL_TEST
		shrd[wid] = dyn;
#endif
	}

#ifdef WRITE_ALL_TEST
	//__syncthreads();
	dyn = shrd[wid];
#endif

#ifdef CHECK_OUT_OF_MEMORY
	if(dyn == NULL)
	{
		if(laneid == 0)
		{
			printf("AtomicMalloc: Out of memory!\n");
			g_kernelFail = 1;
		}
		return;
	}
#endif

#ifdef WRITE_ALL_TEST
	writeAllTest(dyn, allocated, tid, laneid);
	//__threadfence();
#ifdef READ_ALL_TEST
	readAllTest(dyn, allocated, tid, laneid);
#endif
#endif

	if(laneid == 0)
	{
#ifdef WRITE_READ_TEST
		readWriteTest(dyn, tid);
#endif

#ifdef SAVE_PTR_TEST
		*(volatile int*)dyn = tid;
		g_ptrArray[tid / WARP_SIZE] = dyn;
#endif
	}
}

//------------------------------------------------------------------------

extern "C" __global__ void AtomicMalloc_AllocCycleDealloc(void)
{
	uint laneid = GPUTools::laneid();
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int wid = tid / WARP_SIZE;

	// Only the first thread allocates some memory. In real application coallescing may be used instead
	if(laneid == 0)
	{
		for(int i = 0; i < c_env.testIters; i++)
		{
			int* dyn = (int*)mallocAtomicMalloc(allocSize(i));
			g_ptrArray[wid*c_env.testIters + i] = dyn;
#ifdef CHECK_OUT_OF_MEMORY
			if(dyn == NULL)
			{
				printf("AtomicMalloc: Out of memory!\n");
				g_kernelFail = 1;
				return;
			}
#endif
#if defined(WRITE_READ_TEST) || defined(SAVE_PTR_TEST)
			// Write value
			*(volatile int*)dyn = tid + i;
#endif
		}

		for(int i = 0; i < c_env.testIters; i++)
		{
			int* dyn = g_ptrArray[wid*c_env.testIters + i];
#ifdef WRITE_READ_TEST
			// Read value
			int tidR = *(volatile int*)dyn;
			if(tid + i != tidR)
			{
				printf("Allocation followed by read and write error inside kernel for tid %d.\n", tid);
				g_kernelFail = 1;
			}
#endif
		}
	}
}

//------------------------------------------------------------------------

extern "C" __global__ void AtomicMalloc_Probability(void)
{
	uint laneid = GPUTools::laneid();
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int wid = tid / WARP_SIZE;

	// Get random number for this iteration
	float random = g_random[wid*c_env.testIters + g_iter];

	// Only the first thread allocates some memory. In real application coallescing may be used instead
	if(laneid == 0)
	{
		int* dyn = g_ptrArray[wid]; // Read pointer from memory

		if(dyn == NULL && random > c_env.pAlloc) // If not already allocated and random test succeeds
		{
			dyn = (int*)mallocAtomicMalloc(allocSize(c_env.testIters-g_iter));
#ifdef CHECK_OUT_OF_MEMORY
			if(dyn == NULL)
			{
				printf("AtomicMalloc: Out of memory!\n");
				g_kernelFail = 1;
				return;
			}
#endif

#ifdef WRITE_READ_TEST
			readWriteTest(dyn, tid);
#endif
		}
		// No way to deallocate memory

		g_ptrArray[wid] = dyn;
	}
}

//------------------------------------------------------------------------
// AtomicWrapMalloc - Allocator using a single atomic operation in a circular buffer
//------------------------------------------------------------------------

extern "C" __global__ void AtomicWrapMalloc_Basic(void)
{
	mallocAtomicWrapMalloc(16);
}

//------------------------------------------------------------------------

extern "C" __global__ void AtomicWrapMalloc_AllocDealloc(void)
{
	uint laneid = GPUTools::laneid();
#if defined(WRITE_READ_TEST) || defined(WRITE_ALL_TEST) || defined(SAVE_PTR_TEST)
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
#endif
#ifdef WRITE_ALL_TEST
	int wid = threadIdx.x / WARP_SIZE;
	__shared__ int* shrd[32];
#endif

	// Only the first thread allocates some memory. In real application coallescing may be used instead
	int* dyn = NULL;
	uint allocated = allocSize(g_iter);
	if(laneid == 0)
	{
		dyn = (int*)mallocAtomicWrapMalloc(allocated);
#ifdef WRITE_READ_TEST
		readWriteTest(dyn, tid);
#endif
#ifdef WRITE_ALL_TEST
		shrd[wid] = dyn;
#endif
	}

#ifdef WRITE_ALL_TEST
	//__syncthreads();
	dyn = shrd[wid];
#endif

#ifdef WRITE_ALL_TEST
	writeAllTest(dyn, allocated, tid, laneid);
	//__threadfence();
#ifdef READ_ALL_TEST
	readAllTest(dyn, allocated, tid, laneid);
#endif
#endif

	if(laneid == 0)
	{
#ifdef WRITE_READ_TEST
		readWriteTest(dyn, tid);
#endif

#ifdef SAVE_PTR_TEST
		*(volatile int*)dyn = tid;
		g_ptrArray[tid / WARP_SIZE] = dyn;
#endif
	}
}

//------------------------------------------------------------------------

extern "C" __global__ void AtomicWrapMalloc_AllocCycleDealloc(void)
{
	uint laneid = GPUTools::laneid();
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int wid = tid / WARP_SIZE;

	// Only the first thread allocates some memory. In real application coallescing may be used instead
	if(laneid == 0)
	{
		for(int i = 0; i < c_env.testIters; i++)
		{
			int* dyn = (int*)mallocAtomicMalloc(allocSize(i));
			g_ptrArray[wid*c_env.testIters + i] = dyn;
#if defined(WRITE_READ_TEST) || defined(SAVE_PTR_TEST)
			// Write value
			*(volatile int*)dyn = tid + i;
#endif
		}

		for(int i = 0; i < c_env.testIters; i++)
		{
			int* dyn = g_ptrArray[wid*c_env.testIters + i];
#ifdef WRITE_READ_TEST
			// Read value
			int tidR = *(volatile int*)dyn;
			if(tid + i != tidR)
			{
				printf("Allocation followed by read and write error inside kernel for tid %d.\n", tid);
				g_kernelFail = 1;
			}
#endif
		}
	}
}

//------------------------------------------------------------------------

extern "C" __global__ void AtomicWrapMalloc_Probability(void)
{
	uint laneid = GPUTools::laneid();
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int wid = tid / WARP_SIZE;

	// Get random number for this iteration
	float random = g_random[wid*c_env.testIters + g_iter];

	// Only the first thread allocates some memory. In real application coallescing may be used instead
	if(laneid == 0)
	{
		int* dyn = g_ptrArray[wid]; // Read pointer from memory

		if(dyn == NULL && random > c_env.pAlloc) // If not already allocated and random test succeeds
		{
			dyn = (int*)mallocAtomicWrapMalloc(allocSize(c_env.testIters-g_iter));

#ifdef WRITE_READ_TEST
			readWriteTest(dyn, tid);
#endif
		}
		// No way to deallocate memory

		g_ptrArray[wid] = dyn;
	}
}

//------------------------------------------------------------------------
// CircularMalloc - Allocator using a circular linked list of chunks
//------------------------------------------------------------------------

extern "C" __global__ void CircularMalloc_Basic(void)
{
	int* dyn = (int*)mallocCircularMalloc(16);
	freeCircularMalloc(dyn);
}

//------------------------------------------------------------------------

// Allocator using a circular linked list of chunks
extern "C" __global__ void CircularMalloc_AllocDealloc(void)
{
	uint laneid = GPUTools::laneid();
#if defined(WRITE_READ_TEST) || defined(WRITE_ALL_TEST) || defined(SAVE_PTR_TEST)
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
#endif
#ifdef WRITE_ALL_TEST
	int wid = threadIdx.x / WARP_SIZE;
	__shared__ int* shrd[32];
#endif

	// Only the first thread allocates some memory. In real application coallescing may be used instead
	int* dyn = NULL;
	uint allocated = allocSize(g_iter);
	if(laneid == 0)
	{
		dyn = (int*)mallocCircularMalloc(allocated);
#ifdef WRITE_ALL_TEST
		shrd[wid] = dyn;
#endif
	}

#ifdef WRITE_ALL_TEST
	//__syncthreads();
	dyn = shrd[wid];
#endif

#ifdef CHECK_OUT_OF_MEMORY
	if(dyn == NULL)
	{
		if(laneid == 0)
		{
			printf("CircularMalloc: Out of memory!\n");
			g_kernelFail = 1;
		}
		return;
	}
#endif

#ifdef WRITE_ALL_TEST
	writeAllTest(dyn, allocated, tid, laneid);
	//__threadfence();
#ifdef READ_ALL_TEST
	readAllTest(dyn, allocated, tid, laneid);
#endif
#endif

	if(laneid == 0)
	{
#ifdef WRITE_READ_TEST
		readWriteTest(dyn, tid);
#endif

#ifdef SAVE_PTR_TEST
		*(volatile int*)dyn = tid;
		g_ptrArray[tid / WARP_SIZE] = dyn;
#else
		freeCircularMalloc(dyn);
#endif
	}
}

//------------------------------------------------------------------------

extern "C" __global__ void CircularMalloc_AllocCycleDealloc(void)
{
	uint laneid = GPUTools::laneid();
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int wid = tid / WARP_SIZE;

	// Only the first thread allocates some memory. In real application coallescing may be used instead
	if(laneid == 0)
	{
		for(int i = 0; i < c_env.testIters; i++)
		{
			int* dyn = (int*)mallocCircularMalloc(allocSize(i));
			g_ptrArray[wid*c_env.testIters + i] = dyn;
#ifdef CHECK_OUT_OF_MEMORY
			if(dyn == NULL)
			{
				printf("CircularMalloc: Out of memory!\n");
				g_kernelFail = 1;
				return;
			}
#endif
#if defined(WRITE_READ_TEST) || defined(SAVE_PTR_TEST)
			// Write value
			*(volatile int*)dyn = tid + i;
#endif
		}

		for(int i = 0; i < c_env.testIters; i++)
		{
			int* dyn = g_ptrArray[wid*c_env.testIters + i];
#ifdef WRITE_READ_TEST
			// Read value
			int tidR = *(volatile int*)dyn;
			if(tid + i != tidR)
			{
				printf("Allocation followed by read and write error inside kernel for tid %d.\n", tid);
				g_kernelFail = 1;
			}
#endif

#ifndef SAVE_PTR_TEST
			freeCircularMalloc(dyn);
#endif
		}
	}
}

//------------------------------------------------------------------------

extern "C" __global__ void CircularMalloc_Probability(void)
{
	uint laneid = GPUTools::laneid();
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int wid = tid / WARP_SIZE;

	// Get random number for this iteration
	float random = g_random[wid*c_env.testIters + g_iter];

	// Only the first thread allocates some memory. In real application coallescing may be used instead
	if(laneid == 0)
	{
		int* dyn = g_ptrArray[wid]; // Read pointer from memory

		if(dyn == NULL && random > c_env.pAlloc) // If not already allocated and random test succeeds
		{
			dyn = (int*)mallocCircularMalloc(allocSize(c_env.testIters-g_iter));
#ifdef CHECK_OUT_OF_MEMORY
			if(dyn == NULL)
			{
				printf("CircularMalloc: Out of memory!\n");
				g_kernelFail = 1;
				return;
			}
#endif

#ifdef WRITE_READ_TEST
			readWriteTest(dyn, tid);
#endif
		}
		else if(dyn != NULL && random > c_env.pFree) // If allocated and random test succeeds
		{
			freeCircularMalloc(dyn);
			dyn = NULL;
		}

		g_ptrArray[wid] = dyn;
	}
}

//------------------------------------------------------------------------
// CircularFusedMalloc - Allocator using a circular linked list of chunks with fused header and next pointer
//------------------------------------------------------------------------

extern "C" __global__ void CircularFusedMalloc_Basic(void)
{
	int* dyn = (int*)mallocCircularFusedMalloc(16);
	freeCircularFusedMalloc(dyn);
}

//------------------------------------------------------------------------

// Allocator using a circular linked list of chunks
extern "C" __global__ void CircularFusedMalloc_AllocDealloc(void)
{
	uint laneid = GPUTools::laneid();
#if defined(WRITE_READ_TEST) || defined(WRITE_ALL_TEST) || defined(SAVE_PTR_TEST)
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
#endif
#ifdef WRITE_ALL_TEST
	int wid = threadIdx.x / WARP_SIZE;
	__shared__ int* shrd[32];
#endif

	// Only the first thread allocates some memory. In real application coallescing may be used instead
	int* dyn = NULL;
	uint allocated = allocSize(g_iter);
	if(laneid == 0)
	{
		dyn = (int*)mallocCircularFusedMalloc(allocated);
#ifdef WRITE_ALL_TEST
		shrd[wid] = dyn;
#endif
	}

#ifdef WRITE_ALL_TEST
	//__syncthreads();
	dyn = shrd[wid];
#endif

#ifdef CHECK_OUT_OF_MEMORY
	if(dyn == NULL)
	{
		if(laneid == 0)
		{
			printf("CircularFusedMalloc: Out of memory!\n");
			g_kernelFail = 1;
		}
		return;
	}
#endif

#ifdef WRITE_ALL_TEST
	writeAllTest(dyn, allocated, tid, laneid);
	//__threadfence();
#ifdef READ_ALL_TEST
	readAllTest(dyn, allocated, tid, laneid);
#endif
#endif

	if(laneid == 0)
	{
#ifdef WRITE_READ_TEST
		readWriteTest(dyn, tid);
#endif

#ifdef SAVE_PTR_TEST
		*(volatile int*)dyn = tid;
		g_ptrArray[tid / WARP_SIZE] = dyn;
#else
		freeCircularFusedMalloc(dyn);
#endif
	}
}

//------------------------------------------------------------------------

extern "C" __global__ void CircularFusedMalloc_AllocCycleDealloc(void)
{
	uint laneid = GPUTools::laneid();
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int wid = tid / WARP_SIZE;

	// Only the first thread allocates some memory. In real application coallescing may be used instead
	if(laneid == 0)
	{
		for(int i = 0; i < c_env.testIters; i++)
		{
			int* dyn = (int*)mallocCircularFusedMalloc(allocSize(i));
			g_ptrArray[wid*c_env.testIters + i] = dyn;
#ifdef CHECK_OUT_OF_MEMORY
			if(dyn == NULL)
			{
				printf("CircularFusedMalloc: Out of memory!\n");
				g_kernelFail = 1;
				return;
			}
#endif
#if defined(WRITE_READ_TEST) || defined(SAVE_PTR_TEST)
			// Write value
			*(volatile int*)dyn = tid + i;
#endif
		}

		for(int i = 0; i < c_env.testIters; i++)
		{
			int* dyn = g_ptrArray[wid*c_env.testIters + i];
#ifdef WRITE_READ_TEST
			// Read value
			int tidR = *(volatile int*)dyn;
			if(tid + i != tidR)
			{
				printf("Allocation followed by read and write error inside kernel for tid %d.\n", tid);
				g_kernelFail = 1;
			}
#endif

#ifndef SAVE_PTR_TEST
			freeCircularFusedMalloc(dyn);
#endif
		}
	}
}

//------------------------------------------------------------------------

extern "C" __global__ void CircularFusedMalloc_Probability(void)
{
	uint laneid = GPUTools::laneid();
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int wid = tid / WARP_SIZE;

	// Get random number for this iteration
	float random = g_random[wid*c_env.testIters + g_iter];

	// Only the first thread allocates some memory. In real application coallescing may be used instead
	if(laneid == 0)
	{
		int* dyn = g_ptrArray[wid]; // Read pointer from memory

		if(dyn == NULL && random > c_env.pAlloc) // If not already allocated and random test succeeds
		{
			dyn = (int*)mallocCircularFusedMalloc(allocSize(c_env.testIters-g_iter));
#ifdef CHECK_OUT_OF_MEMORY
			if(dyn == NULL)
			{
				printf("CircularFusedMalloc: Out of memory!\n");
				g_kernelFail = 1;
				return;
			}
#endif

#ifdef WRITE_READ_TEST
			readWriteTest(dyn, tid);
#endif
		}
		else if(dyn != NULL && random > c_env.pFree) // If allocated and random test succeeds
		{
			freeCircularFusedMalloc(dyn);
			dyn = NULL;
		}

		g_ptrArray[wid] = dyn;
	}
}

//------------------------------------------------------------------------
// CircularMultiMalloc - Allocator using a circular linked list of chunks
//------------------------------------------------------------------------

extern "C" __global__ void CircularMultiMalloc_Basic(void)
{
	int* dyn = (int*)mallocCircularMultiMalloc(16);
	freeCircularMultiMalloc(dyn);
}

//------------------------------------------------------------------------

// Allocator using a circular linked list of chunks
extern "C" __global__ void CircularMultiMalloc_AllocDealloc(void)
{
	uint laneid = GPUTools::laneid();
#if defined(WRITE_READ_TEST) || defined(WRITE_ALL_TEST) || defined(SAVE_PTR_TEST)
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
#endif
#ifdef WRITE_ALL_TEST
	int wid = threadIdx.x / WARP_SIZE;
	__shared__ int* shrd[32];
#endif

	// Only the first thread allocates some memory. In real application coallescing may be used instead
	int* dyn = NULL;
	uint allocated = allocSize(g_iter);
	if(laneid == 0)
	{
		dyn = (int*)mallocCircularMultiMalloc(allocated);
#ifdef WRITE_ALL_TEST
		shrd[wid] = dyn;
#endif
	}

#ifdef WRITE_ALL_TEST
	//__syncthreads();
	dyn = shrd[wid];
#endif

#ifdef CHECK_OUT_OF_MEMORY
	if(dyn == NULL)
	{
		if(laneid == 0)
		{
			printf("CircularMultiMalloc: Out of memory!\n");
			g_kernelFail = 1;
		}
		return;
	}
#endif

#ifdef WRITE_ALL_TEST
	writeAllTest(dyn, allocated, tid, laneid);
	//__threadfence();
#ifdef READ_ALL_TEST
	readAllTest(dyn, allocated, tid, laneid);
#endif
#endif

	if(laneid == 0)
	{
#ifdef WRITE_READ_TEST
		readWriteTest(dyn, tid);
#endif

#ifdef SAVE_PTR_TEST
		*(volatile int*)dyn = tid;
		g_ptrArray[tid / WARP_SIZE] = dyn;
#else
		freeCircularMultiMalloc(dyn);
#endif
	}
}

//------------------------------------------------------------------------

extern "C" __global__ void CircularMultiMalloc_AllocCycleDealloc(void)
{
	uint laneid = GPUTools::laneid();
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int wid = tid / WARP_SIZE;

	// Only the first thread allocates some memory. In real application coallescing may be used instead
	if(laneid == 0)
	{
		for(int i = 0; i < c_env.testIters; i++)
		{
			int* dyn = (int*)mallocCircularMultiMalloc(allocSize(i));
			g_ptrArray[wid*c_env.testIters + i] = dyn;
#ifdef CHECK_OUT_OF_MEMORY
			if(dyn == NULL)
			{
				printf("CircularMultiMalloc: Out of memory!\n");
				g_kernelFail = 1;
				return;
			}
#endif
#if defined(WRITE_READ_TEST) || defined(SAVE_PTR_TEST)
			// Write value
			*(volatile int*)dyn = tid + i;
#endif
		}

		for(int i = 0; i < c_env.testIters; i++)
		{
			int* dyn = g_ptrArray[wid*c_env.testIters + i];
#ifdef WRITE_READ_TEST
			// Read value
			int tidR = *(volatile int*)dyn;
			if(tid + i != tidR)
			{
				printf("Allocation followed by read and write error inside kernel for tid %d.\n", tid);
				g_kernelFail = 1;
			}
#endif

#ifndef SAVE_PTR_TEST
			freeCircularMultiMalloc(dyn);
#endif
		}
	}
}

//------------------------------------------------------------------------

extern "C" __global__ void CircularMultiMalloc_Probability(void)
{
	uint laneid = GPUTools::laneid();
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int wid = tid / WARP_SIZE;

	// Get random number for this iteration
	float random = g_random[wid*c_env.testIters + g_iter];

	// Only the first thread allocates some memory. In real application coallescing may be used instead
	if(laneid == 0)
	{
		int* dyn = g_ptrArray[wid]; // Read pointer from memory

		if(dyn == NULL && random > c_env.pAlloc) // If not already allocated and random test succeeds
		{
			dyn = (int*)mallocCircularMultiMalloc(allocSize(c_env.testIters-g_iter));
#ifdef CHECK_OUT_OF_MEMORY
			if(dyn == NULL)
			{
				printf("CircularMultiMalloc: Out of memory!\n");
				g_kernelFail = 1;
				return;
			}
#endif

#ifdef WRITE_READ_TEST
			readWriteTest(dyn, tid);
#endif
		}
		else if(dyn != NULL && random > c_env.pFree) // If allocated and random test succeeds
		{
			freeCircularMultiMalloc(dyn);
			dyn = NULL;
		}

		g_ptrArray[wid] = dyn;
	}
}

//------------------------------------------------------------------------
// CircularFusedMultiMalloc - Allocator using a circular linked list of chunks with fused header and next pointer
//------------------------------------------------------------------------

extern "C" __global__ void CircularFusedMultiMalloc_Basic(void)
{
	int* dyn = (int*)mallocCircularFusedMultiMalloc(16);
	freeCircularFusedMultiMalloc(dyn);
}

//------------------------------------------------------------------------

// Allocator using a circular linked list of chunks
extern "C" __global__ void CircularFusedMultiMalloc_AllocDealloc(void)
{
	uint laneid = GPUTools::laneid();
#if defined(WRITE_READ_TEST) || defined(WRITE_ALL_TEST) || defined(SAVE_PTR_TEST)
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
#endif
#ifdef WRITE_ALL_TEST
	int wid = threadIdx.x / WARP_SIZE;
	__shared__ int* shrd[32];
#endif

	// Only the first thread allocates some memory. In real application coallescing may be used instead
	int* dyn = NULL;
	uint allocated = allocSize(g_iter);
	if(laneid == 0)
	{
		dyn = (int*)mallocCircularFusedMultiMalloc(allocated);
#ifdef WRITE_ALL_TEST
		shrd[wid] = dyn;
#endif
	}

#ifdef WRITE_ALL_TEST
	//__syncthreads();
	dyn = shrd[wid];
#endif

#ifdef CHECK_OUT_OF_MEMORY
	if(dyn == NULL)
	{
		if(laneid == 0)
		{
			printf("CircularFusedMultiMalloc: Out of memory!\n");
			g_kernelFail = 1;
		}
		return;
	}
#endif

#ifdef WRITE_ALL_TEST
	writeAllTest(dyn, allocated, tid, laneid);
	//__threadfence();
#ifdef READ_ALL_TEST
	readAllTest(dyn, allocated, tid, laneid);
#endif
#endif

	if(laneid == 0)
	{
#ifdef WRITE_READ_TEST
		readWriteTest(dyn, tid);
#endif

#ifdef SAVE_PTR_TEST
		*(volatile int*)dyn = tid;
		g_ptrArray[tid / WARP_SIZE] = dyn;
#else
		freeCircularFusedMultiMalloc(dyn);
#endif
	}
}

//------------------------------------------------------------------------

extern "C" __global__ void CircularFusedMultiMalloc_AllocCycleDealloc(void)
{
	uint laneid = GPUTools::laneid();
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int wid = tid / WARP_SIZE;

	// Only the first thread allocates some memory. In real application coallescing may be used instead
	if(laneid == 0)
	{
		for(int i = 0; i < c_env.testIters; i++)
		{
			int* dyn = (int*)mallocCircularFusedMultiMalloc(allocSize(i));
			g_ptrArray[wid*c_env.testIters + i] = dyn;
#ifdef CHECK_OUT_OF_MEMORY
			if(dyn == NULL)
			{
				printf("CircularFusedMultiMalloc: Out of memory!\n");
				g_kernelFail = 1;
				return;
			}
#endif
#if defined(WRITE_READ_TEST) || defined(SAVE_PTR_TEST)
			// Write value
			*(volatile int*)dyn = tid + i;
#endif
		}

		for(int i = 0; i < c_env.testIters; i++)
		{
			int* dyn = g_ptrArray[wid*c_env.testIters + i];
#ifdef WRITE_READ_TEST
			// Read value
			int tidR = *(volatile int*)dyn;
			if(tid + i != tidR)
			{
				printf("Allocation followed by read and write error inside kernel for tid %d.\n", tid);
				g_kernelFail = 1;
			}
#endif

#ifndef SAVE_PTR_TEST
			freeCircularFusedMultiMalloc(dyn);
#endif
		}
	}
}

//------------------------------------------------------------------------

extern "C" __global__ void CircularFusedMultiMalloc_Probability(void)
{
	uint laneid = GPUTools::laneid();
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int wid = tid / WARP_SIZE;

	// Get random number for this iteration
	float random = g_random[wid*c_env.testIters + g_iter];

	// Only the first thread allocates some memory. In real application coallescing may be used instead
	if(laneid == 0)
	{
		int* dyn = g_ptrArray[wid]; // Read pointer from memory

		if(dyn == NULL && random > c_env.pAlloc) // If not already allocated and random test succeeds
		{
			dyn = (int*)mallocCircularFusedMultiMalloc(allocSize(c_env.testIters-g_iter));
#ifdef CHECK_OUT_OF_MEMORY
			if(dyn == NULL)
			{
				printf("CircularFusedMultiMalloc: Out of memory!\n");
				g_kernelFail = 1;
				return;
			}
#endif

#ifdef WRITE_READ_TEST
			readWriteTest(dyn, tid);
#endif
		}
		else if(dyn != NULL && random > c_env.pFree) // If allocated and random test succeeds
		{
			freeCircularFusedMultiMalloc(dyn);
			dyn = NULL;
		}

		g_ptrArray[wid] = dyn;
	}
}

//------------------------------------------------------------------------
// ScatterAlloc - Allocator using ScatterAlloc
//------------------------------------------------------------------------

extern "C" __global__ void ScatterAlloc_Basic(void)
{
	int* dyn = (int*)mallocScatterAlloc(16);
	freeScatterAlloc(dyn);
}

//------------------------------------------------------------------------

extern "C" __global__ void ScatterAlloc_AllocDealloc(void)
{
	uint laneid = GPUTools::laneid();
#if defined(WRITE_READ_TEST) || defined(WRITE_ALL_TEST) || defined(SAVE_PTR_TEST)
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
#endif
#ifdef WRITE_ALL_TEST
	int wid = threadIdx.x / WARP_SIZE;
	__shared__ int* shrd[32];
#endif

	// Only the first thread allocates some memory. In real application coallescing may be used instead
	int* dyn = NULL;
	uint allocated = allocSize(g_iter);
	if(laneid == 0)
	{
		dyn = (int*)mallocScatterAlloc(allocated);
#ifdef WRITE_ALL_TEST
		shrd[wid] = dyn;
#endif
	}

#ifdef WRITE_ALL_TEST
	//__syncthreads();
	dyn = shrd[wid];
#endif

#ifdef CHECK_OUT_OF_MEMORY
	if(dyn == NULL)
	{
		if(laneid == 0)
		{
			printf("ScatterAlloc: Out of memory!\n");
			g_kernelFail = 1;
		}
		return;
	}
#endif

#ifdef WRITE_ALL_TEST
	writeAllTest(dyn, allocated, tid, laneid);
	//__threadfence();
#ifdef READ_ALL_TEST
	readAllTest(dyn, allocated, tid, laneid);
#endif
#endif

	if(laneid == 0)
	{
#ifdef WRITE_READ_TEST
		readWriteTest(dyn, tid);
#endif

#ifdef SAVE_PTR_TEST
		*(volatile int*)dyn = tid;
		g_ptrArray[tid / WARP_SIZE] = dyn;
#else
		freeScatterAlloc(dyn);
#endif
	}
}

//------------------------------------------------------------------------

extern "C" __global__ void ScatterAlloc_AllocCycleDealloc(void)
{
	uint laneid = GPUTools::laneid();
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int wid = tid / WARP_SIZE;

	// Only the first thread allocates some memory. In real application coallescing may be used instead
	if(laneid == 0)
	{
		for(int i = 0; i < c_env.testIters; i++)
		{
			int* dyn = (int*)mallocScatterAlloc(allocSize(i));
			g_ptrArray[wid*c_env.testIters + i] = dyn;
#ifdef CHECK_OUT_OF_MEMORY
			if(dyn == NULL)
			{
				printf("ScatterAlloc: Out of memory!\n");
				g_kernelFail = 1;
				return;
			}
#endif
#if defined(WRITE_READ_TEST) || defined(SAVE_PTR_TEST)
			// Write value
			*(volatile int*)dyn = tid + i;
#endif
		}

		for(int i = 0; i < c_env.testIters; i++)
		{
			int* dyn = g_ptrArray[wid*c_env.testIters + i];
#ifdef WRITE_READ_TEST
			// Read value
			int tidR = *(volatile int*)dyn;
			if(tid + i != tidR)
			{
				printf("Allocation followed by read and write error inside kernel for tid %d.\n", tid);
				g_kernelFail = 1;
			}
#endif

#ifndef SAVE_PTR_TEST
			freeScatterAlloc(dyn);
#endif
		}
	}
}

//------------------------------------------------------------------------

extern "C" __global__ void ScatterAlloc_Probability(void)
{
	uint laneid = GPUTools::laneid();
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int wid = tid / WARP_SIZE;

	// Get random number for this iteration
	float random = g_random[wid*c_env.testIters + g_iter];

	// Only the first thread allocates some memory. In real application coallescing may be used instead
	if(laneid == 0)
	{
		int* dyn = g_ptrArray[wid]; // Read pointer from memory

		if(dyn == NULL && random > c_env.pAlloc) // If not already allocated and random test succeeds
		{
			dyn = (int*)mallocScatterAlloc(allocSize(c_env.testIters-g_iter));
#ifdef CHECK_OUT_OF_MEMORY
			if(dyn == NULL)
			{
				printf("ScatterAlloc: Out of memory!\n");
				g_kernelFail = 1;
				return;
			}
#endif

#ifdef WRITE_READ_TEST
			readWriteTest(dyn, tid);
#endif
		}
		else if(dyn != NULL && random > c_env.pFree) // If allocated and random test succeeds
		{
			freeScatterAlloc(dyn);
			dyn = NULL;
		}

		g_ptrArray[wid] = dyn;
	}
}

//------------------------------------------------------------------------
// FDGMalloc - Allocator using FDGMalloc
//------------------------------------------------------------------------

extern "C" __global__ void FDGMalloc_Basic(void)
{
	FDG::Warp* warp = FDG::Warp::start();
	mallocFDGMalloc(warp, 16);
	freeFDGMalloc(warp);
}

//------------------------------------------------------------------------

extern "C" __global__ void FDGMalloc_AllocDealloc(void)
{
	uint laneid = GPUTools::laneid();
#if defined(WRITE_READ_TEST) || defined(WRITE_ALL_TEST) || defined(SAVE_PTR_TEST)
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
#endif
#ifdef WRITE_ALL_TEST
	int wid = threadIdx.x / WARP_SIZE;
	__shared__ int* shrd[32];
#endif

	// Only the first thread allocates some memory. In real application coallescing may be used instead
	int* dyn = NULL;
	uint allocated = allocSize(g_iter);
	FDG::Warp* warp;
	if(laneid == 0)
	{
		// Get WarpHeader
		warp = FDG::Warp::start();
#ifdef CHECK_OUT_OF_MEMORY
		if(warp == NULL)
		{
			printf("FDGMalloc: Warp out of memory!\n");
			g_kernelFail = 1;
		}
#endif
		if(warp != NULL)
			dyn = (int*)mallocFDGMalloc(warp, allocated);
#ifdef WRITE_ALL_TEST
		shrd[wid] = dyn;
#endif
	}

#ifdef WRITE_ALL_TEST
	//__syncthreads();
	dyn = shrd[wid];
#endif

#ifdef CHECK_OUT_OF_MEMORY
	if(dyn == NULL)
	{
		if(laneid == 0)
		{
			printf("FDGMalloc: Out of memory!\n");
			g_kernelFail = 1;
		}
		return;
	}
#endif

#ifdef WRITE_ALL_TEST
	writeAllTest(dyn, allocated, tid, laneid);
	//__threadfence();
#ifdef READ_ALL_TEST
	readAllTest(dyn, allocated, tid, laneid);
#endif
#endif

	if(laneid == 0)
	{
#ifdef WRITE_READ_TEST
		readWriteTest(dyn, tid);
#endif

#ifdef SAVE_PTR_TEST
		*(volatile int*)dyn = tid;
		g_ptrArray[tid / WARP_SIZE] = dyn; // Memory leak, (int*)warp should be passed and freed later;
#else
		freeFDGMalloc(warp);
#endif
	}
}

//------------------------------------------------------------------------

extern "C" __global__ void FDGMalloc_AllocCycleDealloc(void)
{
	uint laneid = GPUTools::laneid();
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int wid = tid / WARP_SIZE;

	// Only the first thread allocates some memory. In real application coallescing may be used instead
	if(laneid == 0)
	{
		// Get WarpHeader
		FDG::Warp* warp = FDG::Warp::start();
#ifdef CHECK_OUT_OF_MEMORY
		if(warp == NULL)
		{
			printf("FDGMalloc: Warp out of memory!\n");
			g_kernelFail = 1;
			return;
		}
#endif
		for(int i = 0; i < c_env.testIters; i++)
		{
			int* dyn = (int*)mallocFDGMalloc(warp, allocSize(i));
			g_ptrArray[wid*c_env.testIters + i] = dyn;
#ifdef CHECK_OUT_OF_MEMORY
			if(dyn == NULL)
			{
				printf("FDGMalloc: Out of memory!\n");
				g_kernelFail = 1;
				return;
			}
#endif
#if defined(WRITE_READ_TEST) || defined(SAVE_PTR_TEST)
			// Write value
			*(volatile int*)dyn = tid + i;
#endif
		}

		for(int i = 0; i < c_env.testIters; i++)
		{
			int* dyn = g_ptrArray[wid*c_env.testIters + i];
#ifdef WRITE_READ_TEST
			// Read value
			int tidR = *(volatile int*)dyn;
			if(tid + i != tidR)
			{
				printf("Allocation followed by read and write error inside kernel for tid %d.\n", tid);
				g_kernelFail = 1;
			}
#endif
		}
#ifndef SAVE_PTR_TEST
		freeFDGMalloc(warp);
#endif
	}
}

//------------------------------------------------------------------------

extern "C" __global__ void FDGMalloc_Probability(void)
{
	uint laneid = GPUTools::laneid();
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int wid = tid / WARP_SIZE;

	// Get random number for this iteration
	float random = g_random[wid*c_env.testIters + g_iter];

	// Only the first thread allocates some memory. In real application coallescing may be used instead
	if(laneid == 0)
	{
		FDG::Warp* warp = (FDG::Warp*)g_ptrArray[wid]; // Read pointer from memory

		if(warp == NULL && random > c_env.pAlloc) // If not already allocated and random test succeeds
		{
			warp = FDG::Warp::start();
#ifdef CHECK_OUT_OF_MEMORY
			if(warp == NULL)
			{
				printf("FDGMalloc: Warp out of memory!\n");
				g_kernelFail = 1;
				return;
			}
#endif
			int* dyn = (int*)mallocFDGMalloc(warp, allocSize(c_env.testIters-g_iter));
#ifdef CHECK_OUT_OF_MEMORY
			if(dyn == NULL)
			{
				printf("FDGMalloc: Out of memory!\n");
				g_kernelFail = 1;
				return;
			}
#endif

#ifdef WRITE_READ_TEST
			readWriteTest(dyn, tid);
#endif
		}
		else if(warp != NULL && random > c_env.pFree) // If allocated and random test succeeds
		{
			freeFDGMalloc(warp);
			warp = NULL;
		}

		g_ptrArray[wid] = (int*)warp;
	}
}

//------------------------------------------------------------------------
// Halloc - Allocator using Halloc
//------------------------------------------------------------------------

extern "C" __global__ void Halloc_Basic(void)
{
	int* dyn = (int*)mallocHalloc(16);
	freeHalloc(dyn);
}

//------------------------------------------------------------------------

extern "C" __global__ void Halloc_AllocDealloc(void)
{
	uint laneid = GPUTools::laneid();
#if defined(WRITE_READ_TEST) || defined(WRITE_ALL_TEST) || defined(SAVE_PTR_TEST)
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
#endif
#ifdef WRITE_ALL_TEST
	int wid = threadIdx.x / WARP_SIZE;
	__shared__ int* shrd[32];
#endif

	// Only the first thread allocates some memory. In real application coallescing may be used instead
	int* dyn = NULL;
	uint allocated = allocSize(g_iter);
	if(laneid == 0)
	{
		dyn = (int*)mallocHalloc(allocated);
#ifdef WRITE_ALL_TEST
		shrd[wid] = dyn;
#endif
	}

#ifdef WRITE_ALL_TEST
	//__syncthreads();
	dyn = shrd[wid];
#endif

#ifdef CHECK_OUT_OF_MEMORY
	if(dyn == NULL)
	{
		if(laneid == 0)
		{
			printf("Halloc: Out of memory!\n");
			g_kernelFail = 1;
		}
		return;
	}
#endif

#ifdef WRITE_ALL_TEST
	writeAllTest(dyn, allocated, tid, laneid);
	//__threadfence();
#ifdef READ_ALL_TEST
	readAllTest(dyn, allocated, tid, laneid);
#endif
#endif

	if(laneid == 0)
	{
#ifdef WRITE_READ_TEST
		readWriteTest(dyn, tid);
#endif

#ifdef SAVE_PTR_TEST
		*(volatile int*)dyn = tid;
		g_ptrArray[tid / WARP_SIZE] = dyn;
#else
		freeHalloc(dyn);
#endif
	}
}

//------------------------------------------------------------------------

extern "C" __global__ void Halloc_AllocCycleDealloc(void)
{
	uint laneid = GPUTools::laneid();
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int wid = tid / WARP_SIZE;

	// Only the first thread allocates some memory. In real application coallescing may be used instead
	if(laneid == 0)
	{
		for(int i = 0; i < c_env.testIters; i++)
		{
			int* dyn = (int*)mallocHalloc(allocSize(i));
			g_ptrArray[wid*c_env.testIters + i] = dyn;
#ifdef CHECK_OUT_OF_MEMORY
			if(dyn == NULL)
			{
				printf("Halloc: Out of memory!\n");
				g_kernelFail = 1;
				return;
			}
#endif
#if defined(WRITE_READ_TEST) || defined(SAVE_PTR_TEST)
			// Write value
			*(volatile int*)dyn = tid + i;
#endif
		}

		for(int i = 0; i < c_env.testIters; i++)
		{
			int* dyn = g_ptrArray[wid*c_env.testIters + i];
#ifdef WRITE_READ_TEST
			// Read value
			int tidR = *(volatile int*)dyn;
			if(tid + i != tidR)
			{
				printf("Allocation followed by read and write error inside kernel for tid %d.\n", tid);
				g_kernelFail = 1;
			}
#endif

#ifndef SAVE_PTR_TEST
			freeHalloc(dyn);
#endif
		}
	}
}

//------------------------------------------------------------------------

extern "C" __global__ void Halloc_Probability(void)
{
	uint laneid = GPUTools::laneid();
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int wid = tid / WARP_SIZE;

	// Get random number for this iteration
	float random = g_random[wid*c_env.testIters + g_iter];

	// Only the first thread allocates some memory. In real application coallescing may be used instead
	if(laneid == 0)
	{
		int* dyn = g_ptrArray[wid]; // Read pointer from memory

		if(dyn == NULL && random > c_env.pAlloc) // If not already allocated and random test succeeds
		{
			dyn = (int*)mallocHalloc(allocSize(c_env.testIters-g_iter));
#ifdef CHECK_OUT_OF_MEMORY
			if(dyn == NULL)
			{
				printf("Halloc: Out of memory!\n");
				g_kernelFail = 1;
				return;
			}
#endif

#ifdef WRITE_READ_TEST
			readWriteTest(dyn, tid);
#endif
		}
		else if(dyn != NULL && random > c_env.pFree) // If allocated and random test succeeds
		{
			freeHalloc(dyn);
			dyn = NULL;
		}

		g_ptrArray[wid] = dyn;
	}
}

//------------------------------------------------------------------------
// Test functions
//------------------------------------------------------------------------

#ifdef SAVE_PTR_TEST
extern "C" __global__ void SavePtrTest(void)
{
	int wid = blockIdx.x * blockDim.x + threadIdx.x;
	
	for(int i = 0; i < c_env.testIters; i++)
	{
		int* ptr = g_ptrArray[wid*c_env.testIters + i];
		if(ptr == NULL) // Unallocated chunk
		{
			printf("Unallocated chunk for wid %d.\n", wid);
			return;
		}

		if(*ptr != wid*WARP_SIZE + i) // Unexpected value
		{
			printf("Allocation followed by read and write error across kernels for wid %d.\n", wid);
		}
	}
}
#endif

extern "C" __global__ void CudaMallocInit(void)
{
	int wid = blockIdx.x * blockDim.x + threadIdx.x;
	
	if(wid == 0)
	{
		void* dyn = malloc(1);
	}
}

//------------------------------------------------------------------------