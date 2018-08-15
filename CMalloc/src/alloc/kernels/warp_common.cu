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
    Warp wide utility functions.
*/

#pragma once
#include "warp_common.cuh"


//------------------------------------------------------------------------

// Loads data cached at all levels - default
template<>
__device__ __forceinline__ int loadCA<int>(const int* address)
{
	int val;
	asm("{\n\t"
		"ld.ca.s32\t%0, [%1];\n\t"
		"}"
#ifndef _M_X64 // x86
		: "=r"(val) : "r"(address));
#else // x64
		: "=r"(val) : "l"(address));
#endif

	return val;
}

template<>
__device__ __forceinline__ unsigned int loadCA<unsigned int>(const unsigned int* address)
{
	unsigned int val;
	asm("{\n\t"
		"ld.ca.u32\t%0, [%1];\n\t"
		"}"
#ifndef _M_X64 // x86
		: "=r"(val) : "r"(address));
#else // x64
		: "=r"(val) : "l"(address));
#endif

	return val;
}

template<>
__device__ __forceinline__ uint2 loadCA<uint2>(const uint2* address)
{
	uint2 val;
	asm("{\n\t"
		"ld.ca.v2.u32\t{%0, %1}, [%2];\n\t"
		"}"
#ifndef _M_X64 // x86
		: "=r"(val.x), "=r"(val.y) : "r"(address));
#else // x64
		: "=r"(val.x), "=r"(val.y) : "l"(address));
#endif

	return val;
}

template<>
__device__ __forceinline__ uint4 loadCA<uint4>(const uint4* address)
{
	uint4 val;
	asm("{\n\t"
		"ld.ca.v4.u32\t{%0, %1, %2, %3}, [%4];\n\t"
		"}"
#ifndef _M_X64 // x86
		: "=r"(val.x), "=r"(val.y), "=r"(val.z), "=r"(val.w) : "r"(address));
#else // x64
		: "=r"(val.x), "=r"(val.y), "=r"(val.z), "=r"(val.w) : "l"(address));
#endif

	return val;
}

template<>
__device__ __forceinline__ float loadCA<float>(const float* address)
{
	float val;
	asm("{\n\t"
		"ld.ca.f32\t%0, [%1];\n\t"
		"}"
#ifndef _M_X64 // x64
		: "=f"(val) : "r"(address));
#else // x64
		: "=f"(val) : "l"(address));
#endif

	return val;
}

//------------------------------------------------------------------------

// Loads data cached at L2 and above
template<>
__device__ __forceinline__ int loadCG<int>(const int* address)
{
	int val;
	asm("{\n\t"
		"ld.cg.s32\t%0, [%1];\n\t"
		"}"
#ifndef _M_X64 // x86
		: "=r"(val) : "r"(address));
#else // x64
		: "=r"(val) : "l"(address));
#endif

	return val;
}

template<>
__device__ __forceinline__ unsigned int loadCG<unsigned int>(const unsigned int* address)
{
	unsigned int val;
	asm("{\n\t"
		"ld.cg.u32\t%0, [%1];\n\t"
		"}"
#ifndef _M_X64 // x86
		: "=r"(val) : "r"(address));
#else // x64
		: "=r"(val) : "l"(address));
#endif

	return val;
}

template<>
__device__ __forceinline__ uint2 loadCG<uint2>(const uint2* address)
{
	uint2 val;
	asm("{\n\t"
		"ld.cg.v2.u32\t{%0, %1}, [%2];\n\t"
		"}"
#ifndef _M_X64 // x86
		: "=r"(val.x), "=r"(val.y) : "r"(address));
#else // x64
		: "=r"(val.x), "=r"(val.y) : "l"(address));
#endif

	return val;
}

template<>
__device__ __forceinline__ uint4 loadCG<uint4>(const uint4* address)
{
	uint4 val;
	asm("{\n\t"
		"ld.cg.v4.u32\t{%0, %1, %2, %3}, [%4];\n\t"
		"}"
#ifndef _M_X64 // x86
		: "=r"(val.x), "=r"(val.y), "=r"(val.z), "=r"(val.w) : "r"(address));
#else // x64
		: "=r"(val.x), "=r"(val.y), "=r"(val.z), "=r"(val.w) : "l"(address));
#endif

	return val;
}

//------------------------------------------------------------------------

// Loads data cached at all levels, with evict-first policy
template<>
__device__ __forceinline__ int loadCS<int>(const int* address)
{
	int val;
	asm("{\n\t"
		"ld.cs.s32\t%0, [%1];\n\t"
		"}"
#ifndef _M_X64 // x86
		: "=r"(val) : "r"(address));
#else // x64
		: "=r"(val) : "l"(address));
#endif

	return val;
}

template<>
__device__ __forceinline__ unsigned int loadCS<unsigned int>(const unsigned int* address)
{
	unsigned int val;
	asm("{\n\t"
		"ld.cs.u32\t%0, [%1];\n\t"
		"}"
#ifndef _M_X64 // x86
		: "=r"(val) : "r"(address));
#else // x64
		: "=r"(val) : "l"(address));
#endif

	return val;
}

template<>
__device__ __forceinline__ uint2 loadCS<uint2>(const uint2* address)
{
	uint2 val;
	asm("{\n\t"
		"ld.cs.v2.u32\t{%0, %1}, [%2];\n\t"
		"}"
#ifndef _M_X64 // x86
		: "=r"(val.x), "=r"(val.y) : "r"(address));
#else // x64
		: "=r"(val.x), "=r"(val.y) : "l"(address));
#endif

	return val;
}

template<>
__device__ __forceinline__ uint4 loadCS<uint4>(const uint4* address)
{
	uint4 val;
	asm("{\n\t"
		"ld.cs.v4.u32\t{%0, %1, %2, %3}, [%4];\n\t"
		"}"
#ifndef _M_X64 // x86
		: "=r"(val.x), "=r"(val.y), "=r"(val.z), "=r"(val.w) : "r"(address));
#else // x64
		: "=r"(val.x), "=r"(val.y), "=r"(val.z), "=r"(val.w) : "l"(address));
#endif

	return val;
}

template<>
__device__ __forceinline__ float loadCS<float>(const float* address)
{
	float val;
	asm("{\n\t"
		"ld.cs.f32\t%0, [%1];\n\t"
		"}"
#ifndef _M_X64 // x86
		: "=f"(val) : "r"(address));
#else // x64
		: "=f"(val) : "l"(address));
#endif

	return val;
}

//------------------------------------------------------------------------

// Loads data similar to loadCS, on local addresses discards L1 cache following the load
template<>
__device__ __forceinline__ int loadLU<int>(const int* address)
{
	int val;
	asm("{\n\t"
		"ld.lu.s32\t%0, [%1];\n\t"
		"}"
#ifndef _M_X64 // x86
		: "=r"(val) : "r"(address));
#else // x64
		: "=r"(val) : "l"(address));
#endif

	return val;
}

template<>
__device__ __forceinline__ unsigned int loadLU<unsigned int>(const unsigned int* address)
{
	unsigned int val;
	asm("{\n\t"
		"ld.lu.u32\t%0, [%1];\n\t"
		"}"
#ifndef _M_X64 // x86
		: "=r"(val) : "r"(address));
#else // x64
		: "=r"(val) : "l"(address));
#endif

	return val;
}

template<>
__device__ __forceinline__ uint2 loadLU<uint2>(const uint2* address)
{
	uint2 val;
	asm("{\n\t"
		"ld.lu.v2.u32\t{%0, %1}, [%2];\n\t"
		"}"
#ifndef _M_X64 // x86
		: "=r"(val.x), "=r"(val.y) : "r"(address));
#else // x64
		: "=r"(val.x), "=r"(val.y) : "l"(address));
#endif

	return val;
}

template<>
__device__ __forceinline__ uint4 loadLU<uint4>(const uint4* address)
{
	uint4 val;
	asm("{\n\t"
		"ld.lu.v4.u32\t{%0, %1, %2, %3}, [%4];\n\t"
		"}"
#ifndef _M_X64 // x86
		: "=r"(val.x), "=r"(val.y), "=r"(val.z), "=r"(val.w) : "r"(address));
#else // x64
		: "=r"(val.x), "=r"(val.y), "=r"(val.z), "=r"(val.w) : "l"(address));
#endif

	return val;
}

template<>
__device__ __forceinline__ float loadLU<float>(const float* address)
{
	float val;
	asm("{\n\t"
		"ld.lu.f32\t%0, [%1];\n\t"
		"}"
#ifndef _M_X64 // x86
		: "=f"(val) : "r"(address));
#else // x64
		: "=f"(val) : "l"(address));
#endif

	return val;
}

//------------------------------------------------------------------------

// Loads data volatilely, again from memory
template<>
__device__ __forceinline__ int loadCV<int>(const int* address)
{
	int val;
	asm("{\n\t"
		"ld.cv.s32\t%0, [%1];\n\t"
		"}"
#ifndef _M_X64 // x86
		: "=r"(val) : "r"(address));
#else // x64
		: "=r"(val) : "l"(address));
#endif

	return val;
}

template<>
__device__ __forceinline__ unsigned int loadCV<unsigned int>(const unsigned int* address)
{
	unsigned int val;
	asm("{\n\t"
		"ld.cv.u32\t%0, [%1];\n\t"
		"}"
#ifndef _M_X64 // x86
		: "=r"(val) : "r"(address));
#else // x64
		: "=r"(val) : "l"(address));
#endif

	return val;
}

template<>
__device__ __forceinline__ uint2 loadCV<uint2>(const uint2* address)
{
	uint2 val;
	asm("{\n\t"
		"ld.cv.v2.u32\t{%0, %1}, [%2];\n\t"
		"}"
#ifndef _M_X64 // x86
		: "=r"(val.x), "=r"(val.y) : "r"(address));
#else // x64
		: "=r"(val.x), "=r"(val.y) : "l"(address));
#endif

	return val;
}

template<>
__device__ __forceinline__ uint4 loadCV<uint4>(const uint4* address)
{
	uint4 val;
	asm("{\n\t"
		"ld.cv.v4.u32\t{%0, %1, %2, %3}, [%4];\n\t"
		"}"
#ifndef _M_X64 // x86
		: "=r"(val.x), "=r"(val.y), "=r"(val.z), "=r"(val.w) : "r"(address));
#else // x64
		: "=r"(val.x), "=r"(val.y), "=r"(val.z), "=r"(val.w) : "l"(address));
#endif

	return val;
}

template<>
__device__ __forceinline__ float loadCV<float>(const float* address)
{
	float val;
	asm("{\n\t"
		"ld.cv.f32\t%0, [%1];\n\t"
		"}"
#ifndef _M_X64 // x86
		: "=f"(val) : "r"(address));
#else // x64
		: "=f"(val) : "l"(address));
#endif

	return val;
}

//------------------------------------------------------------------------

// Saves data cached at all levels - default
template<>
__device__ __forceinline__ void saveWB<int>(const int* address, int val)
{
	asm("{\n\t"
		"st.wb.s32\t[%0], %1;\n\t"
		"}"
#ifndef _M_X64 // x86
		:: "r"(address), "r"(val));
#else // x64
		:: "l"(address), "r"(val));
#endif
}

template<>
__device__ __forceinline__ void saveWB<unsigned int>(const unsigned int* address, unsigned int val)
{
	asm("{\n\t"
		"st.wb.u32\t[%0], %1;\n\t"
		"}"
#ifndef _M_X64 // x86
		:: "r"(address), "r"(val));
#else // x64
		:: "l"(address), "r"(val));
#endif
}

template<>
__device__ __forceinline__ void saveWB<uint2>(const uint2* address, uint2 val)
{
	asm("{\n\t"
		"st.wb.v2.u32\t[%0], {%1, %2};\n\t"
		"}"
#ifndef _M_X64 // x86
		:: "r"(address), "r"(val.x), "r"(val.y));
#else // x64
		:: "l"(address), "r"(val.x), "r"(val.y));
#endif
}

template<>
__device__ __forceinline__ void saveWB<uint4>(const uint4* address, uint4 val)
{
	asm("{\n\t"
		"st.wb.v4.u32\t[%0], {%1, %2, %3, %4};\n\t"
		"}"
#ifndef _M_X64 // x86
		:: "r"(address), "r"(val.x), "r"(val.y), "r"(val.z), "r"(val.w));
#else // x64
		:: "l"(address), "r"(val.x), "r"(val.y), "r"(val.z), "r"(val.w));
#endif
}

template<>
__device__ __forceinline__ void saveWB<float>(const float* address, float val)
{
	asm("{\n\t"
		"st.wb.f32\t[%0], %1;\n\t"
		"}"
#ifndef _M_X64 // x86
		:: "r"(address), "f"(val));
#else // x64
		:: "l"(address), "f"(val));
#endif
}

//------------------------------------------------------------------------

// Saves data cached at L2 and above
template<>
__device__ __forceinline__ void saveCG<int>(const int* address, int val)
{
	asm("{\n\t"
		"st.cg.s32\t[%0], %1;\n\t"
		"}"
#ifndef _M_X64 // x86
		:: "r"(address), "r"(val));
#else // x64
		:: "l"(address), "r"(val));
#endif
}

template<>
__device__ __forceinline__ void saveCG<unsigned int>(const unsigned int* address, unsigned int val)
{
	asm("{\n\t"
		"st.cg.u32\t[%0], %1;\n\t"
		"}"
#ifndef _M_X64 // x86
		:: "r"(address), "r"(val));
#else // x64
		:: "l"(address), "r"(val));
#endif
}

template<>
__device__ __forceinline__ void saveCG<uint2>(const uint2* address, uint2 val)
{
	asm("{\n\t"
		"st.cg.v2.u32\t[%0], {%1, %2};\n\t"
		"}"
#ifndef _M_X64 // x86
		:: "r"(address), "r"(val.x), "r"(val.y));
#else // x64
		:: "l"(address), "r"(val.x), "r"(val.y));
#endif
}

template<>
__device__ __forceinline__ void saveCG<uint4>(const uint4* address, uint4 val)
{
	asm("{\n\t"
		"st.cg.v4.u32\t[%0], {%1, %2, %3, %4};\n\t"
		"}"
#ifndef _M_X64 // x86
		:: "r"(address), "r"(val.x), "r"(val.y), "r"(val.z), "r"(val.w));
#else // x64
		:: "l"(address), "r"(val.x), "r"(val.y), "r"(val.z), "r"(val.w));
#endif
}

template<>
__device__ __forceinline__ void saveCG<float>(const float* address, float val)
{
	asm("{\n\t"
		"ld.st.f32\t[%0], %1;\n\t"
		"}"
#ifndef _M_X64 // x86
		:: "r"(address), "f"(val));
#else // x64
		:: "l"(address), "f"(val));
#endif
}

//------------------------------------------------------------------------

// Saves data cached at all levels, with evict-first policy
template<>
__device__ __forceinline__ void saveCS<int>(const int* address, int val)
{
	asm("{\n\t"
		"st.cs.s32\t[%0], %1;\n\t"
		"}"
#ifndef _M_X64 // x86
		:: "r"(address), "r"(val));
#else // x64
		:: "l"(address), "r"(val));
#endif
}

template<>
__device__ __forceinline__ void saveCS<unsigned int>(const unsigned int* address, unsigned int val)
{
	asm("{\n\t"
		"st.cs.u32\t[%0], %1;\n\t"
		"}"
#ifndef _M_X64 // x86
		:: "r"(address), "r"(val));
#else // x64
		:: "l"(address), "r"(val));
#endif
}

template<>
__device__ __forceinline__ void saveCS<uint2>(const uint2* address, uint2 val)
{
	asm("{\n\t"
		"st.cs.v2.u32\t[%0], {%1, %2};\n\t"
		"}"
#ifndef _M_X64 // x86
		:: "r"(address), "r"(val.x), "r"(val.y));
#else // x64
		:: "l"(address), "r"(val.x), "r"(val.y));
#endif
}

template<>
__device__ __forceinline__ void saveCS<uint4>(const uint4* address, uint4 val)
{
	asm("{\n\t"
		"st.cs.v4.u32\t[%0], {%1, %2, %3, %4};\n\t"
		"}"
#ifndef _M_X64 // x86
		:: "r"(address), "r"(val.x), "r"(val.y), "r"(val.z), "r"(val.w));
#else // x64
		:: "l"(address), "r"(val.x), "r"(val.y), "r"(val.z), "r"(val.w));
#endif
}

template<>
__device__ __forceinline__ void saveCS<float>(const float* address, float val)
{
	asm("{\n\t"
		"st.cs.f32\t[%0], %1;\n\t"
		"}"
#ifndef _M_X64 // x86
		:: "r"(address), "f"(val));
#else // x64
		:: "l"(address), "f"(val));
#endif
}

//------------------------------------------------------------------------

// Saves data volatilely, write-through
template<>
__device__ __forceinline__ void saveWT<int>(const int* address, int val)
{
	asm("{\n\t"
		"st.wt.s32\t[%0], %1;\n\t"
		"}"
#ifndef _M_X64 // x86
		:: "r"(address), "r"(val));
#else // x64
		:: "l"(address), "r"(val));
#endif
}

template<>
__device__ __forceinline__ void saveWT<unsigned int>(const unsigned int* address, unsigned int val)
{
	asm("{\n\t"
		"st.wt.u32\t[%0], %1;\n\t"
		"}"
#ifndef _M_X64 // x86
		:: "r"(address), "r"(val));
#else // x64
		:: "l"(address), "r"(val));
#endif
}

template<>
__device__ __forceinline__ void saveWT<uint2>(const uint2* address, uint2 val)
{
	asm("{\n\t"
		"st.wt.v2.u32\t[%0], {%1, %2};\n\t"
		"}"
#ifndef _M_X64 // x86
		:: "r"(address), "r"(val.x), "r"(val.y));
#else // x64
		:: "l"(address), "r"(val.x), "r"(val.y));
#endif
}

template<>
__device__ __forceinline__ void saveWT<uint4>(const uint4* address, uint4 val)
{
	asm("{\n\t"
		"st.wt.v4.u32\t[%0], {%1, %2, %3, %4};\n\t"
		"}"
#ifndef _M_X64 // x86
		:: "r"(address), "r"(val.x), "r"(val.y), "r"(val.z), "r"(val.w));
#else // x64
		:: "l"(address), "r"(val.x), "r"(val.y), "r"(val.z), "r"(val.w));
#endif
}

template<>
__device__ __forceinline__ void saveWT<float>(const float* address, float val)
{
	asm("{\n\t"
		"st.wt.f32\t[%0], %1;\n\t"
		"}"
#ifndef _M_X64 // x86
		:: "r"(address), "f"(val));
#else // x64
		:: "l"(address), "f"(val));
#endif
}

//------------------------------------------------------------------------

__device__ __forceinline__ unsigned int getMemory(unsigned int* ptr)
{
#if (MEM_ACCESS_TYPE == 0)
	return *ptr;
#elif (MEM_ACCESS_TYPE == 1)
	return loadCG<uint>(ptr);
#elif (MEM_ACCESS_TYPE == 2)
	return atomicCAS(ptr, 0xFFFFFFFF, 0); // Use atomic CAS with almost impossible value to get a value
#endif
}

__device__ __forceinline__ uint2 getMemory(uint2* ptr)
{
#if (MEM_ACCESS_TYPE == 0)
	return *ptr;
#elif (MEM_ACCESS_TYPE == 1)
	return loadCG<uint2>(ptr);
#elif (MEM_ACCESS_TYPE == 2)
	unsigned long long int ret = atomicCAS((unsigned long long int*)ptr, 0xFFFFFFFFFFFFFFFF, 0); // Use atomic CAS with almost impossible value to get a value
	return make_uint2(ret & 0xFFFFFFFF, ret >> 32);
#endif
}

__device__ __forceinline__ uint4 getMemory(uint4* ptr)
{
#if (MEM_ACCESS_TYPE == 0)
	return *ptr;
#elif (MEM_ACCESS_TYPE == 1)
	return loadCG<uint4>(ptr);
#elif (MEM_ACCESS_TYPE == 2)
	unsigned long long int ret1 = atomicCAS((unsigned long long int*)ptr, 0xFFFFFFFFFFFFFFFF, 0); // Use atomic CAS with almost impossible value to get a value
	unsigned long long int ret2 = atomicCAS((unsigned long long int*)((char*)ptr+sizeof(unsigned long long int)), 0xFFFFFFFFFFFFFFFF, 0); // Use atomic CAS with almost impossible value to get a value
	return make_uint4(ret1 & 0xFFFFFFFF, ret1 >> 32, ret2 & 0xFFFFFFFF, ret2 >> 32);
#endif
}

//------------------------------------------------------------------------

__device__ __forceinline__ void setMemory(unsigned int* ptr, unsigned int value)
{
#if (MEM_ACCESS_TYPE == 0)
	*ptr = value;
#elif (MEM_ACCESS_TYPE == 1)
	saveCG<uint>(ptr, value);
#elif (MEM_ACCESS_TYPE == 2)
	atomicExch(ptr, value);
#endif
}

__device__ __forceinline__ void setMemory(uint2* ptr, uint2 value)
{
#if (MEM_ACCESS_TYPE == 0)
	*ptr = value;
#elif (MEM_ACCESS_TYPE == 1)
	saveCG<uint2>(ptr, value);
#elif (MEM_ACCESS_TYPE == 2)
	unsigned long long int val;
	val = ((unsigned long long int)value.y << 32) | (unsigned long long int)value.x;
	atomicExch((unsigned long long int*)ptr, val);
#endif
}

__device__ __forceinline__ void setMemory(uint4* ptr, uint4 value)
{
#if (MEM_ACCESS_TYPE == 0)
	*ptr = value;
#elif (MEM_ACCESS_TYPE == 1)
	saveCG<uint4>(ptr, value);
#elif (MEM_ACCESS_TYPE == 2)
	unsigned long long int val;
	val = ((unsigned long long int)value.y << 32) | (unsigned long long int)value.x;
	atomicExch((unsigned long long int*)ptr, val);
	val = ((unsigned long long int)value.w << 32) | (unsigned long long int)value.z;
	atomicExch((unsigned long long int*)((char*)ptr+sizeof(unsigned long long int)), val);
#endif
}

//------------------------------------------------------------------------

// Alignment to multiply of S
template<typename T, int  S>
__device__ __forceinline__ T align(T a)
{
	 return (a+S-1) & ~(S-1);
}

//------------------------------------------------------------------------