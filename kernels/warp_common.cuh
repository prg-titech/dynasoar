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


//------------------------------------------------------------------------
// Inline PTX loading functions with caching.
// For details refer to the PTX ISA manual for "Cache Operators"
//------------------------------------------------------------------------

// Loads data cached at all levels - default
template<typename T>
__device__ __forceinline__ T loadCA(const T* address);

// Loads data cached at L2 and above
template<typename T>
__device__ __forceinline__ T loadCG(const T* address);

// Loads data cached at all levels, with evict-first policy
template<typename T>
__device__ __forceinline__ T loadCS(const T* address);

// Loads data similar to loadCS, on local addresses discards L1 cache following the load
template<typename T>
__device__ __forceinline__ T loadLU(const T* address);

// Loads data volatilely, again from memory
template<typename T>
__device__ __forceinline__ T loadCV(const T* address);


//------------------------------------------------------------------------
// Inline PTX saving functions with caching.
// For details refer to the PTX ISA manual for "Cache Operators"
//------------------------------------------------------------------------

// Saves data cached at all levels - default
template<typename T>
__device__ __forceinline__ void saveWB(const T* address, T val);

// Saves data cached at L2 and above
template<typename T>
__device__ __forceinline__ void saveCG(const T* address, T val);

// Saves data cached at all levels, with evict-first policy
template<typename T>
__device__ __forceinline__ void saveCS(const T* address, T val);

// Saves data volatilely, write-through
template<typename T>
__device__ __forceinline__ void saveWT(const T* address, T val);


//------------------------------------------------------------------------
// Functions for reading and saving memory in a safe way.
//------------------------------------------------------------------------

// Read data from global memory
template<typename T>
__device__ __forceinline__ T getMemory(T* ptr);

// Save data to global memory
template<typename T>
__device__ __forceinline__ void setMemory(T* ptr, T value);


//------------------------------------------------------------------------
// Utility.
//------------------------------------------------------------------------


// Alignment to multiply of S
template<typename T, int  S>
__device__ __forceinline__ T align(T a);

//------------------------------------------------------------------------