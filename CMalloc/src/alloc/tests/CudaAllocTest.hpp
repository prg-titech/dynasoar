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

#include "base/Timer.hpp"
#include "gpu/Buffer.hpp"
#include "gpu/CudaCompiler.hpp"
#include "util/Environment.h"
#include "util/AppEnvironment.h"

namespace FW
{

/**
 * \brief Cuda allocator test class.
 */
class CudaAllocTest
{
public:
	/**
	 * \brief Constructor.
	 */
	CudaAllocTest();

	/**
	 * \brief Destructor.
	 */
	~CudaAllocTest();

	/**
	 * \brief Initializes memory pool according to the chosen allocator method.
	 * \return Duration of the initialization on the GPU.
	 */
	F32 initAlloc();

	/**
	 * \brief Performs the chosen allocator test.
	 * \return Duration of the test on the GPU.
	 */
	F32 testAlloc();

	/** 
	 * \brief Performs specified number of allocator test repetitions.
	 * \details Measures minimum, maximum, average and variance of allocator initialization and test time and also memory fragmentation across defined number of repetitions.
 	 */
	void measure();

	/**
	 * \return Current CPU time value.
	 */
	F32 getCPUTime() { return m_cpuTime; }

	/**
	 * \return Current GPU time value.
	 */
	F32 getGPUTime() { return m_gpuTime; }

private:

	/**
	 * \brief Updates allocator and allocator test settings constants and uploads them to the GPU.
	 */
	void updateConstants();

	/**
	 * \brief Allocates and prepares dynamic memory according to the chosen method.
	 * \return Time spent on the GPU on the memory pool preparation.
	 */
	F32 prepareDynamicMemory();

	/**
	 * \brief Prepares memory for the Atomic Malloc method.
	 * \return Time spent on the GPU on the memory allocation and preparation. 
	 */
	F32 prepareAtomicMalloc();

	/**
	 * \brief Sets the allocation header for the Circular Malloc method. 
	 */
	void setCircularMallocHeader(bool set, U32 ofs, U32 prevOfs, U32 nextOfs);

	/**
	* \brief Prepares memory for the Circular Malloc method.
	* \return Time spent on the GPU on the memory allocation and preparation.
	*/
	F32 prepareCircularMalloc();

	/**
	* \brief Prepares memory for the Scatter Alloc method.
	* \return Time spent on the GPU on the memory allocation and preparation.
	*/
	F32 prepareScatterAlloc();

	/**
	* \brief Prepares memory for the Halloc method.
	* \return Time spent on the GPU on the memory allocation and preparation.
	*/
	F32 prepareHalloc();

	/**
	 * \brief Prints state of the memory according to the used allocation method.
	 */
	void printState();

	/**
	 * \brief Prints state of the memory allocated via the Atomic Malloc method.
	 */
	void printStateAtomicMalloc();

	/**
	 * \brief Prints state of the memory allocated via the Circular Malloc method.
	 */
	void printStateCircularMalloc();

	/**
	 * \brief Prints test time statistics (min, max, avg and variance).
	 * \return Minimum time.
	 */
	F32 outputStatistics(const char* name, Array<F32>& samples);

	/**
	 * \brief Generates random numbers for each warp and each iteration of the test.
	 */
	void initRandom();

	/**
	 * \brief Computes memory fragmentation.
	 * \param[out] fInt Internal fragmentation ratio.
	 * \param[out] fExt External fragmentation ratio.
	 */
	void computeFragmentation(F32& fInt, F32& fExt);

private:
	// GPU heap
	String       m_method;			//!< Allocation method.
	Buffer       m_mallocData;		//!< Memory pool.
	U32          m_heapPtr;			//!< Heap pointer.

	CudaCompiler m_compiler;		//!< Cuda kernel compiler.
	CudaModule*  m_module;			//!< Cuda module.
	Buffer       m_multiOffset;		//!< Heap Offsets for each multiprocessor when using the Circular Multi Malloc allocator.

	// Debug buffers
	Buffer       m_debug;			//!< Buffer holdind debugging informations.
	// Benchmark
	String       m_test;			//!< Test type.
	Buffer       m_ptr;				//!< Debugging buffer holding allocated pointers.
	Buffer       m_random;			//!< Random number buffer.
	Buffer       m_interFragSum;	//!< Buffer holding internal fragmentation sum for each thread.

	// Statistics
	Timer        m_timer;			//!< Timer.
	F32          m_cpuTime;			//!< CPU time.
	F32          m_gpuTime;			//!< GPU time.
	U32          m_numAlloc;		//!< Number of allocations.
	U32          m_numFree;			//!< Number of deallocations.

	bool         m_firstRun;		//!< Flag whether this run is the first.
	int          m_numSM;			//!< Number of Streaming Multiprocessors on the device.

	Environment*		m_env;		//!< Environment settings.
};

}