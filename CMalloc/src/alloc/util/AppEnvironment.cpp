/* 
 *  Copyright (c) 2013, Faculty of Informatics, Masaryk University
 *  All rights reserved.
 *  
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions are met:
 *      * Redistributions of source code must retain the above copyright
 *        notice, this list of conditions and the following disclaimer.
 *      * Redistributions in binary form must reproduce the above copyright
 *        notice, this list of conditions and the following disclaimer in the
 *        documentation and/or other materials provided with the distribution.
 *      * Neither the name of the <organization> nor the
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
 *  Tomas Kopal, 1996
 *  Vilem Otte <vilem.otte@post.cz>
 *
 */

/*! \file
 *  \brief Environment variables class for this app.
 */

#include "AppEnvironment.h"

void AppEnvironment::RegisterOptions()
{
	/*************************************************************************/
	/*    CudaAllocator                                                      */
	/*************************************************************************/

	 RegisterOption("CudaAlloc.method",
		 optString,
		 "cuda_alloc_method=",
		 "CudaMalloc");

	 RegisterOption("CudaAlloc.heapSize",
		 optFloat,
		 "cuda_alloc_heap_size=",
		 "8388608");

	 RegisterOption("CudaAlloc.payload",
		 optInt,
		 "cuda_alloc_payload=",
		 "4");

	 RegisterOption("CudaAlloc.maxFrag",
		 optFloat,
		 "cuda_alloc_max_frag=",
		 "2");

	 RegisterOption("CudaAlloc.chunkRatio",
		 optFloat,
		 "cuda_alloc_chunk_ratio=",
		 "1");

	 RegisterOption("CudaAlloc.allocScale",
		 optFloat,
		 "cuda_alloc_alloc_scale=",
		 "0");

	 RegisterOption("CudaAlloc.threadsPerBlock",
		 optInt,
		 "cuda_alloc_threads_per_block=",
		 "256");

	 RegisterOption("CudaAlloc.numBlocks",
		 optInt,
		 "cuda_alloc_num_blocks=",
		 "1200");

	 RegisterOption("CudaAlloc.numRepeats",
		 optInt,
		 "cuda_alloc_num_repeats=",
		 "1");

	 RegisterOption("CudaAlloc.test",
		 optString,
		 "cuda_alloc_test=",
		 "AllocDealloc");

	 RegisterOption("CudaAlloc.testIters",
		 optInt,
		 "cuda_alloc_test_iters=",
		 "32");

	 RegisterOption("CudaAlloc.pAlloc",
		 optFloat,
		 "cuda_alloc_palloc=",
		 "0.75");

	 RegisterOption("CudaAlloc.pFree",
		 optFloat,
		 "cuda_alloc_pfree=",
		 "0.75");
}

