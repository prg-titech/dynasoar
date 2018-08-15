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

#include <string>
#include <ctime>
#include "base/Main.hpp"
#include "base/Random.hpp"
#include "CudaAllocTest.hpp"
#include "AllocTest.hpp"

using namespace FW;

#define CIRCULAR_MALLOC_PREALLOC 3
#define CPU 0
#define GPU 1
#define CIRCULAR_MALLOC_INIT GPU
#define BENCHMARK

// Alignment to multiply of S
template<typename T, int  S>
T align(T a)
{
	 return (a+S-1) & -S;
}

//------------------------------------------------------------------------

CudaAllocTest::CudaAllocTest()
{
	m_env = new AppEnvironment();
	
	if (!m_env->Parse(argc, argv, false, NULL, "config.conf")) {
		cerr<<"Error: when parsing environment file!";
		m_env->PrintUsage(cout);
		exit(1);
	}
	m_env->SetStaticOptions();
	Environment::SetSingleton(m_env);
}

//------------------------------------------------------------------------

CudaAllocTest::~CudaAllocTest()
{
	delete m_env;
}

//------------------------------------------------------------------------

F32 CudaAllocTest::initAlloc()
{
	updateConstants();

	// Start the timer
	m_timer.clearTotal();
	m_timer.unstart();
	m_timer.start();
	
	m_gpuTime = prepareDynamicMemory();
	m_timer.end();
	m_cpuTime = m_timer.getTotal();

	return m_gpuTime;
}

//------------------------------------------------------------------------

F32 CudaAllocTest::testAlloc()
{
	int threadsPerBlock = Environment::GetSingleton()->GetInt("CudaAlloc.threadsPerBlock");
	int numBlocks = Environment::GetSingleton()->GetInt("CudaAlloc.numBlocks");
	int testIters = Environment::GetSingleton()->GetInt("CudaAlloc.testIters");

	int numPtrs = 1;
	if(m_test == "AllocCycleDealloc")
	{
		numPtrs = testIters; // Multiple pointers in one iteration
		testIters = 1; // Only one kernel run
	}

	CudaKernel kernel = m_module->getKernel(m_method+"_"+m_test);

	Vec2i threadsSize(threadsPerBlock*numBlocks, 1);
	Vec2i blockSize(threadsPerBlock, 1);

	m_ptr.resizeDiscard((threadsPerBlock / WARP_SIZE) * numBlocks * numPtrs * sizeof(int*));
	m_ptr.clear(NULL);
	int**& ptrArray = *(int***)m_module->getGlobal("g_ptrArray").getMutablePtr();
	ptrArray = (int**)m_ptr.getMutableCudaPtr();
	
	float*& random = *(float**)m_module->getGlobal("g_random").getMutablePtr();
	random = (float*)m_random.getMutableCudaPtr();

	int& iter = *(int*)m_module->getGlobal("g_iter").getMutablePtr();

	int& fail = *(int*)m_module->getGlobal("g_kernelFail").getMutablePtr();
	fail = 0;

#ifdef CIRCULAR_MALLOC_CHECK_INTERNAL_FRAGMENTATION
	// Prepare memory for internal fragmentation data
	m_interFragSum.resizeDiscard(threadsPerBlock * numBlocks * sizeof(float));
	m_interFragSum.clear(0);
	float*& interFragSum = *(float**)m_module->getGlobal("g_interFragSum").getMutablePtr();
	interFragSum = (float*)m_interFragSum.getMutableCudaPtr();
#endif

	// Start the timer
	m_timer.clearTotal();
	m_timer.unstart();
	m_timer.start();

	m_gpuTime = 0;

	for(iter = 0; iter < testIters; iter++)
	{
		m_gpuTime += kernel.launchTimed(threadsSize, blockSize);	
	}

	m_timer.end();
	m_cpuTime = m_timer.getTotal();

	int failed = *(int*)m_module->getGlobal("g_kernelFail").getMutablePtr();
	if(failed != 0)
	{
		FW::fail("Error occured during the test! Check the memory requirements.");
	}


#ifdef SAVE_PTR_TEST
	if(m_test != "Probability")
	{
		kernel = m_module->getKernel("SavePtrTest");
		if (!kernel)
			FW::fail("%s kernel not found!", m_method.getPtr());

		m_module->launchKernelTimed(kernel, blockSize.x/WARP_SIZE, numBlocks);
	}
#endif

	return m_gpuTime;
}

//------------------------------------------------------------------------

void CudaAllocTest::measure()
{
	std::string method, test;
	Environment::GetSingleton()->GetStringValue("CudaAlloc.method", method);
	m_method.set(method.c_str());
	Environment::GetSingleton()->GetStringValue("CudaAlloc.test", test);
	m_test.set(test.c_str());
	int numRepeats = Environment::GetSingleton()->GetInt("CudaAlloc.numRepeats");

	// Check settings
	int testIters = Environment::GetSingleton()->GetInt("CudaAlloc.testIters");

	if(m_test == "AllocDealloc" && testIters != 1)
	{
#ifdef SAVE_PTR_TEST
		printf("Warning: Current setting will result in multiple allocations without corresponding free -> memory leaks!\nSetting \"testIters=1\"\n\n");
		Environment::GetSingleton()->SetInt("CudaAlloc.testIters", 1);
#else
		printf("Warning: Current setting will result in multiple runs of the same benchmark.\n\n");
#endif
	}

#ifdef SAVE_PTR_TEST
	if(m_test == "Probability")
	{
		printf("Warning: Ignoring \"SAVE_PTR_TEST\" for test \"%s\"\n\n", m_test);
	}
#endif

	Array<F32> initTimesGPU, initTimesCPU;
	Array<F32> testTimesGPU, testTimesCPU;
	Array<F32> fragIntern, fragExtern;
	m_firstRun = true;
	for(int i = 0; i < numRepeats; i++)
	{
		// init
		CudaModule::staticInit();
		m_compiler.clearDefines();
		m_compiler.clearOptions();
		//m_compiler.addOptions("-use_fast_math");
		//m_compiler.addOptions("-Xptxas -dlcm=cg");
		//m_compiler.addOptions("-Xptxas -dlcm=cg -Xptxas -dscm=wb -Xptxas -maxrregcount=64");

		// Compile the kernel
		m_compiler.setCachePath("cudacache"); // On the first compilation the cache path becames absolute which kills the second compilation
		m_compiler.setSourceFile("src/alloc/tests/alloc_test.cu");
		m_module = m_compiler.compile();
		failIfError();

		// Init random number for each thread
		initRandom();

		initAlloc();
		initTimesGPU.add(getGPUTime());
		initTimesCPU.add(getCPUTime());

		testAlloc();
		testTimesGPU.add(getGPUTime());
		testTimesCPU.add(getCPUTime());
		m_firstRun = false;

		// Compute Fragmentations
		F32 fInt, fExt;
		computeFragmentation(fInt, fExt);
		fragIntern.add(fInt);
		fragExtern.add(fExt);

#ifndef BENCHMARK
		printState();
#endif

		m_module->freeGlobals();
		m_mallocData.reset();
		m_debug.reset();
		m_ptr.reset();
		m_random.reset();
		m_interFragSum.reset();
		m_multiOffset.reset();
		
		CudaCompiler::staticDeinit();
		CudaModule::staticDeinit();
	}

	cuCtxSynchronize(); // Flushes printfs

	outputStatistics("GPU init time", initTimesGPU);
	outputStatistics("CPU init time", initTimesCPU);

	outputStatistics("GPU test time", testTimesGPU);
	outputStatistics("CPU test time", testTimesCPU);

	outputStatistics("internal fragmentation", fragIntern);
	outputStatistics("external fragmentation", fragExtern);
}

//------------------------------------------------------------------------

void CudaAllocTest::updateConstants()
{
	AllocInfo& allocInfo = *(AllocInfo*)m_module->getGlobal("c_alloc").getMutablePtr();
	allocInfo.heapSize = (unsigned int)Environment::GetSingleton()->GetDouble("CudaAlloc.heapSize");
	allocInfo.payload = (unsigned int)Environment::GetSingleton()->GetDouble("CudaAlloc.payload");
	allocInfo.maxFrag = Environment::GetSingleton()->GetDouble("CudaAlloc.maxFrag");
	allocInfo.chunkRatio = Environment::GetSingleton()->GetDouble("CudaAlloc.chunkRatio");

	AllocTestInfo& allocTestInfo = *(AllocTestInfo*)m_module->getGlobal("c_env").getMutablePtr();
	allocTestInfo.allocScale = (float)Environment::GetSingleton()->GetDouble("CudaAlloc.allocScale");
	allocTestInfo.testIters = Environment::GetSingleton()->GetInt("CudaAlloc.testIters");
	allocTestInfo.pAlloc = Environment::GetSingleton()->GetFloat("CudaAlloc.pAlloc");
	allocTestInfo.pFree = Environment::GetSingleton()->GetFloat("CudaAlloc.pFree");
}

//------------------------------------------------------------------------

F32 CudaAllocTest::prepareDynamicMemory()
{
	// Set the memory limit
	U64 allocSize = (U64)Environment::GetSingleton()->GetDouble("CudaAlloc.heapSize");

	// Allocate the memory for the heap
	if(m_method == "CudaMalloc" || m_method == "FDGMalloc")
	{
		CudaModule::checkError("cuCtxSetLimit", cuCtxSetLimit(CU_LIMIT_MALLOC_HEAP_SIZE, (size_t)allocSize));
	}
	else if(m_method == "AtomicMalloc" || m_method == "AtomicWrapMalloc" || m_method == "CircularMalloc" || m_method == "CircularFusedMalloc" || m_method == "CircularMultiMalloc" || m_method == "CircularFusedMultiMalloc" || m_method == "ScatterAlloc")
	{
		m_mallocData.resizeDiscard(allocSize);
	}
	
	F32 kernelTime = 0.f;
	// Prepare the memory for the heap
	if(m_method == "AtomicMalloc" || m_method == "AtomicWrapMalloc")
	{
		kernelTime = prepareAtomicMalloc();
	}
	else if(m_method == "CircularMalloc" || m_method == "CircularFusedMalloc" || m_method == "CircularMultiMalloc" || m_method == "CircularFusedMultiMalloc")
	{
		kernelTime = prepareCircularMalloc();
	}
	else if(m_method == "ScatterAlloc")
	{
		kernelTime = prepareScatterAlloc();
	}
	else if(m_method == "Halloc")
	{
		kernelTime = prepareHalloc();
	}

	return kernelTime;
}

//------------------------------------------------------------------------

F32 CudaAllocTest::prepareAtomicMalloc()
{
	// Set the heapBase
	char*& heapBase = *(char**)m_module->getGlobal("g_heapBase").getMutablePtr();
	heapBase = (char*)m_mallocData.getMutableCudaPtr();
	
	// Init the heapOffset
	S32& heapOffset = *(S32*)m_module->getGlobal("g_heapOffset").getMutablePtr();
	heapOffset = 0;

	return 0.f;
}

//------------------------------------------------------------------------

void CudaAllocTest::setCircularMallocHeader(bool set, U32 ofs, U32 prevOfs, U32 nextOfs)
{
	if(m_method == "CircularMalloc" || m_method == "CircularMultiMalloc")
	{
		AllocatorLockType type;
		if(set)
			type = AllocatorLockType_Set;
		else
			type = AllocatorLockType_Free;

#ifndef CIRCULAR_MALLOC_DOUBLY_LINKED
		Vec2i header(type, nextOfs);
		m_mallocData.setRange(ofs, &header, sizeof(Vec2i));
#else
		Vec4i header(type, prevOfs, nextOfs, 0);
		m_mallocData.setRange(ofs, &header, sizeof(Vec4i));
#endif
	}
	else
	{
		const U32 flagMask = 0x80000000u;
		if(set)
			nextOfs = nextOfs | flagMask;

		m_mallocData.setRange(ofs, &nextOfs, sizeof(U32));
	}
}

//------------------------------------------------------------------------

F32 CudaAllocTest::prepareCircularMalloc()
{
	F32 initTime = 0.f;

	// Payload size
	U32 payload = (unsigned int)Environment::GetSingleton()->GetDouble("CudaAlloc.payload");

	// Chunk ratio
	F64 chunkRatio = Environment::GetSingleton()->GetDouble("CudaAlloc.chunkRatio");

	// Set the heapBase
	char*& heapBase = *(char**)m_module->getGlobal("g_heapBase").getMutablePtr();
	heapBase = (char*)m_mallocData.getMutableCudaPtr();
	
	// Init the heapOffset
	U32& heapOffset = *(U32*)m_module->getGlobal("g_heapOffset").getMutablePtr();
	heapOffset = 0;

	// Init the heapMultiOffset
	CUdevice device;
	int m_numSM;
	CudaModule::checkError("cuCtxGetDevice", cuCtxGetDevice(&device));
	CudaModule::checkError("cuDeviceGetAttribute", cuDeviceGetAttribute(&m_numSM, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, device));
	U32& numSM = *(U32*)m_module->getGlobal("g_numSM").getMutablePtr();
	numSM = m_numSM;

	m_multiOffset.resizeDiscard(m_numSM * sizeof(unsigned int*));
	unsigned int*& heapMultiOffset = *(unsigned int**)m_module->getGlobal("g_heapMultiOffset").getMutablePtr();

	// Init the heapSize
	U32 heapSize = (U32)m_mallocData.getSize();

	U32 headerSize;
	if(m_method == "CircularMalloc" || m_method == "CircularMultiMalloc")
		headerSize = CIRCULAR_MALLOC_HEADER_SIZE;
	else
		headerSize = sizeof(U32);

	U32& heapLock = *(U32*)m_module->getGlobal("g_heapLock").getMutablePtr();
	heapLock = 0;

	// Set the chunk size
	U32 numChunks = 0;
	U32 chunkSize = align<U32, ALIGN>((U32)((headerSize + payload)*chunkRatio));

#if (CIRCULAR_MALLOC_INIT == CPU)
	// Prepare the buffer on the CPU
	//m_mallocData.getMutablePtr();
	heapMultiOffset = (unsigned int*)m_multiOffset.getMutablePtr();

	// Offset of the division
	U32 ofs = 0;
	U32 prevOfs = heapSize-headerSize;

#if (CIRCULAR_MALLOC_PREALLOC == 1)
	// Create regular chunks
	ofs = 0;

	numChunks = (U32)((m_mallocData.getSize()-(chunkSize+headerSize))/chunkSize);
	U32 chunksPerSM = (U32)ceil((F32)numChunks/(F32)m_numSM);
	for(U32 i = 0; i < numChunks; i++)
	{
		setCircularMallocHeader(false, i*chunkSize, prevOfs, (i+1)*chunkSize); // Unlocked, next at the multiple of chunkSize

		// Set the heap offsets
		if(i % chunksPerSM == 0)
			heapMultiOffset[i / chunksPerSM] = ofs;

		prevOfs = ofs;
		ofs += chunkSize;
	}

#elif (CIRCULAR_MALLOC_PREALLOC == 2)
	if(m_method == "CircularMultiMalloc" || m_method == "CircularFusedMultiMalloc")
		fail("Not yet implemented for MultiMalloc!\n");

	// Create exponential chunks
#if 1
	for(ofs = 0; ofs+chunkSize < heapSize-2*headerSize && ofs+chunkSize > ofs;)
#else
	U32 minChunkSize = chunkSize;
	U32 expChunks = log2((float)(heapSize-2*headerSize)/(float)chunkSize) - 0.5f; // Temporary
	chunkSize = (1 << expChunks) * minChunkSize;
	for(ofs = 0; ofs+chunkSize < heapSize-2*headerSize && ofs+chunkSize > ofs && chunkSize >= minChunkSize;)
#endif
	{
		setCircularMallocHeader(false, ofs, prevOfs, ofs+chunkSize); // Unlocked, next at the multiple of chunkSize

		//printf("Ofs %u Chunk size %u\n", ofs, chunkSize);
		//printf("Ofs %u Chunk size %u. Ofs + Chunk size %u, Heap %u\n", ofs, chunkSize, ofs + chunkSize, heapSize-2*headerSize);
		prevOfs = ofs;
		ofs += chunkSize;
#if 1
		chunkSize = align<U32, ALIGN>(chunkSize*2);
#else
		chunkSize = align<U32, ALIGN>(chunkSize/2);
#endif
		numChunks++;
	}

#elif (CIRCULAR_MALLOC_PREALLOC == 3)
	// Create hierarchical chunks
	U32 minChunkSize = chunkSize;
	F32 treeMem = minChunkSize;
	F32 heapTotalMem = heapSize-2*headerSize;
	F32 heapMem;
	int repeats = 1;
	if(m_method == "CircularMalloc" || m_method == "CircularFusedMalloc")
	{
		heapMem = heapTotalMem;
		repeats = 1;
	}
	else
	{
		heapMem = heapTotalMem/(F32)m_numSM;
		repeats = m_numSM;
	}

	U32 i = 1;
	for(; treeMem < heapMem; i++)
	{
		treeMem = ((float)(i+1))*((float)(1 << i))*((float)minChunkSize);
	}

	U32 maxChunkSize = (1 << (i-2))*minChunkSize;

	for(int r = 0; r < repeats; r++)
	{
		// Set the heap offsets
		heapMultiOffset[r] = ofs;
		chunkSize = maxChunkSize;

		setCircularMallocHeader(false, ofs, prevOfs, ofs+chunkSize); // Unlocked, root chunk
		numChunks++;

		//printf("Ofs %u Chunk size %u\n", ofs, chunkSize);

		prevOfs = ofs;
		ofs += chunkSize;
		U32 localOfs = chunkSize;

		i = 0;
		U32 lvl = 1;
		for(; true; ofs += chunkSize, localOfs += chunkSize, i++)
		{
			if(i == 0 || i == lvl) // New level in BFS order
			{
				chunkSize = align<U32, ALIGN>(chunkSize/2);
				i = 0;
				lvl *= 2;
			}

			if(localOfs+chunkSize >= heapMem || chunkSize < minChunkSize) // We cannot make another chunk
				break;

			//printf("Ofs %u Chunk size %u\n", ofs, chunkSize);

			setCircularMallocHeader(false, ofs, prevOfs, ofs+chunkSize); // Unlocked, next at ofs+chunkSize

			prevOfs = ofs;
			numChunks++;
		}

		if(ofs+chunkSize >= heapTotalMem) // We cannot make another chunk
			break;
	}

#else
#error Unsupported CPU initialization method!
#endif

	//printf("Ofs %u Chunk size %u\n", ofs, (heapSize-headerSize)-ofs);
	numChunks++;

	setCircularMallocHeader(false, ofs, prevOfs, heapSize-headerSize); // Unlocked, next at the end of allocated memory

	setCircularMallocHeader(true, heapSize-headerSize, ofs, 0); // Locked, next at the start of heap

	// Transfer the buffer to the GPU
	//m_mallocData.getMutableCudaPtr();
	heapMultiOffset = (unsigned int*)m_multiOffset.getMutableCudaPtr();

#elif (CIRCULAR_MALLOC_INIT == GPU)
	heapMultiOffset = (unsigned int*)m_multiOffset.getMutableCudaPtr();

#if (CIRCULAR_MALLOC_PREALLOC == 1)
	// Create regular chunks
	CudaKernel initHeap = m_module->getKernel(m_method+"Prepare1");
	if(!initHeap)
		fail((m_method+" initialization kernel not found!").getPtr());

	numChunks = (m_mallocData.getSize()-headerSize)/chunkSize;
	int tpb = 256;
	initHeap.setParams(numChunks);
	initTime = initHeap.launchTimed(numChunks, Vec2i(tpb,1));

#ifndef BENCHMARK
	if(m_firstRun)
		printf("Grid dimensions tpb %d, gridDim.x %d\n", tpb, gridSize.x);
#endif

#elif (CIRCULAR_MALLOC_PREALLOC == 2)
	if(m_method == "CircularMultiMalloc" || m_method == "CircularFusedMultiMalloc")
		fail("Not yet implemented for MultiMalloc!\n");

	// Create exponential chunks
	CudaKernel initHeap = m_module->getKernel(m_method+"Prepare2");
	if(!initHeap)
		fail((m_method+" initialization kernel not found!").getPtr());

	numChunks = ceil(log2((float)(heapSize-2*headerSize)/(float)chunkSize));
	int tpb = 256;
	initHeap.setParams(numChunks);
	initTime = initHeap.launchTimed(numChunks, Vec2i(tpb,1));

#elif (CIRCULAR_MALLOC_PREALLOC == 3)
	// Create hierarchical chunks
	U32 minChunkSize = chunkSize;
	F32 treeMem = (F32)minChunkSize;
	F32 heapTotalMem = (F32)(heapSize-2*headerSize);
	F32 heapMem;
	int repeats = 1;
	if(m_method == "CircularMalloc" || m_method == "CircularFusedMalloc")
	{
		heapMem = heapTotalMem;
		repeats = 1;
	}
	else
	{
		heapMem = heapTotalMem/(F32)m_numSM;
		repeats = m_numSM;
	}

	U32 i = 1;
	for(; treeMem < heapMem; i++)
	{
		treeMem = ((float)(i+1))*((float)(1 << i))*((float)minChunkSize);
	}

	chunkSize = (1 << (i-2))*minChunkSize;

	CudaKernel initHeap = m_module->getKernel(m_method+"Prepare3");

	if(m_method == "CircularMalloc" || m_method == "CircularFusedMalloc")
		numChunks = (1 << (i-1)); // Number of nodes of the tree + 1 for the rest
	else
		numChunks = ((1 << (i-1)) - 1) * m_numSM + 1; // Number of nodes of the tree times numSM + 1 for the rest

	int tpb = 256;
	initHeap.setParams(numChunks, chunkSize);
	initTime = initHeap.launchTimed(numChunks, Vec2i(tpb,1));

#else
#error Unsupported GPU initialization method!
#endif
#endif

#ifndef BENCHMARK
	if(m_firstRun)
		printf("Init heap executed for heap size %lld, headerSize %d, chunkSize %u, numChunks %u\n", m_mallocData.getSize(), headerSize, chunkSize, numChunks);
#endif

	return initTime;
}

//------------------------------------------------------------------------

F32 CudaAllocTest::prepareScatterAlloc()
{
	// CUDA Driver API cannot deal with templates -> use C++ mangled name
	CudaKernel initHeap = m_module->getKernel("_ZN8GPUTools8initHeapILj" STR(SCATTER_ALLOC_PAGESIZE) "ELj" STR(SCATTER_ALLOC_ACCESSBLOCKS)
		"ELj" STR(SCATTER_ALLOC_REGIONSIZE) "ELj" STR(SCATTER_ALLOC_WASTEFACTOR) "ELb" STR(SCATTER_ALLOC_COALESCING) "ELb" STR(SCATTER_ALLOC_RESETPAGES)
		"EEEvPNS_10DeviceHeapIXT_EXT0_EXT1_EXT2_EXT3_EXT4_EEEPvj");
	initHeap.setParams(m_module->getGlobal("theHeap").getMutableCudaPtr(), m_mallocData.getMutableCudaPtr(), (U32)m_mallocData.getSize());
	
	unsigned int numregions = (unsigned int)(((unsigned long long)m_mallocData.getSize())/( ((unsigned long long)SCATTER_ALLOC_REGIONSIZE)*(3*sizeof(unsigned int)+SCATTER_ALLOC_PAGESIZE)+sizeof(unsigned int)));
    unsigned int numpages = numregions*SCATTER_ALLOC_REGIONSIZE;
	const int tpb = 256;
	F32 initTime = initHeap.launchTimed(numpages, Vec2i(tpb,1));

	return initTime;
}

//------------------------------------------------------------------------

F32 CudaAllocTest::prepareHalloc()
{
	// Set the memory limit
	size_t allocSize = (size_t)Environment::GetSingleton()->GetDouble("CudaAlloc.heapSize");
	// Memory pool size must be larger than 256MB, otherwise allocation always fails
	// May be possibly tweaked by changing halloc_opts_t.sb_sz_sh
	allocSize = (size_t)max((U64)allocSize, (U64)256ULL*1024ULL*1024ULL);
	halloc_opts_t opts = halloc_opts_t(allocSize);

	// TODO: initialize all devices
	// get total device memory (in bytes) & total number of superblocks
	uint64 dev_memory;
	cudaDeviceProp dev_prop;
	int dev;
	cucheck(cudaGetDevice(&dev));
	cucheck(cudaGetDeviceProperties(&dev_prop, dev));
	dev_memory = dev_prop.totalGlobalMem;
	uint sb_sz = 1 << opts.sb_sz_sh;

	// set cache configuration
	cucheck(cudaDeviceSetCacheConfig(cudaFuncCachePreferL1));

	// limit memory available to 3/4 of device memory
	opts.memory = min((uint64)opts.memory, 3ull * dev_memory / 4ull);

	// split memory between halloc and CUDA allocator
	uint64 halloc_memory = opts.halloc_fraction * opts.memory;
	uint64 cuda_memory = opts.memory - halloc_memory;
	cucheck(cudaDeviceSetLimit(cudaLimitMallocHeapSize, cuda_memory));
	cuset(cuda_mem_g, uint64, cuda_memory);
	cuset(total_mem_g, uint64, halloc_memory + cuda_memory);

	// set the number of slabs
	//uint nsbs = dev_memory / sb_sz;
	uint nsbs = halloc_memory / sb_sz;
	cuset(nsbs_g, uint, nsbs);
	cuset(sb_sz_g, uint, sb_sz);
	cuset(sb_sz_sh_g, uint, opts.sb_sz_sh);

	// allocate a fixed number of superblocks, copy them to device
	uint nsbs_alloc = (uint)min((uint64)nsbs, (uint64)halloc_memory / sb_sz);
	size_t sbs_sz = MAX_NSBS * sizeof(superblock_t);
	size_t sb_ptrs_sz = MAX_NSBS * sizeof(void *);
	superblock_t *sbs = (superblock_t *)malloc(sbs_sz);
	void **sb_ptrs = (void **)malloc(sb_ptrs_sz);
	memset(sbs, 0, sbs_sz);
	memset(sb_ptrs, 0, sb_ptrs_sz);
	uint *sb_counters = (uint *)malloc(MAX_NSBS * sizeof(uint));
	memset(sbs, 0xff, MAX_NSBS * sizeof(uint));
	char *base_addr = (char *)~0ull;
	for(uint isb = 0; isb < nsbs_alloc; isb++) {
		sb_counters[isb] = sb_counter_val(0, false, SZ_NONE, SZ_NONE);
		sbs[isb].size_id = SZ_NONE;
		sbs[isb].chunk_id = SZ_NONE;
		sbs[isb].is_head = 0;
		//sbs[isb].flags = 0;
		sbs[isb].chunk_sz = 0;
		//sbs[isb].chunk_id = SZ_NONE;
		//sbs[isb].state = SB_FREE;
		//sbs[isb].mutex = 0;
		cucheck(cudaMalloc(&sbs[isb].ptr, sb_sz));
		sb_ptrs[isb] = sbs[isb].ptr;
		base_addr = (char *)min((uint64)base_addr, (uint64)sbs[isb].ptr);
	}
	//cuset_arr(sbs_g, (superblock_t (*)[MAX_NSBS])&sbs);
	cuset_arr(sbs_g, (superblock_t (*)[MAX_NSBS])sbs);
	cuset_arr(sb_counters_g, (uint (*)[MAX_NSBS])sb_counters);
	cuset_arr(sb_ptrs_g, (void* (*)[MAX_NSBS])sb_ptrs);
	// also mark free superblocks in the set
	sbset_t free_sbs;
	memset(free_sbs, 0, sizeof(free_sbs));
	for(uint isb = 0; isb < nsbs_alloc; isb++) {
		uint iword = isb / WORD_SZ, ibit = isb % WORD_SZ;
		free_sbs[iword] |= 1 << ibit;
	}
	free_sbs[SB_SET_SZ - 1] = nsbs_alloc;
	cuset_arr(free_sbs_g, &free_sbs);
	base_addr = (char *)((uint64)base_addr / sb_sz * sb_sz);
	if((uint64)base_addr < dev_memory)
		base_addr = 0;
	else
		base_addr -= dev_memory;
	cuset(base_addr_g, void *, base_addr);

	// allocate block bits and zero them out
	void *bit_blocks, *alloc_sizes;
	uint nsb_bit_words = sb_sz / (BLOCK_STEP * WORD_SZ),
		nsb_alloc_words = sb_sz / (BLOCK_STEP * 4);
	// TODO: move numbers into constants
	uint nsb_bit_words_sh = opts.sb_sz_sh - (4 + 5);
	cuset(nsb_bit_words_g, uint, nsb_bit_words);
	cuset(nsb_bit_words_sh_g, uint, nsb_bit_words_sh);
	cuset(nsb_alloc_words_g, uint, nsb_alloc_words);
	size_t bit_blocks_sz = nsb_bit_words * nsbs * sizeof(uint), 
		alloc_sizes_sz = nsb_alloc_words * nsbs * sizeof(uint);
	cucheck(cudaMalloc(&bit_blocks, bit_blocks_sz));
	cucheck(cudaMemset(bit_blocks, 0, bit_blocks_sz));
	cuset(block_bits_g, uint *, (uint *)bit_blocks);
	cucheck(cudaMalloc(&alloc_sizes, alloc_sizes_sz));
	cucheck(cudaMemset(alloc_sizes, 0, alloc_sizes_sz));
	cuset(alloc_sizes_g, uint *, (uint *)alloc_sizes);

	// set sizes info
	//uint nsizes = (MAX_BLOCK_SZ - MIN_BLOCK_SZ) / BLOCK_STEP + 1;
	uint nsizes = 2 * NUNITS;
	cuset(nsizes_g, uint, nsizes);
	size_info_t size_infos[MAX_NSIZES];
	memset(size_infos, 0, MAX_NSIZES * sizeof(size_info_t));
	for(uint isize = 0; isize < nsizes; isize++) {
		uint iunit = isize / 2, unit = 1 << (iunit + 3);
		size_info_t *size_info = &size_infos[isize];
		//size_info->block_sz = isize % 2 ? 3 * unit : 2 * unit;
		uint block_sz = isize % 2 ? 3 * unit : 2 * unit;
		uint nblocks = sb_sz / block_sz;
		// round #blocks to a multiple of THREAD_MOD
		uint tmod = tmod_by_size(isize);
		nblocks = nblocks / tmod * tmod;
		//nblocks = nblocks / THREAD_MOD * THREAD_MOD;
		size_info->chunk_id = isize % 2 + (isize < nsizes / 2 ? 0 : 2);
		uint chunk_sz = (size_info->chunk_id % 2 ? 3 : 2) * 
			(size_info->chunk_id / 2 ? 128 : 8);
		size_info->chunk_sz = chunk_val(chunk_sz);
		size_info->nchunks_in_block = block_sz / chunk_sz;
		size_info->nchunks = nblocks * size_info->nchunks_in_block;
		// TODO: use a better hash step
		size_info->hash_step = size_info->nchunks_in_block *
		 	max_prime_below(nblocks / 256 + nblocks / 64, nblocks);
		//size_info->hash_step = size_info->nchunks_in_block * 17;
		// printf("block = %d, step = %d, nchunks = %d, nchunks/block = %d\n", 
		// 			 block_sz, size_info->hash_step, size_info->nchunks, 
		// 			 size_info->nchunks_in_block);
		size_info->roomy_threshold = opts.roomy_fraction * size_info->nchunks;
		size_info->busy_threshold = opts.busy_fraction * size_info->nchunks;
		size_info->sparse_threshold = opts.sparse_fraction * size_info->nchunks;
	}  // for(each size)
	cuset_arr(size_infos_g, &size_infos);

	// set grid info
	uint64 sb_grid[2 * MAX_NSBS];
	for(uint icell = 0; icell < 2 * MAX_NSBS; icell++) 
		sb_grid[icell] = grid_cell_init();
	for(uint isb = 0; isb < nsbs_alloc; isb++)
		grid_add_sb(sb_grid, base_addr, isb, sbs[isb].ptr, sb_sz);
	cuset_arr(sb_grid_g, &sb_grid);
	
	// zero out sets (but have some of the free set)
	//fprintf(stderr, "started cuda-memsetting\n");
	//cuvar_memset(unallocated_sbs_g, 0, sizeof(unallocated_sbs_g));
	cuvar_memset(busy_sbs_g, 0, sizeof(roomy_sbs_g));
	cuvar_memset(roomy_sbs_g, 0, sizeof(roomy_sbs_g));
	cuvar_memset(sparse_sbs_g, 0, sizeof(sparse_sbs_g));
	//cuvar_memset(roomy_sbs_g, 0, (MAX_NSIZES * SB_SET_SZ * sizeof(uint)));
	cuvar_memset(head_sbs_g, ~0, sizeof(head_sbs_g));
	cuvar_memset(cached_sbs_g, ~0, sizeof(head_sbs_g));
	cuvar_memset(head_locks_g, 0, sizeof(head_locks_g));
	cuvar_memset(sb_locks_g, 0, sizeof(sb_locks_g));
	//cuvar_memset(counters_g, 1, sizeof(counters_g));
	cuvar_memset(counters_g, 11, sizeof(counters_g));
	//fprintf(stderr, "finished cuda-memsetting\n");
	cucheck(cudaStreamSynchronize(0));

	// free all temporary data structures
	free((void*)sbs);
	free((void*)sb_counters);

	return 0.f;
}

//------------------------------------------------------------------------

void CudaAllocTest::printState()
{
	// Allocate the memory for the heap
	if(m_method == "CudaMalloc")
	{
		printf("No state to print.\n\n");
	}
	else if(m_method == "AtomicMalloc" || m_method == "AtomicWrapMalloc")
	{
		printStateAtomicMalloc();
	}
	else if(m_method == "CircularMalloc" || m_method == "CircularFusedMalloc" || m_method == "CircularMultiMalloc" || m_method == "CircularFusedMultiMalloc")
	{
		printStateCircularMalloc();
	}

	else if(m_method == "ScatterAlloc")
	{
		printf("No state to print.\n\n");
	}
}

//------------------------------------------------------------------------

void CudaAllocTest::printStateAtomicMalloc()
{
	char*& heapBase = *(char**)m_module->getGlobal("g_heapBase").getPtr();
	printf("Heap base %p\n", heapBase);

	U32& heapOffset = *(U32*)m_module->getGlobal("g_heapOffset").getPtr();
	printf("Heap offset %u\n\n", heapOffset);
}

//------------------------------------------------------------------------

void CudaAllocTest::printStateCircularMalloc()
{
	char*& heapBase = *(char**)m_module->getGlobal("g_heapBase").getPtr();
	printf("Heap base %p\n", heapBase);

	if(m_method == "CircularMalloc" || m_method == "CircularFusedMalloc")
	{
		U32& heapOffset = *(U32*)m_module->getGlobal("g_heapOffset").getPtr();
		printf("Heap offset %u\n", heapOffset);
	}
	else
	{
		U32*& heapMultiOffset = *(U32**)m_module->getGlobal("g_heapMultiOffset").getPtr();
		for(int i = 0; i < m_numSM; i++)
			printf("Heap offset[%d] %u\n", i, heapMultiOffset[i]);
	}

	S32& heapLock = *(S32*)m_module->getGlobal("g_heapLock").getPtr();
	printf("Heap lock %d\n\n", heapLock);
}

//------------------------------------------------------------------------

F32 CudaAllocTest::outputStatistics(const char* name, Array<F32>& samples)
{
	F32 minTime = FLT_MAX;
	F32 maxTime = -FLT_MAX;
	F64 mean = 0.f;
	F64 variance = 0.f;
	int numRepeats = samples.getSize();

	// Compute average
	for(int i = 0; i < numRepeats; i++)
	{
		minTime = min(minTime, samples[i]);
		maxTime = max(maxTime, samples[i]);
		mean += samples[i];
#ifndef BENCHMARK
		printf("Sample %d %s from %d runs = %f\n", i, name, numRepeats, samples[i]);
#endif
	}
	mean /= numRepeats;
	// Compute variance
	for(int i = 0; i < numRepeats; i++)
	{
		variance += (samples[i]-mean)*(samples[i]-mean);
	}
	variance /= numRepeats;

	// Screen output
	printf("\n");
	printf("Minimum %s from %d runs = %f\n", name, numRepeats, minTime);
	printf("Average %s from %d runs = %f\n", name, numRepeats, mean);
	printf("Maximum %s from %d runs = %f\n", name, numRepeats, maxTime);
	printf("Variance of %s from %d runs = %f\n", name, numRepeats, variance);
	printf("\n");

	return minTime;
}

//------------------------------------------------------------------------

void CudaAllocTest::initRandom()
{
	srand(0);

	int threadsPerBlock = Environment::GetSingleton()->GetInt("CudaAlloc.threadsPerBlock");
	int numBlocks = Environment::GetSingleton()->GetInt("CudaAlloc.numBlocks");
	int testIters = Environment::GetSingleton()->GetInt("CudaAlloc.testIters");
	F32 pAlloc = Environment::GetSingleton()->GetFloat("CudaAlloc.pAlloc");
	F32 pFree = Environment::GetSingleton()->GetFloat("CudaAlloc.pFree");

	S64 numRandom = (threadsPerBlock / WARP_SIZE) * numBlocks * testIters;
	m_random.resizeDiscard(numRandom * sizeof(F32));
	F32 *rnd = (F32*)m_random.getPtr();
	bool canAlloc = true;
	m_numAlloc = 0;
	m_numFree = 0;

	for(int i = 0; i < numRandom; i++)
	{
		if(i % testIters == 0)
			canAlloc = true;

		*rnd = (F32)rand() / (F32)RAND_MAX;
		
		if(canAlloc && *rnd > pAlloc)
		{
			m_numAlloc++;
			canAlloc = false;
		}
		else if(!canAlloc && *rnd > pFree)
		{
			m_numFree++;
			canAlloc = true;
		}

		rnd++;
	}

	// For these tests all threads in all iterations do allocations and deallocations
	if(m_test == "AllocDealloc" || m_test == "AllocCycleDealloc")
	{
		m_numAlloc = m_numFree = (U32)numRandom;
	}

#ifndef BENCHMARK
	printf("Number of allocations %u\nNumber of frees %u\n\n", m_numAlloc, m_numFree);
#endif
}

//------------------------------------------------------------------------

void CudaAllocTest::computeFragmentation(F32& fInt, F32& fExt)
{	
	int threadsPerBlock = Environment::GetSingleton()->GetInt("CudaAlloc.threadsPerBlock");
	int numBlocks = Environment::GetSingleton()->GetInt("CudaAlloc.numBlocks");

	int numThreads = threadsPerBlock * numBlocks;

	fInt = -1.f; // Fragmentation not measured
	fExt = -1.f; // Fragmentation not measured

	// Currently fragmentation is only measured for CircularMalloc and its variants
#if defined(CIRCULAR_MALLOC_CHECK_INTERNAL_FRAGMENTATION) || defined(CIRCULAR_MALLOC_CHECK_EXTERNAL_FRAGMENTATION)
	if(m_method != "CircularMalloc" && m_method != "CircularMallocFused" && m_method != "CircularMultiMalloc" && m_method != "CircularMultiMallocFused")
#endif
		return;

#ifdef CIRCULAR_MALLOC_CHECK_INTERNAL_FRAGMENTATION
	// Sum internal fragmentations over all threads
	F32* interFragSum = (F32*)m_interFragSum.getPtr();
	F32 sumInter = 0.f;
	for(int i = 0; i < numThreads; i++)
		sumInter += interFragSum[i];

	// Divide by number of allocations
	fInt = sumInter / m_numAlloc;
#endif

#ifdef CIRCULAR_MALLOC_CHECK_EXTERNAL_FRAGMENTATION
	// Walk the list of chunks
	// Sum free chunks and keep track of the maximal free chunk
	F32 maxFree = 0.f;
	F32 sumFree = 0.f;

	// Set the heapBase
	char* heapBase = (char*)m_mallocData.getPtr();

	U32 ofs = 0;
	U32 next;
	bool free;
	do
	{
		if(m_method == "CircularMalloc" || m_method == "CircularMultiMalloc")
		{
#ifndef CIRCULAR_MALLOC_DOUBLY_LINKED
			next = *(U32*)(&heapBase[ofs+sizeof(U32)]);
#else
			next = *(U32*)(&heapBase[ofs+2*sizeof(U32)]);
#endif
			free = (*(AllocatorLockType*)(&heapBase[ofs]) == AllocatorLockType_Free);
		}
		else
		{
			U32 header = *(U32*)(&heapBase[ofs]);
			const U32 flagMask = 0x80000000u;
			next = header & (~flagMask);
			free = (header & flagMask) == 0;
		}

		if(free)
		{
			U32 csize = next - ofs;
			maxFree = max<F32>(maxFree, csize);
			sumFree += csize;
		}

		ofs = next;
	} while(ofs != 0); // Until wrap

	// Divide maximal free chunk by the sum of free space
	if(sumFree == 0.f)
		fExt = 0.f;
	else
		fExt = 1.f - maxFree / sumFree;
#endif
}

//------------------------------------------------------------------------

void FW::init(void)
{
	CudaAllocTest allocTest;
	allocTest.measure();
}

//------------------------------------------------------------------------