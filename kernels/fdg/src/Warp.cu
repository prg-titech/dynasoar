#ifndef __FDGMALLOC_WARP_CU
#define __FDGMALLOC_WARP_CU

/*!	\file		Warp.cu
 *	\author		Sven Widmer <sven.widmer@gris.informatik.tu-darmstadt.de>
 *	\author		Dominik Wodniok <dominik.wodniok@gris.informatik.tu-darmstadt.de>
 *	\author		Nicolas Weber <nicolas.weber@gris.informatik.tu-darmstadt.de>
 *	\author		Michael Goesele <michael.goesele@gris.informatik.tu-darmstadt.de>
 *	\version	1.0
 *	\date		06-12-2012 
 */

//#if __CUDA_ARCH__ < 300
#define FDG__USE_SHARED
__shared__ void* sharedPtr[FDG__WARPSIZE];
//#endif

//-----------------------------------------------------------------------------
__device__ bool Warp::isWorkerThread(uint8_t* workerId, uint32_t* count) {
	const uint8_t id = FDG__THREADIDINWARP;

	// perform voting
	uint32_t vote = __ballot(1);
	uint32_t wId;

	// find worker thread
	FDG__bfind(wId, vote);

	// check if we have to set a count depending on the participating threads
	if(count != 0) {
		FDG__clz(*count, vote);
		*count = FDG__WARPSIZE - *count;
	}

	// return worker id
	if(workerId != 0) {
		*workerId = (uint8_t)wId;
	}

	return id == wId;
}

//-----------------------------------------------------------------------------
__device__ Warp* Warp::start(uint32_t count) {
	// init variables
	Warp* warp = 0;

#ifdef COALESCE_WARP
	// determine worker thread
	const uint8_t id = FDG__THREADIDINWARP;
	uint8_t workerId;
	isWorkerThread(&workerId, count == 0 ? &count : 0);

	// allocate WarpHeader
	if(workerId == id) {
		warp = (Warp*)FDG__MALLOC(sizeof(Warp));
		
		// init if allocation was successful
		if(warp != 0)	
			warp->init(count);
	}
	
	// exchange WarpHeader to other threads
	warp = (Warp*)exchangePointer(warp, workerId, id);
#else
	warp = (Warp*)FDG__MALLOC(sizeof(Warp));
	// init if allocation was successful
	if(warp != 0)	
		warp->init(1);
#endif

	return warp;
}

//-----------------------------------------------------------------------------
__device__ List_t * Warp::allocateList(void) {
	// allocate a list
	List_t* ptr = (List_t*)FDG__MALLOC(sizeof(List_t));
		
	// check if the list was allocated
	if(ptr == 0)
		return 0;	
			
	// init list
	ptr->init();

	// return pointer
	return ptr;
}

//-----------------------------------------------------------------------------
__device__ void* Warp::allocateSuperBlock(const uint32_t size) {
	// allocate memory
	void* ptr = FDG__MALLOC(size);
		
	// check if the allocation was succesful
	if(ptr == 0)
		return 0;

	// add size to peak
	m_peak += size;

	// try to append to list
	if(!appendToList(ptr, false)) {
		FDG__FREE(ptr);
		ptr = 0;
	}

	// return pointer
	return ptr;
}

//-----------------------------------------------------------------------------
__device__ uint32_t Warp::getPeak(void) {
	return m_peak;
}

//-----------------------------------------------------------------------------
__device__ void Warp::init(const uint32_t count) {
	m_list			= 0;
	m_superBlock	= 0;
	m_peak			= sizeof(Warp);

	m_count  = count;
	m_active = count;

	m_request[ 0] = 0; m_request[ 1] = 0;
	m_request[ 2] = 0; m_request[ 3] = 0; 
	m_request[ 4] = 0; m_request[ 5] = 0;
	m_request[ 6] = 0; m_request[ 7] = 0;
	m_request[ 8] = 0; m_request[ 9] = 0; 
	m_request[10] = 0; m_request[11] = 0;
	m_request[12] = 0; m_request[13] = 0;
	m_request[14] = 0; m_request[15] = 0;
	m_request[16] = 0; m_request[17] = 0;
	m_request[18] = 0; m_request[19] = 0;
	m_request[20] = 0; m_request[21] = 0;
	m_request[22] = 0; m_request[23] = 0;
	m_request[24] = 0; m_request[25] = 0;
	m_request[26] = 0; m_request[27] = 0;
	m_request[28] = 0; m_request[29] = 0;
	m_request[30] = 0; m_request[31] = 0;
}

//-----------------------------------------------------------------------------
__device__ void Warp::end(void) {
	const uint32_t remaining = atomicSub(&m_active,1);

	if(remaining == 1) {
		if(m_list != 0) {
			List_t* list = m_list;
			m_list = 0;

			while(list != 0) {
				list = list->freeList();
			}
		}

		FDG__FREE(this);
	}
}

//-----------------------------------------------------------------------------
__device__ void Warp::tidyUp(void) {
	const uint32_t remaining = atomicSub(&m_active,1);

	if(remaining == 1) {
		if(m_list != 0) {
			List_t* list = m_list;
			
			while(list != 0) {
				list = list->freeList(list != m_list);
			}
		}

		m_list->init();
		m_superBlock	= 0;
		m_peak			= sizeof(Warp);
		m_active		= m_count;
	}
}

//-----------------------------------------------------------------------------
__device__ bool Warp::appendToList(void* ptr, bool performVoting) {
	// check if we can add this ptr to the list.
	if(m_list == 0 || !m_list->append(ptr)) {
		// check if this is the worker thread
#ifdef COALESCE_WARP
		if(!performVoting || isWorkerThread()) 
#endif
		{
			// allocate new list
			List_t* newList = allocateList();

			// check if the list was allocated
			if(newList == 0)
				return false;

			// set links
			newList->setPrevious(m_list);
			m_list = newList;
		}

		// append ptr to list
		m_list->append(ptr);
	}

	return true;
}

//-----------------------------------------------------------------------------
__device__ void* Warp::exchangePointer(void* ptr, const uint8_t workerId, const uint8_t id) {
	// we can only use shfl if CC is 3.0 or higher and if we are on a 32 bit system.
	// shfl only supports 4 Byte values
	#ifndef FDG__USE_SHARED
		#if defined(_M_X64) || defined(__amd64__)
			uint64_t ptr64 = (uint64_t)ptr;
			ptr64 = ((uint64_t)__shfl((int32_t)(ptr64 >> 32),	(uint32_t)workerId, FDG__WARPSIZE)) << 32;
			ptr64 |= (uint64_t)__shfl((int32_t)ptr,				(uint32_t)workerId, FDG__WARPSIZE);

			ptr = (void*)ptr64;
		#else
			ptr = (void*)__shfl((int32_t)ptr, (uint32_t)workerId, FDG__WARPSIZE);
		#endif
	#else
		if(workerId == id)
			sharedPtr[FDG__PSEUDOWARPIDINBLOCK] = ptr;

		ptr = sharedPtr[FDG__PSEUDOWARPIDINBLOCK];
	#endif

	return ptr;
}

//-----------------------------------------------------------------------------
__device__ void* Warp::alloc(const uint32_t size) {
#ifdef COALESCE_WARP
	// init vars
	const uint8_t id	= FDG__THREADIDINWARP;
	uint8_t workerId	= 0;
	void* ptr			= 0;

	// determine a worker, we will need it later
	isWorkerThread(&workerId);

	// calculate required blocks
	m_request[id] = size;
	
	// enforce alignment
	if(m_request[id] % FDG__MIN_ALLOC_SIZE != 0) {
		m_request[id] += FDG__MIN_ALLOC_SIZE - (m_request[id] % FDG__MIN_ALLOC_SIZE);
	}

	// calculate prefix sum
	uint32_t offset = 0;
	uint32_t total	= 0;
	for(uint32_t i = 0; i < FDG__WARPSIZE; i++) {
		if(i < id)
			offset += m_request[i];

		total += m_request[i];
	}

	// is the total request bigger than a superblock?
	if(total >= FDG__SUPERBLOCK_SIZE) {
		uint8_t* data = 0;

		// let worker get a new data chunk
		if(workerId == id)
			data = (uint8_t*)allocateSuperBlock(total);

		// exchange pointer
		data = (uint8_t*)exchangePointer(data, workerId, id);

		// get pointer if request was successful
		if(data != 0)
			ptr = (void*)&data[offset];
	} else {
		// check if the first superBlock has been allocated
		if(m_superBlock == 0) {
			if(workerId == id) {
				m_superBlock = (SuperBlock_t*)allocateSuperBlock(sizeof(SuperBlock_t));

				if(m_superBlock != 0)
					m_superBlock->init();
			}
		}

		// repeat until we have sufficed all requests.
		do {
			// check if the allocation was successful
			if(m_superBlock == 0)
				return 0;

			// allocate memory inside super block
			ptr = m_superBlock->alloc(m_request[id], offset, workerId, id);

			// check if allocation was successful
			if(ptr == 0) {
				// the worker is always the last one, so we do not need to determine a new 
				// worker here.
				if(workerId == id) {
					m_superBlock = (SuperBlock_t*)allocateSuperBlock(sizeof(SuperBlock_t));

					if(m_superBlock != 0)
						m_superBlock->init();
				}
			}
		} while(ptr == 0);
	}
#else
	void* ptr			= 0;
	uint32_t total	= size;
	// enforce alignment
	if(total % FDG__MIN_ALLOC_SIZE != 0) {
		total += FDG__MIN_ALLOC_SIZE - (total % FDG__MIN_ALLOC_SIZE);
	}
	m_request[0] = total;

	// is the total request bigger than a superblock?
	if(total >= FDG__SUPERBLOCK_SIZE) {
		uint8_t* data = 0;

		// let worker get a new data chunk
		data = (uint8_t*)allocateSuperBlock(total);

		// get pointer if request was successful
		if(data != 0)
			ptr = (void*)&data;
	} else {
		// check if the first superBlock has been allocated
		if(m_superBlock == 0) {
			m_superBlock = (SuperBlock_t*)allocateSuperBlock(sizeof(SuperBlock_t));

			if(m_superBlock != 0)
				m_superBlock->init();
		}

		// repeat until we have sufficed all requests.
		do {
			// check if the allocation was successful
			if(m_superBlock == 0)
				return 0;

			// allocate memory inside super block
			ptr = m_superBlock->alloc(total, 0, 0, 0);

			// check if allocation was successful
			if(ptr == 0) {
				m_superBlock = (SuperBlock_t*)allocateSuperBlock(sizeof(SuperBlock_t));

				if(m_superBlock != 0)
					m_superBlock->init();
			}
		} while(ptr == 0);
	}
#endif
	
	return ptr;
}

#endif