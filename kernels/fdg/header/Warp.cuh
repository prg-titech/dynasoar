#ifndef __FDGMALLOC_WARP_CUH
#define __FDGMALLOC_WARP_CUH

/*!	\class		Warp
 *	\brief		Public WarpHeader interface for FDGMalloc
 *	\author		Sven Widmer <sven.widmer@gris.informatik.tu-darmstadt.de>
 *	\author		Dominik Wodniok <dominik.wodniok@gris.informatik.tu-darmstadt.de>
 *	\author		Nicolas Weber <nicolas.weber@gris.informatik.tu-darmstadt.de>
 *	\author		Michael Goesele <michael.goesele@gris.informatik.tu-darmstadt.de>
 *	\version	1.0
 *	\date		06-12-2012 
 */
class __align__(128) Warp {
private:
	/*! stores the peak memory consumption for this Warp. */
	uint32_t		m_peak;

	/*! stores the count of threads participating. */
	uint32_t		m_count;

	/*! stores the count of threads that have not executed Warp::end(). */
	uint32_t		m_active;

	/*! stores the current List. */
	List_t*			m_list;

	/*! stores a pointer to the current SuperBlock. */
	SuperBlock_t*	m_superBlock;

	/*! register used to exchange allocation requests. */
	uint32_t		m_request[FDG__WARPSIZE];

	/*!	\brief						Initializes this WarpHeader.
	 *	\param[in]					Count of participating threads. */
	__device__ void				init				(uint32_t count);

	/*! \brief						Allocates a new List.
	 *	\return						Returns pointer to new List. */
	__device__ List_t*			allocateList		(void);

	/*! \brief						Allocates a new SuperBlock.
	 *	\return						Pointer to new SuperBlock.
	 *	\param[in]	size			Size to allocate. */
	__device__ void*			allocateSuperBlock	(const uint32_t size);

	/*! \brief						Appends a pointer to the List
	 *	\return						True if append was successful.
	 *	\param[in]	ptr				Pointer that shall be stored in the List.
	 *	\param[in]	performVoting	Determines if there shall be a voting for a worker thread for this operation */
	__device__ bool				appendToList		(void* ptr, bool performVoting = true);
	
	/*!	\brief						Static method to exchange pointer between threads.
	 *								For compute capability >= 3.0 we use __shfl, for all other we use shared memory.
	 *	\return						Exchanged pointer.
	 *	\param[in]	ptr				Pointer that shall be exchanged.
	 *	\param[in]	workerId		Id of worker thread.
	 *	\param[in]	od				Id of current thread. */
	__device__ static void*		exchangePointer		(void* ptr, const uint8_t workerId, const uint8_t id);

	/*!	\brief						Static method to perform a voting to determine a worker thread
	 *	\return						True if this thread is the worker thread.
	 *	\param[out]	workerId		Id of worker thread.
	 *	\param[out]	count			Count of participating threads. */
	__device__ static bool isWorkerThread(uint8_t* workerId = 0, uint32_t* count = 0);

public:
	/*!	\brief						Allocates a new WarpHeader and returns a pointer to it.
	 *	\return						Pointer to WarpHeader.
	 *	\param[in]	count			Count of participating Threads. If count == 0, the value will be determined
	 *								automatically. */
	__device__ static Warp* start(uint32_t count = 0);

	/*!	\brief						Allocates memory.
	 *	\return						Pointer to allocated memory.
	 *	\param[in] size				Count of Bytes to allocate. */
	__device__ void* alloc(const uint32_t size);

	/*!	\brief						Frees all memory that was allocated by this warp, as well as the WarpHeader. */
	__device__ void end(void);

	/*!	\brief						Cleans all memory that was allocated but keeps the WarpHeader intact. */
	__device__ void tidyUp(void);

	/*!	\brief						Returns the current amount of memory that has been allocated by this WarpHeader.
	 *	\return						Memory amount. */
	__device__ inline uint32_t getPeak(void);
};

#endif