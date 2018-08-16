#ifndef __FDGMALLOC_LIST_CUH
#define __FDGMALLOC_LIST_CUH

/*!	\class		List
 *	\brief		Implements a list to store allocated pointer. Lists can be linked together.
 *	\author		Sven Widmer <sven.widmer@gris.informatik.tu-darmstadt.de>
 *	\author		Dominik Wodniok <dominik.wodniok@gris.informatik.tu-darmstadt.de>
 *	\author		Nicolas Weber <nicolas.weber@gris.informatik.tu-darmstadt.de>
 *	\author		Michael Goesele <michael.goesele@gris.informatik.tu-darmstadt.de>
 *	\version	1.0
 *	\date		06-12-2012 
 */
template<uint32_t SIZE>
class __align__(128) List {
private:
	/*! stores the index of the next empty item */
	uint32_t	m_current;

	/*! stores a pointer to the previous list */
	List<SIZE>*	m_previous;

	/*! storage for pointer */
	void*		m_items[SIZE];

public:
	/*!	\brief	Initializes the List */
	__device__ void init		(void);

	/*! \brief					Appends a pointer to the List 
	 *	\return					Returns true, except the List is full
	 *	\param[in]	ptr			Pointer to store. */
	__device__ bool append		(void* ptr);

	/*!	\brief					Frees all stored pointers.
	 *	\return					Pointer to previous List.
	 *	\param[in]	freeList	Determines if the list shall be freed as well. */
	__device__ List<SIZE>* freeList	(const bool freeList = true);

	/*!	\brief					Sets the pointer to the previous List
	 *	\param[in]	list		Pointer to previous List. */
	__device__ void setPrevious	(List<SIZE>* list);
};

/*! \typedef	List_t
 *	\brief		Typedef for a List with default size. */
typedef List<FDG__LIST_SIZE> List_t;

#endif