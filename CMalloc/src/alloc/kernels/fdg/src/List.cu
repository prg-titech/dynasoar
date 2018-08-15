#ifndef __FDGMALLOC_LIST_CU
#define __FDGMALLOC_LIST_CU

/*!	\file		List.cu
 *	\author		Sven Widmer <sven.widmer@gris.informatik.tu-darmstadt.de>
 *	\author		Dominik Wodniok <dominik.wodniok@gris.informatik.tu-darmstadt.de>
 *	\author		Nicolas Weber <nicolas.weber@gris.informatik.tu-darmstadt.de>
 *	\author		Michael Goesele <michael.goesele@gris.informatik.tu-darmstadt.de>
 *	\version	1.0
 *	\date		06-12-2012 
 */

//-----------------------------------------------------------------------------
template<uint32_t SIZE>
__device__ void List<SIZE>::init(void) {
	m_previous	= 0;
	m_current	= 0;
}

//-----------------------------------------------------------------------------
template<uint32_t SIZE>
__device__ bool List<SIZE>::append(void* ptr) {
	// check if List is full
	if(m_current >= SIZE) { 
		return false;
	} else {
		// set pointer
		m_items[m_current] = ptr;

		// increase counter
		m_current++;

		return true;
	}
}

//-----------------------------------------------------------------------------
template<uint32_t SIZE>
__device__ List<SIZE>* List<SIZE>::freeList(const bool freeList) {
	// determine max element
	const uint32_t max = (m_current > SIZE ? SIZE : m_current);

	// free each element
	for(uint32_t i = 0; i < max; i++) {
		FDG__FREE(m_items[i]);
	}

	// get previous list pointer so we can return it.
	List<SIZE>* previous = m_previous;

	// free this List
	if(freeList)
		FDG__FREE(this);

	return previous;
}

//-----------------------------------------------------------------------------
template<uint32_t SIZE>
__device__ void List<SIZE>::setPrevious(List<SIZE>* list) {
	m_previous = list;
}

#endif