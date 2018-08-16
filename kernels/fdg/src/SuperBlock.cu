#ifndef __FDGMALLOC_SUPERBLOCK_CU
#define __FDGMALLOC_SUPERBLOCK_CU

/*!	\file		SuperBlock.cu
 *	\author		Sven Widmer <sven.widmer@gris.informatik.tu-darmstadt.de>
 *	\author		Dominik Wodniok <dominik.wodniok@gris.informatik.tu-darmstadt.de>
 *	\author		Nicolas Weber <nicolas.weber@gris.informatik.tu-darmstadt.de>
 *	\author		Michael Goesele <michael.goesele@gris.informatik.tu-darmstadt.de>
 *	\version	1.0
 *	\date		06-12-2012 
 */

//-----------------------------------------------------------------------------
template<size_t SIZE>
__device__ void SuperBlock<SIZE>::init(void) {
	current = 0;
}

//-----------------------------------------------------------------------------
template<size_t SIZE>
__device__ void* SuperBlock<SIZE>::alloc(const uint32_t size, const uint32_t offset, const uint8_t workerId, const uint8_t id) {
	const uint32_t pos 	= current + offset;

	if((pos + size) > (SIZE))
		return 0;

	if(id == workerId) {
		current = pos + size;	
	}

	return (void*)&items[pos];
}

#endif