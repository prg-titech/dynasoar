#ifndef __FDGMALLOC_SUPERBLOCK_CUH
#define __FDGMALLOC_SUPERBLOCK_CUH

/*!	\class		SuperBlock
 *	\brief		Implements a basic SuperBlock structure with minimal header.
 *	\author		Sven Widmer <sven.widmer@gris.informatik.tu-darmstadt.de>
 *	\author		Dominik Wodniok <dominik.wodniok@gris.informatik.tu-darmstadt.de>
 *	\author		Nicolas Weber <nicolas.weber@gris.informatik.tu-darmstadt.de>
 *	\author		Michael Goesele <michael.goesele@gris.informatik.tu-darmstadt.de>
 *	\version	1.0
 *	\date		06-12-2012 
 */
template<size_t SIZE>
class __align__(2048) SuperBlock {
private:
	/*! SuperBlock storrage */
	uint8_t items[SIZE];
	
	/*! index of the next unused Byte */
	uint32_t current;
	
public:
	/*! \brief		Initializes the SuperBlock */
	__device__ void		init(void);
	
	/*!	\brief		Allocates memory inside the SuperBlock
	 *	\return		A pointer if the request was successful. Otherwise 0 is returned.
	 *	\param[in]	size		Allocation size in Bytes.
	 *	\param[in]	offset		Offset for this thread in Bytes.
	 *	\param[in]	workerId	WorkerId for this Warp.
	 *	\param[in]	id			Id of this thread. */
	__device__ void*	alloc(const uint32_t size, const uint32_t offset, const uint8_t workerId, const uint8_t id);
};

/*! \typedef	SuperBlock_t
 *	\brief		Typedef for a SuperBlock with default size. */
typedef SuperBlock<FDG__SUPERBLOCK_SIZE> SuperBlock_t;

#endif
