#ifndef __FDGMALLOC_CU
#define __FDGMALLOC_CU

/*!	\file		FDGMalloc.cu
 *	\brief		Includes the implementation of FDGMalloc.
 *	\author		Sven Widmer <sven.widmer@gris.informatik.tu-darmstadt.de>
 *	\author		Dominik Wodniok <dominik.wodniok@gris.informatik.tu-darmstadt.de>
 *	\author		Nicolas Weber <nicolas.weber@gris.informatik.tu-darmstadt.de>
 *	\author		Michael Goesele <michael.goesele@gris.informatik.tu-darmstadt.de>
 *	\version	1.0
 *	\date		06-12-2012 
 */

namespace FDG {
	#include "src/SuperBlock.cu"
	#include "src/List.cu"
	#include "src/Warp.cu"
}

#endif