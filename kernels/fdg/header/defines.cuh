#ifndef __FDGMALLOC_DEFINES_CUH
#define __FDGMALLOC_DEFINES_CUH

/*!	\file		defines.cuh
 *	\brief		Defines required preprocessor definitions.
 *	\author		Sven Widmer <sven.widmer@gris.informatik.tu-darmstadt.de>
 *	\author		Dominik Wodniok <dominik.wodniok@gris.informatik.tu-darmstadt.de>
 *	\author		Nicolas Weber <nicolas.weber@gris.informatik.tu-darmstadt.de>
 *	\author		Michael Goesele <michael.goesele@gris.informatik.tu-darmstadt.de>
 *	\version	1.0
 *	\date		06-12-2012 
 */

#define FDG__WARPSIZE 32

#define FDG__MALLOC(__SIZE) malloc(__SIZE)
#define FDG__FREE(__PTR) free(__PTR)

#define FDG__THREADID				(threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * (blockDim.x * blockDim.y))
#define FDG__BLOCKID				(blockIdx.x  + blockIdx.y  * gridDim.x  + blockIdx.z  * (gridDim.x  * gridDim.y ))
#define FDG__THREADIDINWARP			(FDG__THREADID % FDG__WARPSIZE)
#define FDG__PSEUDOWARPID			(FDG__BLOCKID * (blockDim.x * blockDim.y * blockDim.z) + FDG__THREADID) / FDG__WARPSIZE)
#define FDG__PSEUDOWARPIDINBLOCK	(FDG__THREADID / FDG__WARPSIZE)

#define FDG__bfind(out, in)			asm("bfind.u32 %0, %1;" : "=r"(out) : "r"(in))
#define FDG__clz(out, in)			asm("clz.b32 %0, %1;" : "=r"(out) : "r"(in));

#endif