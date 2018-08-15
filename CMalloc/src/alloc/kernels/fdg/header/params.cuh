#ifndef __FDGMALLOC_PARAMS_CUH
#define __FDGMALLOC_PARAMS_CUH

/*!	\file		params.cuh
 *	\brief		Defines adjustable parameters for FDGMalloc.
 *	\author		Sven Widmer <sven.widmer@gris.informatik.tu-darmstadt.de>
 *	\author		Dominik Wodniok <dominik.wodniok@gris.informatik.tu-darmstadt.de>
 *	\author		Nicolas Weber <nicolas.weber@gris.informatik.tu-darmstadt.de>
 *	\author		Michael Goesele <michael.goesele@gris.informatik.tu-darmstadt.de>
 *	\version	1.0
 *	\date		06-12-2012 
 */

/*!	\define FDG__LIST_SIZE
 *	\brief	Number of entries in a List */
#define FDG__LIST_SIZE			126

/*!	\define	FDG__MIN_ALLOC_SIZE
 *	\brief	Minimum allocation size if Bytes. Each allocation size will be a multiple of
 *			this size to ensure alignment. */
#define FDG__MIN_ALLOC_SIZE		16

/*!	\define	FDG__SUPERBLOCK_SIZE
 *	\brief	Size of a SuperBlock in Bytes. */
#define FDG__SUPERBLOCK_SIZE	8188

/*!	\define	FDG__WARPSIZE
 *	\brief	Number of threads per warp. Default for CUDA is 32. */
#define FDG__WARPSIZE 32

#endif