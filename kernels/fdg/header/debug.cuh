#ifndef __FDGMALLOC_DEBUG_CUH
#define __FDGMALLOC_DEBUG_CUH

#define FDG__ERROR(cond,S)
#define FDG__LOG(S)
#define FDG__WARNING(s)

#ifdef FDG__DEBUG_ERROR
	#undef FDG__ERROR
	#define FDG__ERROR(cond,s) if(cond) { printf("W:%3i T:%2i Error: %s\n",FDG__PSEUDOWARPID,FDG__THREADIDINWARP,s); assert(0); }
#endif

#ifdef FDG__DEBUG_LOG
	#undef FDG__LOG
	#define FDG__LOG(s)	printf("W:%3i T:%2i Log:     %s\n",FDG__PSEUDOWARPID,FDG__THREADIDINWARP,s)
#endif

#ifdef FDG__DEBUG_WARNING
	#undef FDG__WARNING
	#define FDG__WARNING(s) printf("W:%3i T:%2i Warn:    %s\n",FDG__PSEUDOWARPID,FDG__THREADIDINWARP,s)
#endif

#ifdef FDG_DEBUG
	#pragma message("WARNING: You should deactivate FDG_DEBUG in a productive environment.")
#endif

#endif