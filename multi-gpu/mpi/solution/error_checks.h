// This header provides two helper macros for error checking
// See the exercise skeletons and answers for usage examples.

#ifndef COURSE_UTIL_H_
#define COURSE_UTIL_H_

#include <cstdio>
#include <cstdlib>

#define HIP_CHECK(errarg)   __checkErrorFunc(errarg, __FILE__, __LINE__)
#define CHECK_ERROR_MSG(errstr) __checkErrMsgFunc(errstr, __FILE__, __LINE__)

inline void __checkErrorFunc(hipError_t errarg, const char* file, 
			     const int line)
{
    if(errarg) {
	fprintf(stderr, "Error at %s(%i)\n", file, line);
	exit(EXIT_FAILURE);
    }
}


inline void __checkErrMsgFunc(const char* errstr, const char* file, 
			      const int line)
{
    hipError_t err = hipGetLastError();
    if(err != hipSuccess) {
	fprintf(stderr, "Error: %s at %s(%i): %s\n", 
		errstr, file, line, hipGetErrorString(err));
	exit(EXIT_FAILURE);
    }
}

#endif