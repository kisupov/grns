/*
 *  Common useful routines and macros
 */

#ifndef GRNS_COMMON_CUH
#define GRNS_COMMON_CUH

#include <stdio.h>
#include <string>
#include <sstream>
#include <iostream>

/*
 * Macros that define inline specifiers for gcc and nvcc
 */
#define GCC_FORCEINLINE __attribute__((always_inline)) inline
#define DEVICE_CUDA_FORCEINLINE __device__ __forceinline__

/*
 * Checking CUDA results
 */
#define checkDeviceHasErrors(ans) { gpuAssert((ans), __FILE__, __LINE__); }

inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = false) {
    if (code != cudaSuccess) {
        fprintf(stderr, "%s %s:%d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

#define cudaCheckErrors() {                                        \
 cudaError_t e=cudaGetLastError();                                 \
 if(e!=cudaSuccess) {                                              \
   printf("CUDA Runtime Error %s:%d: '%s'\n",__FILE__,__LINE__,cudaGetErrorString(e)); \
   exit(0); \
 }                                                                 \
}

#endif //GRNS_COMMON_CUH