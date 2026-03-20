#pragma once

#include <cstdio>
#include <cstdlib>

#ifdef CUDA_MS_HAVE_CUDA
#  include <cuda_runtime.h>

#  define CUDA_CHECK(call)                                                      \
     do {                                                                       \
         cudaError_t _err = (call);                                             \
         if (_err != cudaSuccess) {                                             \
             fprintf(stderr, "CUDA error at %s:%d — %s\n",                     \
                     __FILE__, __LINE__, cudaGetErrorString(_err));             \
             exit(EXIT_FAILURE);                                                \
         }                                                                      \
     } while (0)

inline bool cuda_device_available() {
    int count = 0;
    return cudaGetDeviceCount(&count) == cudaSuccess && count > 0;
}

inline void print_device_info() {
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess || count == 0) {
        printf("  (no CUDA devices)\n");
        return;
    }
    for (int i = 0; i < count; ++i) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);
        printf("  [%d] %s  (SM %d.%d, %.0f MB)\n",
               i, prop.name, prop.major, prop.minor,
               prop.totalGlobalMem / (1024.0 * 1024.0));
    }
}

#else

#  define CUDA_CHECK(call) ((void)0)

inline bool cuda_device_available() { return false; }
inline void print_device_info() { printf("  (CUDA not compiled)\n"); }

#endif
