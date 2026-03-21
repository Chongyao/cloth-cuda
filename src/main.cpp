#include "utils/cuda_helper.h"
#include <cstdio>

int main(int /*argc*/, char* /*argv*/[]) {
    printf("=== CUDA Devices ===\n");
    print_device_info();
    printf("\ncuda-ms: simulation not yet implemented.\n");
    return 0;
}
