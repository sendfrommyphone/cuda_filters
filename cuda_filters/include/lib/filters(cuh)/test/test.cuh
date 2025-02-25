#ifndef TEST_CUH
#define TEST_CUH

#include <cuda_runtime.h>

namespace test {
    // º¯ÊıÉùÃ÷
    cudaError_t testAllOperators(unsigned char* input_image,
        unsigned char* output_image,
        int width, int height,
        float lowThreshold, float highThreshold);
}

#endif