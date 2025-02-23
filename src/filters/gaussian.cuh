#pragma once
#include <cuda_runtime.h>

namespace Gaussian {
    __global__ void kernel(float* input, float* output, int width, int height);
    
    cudaError_t launch(float* input, float* output, 
                      int width, int height, 
                      dim3 gridDim, dim3 blockDim);
}
