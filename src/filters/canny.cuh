#pragma once
#include <cuda_runtime.h>

namespace Canny {
    // 非极大值抑制
    __global__ void nonMaxSuppression(float* grad_x, float* grad_y, 
                                    float* output, int width, int height);
    
    // 双阈值处理
    __global__ void edgeTracking(float* img, int width, int height,
                                float low, float high);
    
    cudaError_t fullPipeline(float* input, float* output,
                            int width, int height,
                            float lowThresh, float highThresh);
}
