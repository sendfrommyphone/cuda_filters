#include<cuda_runtime.h>
#include<device_launch_parameters.h>
#include<cmath>
#include<iostream>

namespace Canny {

    // 辅助函数：计算梯度方向
    __device__ float computeGradientDirection(float gx, float gy);

    // 辅助函数：非极大值抑制
    __global__ void nonMaxSuppression(float* input, float* grad_x, float* grad_y, float* output, int width, int height);

    // 辅助函数：双阈值处理
    __global__ void doubleThreshold(float* input, float* output, int width, int height, float lowThreshold, float highThreshold);

    // 辅助函数：边缘连接
    __global__ void edgeTrackingByHysteresis(float* input, float* output, int width, int height);

    // Canny 边缘检测计算函数
    cudaError_t cannyEdgeDetection(float* input, float* output, int width, int height, float lowThreshold, float highThreshold);

}