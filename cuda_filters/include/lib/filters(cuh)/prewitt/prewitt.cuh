#pragma once
#include <cuda_runtime.h>
#include"device_launch_parameters.h"
#include<cmath>
#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>
namespace Prewitt {
    // �ݶȼ����ں�
    __global__ void gradientKernel(float* input,
        float* output,
        float* grad_x,
        float* grad_y,
        int width,
        int height);

    // �����������̷�װ
    cudaError_t compute(float* input,
        float* output,
        float* grad_x,   // �����ݶȷ���
        float* grad_y,
        int width,
        int height);
}