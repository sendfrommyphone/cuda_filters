#pragma once
#include <cuda_runtime.h>
#include"device_launch_parameters.h"
#include<cmath>
#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>
namespace Laplacian {
    // �ݶȼ����ں�
    __global__ void laplacianKernel(float* input,
        float* output,
        int width,
        int height);

    // �����������̷�װ
    cudaError_t compute(float* input,
        float* output,   // �����ݶȷ���
        int width,
        int height);
}