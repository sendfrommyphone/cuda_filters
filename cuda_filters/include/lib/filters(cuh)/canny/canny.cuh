#include<cuda_runtime.h>
#include<device_launch_parameters.h>
#include<cmath>
#include<iostream>

namespace Canny {

    // ���������������ݶȷ���
    __device__ float computeGradientDirection(float gx, float gy);

    // �����������Ǽ���ֵ����
    __global__ void nonMaxSuppression(float* input, float* grad_x, float* grad_y, float* output, int width, int height);

    // ����������˫��ֵ����
    __global__ void doubleThreshold(float* input, float* output, int width, int height, float lowThreshold, float highThreshold);

    // ������������Ե����
    __global__ void edgeTrackingByHysteresis(float* input, float* output, int width, int height);

    // Canny ��Ե�����㺯��
    cudaError_t cannyEdgeDetection(float* input, float* output, int width, int height, float lowThreshold, float highThreshold);

}