#include "lib/filters(cuh)/gaussian/gaussian.cuh"
#include "lib/filters(cuh)/prewitt/prewitt.cuh"
#include<src\filters(cu)\gaussian\gaussian.cu>
#include<src/filters(cu)/prewitt/prewitt.cu>
#include <stdio.h>
#include <cmath>

#define M_PI 3.1415926


__device__ float computeGradientDirection(float gx, float gy) {
    return atan2f(gy, gx) * 180.0f / M_PI; 
}


__global__ void nonMaxSuppression(float* input, float* grad_x, float* grad_y, float* output, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;


    if (x <= 0 || x >= height - 1 || y <= 0 || y >= width - 1) {
        return;
    }


    float angle = computeGradientDirection(grad_x[y * width + x], grad_y[y * width + x]);


    if ((angle >= -22.5f && angle < 22.5f) || (angle >= 157.5f || angle < -157.5f)) {

        if (input[y * width + x] < input[y * width + x - 1] || input[y * width + x] < input[y * width + x + 1]) {
            output[y * width + x] = 0;
        }
    }
    else if ((angle >= 22.5f && angle < 67.5f) || (angle >= -157.5f && angle < -112.5f)) {

        if (input[y * width + x] < input[(y - 1) * width + x + 1] || input[y * width + x] < input[(y + 1) * width + x - 1]) {
            output[y * width + x] = 0;
        }
    }
    else if ((angle >= 67.5f && angle < 112.5f) || (angle >= -112.5f && angle < -67.5f)) {
     
        if (input[y * width + x] < input[(y - 1) * width + x] || input[y * width + x] < input[(y + 1) * width + x]) {
            output[y * width + x] = 0;
        }
    }
    else {
        
        if (input[y * width + x] < input[(y - 1) * width + x - 1] || input[y * width + x] < input[(y + 1) * width + x + 1]) {
            output[y * width + x] = 0;
        }
    }
}


__global__ void doubleThreshold(float* input, float* output, int width, int height, float lowThreshold, float highThreshold) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

   
    if (x <= 0 || x >= height - 1 || y <= 0 || y >= width - 1) {
        return;
    }

    
    float value = input[y * width + x];

   
    if (value > highThreshold) {
        output[y * width + x] = 255;
    }
    
    else if (value < lowThreshold) {
        output[y * width + x] = 0;
    }
    
    else {
        output[y * width + x] = 128;
    }
}


__global__ void edgeTrackingByHysteresis(float* input, float* output, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

   
    if (x <= 0 || x >= height - 1 || y <= 0 || y >= width - 1) {
        return;
    }

    
    if (input[y * width + x] == 255) {
        output[y * width + x] = 255;
    }
    
    else if (input[y * width + x] == 128) {
        if (input[(y - 1) * width + x] == 255 || input[(y + 1) * width + x] == 255 ||
            input[y * width + x - 1] == 255 || input[y * width + x + 1] == 255) {
            output[y * width + x] = 255;
        }
        else {
            output[y * width + x] = 0;
        }
    }
    else {
        output[y * width + x] = 0;
    }
}


cudaError_t cannyEdgeDetection(float* input, float* output, int width, int height, float lowThreshold, float highThreshold) {
    float* dev_input = nullptr, * dev_output = nullptr, * dev_grad_x = nullptr, * dev_grad_y = nullptr;
    const size_t size = sizeof(float) * width * height;

    CHECK_CUDA(cudaMalloc(&dev_input, size));
    CHECK_CUDA(cudaMalloc(&dev_output, size));
    CHECK_CUDA(cudaMalloc(&dev_grad_x, size));
    CHECK_CUDA(cudaMalloc(&dev_grad_y, size));

   
    dim3 block(16, 16);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);

    
    CHECK_CUDA(cudaMemcpy(dev_input, input, size, cudaMemcpyHostToDevice));

   
    CHECK_CUDA(Gaussian::compute(dev_input, dev_input, width, height));

    
    float* grad_x = new float[width * height];
    float* grad_y = new float[width * height];
    CHECK_CUDA(Prewitt::compute(dev_input, dev_output, grad_x, grad_y, width, height));

    
    nonMaxSuppression << <grid, block >> > (dev_output, dev_grad_x, dev_grad_y, dev_output, width, height);
    CHECK_CUDA(cudaDeviceSynchronize());

    
    doubleThreshold << <grid, block >> > (dev_output, dev_output, width, height, lowThreshold, highThreshold);
    CHECK_CUDA(cudaDeviceSynchronize());

    
    edgeTrackingByHysteresis << <grid, block >> > (dev_output, dev_output, width, height);
    CHECK_CUDA(cudaDeviceSynchronize());

   
    CHECK_CUDA(cudaMemcpy(output, dev_output, size, cudaMemcpyDeviceToHost));

cleanup:
    cudaFree(dev_input);
    cudaFree(dev_output);
    cudaFree(dev_grad_x);
    cudaFree(dev_grad_y);

    return cudaSuccess;
}


void matToFloatArray(const cv::Mat& mat, float* output) {
    for (int y = 0; y < mat.rows; ++y) {
        for (int x = 0; x < mat.cols; ++x) {
            output[y * mat.cols + x] = mat.at<uchar>(y, x);  
        }
    }
}


cv::Mat floatArrayToMat(float* input, int width, int height) {
    cv::Mat mat(height, width, CV_8UC1);  
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            mat.at<uchar>(y, x) = static_cast<uchar>(input[y * width + x]);
        }
    }
    return mat;
}


