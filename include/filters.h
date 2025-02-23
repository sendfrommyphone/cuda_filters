#ifndef FILTERS_H
#define FILTERS_H

#include<cuda_runtime.h>
#include<device_launch_parameters.h>
#include<stdio.h>
#include<stdlib.h>
#include<iostream>
#include<cufft.h>
#include <opencv2/opencv.hpp>

cudaError_t prewittWithCuda(float* input, float* output,float*grad_x,float*grad_y, int width, int height);
cudaError_t laplacianWithCuda(float* input, float* output, int width, int height);
cudaError_t gaussWithCuda(float* input, float* output, int width, int height);
cudaError_t sobelWithCuda(float* input, float* output, int width, int height);
cudaError_t cannyWithCuda(float*input,float* output,int width, int height, float lowThreshold, float highThreshold);



#endif
