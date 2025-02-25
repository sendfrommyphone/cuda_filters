#include "lib/filters(cuh)/prewitt/prewitt.cuh"
#include <stdio.h>




__constant__ float Px[9] = { -1, 0, 1, -1, 0, 1, -1, 0, 1 };
__constant__ float Py[9] = { 1, 1, 1, 0, 0, 0, -1, -1, -1 };

float gx = 0.0f, gy = 0.0f;

__global__ void Prewitt::gradientKernel(float* input,
    float* output,
    float* grad_x,
    float* grad_y,
    int width,
    int height) {
    
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x <=0 || x >= height-1 || y<=0 || y >= width-1) {
        return;
    }

    float gx = 0.0f, gy = 0.0f;
    #pragma unroll
    for (int ky = -1; ky <= 1; ky++) {
        #pragma unroll
        for (int kx = -1; kx <= 1; kx++) {
            const int kernel_idx = (ky + 1) * 3 + (kx + 1);

            const int px = fminf(fmaxf(x + kx, 0), width - 1);
            const int py = fminf(fmaxf(y + ky, 0),height - 1);

            const float pixel = input[py * width + px];
            gx += pixel * Px[kernel_idx];
            gy += pixel * Py[kernel_idx];
        }
    }

    const float magnitude = fminf(fabsf(gx) + fabsf(gy), 255.0f);
    
    int idx = y * width + x;
    output[idx] = magnitude;
    grad_x[idx] = gx;
    grad_y[idx] = gy;
}

cudaError_t Prewitt::compute(float* input,
    float* output,
    float* grad_x,
    float* grad_y,
    int width,
    int height) {

    float* dev_input=nullptr,  *dev_output=nullptr, *dev_grad_x=nullptr, *dev_grad_y=nullptr;

    const size_t size = sizeof(float) * width * height;
    CHECK_CUDA(cudaMalloc(&dev_input, size));
    CHECK_CUDA(cudaMalloc(&dev_output, size));
    CHECK_CUDA(cudaMalloc(&dev_grad_x, size));
    CHECK_CUDA(cudaMalloc(&dev_grad_y, size));


    dim3 block(16, 16);
    dim3 grid((width + block.x - 1) / block.x,(height + block.y - 1) / block.y);

    CHECK_CUDA(cudaMemcpy(dev_input, input, size, cudaMemcpyHostToDevice));

    gradientKernel << <grid,block >> > (dev_input, dev_output,
        dev_grad_x, dev_grad_y,
        width, height);

    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaGetLastError());
    
    CHECK_CUDA(cudaMemcpy(output, dev_output, size, cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(grad_x, dev_grad_x, size, cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(grad_y, dev_grad_y, size, cudaMemcpyDeviceToHost));

cleanup:
    // 7. �ͷ��豸�ڴ�
    cudaFree(dev_input);
    cudaFree(dev_output);
    cudaFree(dev_grad_x);
    cudaFree(dev_grad_y);

    return cudaSuccess;;
}


