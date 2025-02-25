#include "lib/filters(cuh)/sobel/sobel.cuh"
#include <stdio.h>


__constant__ float Sx[9] = { -1,0,1,
                            -2,0,2,
                            -1,0,1 };


__constant__ float Sy[9] = { 1,2,1,
                             0,0,0,
                            - 1,- 2,- 1 };
                            



__global__ void Sobel::sobelKernel(float* input,
    float* output,
    int width,
    int height) {
    
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x <=0 || x >= height-1 || y<=0 || y >= width-1) {
        return;
    }

    float grad_x = 0.0f, grad_y = 0.0f;

    #pragma unroll
    for (int ky = -1; ky <= 1; ky++) {
        #pragma unroll
        for (int kx = -1; kx <= 1; kx++) {

            const int kernel_idx = (ky + 1) * 3 + (kx + 1);

            const int px = fminf(fmaxf(x + kx, 0), width - 1);
            const int py = fminf(fmaxf(y + ky, 0),height - 1);

            const float pixel = input[py * width + px];
            grad_x += pixel * Sx[kernel_idx];
            grad_y += pixel * Sy[kernel_idx];
        }
    }

    
    int idx = y * width + x;
    float magnitude = fminf(sqrt(grad_x * grad_x + grad_y * grad_y), 255);
    output[idx] = magnitude;
}

cudaError_t Sobel::compute(float* input,
    float* output,
    int width,
    int height) {

    float* dev_input=nullptr,  *dev_output=nullptr;

    const size_t size = sizeof(float) * width * height;
    CHECK_CUDA(cudaMalloc(&dev_input, size));
    CHECK_CUDA(cudaMalloc(&dev_output, size));

    dim3 block(16, 16);
    dim3 grid((width + block.x - 1) / block.x,(height + block.y - 1) / block.y);

    CHECK_CUDA(cudaMemcpy(dev_input, input, size, cudaMemcpyHostToDevice));

    sobelKernel << <grid,block >> > (dev_input, dev_output,
        width, height);

    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaGetLastError());
    
    CHECK_CUDA(cudaMemcpy(output, dev_output, size, cudaMemcpyDeviceToHost));

cleanup:
    // 7. �ͷ��豸�ڴ�
    cudaFree(dev_input);
    cudaFree(dev_output);

    return cudaSuccess;;
}

