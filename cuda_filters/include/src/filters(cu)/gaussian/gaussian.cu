#include "lib/filters(cuh)/gaussian/gaussian.cuh"
#include <stdio.h>



__constant__ float G[9] = { 0.06,0.12,0.06,
                            0.12,0.24,0.12,
                            0.06,0.12,0.06 };






__global__ void Gaussian::gaussKernel(float* input,
    float* output,
    int width,
    int height) {
    
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x <=0 || x >= height-1 || y<=0 || y >= width-1) {
        return;
    }

    float result = 0.0f;

    #pragma unroll
    for (int ky = -1; ky <= 1; ky++) {
        #pragma unroll
        for (int kx = -1; kx <= 1; kx++) {

            const int kernel_idx = (ky + 1) * 3 + (kx + 1);

            const int px = fminf(fmaxf(x + kx, 0), width - 1);
            const int py = fminf(fmaxf(y + ky, 0),height - 1);

            const float pixel = input[py * width + px];
            
            result += pixel * G[kernel_idx];
        }
    }

    
    int idx = y * width + x;
    output[idx] = fminf(fabsf(result),255);
}

cudaError_t Gaussian::compute(float* input,
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

    gaussKernel << <grid,block >> > (dev_input, dev_output,
        width, height);

    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaGetLastError());
    
    CHECK_CUDA(cudaMemcpy(output, dev_output, size, cudaMemcpyDeviceToHost));

cleanup:
    cudaFree(dev_input);
    cudaFree(dev_output);

    return cudaSuccess;;
}



