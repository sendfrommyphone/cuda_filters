#include "gaussian.cuh"

__constant__ float gauss_kernel[9] = {
    0.06, 0.12, 0.06,
    0.12, 0.24, 0.12,
    0.06, 0.12, 0.06
};

__global__ void Gaussian::kernel(float* input, float* output, 
                                int width, int height) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
 
    if (x >= width || y >= height) return;

    float sum = 0;
    #pragma unroll 
    for (int ky = -1; ky <= 1; ++ky) {
        #pragma unroll
        for (int kx = -1; kx <= 1; ++kx) {
            const int px = min(max(x + kx, 0), width-1);
            const int py = min(max(y + ky, 0), height-1);
            sum += input[py * width + px] * gauss_kernel[(ky+1)*3 + (kx+1)];
        }
    }
    output[y * width + x] = fminf(sum, 255.0f);
}

cudaError_t Gaussian::launch(float* input, float* output,
                            int width, int height) {
  
    const dim3 block(16, 16);
    const dim3 grid((width + block.x - 1)/block.x, 
                   (height + block.y - 1)/block.y);
    
    kernel<<<grid, block>>>(input, output, width, height);
    return cudaPeekAtLastError();
}
