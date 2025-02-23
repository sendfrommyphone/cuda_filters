#include"filters.h"

#define PI 3.1415926f

	
	__global__ void laplacekernel(float* input, float* output, int width, int height) {
		const float laplace[3][3] = {
			{0,1,1},
			{1,-4,1},
			{0,1,0}
		};


		int x = blockIdx.x * blockDim.x + threadIdx.x;
		int y = blockIdx.y * blockDim.y + threadIdx.y;
		if (x < 1 || x >= width - 1 || y < 1 || y >= height - 1) {
			return;
		}

		float grad = 0;

		for (int ky = -1; ky <= 1; ky++) {
			for (int kx = -1; kx <= 1; kx++) {
				float pixel = input[(y + ky) * width + x + kx];
				grad += pixel * laplace[ky + 1][kx + 1];
			}
		}

		float magnitude = fminf(abs(grad), 255);
		output[y * width + x] = magnitude;
	}

	cudaError_t laplacianWithCuda(float* input, float* output, int width, int height) {
		float* dev_input = 0;
		float* dev_output = 0;

		cudaError_t cudaStatus;

		cudaStatus = cudaMalloc((void**)&dev_input, sizeof(float) * width * height);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMalloc failed for dev_input\n");
			return cudaStatus;
		}

		cudaStatus = cudaMalloc((void**)&dev_output, sizeof(float) * width * height);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMalloc failed for dev_output\n");
			return cudaStatus;
		}

		cudaStatus = cudaMemcpy(dev_input, input, sizeof(float) * width * height, cudaMemcpyHostToDevice);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMemcpy failed for input\n");
			return cudaStatus;
		}

		dim3 blocknum((15 + width) / 16, (15 + height) / 16);
		dim3 threadsperblock(16, 16);

		laplacekernel << <blocknum, threadsperblock >> > (dev_input, dev_output, width, height);


		cudaStatus = cudaGetLastError();
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "prewittWithCuda failed :%s", cudaGetErrorString(cudaStatus));
			return cudaStatus;
		}

		cudaStatus = cudaDeviceSynchronize();
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaDeviceSynchornize failed\n");
			return cudaStatus;
		}

		cudaStatus = cudaMemcpy(output, dev_output, sizeof(float) * width * height, cudaMemcpyDeviceToHost);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMemcpy failed for dev_output\n");
			return cudaStatus;
		}

		cudaFree(dev_input);
		cudaFree(dev_output);

		return cudaSuccess;
	}



	__global__ void sobelkernel(float* input, float* output, int width, int height) {
		const float Gx[3][3] = {
			{-1,0,1},
			{-2,0,2},
			{-1,0,1}
		};

		const float Gy[3][3] = {
			{1,2,1},
			{0,0,0},
			{-1,-2,-1}
		};

		int x = blockIdx.x * blockDim.x + threadIdx.x;
		int y = blockIdx.y * blockDim.y + threadIdx.y;
		if (x < 1 || x >= width - 1 || y < 1 || y >= height - 1) {
			return;
		}

		float grad_x = 0, grad_y = 0;

		for (int ky = -1; ky <= 1; ky++) {
			for (int kx = -1; kx <= 1; kx++) {
				float pixel = input[(y + ky) * width + x + kx];
				grad_x += pixel * Gx[ky + 1][kx + 1];
				grad_y += pixel * Gy[ky + 1][kx + 1];
			}
		}

		float magnitude = fminf(abs(grad_x) + abs(grad_y), 255);
		output[y * width + x] = magnitude;
	}

	cudaError_t sobelWithCuda(float* input, float* output, int width, int height) {
		float* dev_input = 0;
		float* dev_output = 0;

		cudaError_t cudaStatus;

		cudaStatus = cudaMalloc((void**)&dev_input, sizeof(float) * width * height);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMalloc failed for dev_input\n");
			return cudaStatus;
		}

		cudaStatus = cudaMalloc((void**)&dev_output, sizeof(float) * width * height);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMalloc failed for dev_output\n");
			return cudaStatus;
		}

		cudaStatus = cudaMemcpy(dev_input, input, sizeof(float) * width * height, cudaMemcpyHostToDevice);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMemcpy failed for input\n");
			return cudaStatus;
		}

		dim3 blocknum((15 + width) / 16, (15 + height) / 16);
		dim3 threadsperblock(16, 16);

		sobelkernel << <blocknum, threadsperblock >> > (dev_input, dev_output, width, height);


		cudaStatus = cudaGetLastError();
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "prewittWithCuda failed :%s", cudaGetErrorString(cudaStatus));
			return cudaStatus;
		}

		cudaStatus = cudaDeviceSynchronize();
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaDeviceSynchornize failed\n");
			return cudaStatus;
		}

		cudaStatus = cudaMemcpy(output, dev_output, sizeof(float) * width * height, cudaMemcpyDeviceToHost);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMemcpy failed for dev_output\n");
			return cudaStatus;
		}

		cudaFree(dev_input);
		cudaFree(dev_output);

		return cudaSuccess;
	}



	__global__ void gausskernel(float* input, float* output, int width, int height) {
		const float gauss[3][3] = {
			{0.06, 0.12, 0.06},
			{0.12,0.24, 0.12},
			{0.06, 0.12, 0.06}
		};

		int x = blockIdx.x * blockDim.x + threadIdx.x;
		int y = blockIdx.y * blockDim.y + threadIdx.y;

		if (x < 1 || x >= width - 1 || y < 1 || y >= height - 1) {
			return;
		}

		float rgb_value = 0;

		for (int kx = -1; kx <= 1; kx++) {
			for (int ky = -1; ky <= 1; ky++) {
				rgb_value += input[(y + ky) * width + x + kx] * gauss[ky + 1][kx + 1];
			}
		}

		float magnitude = fminf(rgb_value, 255); 
		output[y * width + x] = magnitude;
	}


	cudaError_t gaussWithCuda(float* input, float* output, int width, int height) {
		float* dev_input = 0;
		float* dev_output = 0;

		cudaError_t cudaStatus;

		cudaStatus = cudaMalloc((void**)&dev_input, sizeof(float) * width * height);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMalloc failed for dev_input\n");
			return cudaStatus;
		}

		cudaStatus = cudaMalloc((void**)&dev_output, sizeof(float) * width * height);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMalloc failed for dev_output\n");
			return cudaStatus;
		}

		cudaStatus = cudaMemcpy(dev_input, input, sizeof(float) * width * height, cudaMemcpyHostToDevice);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMemcpy failed for input\n");
			return cudaStatus;
		}

		dim3 blocknum((15 + width) / 16, (15 + height) / 16);
		dim3 threadsperblock(16, 16);

		gausskernel << <blocknum, threadsperblock >> > (dev_input, dev_output, width, height);


		cudaStatus = cudaGetLastError();
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "prewittWithCuda failed :%s", cudaGetErrorString(cudaStatus));
			return cudaStatus;
		}

		cudaStatus = cudaDeviceSynchronize();
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaDeviceSynchornize failed\n");
			return cudaStatus;
		}

		cudaStatus = cudaMemcpy(output, dev_output, sizeof(float) * width * height, cudaMemcpyDeviceToHost);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMemcpy failed for dev_output\n");
			return cudaStatus;
		}

		cudaFree(dev_input);
		cudaFree(dev_output);

		return cudaSuccess;
	}


	__global__ void prewittkernel(float* input, float* output, float* grad_x, float* grad_y, int width, int height) {
		const float Px[3][3] = {
			{-1, 0, 1},
			{-1, 0, 1},
			{-1, 0, 1}
		};

		const float Py[3][3] = {
			{1, 1, 1},
			{0, 0, 0},
			{-1, -1, -1}
		};

		int x = blockIdx.x * blockDim.x + threadIdx.x;
		int y = blockIdx.y * blockDim.y + threadIdx.y;

		if (x < 1 || x >= width - 1 || y < 1 || y >= height - 1) {
			return;
		}

		float grad_x_val = 0.0f;
		float grad_y_val = 0.0f;

		for (int ky = -1; ky <= 1; ky++) {
			for (int kx = -1; kx <= 1; kx++) {
				float pixel = input[(y + ky) * width + x + kx];
				grad_x_val += pixel * Px[ky + 1][kx + 1];
				grad_y_val += pixel * Py[ky + 1][kx + 1];
			}
		}

		grad_x[y * width + x] = grad_x_val;
		grad_y[y * width + x] = grad_y_val;

		float magnitude = fminf(abs(grad_x_val) + abs(grad_y_val), 255.0f);
		output[y * width + x] = magnitude;
	}


	cudaError_t prewittWithCuda(float* input, float* output, float* grad_x, float* grad_y, int width, int height) {
		float* dev_input = nullptr;
		float* dev_output = nullptr;
		float* dev_grad_x = nullptr;
		float* dev_grad_y = nullptr;

		cudaError_t cudaStatus;


		cudaStatus = cudaMalloc((void**)&dev_input, sizeof(float) * width * height);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMalloc failed for dev_input\n");
			return cudaStatus;
		}

		cudaStatus = cudaMalloc((void**)&dev_output, sizeof(float) * width * height);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMalloc failed for dev_output\n");
			return cudaStatus;
		}

		cudaStatus = cudaMalloc((void**)&dev_grad_x, sizeof(float) * width * height);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMalloc failed for dev_grad_x\n");
			return cudaStatus;
		}

		cudaStatus = cudaMalloc((void**)&dev_grad_y, sizeof(float) * width * height);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMalloc failed for dev_grad_y\n");
			return cudaStatus;
		}


		cudaStatus = cudaMemcpy(dev_input, input, sizeof(float) * width * height, cudaMemcpyHostToDevice);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMemcpy failed for input\n");
			return cudaStatus;
		}


		dim3 blockDim(16, 16); 
		dim3 gridDim((width + blockDim.x - 1) / blockDim.x, (height + blockDim.y - 1) / blockDim.y); 

		prewittkernel << <gridDim, blockDim >> > (dev_input, dev_output, dev_grad_x, dev_grad_y, width, height);

		cudaStatus = cudaGetLastError();
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "prewittWithCuda failed: %s\n", cudaGetErrorString(cudaStatus));
			return cudaStatus;
		}

		cudaStatus = cudaDeviceSynchronize();
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaDeviceSynchronize failed\n");
			return cudaStatus;
		}

		cudaStatus = cudaMemcpy(output, dev_output, sizeof(float) * width * height, cudaMemcpyDeviceToHost);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMemcpy failed for dev_output\n");
			return cudaStatus;
		}

		cudaStatus = cudaMemcpy(grad_x, dev_grad_x, sizeof(float) * width * height, cudaMemcpyDeviceToHost);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMemcpy failed for dev_grad_x\n");
			return cudaStatus;
		}

		cudaStatus = cudaMemcpy(grad_y, dev_grad_y, sizeof(float) * width * height, cudaMemcpyDeviceToHost);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMemcpy failed for dev_grad_y\n");
			return cudaStatus;
		}

		cudaFree(dev_input);
		cudaFree(dev_output);
		cudaFree(dev_grad_x);
		cudaFree(dev_grad_y);

		return cudaSuccess;
	}


	__device__ float angletodegrees(float grad_x, float grad_y) {
		float angle = atan2f(grad_y, grad_x) * 180.f / PI;
		if (angle < 0) {
			angle += 180.f;
		}
		return angle;
	}
	__global__ void nonMaxSuppression(float* grad_x, float* grad_y, float* output, int width, int height) {
		int x = blockIdx.x * blockDim.x + threadIdx.x;
		int y = blockIdx.y * blockDim.y + threadIdx.y;

		if (x >= 1 && x < width - 1 && y >= 1 && y < height - 1) {
			float angle = angletodegrees(grad_x[y * width + x], grad_y[y * width + x]);

			float q = 255.0f;
			float r = 255.0f;

			if ((0 <= angle && angle < 22.5f) || (157.5f <= angle && angle < 180.0f)) {
				q = grad_x[y * width + x + 1];
				r = grad_x[y * width + x - 1];
			}
			else if (22.5f <= angle && angle < 67.5f) {
				q = grad_x[(y + 1) * width + x - 1];
				r = grad_x[(y - 1) * width + x + 1];
			}
			else if (67.5f <= angle && angle < 112.5f) {
				q = grad_y[(y + 1) * width + x];
				r = grad_y[(y - 1) * width + x];
			}
			else if (112.5f <= angle && angle < 157.5f) {
				q = grad_x[(y - 1) * width + x - 1];
				r = grad_x[(y + 1) * width + x + 1];
			}

			if (grad_x[y * width + x] >= q && grad_x[y * width + x] >= r) {
				output[y * width + x] = grad_x[y * width + x];
			}
			else {
				output[y * width + x] = 0;
			}
		}
	}

	__global__ void edgeTrackingByHysteresis(float* img, int width, int height, float lowThreshold, float highThreshold) {
		int x = blockIdx.x * blockDim.x + threadIdx.x;
		int y = blockIdx.y * blockDim.y + threadIdx.y;

		if (x >= 1 && x < width - 1 && y >= 1 && y < height - 1) {
			int idx = y * width + x;

			if (img[idx] >= highThreshold) {
				img[idx] = 255.0f;  
			}
			else if (img[idx] <= lowThreshold) {
				img[idx] = 0.0f; 
			}
			else {
				
				bool isEdgeConnected = false;
				for (int j = -1; j <= 1; j++) {
					for (int i = -1; i <= 1; i++) {
						int neighborIdx = (y + j) * width + (x + i);
						if (img[neighborIdx] >= highThreshold) {
							isEdgeConnected = true;
							break;
						}
					}
					if (isEdgeConnected) break;
				}

				if (isEdgeConnected) {
					img[idx] = 255.0f;  
				}
				else {
					img[idx] = 0.0f;    
				}
			}
		}
	}

	cudaError_t cannyWithCuda(float* input, float* output, int width, int height, float lowThreshold, float highThreshold) {
		float* smoothedImage = new float[width * height];
		float* grad_x = new float[width * height];
		float* grad_y = new float[width * height];


		cudaError_t cudaStatus = gaussWithCuda(input, smoothedImage, width, height);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "Error applying Gaussian filter\n");
			delete[] smoothedImage;
			return cudaStatus;
		}


		cudaStatus = prewittWithCuda(smoothedImage, output, grad_x, grad_y, width, height);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "Error calculating gradient\n");
			delete[] smoothedImage;
			return cudaStatus;
		}


		dim3 blockDim(16, 16);
		dim3 gridDim((width + blockDim.x - 1) / blockDim.x, (height + blockDim.y - 1) / blockDim.y);
		nonMaxSuppression << <gridDim, blockDim >> > (grad_x, grad_y, output, width, height);

		cudaError_t cudaStatus2 = cudaGetLastError();
		if (cudaStatus2 != cudaSuccess) {
			fprintf(stderr, "Error in non-max suppression: %s\n", cudaGetErrorString(cudaStatus2));
			delete[] smoothedImage;
			return cudaStatus2;
		}


		edgeTrackingByHysteresis << <gridDim, blockDim >> > (output, width, height, lowThreshold, highThreshold);

		cudaStatus2 = cudaGetLastError();
		if (cudaStatus2 != cudaSuccess) {
			fprintf(stderr, "Error in edge tracking by hysteresis: %s\n", cudaGetErrorString(cudaStatus2));
			delete[] smoothedImage;
			return cudaStatus2;
		}

		cudaDeviceSynchronize();  

		delete[] smoothedImage;
		return cudaSuccess;
	}






