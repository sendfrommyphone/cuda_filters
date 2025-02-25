#include "lib/filters(cuh)/test/test.cuh"
#include "lib/filters(cuh)/gaussian/gaussian.cuh"
#include "lib/filters(cuh)/prewitt/prewitt.cuh"
#include "lib/filters(cuh)/sobel/sobel.cuh"
#include "lib/filters(cuh)/laplacian/laplacian.cuh"
#include "lib/filters(cuh)/canny/canny.cuh"
#include<src/filters(cu)/canny/canny.cu>
#include<src/filters(cu)/prewitt/prewitt.cu>
#include<src/filters(cu)/sobel/sobel.cu>
#include<src/filters(cu)/gaussian/gaussian.cu>
#include<src/filters(cu)/laplacian/laplacian.cu>
#include <iostream>
#include <opencv2/opencv.hpp>

#define CHECK_CUDA(call) {                                             \
    cudaError_t err = call;                                            \
    if (err != cudaSuccess) {                                          \
        std::cerr << "CUDA Error at " << __FILE__ << ":" << __LINE__   \
                  << " - " << cudaGetErrorString(err) << std::endl;    \
        exit(EXIT_FAILURE);                                             \
    }                                                                   \
}

cudaError_t testAllOperators(unsigned char* input_image,
    unsigned char* output_image,
    int width, int height,
    float lowThreshold, float highThreshold) {
        float* dev_input = nullptr;
        float* dev_output = nullptr;
        float* grad_x = nullptr;
        float* grad_y = nullptr;

        const size_t size = sizeof(float) * width * height;
        const size_t grad_size = sizeof(int) * width * height;

        // 分配GPU内存
        cudaError_t err=cudaSuccess;
        err = cudaMalloc(&dev_input, size);
        CHECK_CUDA(err);

        err = cudaMalloc(&dev_output, size);
        CHECK_CUDA(err);

        err = cudaMalloc(&grad_x, grad_size);
        CHECK_CUDA(err);

        err = cudaMalloc(&grad_y, grad_size);
        CHECK_CUDA(err);

        // 将输入图像数据从主机复制到设备
        err = cudaMemcpy(dev_input, input_image, size, cudaMemcpyHostToDevice);
        CHECK_CUDA(err);

        // 1. 高斯滤波
        err = Gaussian::compute(dev_input, dev_output, width, height);
        CHECK_CUDA(err);

        // 将高斯滤波后的结果复制回主机
        err = cudaMemcpy(output_image, dev_output, size, cudaMemcpyDeviceToHost);
        CHECK_CUDA(err);

        // 2. Prewitt算子边缘检测
        err = Prewitt::compute(dev_input, dev_output, grad_x, grad_y, width, height);
        CHECK_CUDA(err);

        // 3. Sobel算子边缘检测
        err = Sobel::compute(dev_input, dev_output, width, height);
        CHECK_CUDA(err);

        // 4. Laplacian算子边缘检测
        err = Laplacian::compute(dev_input, dev_output, width, height);
        CHECK_CUDA(err);

        // 5. Canny边缘检测
        err = Canny::cannyEdgeDetection(dev_input, dev_output, width, height, lowThreshold, highThreshold);
        CHECK_CUDA(err);

        // 将Canny的结果复制回主机
        err = cudaMemcpy(output_image, dev_output, size, cudaMemcpyDeviceToHost);
        CHECK_CUDA(err);

        // 清理GPU内存
        cudaFree(dev_input);
        cudaFree(dev_output);
        cudaFree(grad_x);
        cudaFree(grad_y);

        return cudaSuccess;
 }


int main() {
    // 读取图像
    cv::Mat image = cv::imread("input_image.png", cv::IMREAD_GRAYSCALE); // 读取灰度图像
    if (image.empty()) {
        std::cerr << "Error: Could not load image!" << std::endl;
        return -1;
    }

    int width = image.cols;
    int height = image.rows;

    // 将图像数据从 OpenCV 转换为 unsigned char* 类型
    unsigned char* input_image = image.data;
    unsigned char* output_image = new unsigned char[width * height];

    // 设置Canny边缘检测的阈值
    float lowThreshold = 50.0f;
    float highThreshold = 150.0f;

    // 调用 OperatorTests::testAllOperators 来测试所有算子
    cudaError_t err = test::testAllOperators(input_image, output_image, width, height, lowThreshold, highThreshold);
    if (err != cudaSuccess) {
        std::cerr << "CUDA Error during operator tests!" << std::endl;
        return -1;
    }

    // 创建处理结果的图像
    cv::Mat output_mat(height, width, CV_8UC1, output_image);

    // 显示结果
    cv::imshow("Processed Image", output_mat);
    cv::waitKey(0);

    // 保存输出图像
    cv::imwrite("output_image.png", output_mat);

    // 清理内存
    delete[] output_image;

    return 0;
}

