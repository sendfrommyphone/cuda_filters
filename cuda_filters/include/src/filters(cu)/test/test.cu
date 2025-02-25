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

        // ����GPU�ڴ�
        cudaError_t err=cudaSuccess;
        err = cudaMalloc(&dev_input, size);
        CHECK_CUDA(err);

        err = cudaMalloc(&dev_output, size);
        CHECK_CUDA(err);

        err = cudaMalloc(&grad_x, grad_size);
        CHECK_CUDA(err);

        err = cudaMalloc(&grad_y, grad_size);
        CHECK_CUDA(err);

        // ������ͼ�����ݴ��������Ƶ��豸
        err = cudaMemcpy(dev_input, input_image, size, cudaMemcpyHostToDevice);
        CHECK_CUDA(err);

        // 1. ��˹�˲�
        err = Gaussian::compute(dev_input, dev_output, width, height);
        CHECK_CUDA(err);

        // ����˹�˲���Ľ�����ƻ�����
        err = cudaMemcpy(output_image, dev_output, size, cudaMemcpyDeviceToHost);
        CHECK_CUDA(err);

        // 2. Prewitt���ӱ�Ե���
        err = Prewitt::compute(dev_input, dev_output, grad_x, grad_y, width, height);
        CHECK_CUDA(err);

        // 3. Sobel���ӱ�Ե���
        err = Sobel::compute(dev_input, dev_output, width, height);
        CHECK_CUDA(err);

        // 4. Laplacian���ӱ�Ե���
        err = Laplacian::compute(dev_input, dev_output, width, height);
        CHECK_CUDA(err);

        // 5. Canny��Ե���
        err = Canny::cannyEdgeDetection(dev_input, dev_output, width, height, lowThreshold, highThreshold);
        CHECK_CUDA(err);

        // ��Canny�Ľ�����ƻ�����
        err = cudaMemcpy(output_image, dev_output, size, cudaMemcpyDeviceToHost);
        CHECK_CUDA(err);

        // ����GPU�ڴ�
        cudaFree(dev_input);
        cudaFree(dev_output);
        cudaFree(grad_x);
        cudaFree(grad_y);

        return cudaSuccess;
 }


int main() {
    // ��ȡͼ��
    cv::Mat image = cv::imread("input_image.png", cv::IMREAD_GRAYSCALE); // ��ȡ�Ҷ�ͼ��
    if (image.empty()) {
        std::cerr << "Error: Could not load image!" << std::endl;
        return -1;
    }

    int width = image.cols;
    int height = image.rows;

    // ��ͼ�����ݴ� OpenCV ת��Ϊ unsigned char* ����
    unsigned char* input_image = image.data;
    unsigned char* output_image = new unsigned char[width * height];

    // ����Canny��Ե������ֵ
    float lowThreshold = 50.0f;
    float highThreshold = 150.0f;

    // ���� OperatorTests::testAllOperators ��������������
    cudaError_t err = test::testAllOperators(input_image, output_image, width, height, lowThreshold, highThreshold);
    if (err != cudaSuccess) {
        std::cerr << "CUDA Error during operator tests!" << std::endl;
        return -1;
    }

    // ������������ͼ��
    cv::Mat output_mat(height, width, CV_8UC1, output_image);

    // ��ʾ���
    cv::imshow("Processed Image", output_mat);
    cv::waitKey(0);

    // �������ͼ��
    cv::imwrite("output_image.png", output_mat);

    // �����ڴ�
    delete[] output_image;

    return 0;
}

