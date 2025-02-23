#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>
#include "filters.h"  




int main() {

    cv::Mat image = cv::imread("input.jpg", cv::IMREAD_GRAYSCALE);
    if (image.empty()) {
        std::cerr << "Could not open or find the image\n";
        return -1;
    }


    int width = image.cols;
    int height = image.rows;


    float* input = new float[width * height];
    float* output = new float[width * height](); 
    float* grad_x = new float[width * height]();
    float* grad_y = new float[width * height]();

    
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            input[i * width + j] = static_cast<int>(image.at<uchar>(i, j));
        }
    }

    cudaError_t cudaStatus;


    cudaStatus = prewittWithCuda(input, output,grad_x,grad_y, width, height);
    if (cudaStatus != cudaSuccess) {
        std::cerr << "Prewitt filter with CUDA failed\n";
        delete[] input;
        delete[] output;
        return -1;
    }


    cv::Mat outputImage(height, width, CV_32FC1, output);
    cv::normalize(outputImage, outputImage, 0, 255, cv::NORM_MINMAX);
    outputImage.convertTo(outputImage, CV_8UC1);
    cv::imwrite("output_prewitt.jpg", outputImage);


    std::fill(output, output + width * height, 0);



    cudaStatus = laplacianWithCuda(input, output, width, height);
    if (cudaStatus != cudaSuccess) {
        std::cerr << "Laplacian filter with CUDA failed\n";
        delete[] input;
        delete[] output;
        return -1;
    }

    cv::Mat outputImageLaplacian(height, width, CV_32FC1, output);
    cv::normalize(outputImageLaplacian, outputImageLaplacian, 0, 255, cv::NORM_MINMAX);
    outputImageLaplacian.convertTo(outputImageLaplacian, CV_8UC1);
    cv::imwrite("output_laplacian.jpg", outputImageLaplacian);

    std::fill(output, output + width * height, 0);

    cudaStatus = gaussWithCuda(input, output, width, height);
    if (cudaStatus != cudaSuccess) {
        std::cerr << "Gauss filter with CUDA failed\n";
        delete[] input;
        delete[] output;
        return -1;
    }

    cv::Mat outputImageGauss(height, width, CV_32FC1, output);
    cv::normalize(outputImageGauss, outputImageGauss, 0, 255, cv::NORM_MINMAX);
    outputImageGauss.convertTo(outputImageGauss, CV_8UC1);
    cv::imwrite("output_gauss.jpg", outputImageGauss);


    std::fill(output, output + width * height, 0);


    cudaStatus = sobelWithCuda(input, output, width, height);
    if (cudaStatus != cudaSuccess) {
        std::cerr << "Sobel filter with CUDA failed\n";
        delete[] input;
        delete[] output;
        return -1;
    }

    cv::Mat outputImageSobel(height, width, CV_32FC1, output);
    cv::normalize(outputImageSobel, outputImageSobel, 0, 255, cv::NORM_MINMAX);
    outputImageSobel.convertTo(outputImageSobel, CV_8UC1);
    cv::imwrite("output_sobel.jpg", outputImageSobel);


    std::fill(output, output + width * height, 0);

    float lowThreshold = 50.0f;  
    float highThreshold = 150.0f;  

    cudaStatus = cannyWithCuda(input, output, width, height, lowThreshold, highThreshold);
    if (cudaStatus != cudaSuccess) {
        std::cerr << "Canny edge detection with CUDA failed\n";
        delete[] input;
        delete[] output;
        return -1;
    }


    cv::Mat outputImageCanny(height, width, CV_32FC1, output);
    cv::normalize(outputImageCanny, outputImageCanny, 0, 255, cv::NORM_MINMAX);
    outputImageCanny.convertTo(outputImageCanny, CV_8UC1);
    cv::imwrite("output_canny.jpg", outputImageCanny);


    delete[] input;
    delete[] output;

    std::cout << "Processing complete! Outputs saved to output_prewitt.jpg, output_laplacian.jpg, output_gauss.jpg, output_sobel.jpg,output_canny.jpg\n";
    return 0;
}
