# Image Edge Detection Project  

## Project Overview  
This project implements several edge detection algorithms in C++, including Sobel, Canny, Laplacian, Prewitt, and Gaussian blur, using OpenCV with CUDA acceleration. The goal is to provide a fast and efficient way to detect edges in images, leveraging the power of GPU processing.  

## Features  

- **Sobel Operator**: Computes the gradient of the image intensity, useful for detecting horizontal and vertical edges.  
- **Canny Edge Detection**: A multi-stage algorithm that detects edges efficiently with high accuracy.  
- **Laplacian Operator**: Detects areas of rapid intensity change, useful for finding edges in an image.  
- **Prewitt Operator**: Similar to Sobel, used for detecting horizontal and vertical edges, but with a different kernel.  
- **Gaussian Blur**: Reduces image noise and detail, often used as a preprocessing step before edge detection.  

## Requirements  

- C++ (C++11 or later)  
- OpenCV 4.1.0 with CUDA 12.6  
- CMake (for building the project)  
- A compatible GPU for CUDA acceleration  
