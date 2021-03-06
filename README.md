# cuda_laboratories
This repository contains the code for all the labs in the Parallel Programming Course.

## 1. laboratory - prime numbers

#### Environment
 * Operating System: Windows 10 Home
 * GPU: NVIDIA GTX 660
 * CUDA version: 11

#### Introduction
The lab involved writing a program to check if a given number is prime. The program was to be run on GPU and CPU. As a result, execution times on the CPU and GPU had to be compared. 
#### Results
 - The GPU needs approximately 140 [us] for each number to allocate memory and transfer data to the device
 - For 34-bit numbers and above, the GPU is faster than the CPU 
 - For a 61-bit number, the CPU operation time is about 3.4s while the GPU handles the problem in 0.2s

![image](https://user-images.githubusercontent.com/61761700/154140565-a0d036e3-df5b-4831-aa4c-33a22a1432f7.png)


## 2. laboratory - numerical integration and bitonic sort

#### Environment
 * Operating System: Windows 10 Home
 * GPU: NVIDIA GeForce GT635M
 * CUDA version: 11

#### Introduction
The lab required you to write a program that computes an integral under a field created from a random set of points. Before calculation, set of points must have been sorted ascending. CPU and GPU operation times had to be compared.
Implemented algorithms:
 * Numerical Integration - Trapezoidal Rule
 * Sorting on GPU - Bitonic Sort
 * Sorting on CPU - QuickSort

#### Results

![image](https://user-images.githubusercontent.com/61761700/154142928-1c8d8ceb-4ebb-4baf-9a7e-4a3723a1bf6b.png)


## 3. laboratory - matrix operations

#### Environment
 * Operating System: Windows 10 Home
 * GPU: NVIDIA GTX 660
 * CUDA version: 11

#### Introduction
The lab involved writing a program that performed simple matrix calculations. Matrices A and B have size NxN and their values are randomized.

Task was to calculate the value of the expression:

![image](https://user-images.githubusercontent.com/61761700/154143596-99e152f0-34d2-49ce-984f-ac2862c67e6b.png)

Implemented operations:
* matrix subtraction
* matrix addition
* matrix transposition
* matrix multiplication using shared memory
* matrix multiplication by a scalar

#### Results
* Matrix multiplication is the longest GPU operation
* For the largest tested 1792 x 1792 matrix, the CPU computation time is 27.9 seconds while the GPU the calculation takes only 0.26 second
 
![image](https://user-images.githubusercontent.com/61761700/154145099-c494a1ec-62a8-403b-8cad-a56e6d5e8c8a.png)

> results for two data types

