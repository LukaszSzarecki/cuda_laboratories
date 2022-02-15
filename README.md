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
The lab required you to write a program that computes an integral under a field created from a random set of points. Before calculation, set of points must have been sorted ascending. 
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

#### Results
