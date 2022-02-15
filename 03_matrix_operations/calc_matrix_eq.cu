/*
The lab involved writing a program that performed simple matrix calculations.
Matrices a and b have size NxN and their values are randomized

Calculate the value of the expression:
X = A*B + 8 * transposed(A) + A
*/

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cuda.h>
#include <stdio.h>
#include <iostream>
#include <iomanip>
#include <math.h>
#include <cmath>
#include <cassert>

#define BLOCK_SIZE 32 //in block 16x16 threads max 32x32 threads	
#define N 1024 //array size NxN
//array size =  NxN must be multiple of BLOCK_SIZE for matrixMultiply and matrixMultiplySharedSimple(basic multiplication)
//array size =  NxN could be any value for matrixMultiplySharedAdvanced (advanced multiplication)
#define FLOAT_MIN  0
#define FLOAT_MAX  5
#define INDEX(_row,_col, _width) (((_row)*(_width))+(_col))
#define LOOP(_size) for(int i = 0 ; i < _size; i++)
#define KERNEL_FMUL(_a,_b) __fmul_rn(_a,_b) //	Multiply two floating-point values in round-to-nearest-even mode.

#define SHARED_MEM_SIZE N*4



using namespace std;
//CPU addition
void matrix_add(float *a, float *b, float *c);
void verify_add_result(float *a, float *b, float *c);
//CPU subtraction
void matrix_subtract(float *a, float *b, float *c);
void verify_subtract_result(float *a, float *b, float *c);
//CPU multiplication
void matrix_multply(float *a, float *b, float *c);
void verify_multply_result(float *a, float *b, float *c);
//CPU multiplication by const
void matrix_multply_by_const(float *a, float t, float *c);
void verify_multply_by_const_result(float *a, float t, float *c);
//CPU transposition
void matrix_transpose(float *a, float *c);
void verify_transpose_result(float *a, float *c);

void print_all_matrix(float *a, float *b, float *c, float *x);
void verify_cpu_gpu_result(float *x_cpu, float *x_gpu);




// Dodawanie A+B
__global__ void matrixAdd(float* a, float* b, float* c, int num) {
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	if ((row < num) && (col < num))
		c[row * num + col] = a[row * num + col] + b[row * num + col];
}
//Odejmowani A-B
__global__ void matrixSubtract(float* a, float* b, float* c, int num) {
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	if ((row < num) && (col < num))
		c[row * num + col] = a[row * num + col] - b[row * num + col];
}
//Mnozenie AxB podstawowe
__global__ void matrixMultiply(float* a, float* b, float* c, int num);
//Mnozenie AxB pamiec wspoldzielona dla rozmiaru macierzy bedacego wielokrotnoscia BLOCK_SIZE
__global__ void matrixMultiplySharedSimple(float* a, float* b, float* c, int num);


//Mnozenie AxB pamiec wspoldzielona dla dowolnego rozmiaru macierzy
//Autor: Duksu Kim
__global__ void matrixMultiplySharedAdvanced(float * A, float * B, float * C, int num)
{
	int row = blockDim.x * blockIdx.x + threadIdx.x;
	int col = blockDim.y * blockIdx.y + threadIdx.y;

	float value = 0;
	__shared__ float subA[BLOCK_SIZE][BLOCK_SIZE];
	__shared__ float subB[BLOCK_SIZE][BLOCK_SIZE];

	int local_row = threadIdx.x;
	int local_col = threadIdx.y;
	for (int id = 0; id < ceil((float)num / BLOCK_SIZE); id++) {
		int offset = id*BLOCK_SIZE;

		if (row >= num || offset + local_col >= num)
			subA[local_col][local_row] = 0;
		else
			subA[local_col][local_row] = A[INDEX(row, offset + local_col, num)];

		if (col >= num || offset + local_row >= num)
			subB[local_row][local_col] = 0;
		else
			subB[local_row][local_col] = B[INDEX(offset + local_row, col, num)];

		__syncthreads();

		// compute
		LOOP(BLOCK_SIZE) {
			value += KERNEL_FMUL(subA[i][local_row], subB[i][local_col]);
		}
		__syncthreads();
	}
	if (row >= num || col >= num)
		return;

	C[INDEX(row, col, num)] = value;
}


//Mnozenie t*A
__global__ void matrixMultiplyByConst(float* a, float t, float* c, int num) {
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	if ((row < num) && (col < num))
		c[row * num + col] = t * a[row * num + col];
}
// Transponowanie macierzy
__global__ void matrixTranspose(float* a, float* c, int num) {
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	if ((row < num) && (col < num))
		c[row * num + col] = a[col * num + row];
}

int main()
{
	//================================================
	//Initialize on CPU

	const size_t bytes = sizeof(float) * N * N;
	// Vectors for holding the host-side (CPU-side) data
	float *h_a = new float[N*N];
	float *h_b = new float[N*N];
	float *h_c = new float[N*N];
	float *h_x_gpu = new float[N*N];
	float *h_x_cpu = new float[N*N];
	float *h_pom = new float[N*N];

	// Hosts
	// Allocate pinned memory  
	cudaMallocHost(&h_a, bytes);
	cudaMallocHost(&h_b, bytes);
	cudaMallocHost(&h_c, bytes);
	cudaMallocHost(&h_x_gpu, bytes);
	cudaMallocHost(&h_x_cpu, bytes);
	cudaMallocHost(&h_pom, bytes);


	// Initialize random numbers in each array
	for (int i = 0; i < N*N; i++) {
		h_a[i] = FLOAT_MIN + (float)(rand()) / ((float)(RAND_MAX / (FLOAT_MAX - FLOAT_MIN)));
		h_b[i] = FLOAT_MIN + (float)(rand()) / ((float)(RAND_MAX / (FLOAT_MAX - FLOAT_MIN)));
	}
	//================================================
	// GPU side

	cudaEvent_t startGPU, stopGPU;
	cudaEventCreate(&startGPU);
	cudaEventCreate(&stopGPU);
	cudaEventRecord(startGPU);
	//Device vector pointers
	float *d_a = new float[N*N];
	float *d_b = new float[N*N];
	float *d_c = new float[N*N];
	float *d_x = new float[N*N];

	float constant = 8;

	cudaMalloc(&d_a, bytes);
	cudaMalloc(&d_b, bytes);
	cudaMalloc(&d_c, bytes);
	cudaMalloc(&d_x, bytes);

	// Copy data from the host to the device (CPU -> GPU)
	cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice);

	//Grid size
	int GRID_SIZE = (int)ceil((float)N / BLOCK_SIZE);

	dim3 grid(GRID_SIZE, GRID_SIZE);
	dim3 threads(BLOCK_SIZE, BLOCK_SIZE);

	//Equation: X = A*B + 8 * transposed(A) + A	
	matrixMultiplySharedAdvanced << <grid, threads >> >(d_a, d_b, d_c, N);	//wynikowa C
	matrixTranspose << <grid, threads >> >(d_a, d_b, N);	//wynikowa B
	matrixMultiplyByConst << <grid, threads >> >(d_b, constant, d_b, N);	//wynikowa B
	matrixAdd << <grid, threads >> >(d_c, d_b, d_b, N);		//wynikowa B
	matrixAdd << <grid, threads >> >(d_a, d_b, d_x, N);		//wynikowa X na GPU


															// Copy result matrix from device to host
	cudaMemcpy(h_x_gpu, d_x, bytes, cudaMemcpyDeviceToHost);

	// Free memory on device
	cudaFree(d_a);
	cudaFree(d_b);
	cudaFree(d_c);
	cudaFree(d_x);

	cudaEventRecord(stopGPU);
	cudaEventSynchronize(stopGPU);
	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, startGPU, stopGPU);
	float timeOnGPU = milliseconds * 1000.; //[us]

	//================================================
	//CPU side
	clock_t startCPU, stopCPU;
	startCPU = clock();

	//Equation: X = A*B + 8 * transposed(A) + A

	matrix_multply(h_a, h_b, h_c);	//wynikowa C
	matrix_transpose(h_a, h_pom);		//wynikowa h_pom
	matrix_multply_by_const(h_pom, constant, h_pom);	//wynikowa h_pom
	matrix_add(h_c, h_pom, h_pom);	//wynikowa h_pom
	matrix_add(h_a, h_pom, h_x_cpu);	//wynikowa X na CPU

	stopCPU = clock();
	// print_all_matrix(h_a, h_b, h_x_cpu, h_x_gpu);
	// Compare results
	verify_cpu_gpu_result(h_x_cpu, h_x_gpu);




	//================================================
	double timeOnCPU = double(stopCPU - startCPU) / double(CLOCKS_PER_SEC);
	timeOnCPU = pow(10, 6) * timeOnCPU; //[us]

	printf("Matrix Size: %d x %d \n", N, N);
	printf("Time on GPU: %f  [us]\n", (double)(timeOnGPU));
	printf("Time on CPU: %f  [us]\n", (double)(timeOnCPU));

	cout << "GPU szybsze od CPU " << (float)(timeOnCPU / timeOnGPU) << endl;
	cout << "COMPLETED SUCCESSFULLY\n";


	cout << "Whether to display the A, B matrix and the result on the GPU and CPU?\n'y' - yes\n'n' - no" << endl;

	char znak;
	cin >> znak;
	if (znak == 'y') {
		print_all_matrix(h_a, h_b, h_x_cpu, h_x_gpu);
		system("PAUSE");
	}
	else {
		cout << "Press any key to exit the program\n";
		system("PAUSE");
	}




	return 0;
}

void verify_add_result(float *a, float *b, float *c)
{
	for (int i = 0; i < 10; i++) {
		cout << setprecision(3) << c[i] << " = " << a[i] << " + " << b[i] << "\n";
	}
	for (int i = 0; i < N * N; i++) {
		assert(c[i] == a[i] + b[i]);
	}
}

void matrix_add(float * a, float * b, float * c)
{
	for (int i = 0; i < N * N; i++)
		c[i] = a[i] + b[i];
}

void verify_subtract_result(float *a, float *b, float *c)
{
	for (int i = 0; i < 10; i++) {
		cout << setprecision(3) << c[i] << " = " << a[i] << " - " << b[i] << "\n";
	}
	for (int i = 0; i < N * N; i++) {
		assert(c[i] == a[i] - b[i]);
	}
}

void matrix_subtract(float * a, float * b, float * c)
{
	for (int i = 0; i < N * N; i++)
		c[i] = a[i] - b[i];
}

void verify_multply_result(float *a, float *b, float *c) {
	for (int i = 0; i < N; i++) {
		for (int j = 0; j < N; j++) {
			float tmp = 0;
			for (int m = 0; m < N; m++) {
				tmp += a[i * N + m] * b[m * N + j];
			}
			assert(tmp == c[i * N + j]);
		}
	}
}

void matrix_multply(float * a, float * b, float * c)
{
	for (int i = 0; i < N; i++) {
		for (int j = 0; j < N; j++) {
			float tmp = 0;
			for (int m = 0; m < N; m++) {
				tmp += a[i * N + m] * b[m * N + j];
			}
			c[i * N + j] = tmp;
		}
	}
}

void verify_multply_by_const_result(float *a, float t, float *c) {
	for (int i = 0; i < 10; i++) {
		cout << setprecision(4) << c[i] << " = " << t << " x " << a[i] << "\n";
	}
	for (int i = 0; i < N * N; i++) {
		assert(c[i] == (t * a[i]));
	}
}

void matrix_multply_by_const(float * a, float t, float * c)
{
	for (int i = 0; i < N * N; i++)
		c[i] = t * a[i];
}

void verify_transpose_result(float * a, float * c)
{
	for (int i = 0; i < N; i++) {
		for (int j = 0; j < N; j++) {
			assert(c[i*N + j] == a[j*N + i]);
		}
	}
}
void matrix_transpose(float * a, float * c)
{
	for (int i = 0; i < N; i++) {
		for (int j = 0; j < N; j++) {
			c[i*N + j] = a[j*N + i];
		}
	}
}
void print_all_matrix(float * a, float * b, float * c, float * x)
{
	cout << setprecision(6);
	int setw_value = 10;
	//MAtrix A
	cout << 'A' << endl;
	for (int i = 0; i < N; i++) {
		for (int j = 0; j < N; j++) {
			cout << setw(setw_value) << a[i*N + j] << " ";
		}
		cout << endl;
	}
	//MAtrix B
	cout << 'B' << endl;
	for (int i = 0; i < N; i++) {
		for (int j = 0; j < N; j++) {
			cout << setw(setw_value) << b[i*N + j] << " ";
		}
		cout << endl;
	}
	//MAtrix X cpu
	cout << "X CPU" << endl;
	for (int i = 0; i < N; i++) {
		for (int j = 0; j < N; j++) {
			cout << setw(setw_value) << c[i*N + j] << " ";
		}
		cout << endl;
	}
	//MAtrix X gpu
	cout << "X GPU" << endl;
	for (int i = 0; i < N; i++) {
		for (int j = 0; j < N; j++) {
			cout << setw(setw_value) << x[i*N + j] << " ";
		}
		cout << endl;
	}
}
void verify_cpu_gpu_result(float * x_cpu, float * x_gpu)
{
	for (int i = 0; i < N * N; i++) {
		assert(x_cpu[i] == x_gpu[i]);
	}
}

//Basic and shared memory multiplication - declaration

__global__ void matrixMultiply(float* a, float* b, float* c, int num) {
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	c[row * num + col] = 0;
	// Boundary protection
	if ((row < num) && (col < num)) {
		for (int m = 0; m < num; m++)
			c[row * num + col] += a[row * num + m] * b[m * num + col];
	}
	// __syncthreads();
}
__global__ void matrixMultiplySharedSimple(float* a, float* b, float* c, int num) {
	int size = SHARED_MEM_SIZE;
	if (size > 2000) size = N;
	__shared__ float A[SHARED_MEM_SIZE];
	__shared__ float B[SHARED_MEM_SIZE];

	int tidx = threadIdx.x;
	int tidy = threadIdx.y;
	int bidx = blockIdx.x;
	int bidy = blockIdx.y;

	int row = bidx * BLOCK_SIZE + tidy;
	int col = bidy * BLOCK_SIZE + tidx;
	c[row * num + col] = 0;

	for (int i = 0; i < (num / BLOCK_SIZE); i++)
	{

		A[(tidy*BLOCK_SIZE + tidx)] = a[row * num + i * BLOCK_SIZE + tidx];
		B[(tidy*BLOCK_SIZE + tidx)] = b[(i * BLOCK_SIZE * num + tidy * num + col)];
		__syncthreads();
		for (int j = 0; j < BLOCK_SIZE; j++)
			c[row * num + col] += A[tidy*BLOCK_SIZE + j] * B[BLOCK_SIZE * j + tidx];
		__syncthreads();
	}

}


// Code to check each operation on the matrix separately
//================================================
//GPU side

/*Launch the kernel on the GPU

matrixAdd << <grid, threads >> >(d_a, d_b, d_c, N);
matrixSubtract << <grid, threads >> >(d_a, d_b, d_c, N);
matrixMultiply << <grid, threads >> >(d_a, d_b, d_c, N);
matrixMultiplySharedSimple << <grid, threads >> >(d_a, d_b, d_c, N);
matrixMultiplySharedAdvanced << <grid, threads >> >(d_a, d_b, d_c, N);
matrixMultiplyByConst << <grid, threads >> >(d_a, constant, d_c, N);
matrixTranspose<<<grid, threads >>>(d_a, d_c, N);
*/

/*Check result for errors

verify_add_result(h_a, h_b, h_c);
verify_subtract_result(h_a, h_b, h_c);
verify_multply_result(h_a, h_b, h_c);
verify_multply_by_const_result(h_a, constant, h_c);
verify_transpose_result(h_a, h_c);
print_all_matrix(h_a, h_b, h_c, h_x);
*/

//================================================
//CPU side

/*
matrix_add(h_a, h_b, h_c);
verify_add_result(h_a, h_b, h_c);
matrix_subtract(h_a, h_b, h_c);
verify_subtract_result(h_a, h_b, h_c);
matrix_multply(h_a, h_b, h_c);
verify_multply_result(h_a, h_b, h_c);
matrix_multply_by_const(h_a, constant, h_c);
verify_multply_by_const_result(h_a, constant, h_c);
matrix_transpose(h_a, h_c);
verify_transpose_result(h_a, h_c);
*/