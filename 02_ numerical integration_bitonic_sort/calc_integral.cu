/*
The lab required you to write a program that computes an integral under a field created from a 
 random set of points. Before calculation set of points must have been sorted ascending. 
*/

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>
#include <cstdlib>
#include <random>
#include<vector>
#include <time.h>
#include <iomanip> 
#include <stdio.h>
#include <cassert>

#define MAX_VALUE 100000

#define THREADS_PER_BLOCK 512
#define SHMEM_SIZE THREADS_PER_BLOCK * 4
using namespace std;

struct point
{
	float x;
	float y;
};
struct measurements
{
	double sorting_time_on_cpu;
	double sorting_time_on_gpu;
	double integration_time_on_cpu;
	double integration_time_on_gpu;
};

//CPU functions declarations
void generate_data(int amount, vector<point> &vect);

bool num_drawn(vector<point> &vect, float px);
void save_in_file(vector<point> vect, vector<point> sort_vect, vector<point> sort_vect2);
void save_in_file_times(int arr[], vector<measurements> time_measure);
void cpu_bubble_sort(vector<point> &sort_vect);
void cpu_quick_sort(vector<point> &sort_vect);
void quick_sort(vector<point>& sort_vect, int bottom, int top);
int partition(vector<point>&sort_vect, int bottom, int top);
void cpu_integrate_trapezoid(vector<point>sort_vect, double &h_array_sum);
double gpu_integrate_trapezoid(vector<point> gpu_sorted_points, int n);



//Kernels definitions
__global__ void emptyKernel() {};

__global__ void bitonicSort(point *d_vect, int j, int k)
{
	int tid = blockDim.x * blockIdx.x + threadIdx.x;
	int m = tid^j;
	if (m > tid)
	{
		if (((tid&k) == 0 && (d_vect[tid].x > d_vect[m].x)) || ((tid&k) != 0 && (d_vect[tid].x < d_vect[m].x)))
		{
			point temp = d_vect[tid];
			d_vect[tid] = d_vect[m];
			d_vect[m] = temp;
		}
	}
}

__global__ void integrateTrapezoid(point *d_vect, double *d_area, int amount)
{
	int tid = blockDim.x * blockIdx.x + threadIdx.x;
	if (tid > 0)
	{
		d_area[tid - 1] = (d_vect[tid - 1].y + d_vect[tid].y) * (d_vect[tid].x - d_vect[tid - 1].x) * 0.5;
	}
}

void gpu_bitonic_sort(vector<point> &vect, int amount)
{
	unsigned int padding = 1;
	while (padding < amount)
	{
		padding *= 2;
	}

	int THREADS = THREADS_PER_BLOCK;
	while (THREADS > padding)
	{
		THREADS /= 2;
	}
	int BLOCKS = (padding + THREADS - 1) / THREADS;
	vector<point> padding_vect(vect);
	int dist = padding - amount;
	for (int j = 0; j < dist; j++)
	{
		point q;
		q.x = -2;
		q.y = -2;
		padding_vect.push_back(q);
	}
	//Alocate memory on device
	size_t vect_size = padding * sizeof(point);
	point *d_vect;
	cudaMalloc(&d_vect, vect_size);
	//Copy date from host to device
	cudaMemcpy(d_vect, padding_vect.data(), vect_size, cudaMemcpyHostToDevice);
	//Run kernels
	int j, k;
	for (k = 2; k <= padding; k *= 2)
	{
		for (j = k / 2; j > 0; j /= 2)
		{
			bitonicSort << <BLOCKS, THREADS >> > (d_vect, j, k);
		}
	}
	//Copy date from device to host
	cudaMemcpy(padding_vect.data(), d_vect, vect_size, cudaMemcpyDeviceToHost);
	for (int i = 0; i < amount; i++)
	{
		vect[i] = padding_vect[i + dist];
	}
	cudaFree(d_vect);


}

__global__ void sumReduction(double *v, double *v_r, const int n) {
	__shared__ double partial_sum[SHMEM_SIZE];
	int tid = blockIdx.x * blockDim.x + threadIdx.x;

	if (tid < n - 1)
		partial_sum[threadIdx.x] = v[tid];
	else partial_sum[threadIdx.x] = 0;
	__syncthreads();

	for (int s = 1; s < blockDim.x; s *= 2) {
		int index = 2 * s * threadIdx.x;
		if (index < blockDim.x) {
			partial_sum[index] += partial_sum[index + s];
		}
		__syncthreads();
	}
	if (threadIdx.x == 0) {
		v_r[blockIdx.x] = partial_sum[0];
	}
}



int main()
{
	vector <measurements> time_measure;

	//int numbers[] = { 2000, 5000, 10000, 50000, 100000};
	int numbers[] = { 1000,5000,10000 };

	for (int num = 0; num < sizeof(numbers) / sizeof(*numbers); num++)
	{
		int n = numbers[num];
		measurements t_start;
		t_start.sorting_time_on_cpu = 0; t_start.sorting_time_on_gpu = 0;
		t_start.integration_time_on_cpu = 0; t_start.integration_time_on_gpu = 0;
		time_measure.push_back(t_start);

		vector <point> cpu_points;
		//Generate random vector of data
		generate_data(n, cpu_points);
		vector <point> gpu_sorted_points(cpu_points);
		vector <point> cpu_sorted_points(cpu_points);
		//cout << "Data generated\n";

		//time measures
		cudaEvent_t startGPU, stopGPU;
		cudaEventCreate(&startGPU);
		cudaEventCreate(&stopGPU);
		clock_t startCPU, stopCPU;

		//-------------------------------------------------CPU--time--measure
		//SORTING
		startCPU = clock();
		cpu_quick_sort(cpu_sorted_points);

		stopCPU = clock();
		double timeOnCPU = double(stopCPU - startCPU) / double(CLOCKS_PER_SEC);
		timeOnCPU = pow(10, 6) * timeOnCPU; //[us]


		//-------------------------------------------------CPU--time--measure
		emptyKernel << <1, 1 >> > ();

		//-------------------------------------------------GPU--time--measure
		cudaEventRecord(startGPU);
		gpu_bitonic_sort(gpu_sorted_points, n);

		cudaEventRecord(stopGPU);
		cudaEventSynchronize(stopGPU);
		float miliseconds = 0;
		cudaEventElapsedTime(&miliseconds, startGPU, stopGPU);
		double timeOnGPU = miliseconds * 1000; // [us]
		//-------------------------------------------------GPU--time--measure
		cout << numbers[num] << endl;
		printf("Sorting time on CPU: %f \nSorting time on GPU: %f \n", timeOnCPU, timeOnGPU);
		time_measure[num].sorting_time_on_cpu = timeOnCPU;
		timeOnCPU = 0;
		time_measure[num].sorting_time_on_gpu = timeOnGPU;
		timeOnGPU = 0;

		for (int k = 0; k < gpu_sorted_points.size(); k++)
		{
			assert(cpu_sorted_points[k].x == gpu_sorted_points[k].x);
			assert(cpu_sorted_points[k].y == gpu_sorted_points[k].y);
		}


		//INTEGRATION
		//-------------------------------------------------CPU--time--measure
		startCPU = clock();
		double cpu_array_sum = 0;
		cpu_integrate_trapezoid(cpu_sorted_points, cpu_array_sum);
		stopCPU = clock();
		timeOnCPU = double(stopCPU - startCPU) / double(CLOCKS_PER_SEC);
		timeOnCPU = pow(10, 6) * timeOnCPU; //[us]
		//-------------------------------------------------CPU--time--
		//-------------------------------------------------GPU--time--measure
		cudaEventRecord(startGPU);

		double gpu_array_sum = gpu_integrate_trapezoid(gpu_sorted_points, n);

		cudaEventRecord(stopGPU);
		cudaEventSynchronize(stopGPU);
		miliseconds = 0;
		cudaEventElapsedTime(&miliseconds, startGPU, stopGPU);
		timeOnGPU = miliseconds * 1000; // [us]

		//-------------------------------------------------GPU--time--measure


		printf("Integration time on CPU: %f\nIntegration time on GPU: %f\n", timeOnCPU, timeOnGPU);
		printf("Integrate CPU = %f\n Integrate GPU = %f\n", cpu_array_sum, gpu_array_sum);
		assert(cpu_array_sum == gpu_array_sum);
		time_measure[num].integration_time_on_cpu = timeOnCPU;
		timeOnCPU = 0;
		time_measure[num].integration_time_on_gpu = timeOnGPU;
		timeOnGPU = 0;

		//save_in_file(cpu_points, cpu_sorted_points, gpu_sorted_points);

	}

	save_in_file_times(numbers, time_measure);


	system("PAUSE");

	return 0;
}

void generate_data(int amount, vector<point>& vect)
{

	srand((unsigned)time(NULL));

	 random_device rd;  //Will be used to obtain a seed for the random number engine
	 mt19937 gen(rd()); //Standard mersenne_twister_engine seeded with rd()
	 uniform_int_distribution<> distrib(1, MAX_VALUE);
	float py, px;

	for (int j = 0; j < amount; ++j)
	{
		px = (float)distrib(gen);
		py = (float)distrib(gen);
		
		//py = (float)rand() / MAX_VALUE;
		//px = (float)rand() / MAX_VALUE;
		px = 0.2323 * px;
		py = 0.8467 * py;
		  px = (float(px) / float((MAX_VALUE)) * MAX_VALUE);
		  py = (float(py) / float((MAX_VALUE)) * MAX_VALUE);
		if (num_drawn(vect, px) == false) {
			point p;
			p.x = px;
			p.y = py;
			vect.push_back(p);

		}
		else
		{
			// srand((unsigned) time(NULL));
			j--;
		}
	}
}

bool num_drawn(vector<point> &vect, float px)
{
	int len = vect.size();
	if (len <= 0)
		return false;
	int i = 0;
	do
	{
		if (vect[i].x == px)
			return true;
		i++;
	} while (i < len);
	return false;
}
void save_in_file(vector<point> vect, vector<point> sort_vect, vector<point> sort_vect2)
{
	FILE *file_pointer;
	file_pointer = fopen("results.csv", "w+");

	fprintf(file_pointer, "punkty_x;punkty_y;CPU_x_posortowany;CPU_y_posortowany;GPU_x_posortowany;GPU_y_posortowany\n");
	for (unsigned int i = 0; i < vect.size(); i++) {
		fprintf(file_pointer, "%f;%f;%f;%f;%f;%f\n", vect[i].x, vect[i].y, sort_vect[i].x, sort_vect[i].y, sort_vect2[i].x, sort_vect2[i].y);
	}


	fclose(file_pointer);
}

void save_in_file_times(int arr[], vector<measurements> time_measure)
{
	FILE *file_pointer;
	file_pointer = fopen("time_results.csv", "w+");

	fprintf(file_pointer, " ;Sortowania; ;Ca�kowanie; ;\n");
	fprintf(file_pointer, "n;Czas na CPU;Czas na GPU;Czas na CPU;Czas na GPU\n");
	for (unsigned int i = 0; i < time_measure.size(); i++) {
		fprintf(file_pointer, "%d;%f;%f;%f;%f\n", arr[i], time_measure[i].sorting_time_on_cpu, time_measure[i].sorting_time_on_gpu, time_measure[i].integration_time_on_cpu, time_measure[i].integration_time_on_gpu);
	}
}
//Unused bubblesort
void cpu_bubble_sort(vector<point>& sort_vect)
{
	float buf = 0;
	for (unsigned int i = 1; i < sort_vect.size(); i++)
	{
		for (unsigned int j = 0; j < sort_vect.size() - i; j++)
		{
			if (sort_vect[j].x > sort_vect[j + 1].x)
			{
				buf = sort_vect[j].x;
				sort_vect[j].x = sort_vect[j + 1].x;
				sort_vect[j + 1].x = buf;

				buf = sort_vect[j].y;
				sort_vect[j].y = sort_vect[j + 1].y;
				sort_vect[j + 1].y = buf;
			}
		}
	}
}

void cpu_quick_sort(vector<point>& sort_vect)
{
	int bottom = 0;
	int top = sort_vect.size() - 1;
	quick_sort(sort_vect, bottom, top);
}
void quick_sort(vector<point>& sort_vect, int bottom, int top)
{
	if (bottom < top) {
		int middle_index = partition(sort_vect, bottom, top);
		quick_sort(sort_vect, bottom, middle_index - 1);
		quick_sort(sort_vect, middle_index, top);
	}
}

int partition(vector<point> &sort_vect, int bottom, int top) {
	int middle_index = bottom + (top - bottom) / 2;
	float middle_value = sort_vect[middle_index].x;
	int i = bottom, j = top;
	float bufx;
	float bufy;
	while (i <= j) {
		while (sort_vect[i].x < middle_value) {
			i++;
		}
		while (sort_vect[j].x > middle_value) {
			j--;
		}
		if (i <= j) {
			bufx = sort_vect[i].x;
			sort_vect[i].x = sort_vect[j].x;
			sort_vect[j].x = bufx;
			bufy = sort_vect[i].y;
			sort_vect[i].y = sort_vect[j].y;
			sort_vect[j].y = bufy;
			i++;
			j--;
		}
	}
	return i;
}

void cpu_integrate_trapezoid(vector<point>sort_vect, double &h_array_sum)
{
	for (int i = 1; i < sort_vect.size(); i++)
	{
		h_array_sum += (sort_vect[i - 1].y + sort_vect[i].y) * (sort_vect[i].x - sort_vect[i - 1].x) * 0.5;
	}
}

double gpu_integrate_trapezoid(vector<point> gpu_sorted_points, int n)
{
	int THREADS = THREADS_PER_BLOCK;
	int BLOCKS = (n + THREADS - 1) / THREADS;

	double *d_v, *d_v_r;
	point *d_gpu_points;

	size_t vect_size = n * sizeof(point);
	size_t bytes = n * sizeof(double);

	cudaMalloc(&d_v, bytes);
	cudaMalloc(&d_v_r, bytes);
	cudaMalloc(&d_gpu_points, vect_size);

	cudaMemcpy(d_gpu_points, gpu_sorted_points.data(), vect_size, cudaMemcpyHostToDevice);
	integrateTrapezoid << <BLOCKS, THREADS >> > (d_gpu_points, d_v, n);

	sumReduction << <BLOCKS, THREADS >> > (d_v, d_v_r, n);
	sumReduction << <1, THREADS >> > (d_v_r, d_v_r, n);

	double * h_gpu_array_sum, *h_area;
	h_gpu_array_sum = (double*)malloc(bytes);
	h_area = (double *)malloc(bytes);
	cudaMemcpy(h_area, d_v, bytes, cudaMemcpyDeviceToHost);

	cudaMemcpy(h_gpu_array_sum, d_v_r, bytes, cudaMemcpyDeviceToHost);

	cudaFree(d_v);
	cudaFree(d_v_r);
	cudaFree(d_gpu_points);
	return h_gpu_array_sum[0];
}