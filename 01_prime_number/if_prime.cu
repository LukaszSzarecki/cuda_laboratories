/*
program that checks if a given number is prime
*/

#include <iostream>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <time.h>
#include <math.h>
using namespace std;

#define WATKI_W_BLOKU 640

// GPU kernel to check if a number is prime
__global__ void primeGPU(unsigned long long* n, int* pGPU, unsigned long long n_sqrt)
{
      unsigned long long i = blockIdx.x * blockDim.x + threadIdx.x;
        i+=1;    // i musi zaczynac sie o 1
        if (6 * i <= n_sqrt)
        {
            if (((*n % (6*i - 1)) == 0) || ((*n % (6*i + 1)) == 0))
            {
                *pGPU = 0;
            }
        }
    
}

// CPU function to check if a number is prime
bool primeCPU(long long int n) {
    if (n <= 1) return false;
    if (n == 2 || n == 3) return true;
    if (n % 2 == 0 || n % 3 == 0) return false;
    unsigned long long int i;
    for (i = 5; i * i <= n; i = i + 6)
        if (n % i == 0 || n % (i + 2) == 0)
            return false;
    return true;
}

int main() {
    // Test numbers
    unsigned long long n[] = {524287, 2147483647,274876858369,1125897758834689, 2305843009213693951, 4611686014132420609};
    // Mersenne prime 
    // unsigned long long n[] = {3, 7,31,127, 8191, 131071, 524287, 2147483647, 2305843009213693951};
    // Different prime numbers 2^{12,14,18,20,24,28,32,36,40,61}
    // unsigned long long n[] = {4093, 16363, 262153, 1048613, 16777213, 268435523,4294967231, 68719476713, 999999999937, 2305843009213693951};

    unsigned long long* d_n;
    int* d_pGPU;
    int pGPU = 1;

    for (int i = 0; i < sizeof(n)/sizeof(*n); i++)
    {
         //Time-counting variables on GPU
        cudaEvent_t startGPU, stopGPU;
        cudaEventCreate(&startGPU);
        cudaEventCreate(&stopGPU);

        cudaEventRecord(startGPU);

        unsigned long long n_sqrt = sqrt(n[i]);
        int BLOKI = n_sqrt / WATKI_W_BLOKU + 1;
        if ((n[i] % 2 == 0) || (n[i] % 3 == 0))
            pGPU = 0;
        else
        {   
            //Allocating memory on device
            cudaMalloc(&d_pGPU, sizeof(int));
            cudaMalloc(&d_n, sizeof(unsigned long long));
            //Data transfer to device
            cudaMemcpy(d_pGPU, &pGPU, sizeof(int), cudaMemcpyHostToDevice);
            cudaMemcpy(d_n, &n[i], sizeof(unsigned long long), cudaMemcpyHostToDevice);
            //Calling kernel
            primeGPU << < BLOKI, WATKI_W_BLOKU >> > (d_n, d_pGPU, n_sqrt);
            //Data transfer from device to host
            cudaMemcpy(&pGPU, d_pGPU, sizeof(int), cudaMemcpyDeviceToHost);
            //Releasing memory on a device
            cudaFree(d_n);
            cudaFree(d_pGPU);
        }
        cudaEventRecord(stopGPU);
        cudaEventSynchronize(stopGPU);  
        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, startGPU, stopGPU);
        double timeOnGPU = milliseconds * 1000; //[us]

        if ((n[i] ==2 || n[i] == 3))
            pGPU = 1;

        //Calling a function on CPU
        clock_t startCPU, stopCPU;
        bool pCPU = false;
        startCPU = clock();
        pCPU = primeCPU(n[i]);
        stopCPU = clock();
        double timeOnCPU = double(stopCPU - startCPU) / double(CLOCKS_PER_SEC);
        timeOnCPU = pow(10,6) * timeOnCPU;



        cout << "-------------------------------------------------"<<endl;
        if (pGPU)
            printf("%llu is prime CPU\n", n[i]);
        else
            printf("%llu is not prime CPU\n", n[i]);
        if (pCPU)
            printf("%llu is prime GPU\n", n[i]);
        else
            printf("%llu is not prime GPU\n", n[i]);
        
        printf("Time on GPU %d \t Time on CPU %f \n" ,(int)(timeOnGPU),(double)(timeOnCPU));
        pGPU = 1;
    }


    getchar();
    return 0;
}