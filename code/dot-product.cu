/**
 * Cuda Programming HW Problem One: Dot Product
 * Alex Farrell
*/
#include <stdio.h>
#include <stdlib.h>

#define BLOCK_SIZE 256
#define NUM_BLOCKS 256
#define N 65536


__global__
void dotp(float *u, float *v, float *partialSum, int n) {
    // Partial sums
    __shared__ float localCache[BLOCK_SIZE];
    int tidx = threadIdx.x + blockIdx.x * blockDim.x;
    localCache[threadIdx.x] = u[tidx] * v[tidx];
    __syncthreads();

    // Parallel reduction
    int cacheIndex = threadIdx.x;
    int i = blockDim.x / 2;
    while (i > 0) {
        if (cacheIndex < i) {
            localCache[cacheIndex] = localCache[cacheIndex] + localCache[cacheIndex + i];
            // Need to be careful calling __syncthreads in an if
            // statements to avoid issues such as deadlock. Should
            // be safe for this problem.
            __syncthreads();
        }
        i = i / 2;
    }

    if (cacheIndex == 0) {
        partialSum[blockIdx.x] = localCache[cacheIndex];
    }
}

//----------------------------------------------------------------

int main() {
    srand48(time(0));

    float *U, *V, *partialSum;
    float *dev_U, *dev_V, *dev_partialSum;

    U = (float *) malloc(N * sizeof(float));
    V = (float *) malloc(N * sizeof(float));
    partialSum = (float *) malloc(N * sizeof(float));

    cudaMalloc((void **) &dev_U, N*sizeof(float));
    cudaMalloc((void **) &dev_V, N*sizeof(float));
    cudaMalloc((void **) &dev_partialSum, N*sizeof(float));

    //----------------------------------------------------------------
    // GPU Calculation

    // Testing vectors
    for (int i=0; i<N; ++i) {
        U[i] = (float) drand48();
        V[i] = (float) drand48();
    }

    // *** NOTE: The CUDA events are for getting the elapsed time for the GPU calculation. ***
    cudaEvent_t startWithMemGPU, stopWithMemGPU;
    cudaEvent_t startNoMemGPU_1, stopNoMemGPU_1;
    cudaEvent_t startNoMemGPU_2, stopNoMemGPU_2;
    cudaEventCreate(&startWithMemGPU);
    cudaEventCreate(&stopWithMemGPU);
    cudaEventRecord(startWithMemGPU, 0);

    // Copies data to the GPU so it can perform the calculation.
    cudaMemcpy(dev_U, U, N*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_V, V, N*sizeof(float), cudaMemcpyHostToDevice);

    cudaEventCreate(&startNoMemGPU_1);
    cudaEventCreate(&stopNoMemGPU_1);
    cudaEventRecord(startNoMemGPU_1, 0);

    // GPU Calculation kernel
    dotp<<<NUM_BLOCKS, BLOCK_SIZE>>>(dev_U, dev_V, dev_partialSum, N);

    cudaEventRecord(stopNoMemGPU_1, 0);
    cudaEventSynchronize(stopNoMemGPU_1);

    cudaDeviceSynchronize();
    cudaMemcpy(partialSum, dev_partialSum, NUM_BLOCKS*sizeof(float), cudaMemcpyDeviceToHost);

    cudaEventCreate(&startNoMemGPU_2);
    cudaEventCreate(&stopNoMemGPU_2);
    cudaEventRecord(startNoMemGPU_2, 0);

    // Finish GPU calculation CPU-side by addding the partial sums.
    float gpuResult = 0.0;
    for (int i=0; i<NUM_BLOCKS; ++i) {
        gpuResult = gpuResult + partialSum[i];
    }

    cudaEventRecord(stopNoMemGPU_2, 0);
    cudaEventSynchronize(stopNoMemGPU_2);

    cudaEventRecord(stopWithMemGPU, 0);
    cudaEventSynchronize(stopWithMemGPU);

    // Elapsed Time
    float gpuElapsedWithMem;
    float gpuElapsedNoMem_1, gpuElapsedNoMem_2;
    float gpuElapsedNoMem_total = 0.0;
    cudaEventElapsedTime(&gpuElapsedWithMem, startWithMemGPU, stopWithMemGPU);
    cudaEventElapsedTime(&gpuElapsedNoMem_1, startNoMemGPU_1, stopNoMemGPU_1);
    cudaEventElapsedTime(&gpuElapsedNoMem_2, startNoMemGPU_2, stopNoMemGPU_2);
    gpuElapsedNoMem_total = gpuElapsedNoMem_1 + gpuElapsedNoMem_2;

    cudaEventDestroy(startWithMemGPU);
    cudaEventDestroy(stopWithMemGPU);
    cudaEventDestroy(startNoMemGPU_1);
    cudaEventDestroy(stopNoMemGPU_1);
    cudaEventDestroy(startNoMemGPU_2);
    cudaEventDestroy(stopNoMemGPU_2);
    
    // CUDA Error Check
    cudaError_t err = cudaGetLastError();
    const char *msg = cudaGetErrorName(err);
    printf("error = |%s|\n", msg);

    //----------------------------------------------------------------
    // CPU Calculation (for comparison to GPU Calculation)
    cudaEvent_t startCPU, stopCPU;
    cudaEventCreate(&startCPU);
    cudaEventCreate(&stopCPU);
    cudaEventRecord(startCPU, 0);

    float cpuResult = 0.0;
    for (int i=0; i<N; ++i) {
        cpuResult = cpuResult + (U[i] * V[i]);    
    }

    cudaEventRecord(stopCPU, 0);
    cudaEventSynchronize(stopCPU);

    float cpuElapsed;
    cudaEventElapsedTime(&cpuElapsed, startCPU, stopCPU);
    cudaEventDestroy(startCPU);
    cudaEventDestroy(stopCPU);

    //----------------------------------------------------------------
    // Relative Error
    float relativeError = 0.0;
    relativeError = fabs((gpuResult - cpuResult) / gpuResult);

    //----------------------------------------------------------------
    // Results
    printf("GPU Result: %0.4f\n", gpuResult);
    printf("CPU Result: %0.4f\n", cpuResult);
    printf("Relative Error: %0.8f\n", relativeError);
    printf("CPU Elapsed Time: %0.4f\n", cpuElapsed);
    printf("GPU Elapsed Time (with mem copies): %0.4f\n", gpuElapsedWithMem);
    printf("GPU Elapsed Time (no mem copies): %0.4f\n", gpuElapsedNoMem_total);

    //----------------------------------------------------------------
    // Cleanup
    cudaFree(dev_U);
    cudaFree(dev_V);
    cudaFree(dev_partialSum);

    free(U);
    free(V);
    free(partialSum);

    return 0;
}