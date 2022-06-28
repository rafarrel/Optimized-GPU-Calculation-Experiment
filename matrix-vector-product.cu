/**
 * Cuda Programming HW Problem Two: Matrix-vector Product
 * Alex Farrell
*/
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#define BLOCK_SIZE 256
#define NUM_BLOCKS 256
#define N 5000


__global__
void MxV(float *M, float *x, float *y, size_t pitch, int n) {
    // Dot Product
    int tidx = threadIdx.x + blockIdx.x * blockDim.x;
    int numPerRow = pitch / sizeof(float);

    if (tidx < n) {
        for (int j=0; j<n; ++j) {
            y[tidx] = y[tidx] + M[tidx*numPerRow + j] * x[j];
        }
    }
}

//----------------------------------------------------------------

int main() {
    srand48(time(0));

    float *M, *x, *y;
    float *dev_M, *dev_x, *dev_y;
    size_t pitch;

    M = (float *) malloc(N*N*sizeof(float));
    x = (float *) malloc(N*sizeof(float));
    y = (float *) malloc(N*sizeof(float));

    cudaMallocPitch((void **) &dev_M, &pitch, N*sizeof(float), N);
    cudaMallocPitch((void **) &dev_x, &pitch, N*sizeof(float), N);
    cudaMallocPitch((void **) &dev_y, &pitch, N*sizeof(float), N);

    //----------------------------------------------------------------
    // GPU Calculation

    // Testing matrix and vector
    for (int i=0; i<N; ++i) {
        for (int j=0; j<N; ++j) {
            M[i*N + j] = (float) drand48();
        }
        x[i] = (float) drand48();
    }

    // *** NOTE: The CUDA events are for getting the elapsed time for the GPU calculation. ***
    cudaEvent_t startWithMemGPU, stopWithMemGPU;
    cudaEvent_t startNoMemGPU_1, stopNoMemGPU_1;
    cudaEvent_t startNoMemGPU_2, stopNoMemGPU_2;
    cudaEventCreate(&startWithMemGPU);
    cudaEventCreate(&stopWithMemGPU);
    cudaEventRecord(startWithMemGPU, 0);

    // Copies data to the GPU so it can perform the calculation.
    cudaMemcpy(dev_M, M, N*N*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_x, x, N*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy2D(dev_M, pitch, M, N*sizeof(float), N*sizeof(float), N, cudaMemcpyHostToDevice);
    cudaMemcpy2D(dev_x, pitch, x, N*sizeof(float), N*sizeof(float), 1, cudaMemcpyHostToDevice);

    cudaEventCreate(&startNoMemGPU_1);
    cudaEventCreate(&stopNoMemGPU_1);
    cudaEventRecord(startNoMemGPU_1, 0);

    // GPU Calculation kernel
    MxV<<<NUM_BLOCKS, BLOCK_SIZE>>>(dev_M, dev_x, dev_y, pitch, N);

    cudaEventRecord(stopNoMemGPU_1, 0);
    cudaEventSynchronize(stopNoMemGPU_1);

    cudaDeviceSynchronize();
    cudaMemcpy2D(y, N*sizeof(float), dev_y, pitch, N*sizeof(float), 1, cudaMemcpyDeviceToHost);

    cudaEventCreate(&startNoMemGPU_2);
    cudaEventCreate(&stopNoMemGPU_2);
    cudaEventRecord(startNoMemGPU_2, 0);

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
    // CPU Calculation
    cudaEvent_t startCPU, stopCPU;
    cudaEventCreate(&startCPU);
    cudaEventCreate(&stopCPU);
    cudaEventRecord(startCPU, 0);

    float *cpuResult;
    cpuResult = (float *) malloc(N*sizeof(float));
    for (int i=0; i<N; ++i) {
        cpuResult[i] = 0;
        for(int j=0; j<N; ++j) {
            cpuResult[i] = cpuResult[i] + (M[i*N + j] * x[j]);    
        }
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
    float u_minus_v_norm = 0.0;
    float v_norm = 0.0;
    for (int i=0; i<N; ++i) {
        u_minus_v_norm = u_minus_v_norm + (float) pow((y[i] - cpuResult[i]), 2);
    }
    u_minus_v_norm = (float) sqrt(u_minus_v_norm);

    for (int i=0; i<N; ++i) {
        v_norm = v_norm + (float) pow(cpuResult[i], 2);
    }
    v_norm = (float) sqrt(v_norm);

    relativeError = (float) (u_minus_v_norm / v_norm);

    //----------------------------------------------------------------
    // Results
    printf("GPU Result: [");
    for (int i=0; i<5; ++i) {
        printf("%0.4f ", y[i]);
    }
    printf("... ");
    printf("%0.4f]\n", y[N-1]);

    printf("CPU Result: [");
    for (int i=0; i<5; ++i) {
        printf("%0.4f ", cpuResult[i]);
    }
    printf("... ");
    printf("%0.4f]\n", cpuResult[N-1]);

    printf("Relative Error: %0.8f\n", relativeError);
    printf("CPU Elapsed Time: %0.4f\n", cpuElapsed);
    printf("GPU Elapsed Time (with mem copies): %0.4f\n", gpuElapsedWithMem);
    printf("GPU Elapsed Time (no mem copies): %0.4f\n", gpuElapsedNoMem_total);

    //----------------------------------------------------------------
    // Cleanup
    cudaFree(dev_M);
    cudaFree(dev_x);
    cudaFree(dev_y);

    free(M);
    free(x);
    free(y);
    free(cpuResult);

    return 0;
}