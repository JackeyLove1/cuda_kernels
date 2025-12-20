#include <cub/cub.cuh>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>

int main()
{
    const int N = 100'000'000;

    size_t size_input  = N * sizeof(float);
    size_t size_output = sizeof(float);

    float *h_input  = (float *)malloc(size_input);
    float *h_output = (float *)malloc(size_output);

    // Initialize input with random data
    for (int i = 0; i < N; ++i) {
        h_input[i] = (float)(rand() % 100) / 10.0f;
    }

    float *d_input, *d_output;
    cudaMalloc((void **)&d_input, size_input);
    cudaMalloc((void **)&d_output, size_output);

    cudaMemcpy(d_input, h_input, size_input, cudaMemcpyHostToDevice);
    cudaMemset(d_output, 0, size_output);

    // Determine temporary device storage requirements
    void     *d_temp_storage     = NULL;
    size_t   temp_storage_bytes  = 0;
    cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, d_input, d_output, N);

    // Allocate temporary storage
    cudaMalloc(&d_temp_storage, temp_storage_bytes);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    
    // Run sum-reduction
    cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, d_input, d_output, N);
    
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    printf("Time: %f ms\n", milliseconds);
    printf("Bandwidth: %f GB/s\n", (size_input / 1e9) / (milliseconds / 1000));

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    cudaMemcpy(h_output, d_output, size_output, cudaMemcpyDeviceToHost);

    // Verify
    double expected = 0.0;
    for (int i = 0; i < N; ++i) {
        expected += (double)h_input[i];
    }

    printf("Input size: %d\n", N);
    printf("GPU Result: %f\n", *h_output);
    printf("CPU Result: %f\n", expected);

    // Check for errors (relative error or absolute error depending on magnitude)
    double diff = fabs((double)(*h_output) - expected);
    double relative_error = expected != 0.0 ? diff / fabs(expected) : diff;

    if (relative_error < 1e-4) {
        printf("PASSED\n");
    }
    else {
        printf("FAILED (diff: %f, rel: %e)\n", diff, relative_error);
    }

    cudaFree(d_temp_storage);
    cudaFree(d_input);
    cudaFree(d_output);
    free(h_input);
    free(h_output);

    return 0;
}

/**
Time: 0.579584 ms
Bandwidth: 690.150198 GB/s
Input size: 100000000
GPU Result: 495011456.000000
CPU Result: 495011476.122117
**/
