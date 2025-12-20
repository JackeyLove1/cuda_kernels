#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>


template <typename T, typename K> __host__ __device__ __forceinline__ decltype(auto) CDIV(T a, K b)
{
    return (a + b - 1) / b;
}

__global__ void reduceSumKernel(const float *__restrict__ input, float *__restrict__ output, const int N)
{
    extern __shared__ float sdata[];
    const auto              tid        = threadIdx.x;
    auto                    globalId   = threadIdx.x + (blockDim.x * 2) * blockIdx.x;
    const auto              gridStride = (blockDim.x * 2) * gridDim.x;
    sdata[tid]                         = 0;
    while (globalId < N) {
        sdata[tid] += input[globalId];
        if (globalId + blockDim.x < N) {
            sdata[tid] += input[globalId + blockDim.x];
        }
        globalId += gridStride;
    }
    __syncthreads();

#pragma unroll
    for (auto step = blockDim.x / 2; step > 0; step >>= 1) {
        if (tid < step) {
            sdata[tid] += sdata[tid + step];
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicAdd(output, sdata[0]);
    }
}

extern "C" void solve(const float *input, float *output, int N)
{
    constexpr int threadsPerBlock = 512;
    // Calculate required blocks for one pass, but cap it to avoid atomic contention
    int blocksPerGrid = CDIV(N, threadsPerBlock * 2);
    if (blocksPerGrid > 256) {
        blocksPerGrid = 256;
    }
    reduceSumKernel<<<blocksPerGrid, threadsPerBlock, threadsPerBlock * sizeof(float)>>>(input, output, N);
    cudaDeviceSynchronize();
}

int main()
{
    const int N               = 1000000;
    const int threadsPerBlock = 512;
    // Calculate padded size to handle lack of boundary checks in kernel
    const int blocksPerGrid = CDIV(N, threadsPerBlock);
    const int padded_N      = blocksPerGrid * threadsPerBlock;

    size_t size_input  = padded_N * sizeof(float);
    size_t size_output = sizeof(float);

    float *h_input  = (float *)malloc(size_input);
    float *h_output = (float *)malloc(size_output);

    // Initialize input with random data
    for (int i = 0; i < N; ++i) {
        h_input[i] = (float)(rand() % 100) / 10.0f;
    }
    // Pad the rest with 0
    for (int i = N; i < padded_N; ++i) {
        h_input[i] = 0.0f;
    }

    float *d_input, *d_output;
    cudaMalloc((void **)&d_input, size_input);
    cudaMalloc((void **)&d_output, size_output);

    cudaMemcpy(d_input, h_input, size_input, cudaMemcpyHostToDevice);
    cudaMemset(d_output, 0, size_output);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    solve(d_input, d_output, N);
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
    // Relative error check might be better for large sums
    double relative_error = expected != 0.0 ? diff / fabs(expected) : diff;

    if (relative_error < 1e-4) {
        printf("PASSED\n");
    }
    else {
        printf("FAILED (diff: %f, rel: %e)\n", diff, relative_error);
    }

    cudaFree(d_input);
    cudaFree(d_output);
    free(h_input);
    free(h_output);

    return 0;
}
