#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cuda_runtime.h>

#define WARP_SIZE 32
#define FULL_MASK 0xFFFFFFFF

template <typename T, typename U> __host__ __device__ __forceinline__ auto CDIV(T a, U b) { return (a + b - 1) / b; }

template <typename T> __forceinline__ __device__ auto WarpReduceSum(T value)
{
#pragma unroll
    for (int offsets = WARP_SIZE / 2; offsets > 0; offsets >>= 1) {
        value += __shfl_down_sync(FULL_MASK, value, offsets);
    }
    return value;
}

__global__ __launch_bounds__(256) void kernel(const float *__restrict__ A,
                                              const float *__restrict__ B,
                                              float    *result,
                                              const int N)
{
    __shared__ float smem[WARP_SIZE];
    const auto       tid     = threadIdx.x;
    const auto       stride  = gridDim.x * blockDim.x;
    const auto       idx     = threadIdx.x + blockIdx.x * blockDim.x;
    const auto       lane_id = tid & (WARP_SIZE - 1);
    const auto       warp_id = tid / WARP_SIZE;

    auto *vec_a = reinterpret_cast<const float4 *>(A);
    auto *vec_b = reinterpret_cast<const float4 *>(B);

    float     sum       = 0.f;
    const int vec_limit = N / 4;
    for (int i = idx; i < vec_limit; i += stride) {
        float4 va = vec_a[i];
        float4 vb = vec_b[i];
        sum += va.x * vb.x + va.y * vb.y + va.z * vb.z + va.w * vb.w;
    }

    const int remainder_start = vec_limit * 4;
    for (int i = remainder_start + idx; i < N; i += stride) {
        sum += A[i] * B[i];
    }

    sum = WarpReduceSum(sum);
    if (lane_id == 0) {
        smem[warp_id] = sum;
    }
    __syncthreads();

    if (warp_id == 0) {
        const auto warp_nums = CDIV(blockDim.x, WARP_SIZE);
        auto       val       = (lane_id < warp_nums) ? smem[lane_id] : 0.f;
        auto       block_sum = WarpReduceSum(val);
        if (lane_id == 0) {
            atomicAdd(result, block_sum);
        }
    }
}

// A, B, result are device pointers
extern "C" void solve(const float *A, const float *B, float *result, int N)
{
    constexpr auto threadPerBlock = 256;
    constexpr auto blockSize      = 256;
    kernel<<<blockSize, threadPerBlock>>>(A, B, result, N);
}

// CPU reference implementation for correctness check
float dot_product_cpu(const float *A, const float *B, int N)
{
    double sum = 0.0;
    for (int i = 0; i < N; ++i) {
        sum += (double)A[i] * (double)B[i];
    }
    return (float)sum;
}

int main()
{
    // Test parameters
    const int N               = 1024 * 1024 * 100; // 100M elements
    const int num_iterations  = 100;               // Number of iterations for warmup and timing
    const int num_timing_runs = 50;                // Number of runs for average timing

    printf("=== Dot Product Performance Test ===\n");
    printf("Array size: %d elements (%.2f MB per array)\n", N, N * sizeof(float) / (1024.0f * 1024.0f));
    printf("Total data size: %.2f MB\n", 2 * N * sizeof(float) / (1024.0f * 1024.0f));
    printf("\n");

    // Allocate host memory
    float *h_A      = (float *)malloc(N * sizeof(float));
    float *h_B      = (float *)malloc(N * sizeof(float));
    float *h_result = (float *)malloc(sizeof(float));

    if (!h_A || !h_B || !h_result) {
        printf("Error: Failed to allocate host memory\n");
        return 1;
    }

    // Initialize host data
    for (int i = 0; i < N; ++i) {
        h_A[i] = (float)(i % 100) / 100.0f;
        h_B[i] = (float)((i * 7) % 100) / 100.0f;
    }

    // Allocate device memory
    float *d_A, *d_B, *d_result;
    cudaMalloc(&d_A, N * sizeof(float));
    cudaMalloc(&d_B, N * sizeof(float));
    cudaMalloc(&d_result, sizeof(float));

    if (!d_A || !d_B || !d_result) {
        printf("Error: Failed to allocate device memory\n");
        return 1;
    }

    // Copy data to device
    cudaMemcpy(d_A, h_A, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, N * sizeof(float), cudaMemcpyHostToDevice);

    // Warmup
    printf("Warming up...\n");
    for (int i = 0; i < num_iterations; ++i) {
        cudaMemset(d_result, 0, sizeof(float));
        solve(d_A, d_B, d_result, N);
    }
    cudaDeviceSynchronize();

    // Correctness check
    printf("Checking correctness...\n");
    cudaMemset(d_result, 0, sizeof(float));
    solve(d_A, d_B, d_result, N);
    cudaMemcpy(h_result, d_result, sizeof(float), cudaMemcpyDeviceToHost);

    float cpu_result     = dot_product_cpu(h_A, h_B, N);
    float error          = fabsf(*h_result - cpu_result);
    float relative_error = error / fabsf(cpu_result);

    printf("CPU result: %.6f\n", cpu_result);
    printf("GPU result: %.6f\n", *h_result);
    printf("Absolute error: %.6e\n", error);
    printf("Relative error: %.6e\n", relative_error);

    if (relative_error > 1e-5f) {
        printf("WARNING: Results don't match! Relative error > 1e-5\n");
    }
    else {
        printf("Correctness check passed!\n");
    }
    printf("\n");

    // Performance measurement
    printf("Measuring performance...\n");

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Timing runs
    float total_time_ms = 0.0f;
    for (int i = 0; i < num_timing_runs; ++i) {
        cudaMemset(d_result, 0, sizeof(float));
        cudaEventRecord(start);
        solve(d_A, d_B, d_result, N);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        float elapsed_ms;
        cudaEventElapsedTime(&elapsed_ms, start, stop);
        total_time_ms += elapsed_ms;
    }

    float avg_time_ms = total_time_ms / num_timing_runs;
    float avg_time_s  = avg_time_ms / 1000.0f;

    // Calculate bandwidth
    // Read: A (N floats) + B (N floats) = 2 * N * sizeof(float) bytes
    // Write: result (1 float) = sizeof(float) bytes
    // Total: (2 * N + 1) * sizeof(float) bytes
    size_t total_bytes    = (2 * N + 1) * sizeof(float);
    float  bandwidth_gb_s = (total_bytes / (1024.0f * 1024.0f * 1024.0f)) / avg_time_s;

    // Calculate throughput (operations per second)
    // Each element requires: 1 multiply + 1 add = 2 operations
    float operations      = 2.0f * N;
    float throughput_gops = (operations / 1e9f) / avg_time_s;

    // Print results
    printf("=== Performance Results ===\n");
    printf("Average time: %.4f ms (%.6f s)\n", avg_time_ms, avg_time_s);
    printf("Bandwidth: %.2f GB/s\n", bandwidth_gb_s);
    printf("Throughput: %.2f GFLOP/s\n", throughput_gops);
    printf("\n");

    // Get device properties for reference
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("=== Device Information ===\n");
    printf("Device: %s\n", prop.name);
    printf("Compute Capability: %d.%d\n", prop.major, prop.minor);
    printf("Global Memory Bandwidth (theoretical): %.2f GB/s\n",
           prop.memoryBusWidth / 8.0f * prop.memoryClockRate * 2.0f / 1e6f);
    printf("\n");

    // Cleanup
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_result);
    free(h_A);
    free(h_B);
    free(h_result);

    return 0;
}

/**
Average time: 1.1636 ms (0.001164 s)
Bandwidth: 671.40 GB/s
Throughput: 180.23 GFLOP/s
 */