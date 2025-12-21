#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <type_traits>

#define WARP_SIZE 32
#define FULL_MASK 0xFFFFFFFF
#define M_LOG2E_F 1.44269504089f

#define FLOAT4(value) (reinterpret_cast<float4 *>(&(value))[0])

template <typename T, typename U> __host__ __device__ __forceinline__ auto CDIV(T a, U b) { return (a + b - 1) / b; }

template <typename T> __forceinline__ __device__ auto WarpReduceSum(T value)
{
#pragma unroll
    for (int offsets = WARP_SIZE / 2; offsets > 0; offsets >>= 1) {
        value += __shfl_down_sync(FULL_MASK, value, offsets);
    }
    return value;
}

template<typename T>
__forceinline__ __device__ auto sigmoid(T value){
   if constexpr (std::is_same_v<T, float>) {
        return 1.0f / (1.0f + __expf(-value));
    } else if constexpr (std::is_same_v<T, float4>) {
        const auto v1 = value.x;
        const auto v2 = value.y;
        const auto v3 = value.z;
        const auto v4 = value.w;
        float4 result;
        result.x = sigmoid(v1);
        result.y = sigmoid(v2);
        result.z = sigmoid(v3);
        result.w = sigmoid(v4);
        return result;
    }
}

__global__ void sigmoid_kernel(const float *__restrict__ input, float *__restrict__ output, const int N)
{
    auto       id     = blockDim.x * blockIdx.x + threadIdx.x;
    const auto stride = gridDim.x * blockDim.x;
    for (; id * 4 < N; id += stride) {
        const auto baseIdx   = id * 4;
        const auto remaining = N - baseIdx;
        if (remaining >= 4) {
            const auto a = reinterpret_cast<const float4 *>(input)[id];
            const auto b = sigmoid(a);
            reinterpret_cast<float4*>(output)[id] = b;
        } else {
#pragma unroll
            for (auto j = baseIdx; j < N; ++j) {
                output[j] = sigmoid(input[j]);
            }
        }
    }
}

// input, output are device pointers
extern "C" void solve(const float *input, float *output, int N)
{
    constexpr int threadsPerBlock = 256;
    constexpr int maxBlocks       = 2048;
    int           blocksPerGrid   = std::min((N + threadsPerBlock - 1) / threadsPerBlock, maxBlocks);

    sigmoid_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, output, N);
    cudaDeviceSynchronize();
}

// CPU reference implementation for verification
float sigmoid_cpu(float x) {
    return 1.0f / (1.0f + expf(-x));
}

int main(int argc, char *argv[]) {
    // Default size: 32M elements (128MB)
    int N = 1 << 25;
    
    // Allow size to be specified via command line
    if (argc > 1) {
        N = atoi(argv[1]);
        if (N <= 0) {
            printf("Invalid size, using default: %d\n", 1 << 25);
            N = 1 << 25;
        }
    }
    
    size_t bytes = N * sizeof(float);
    printf("Testing Sigmoid Kernel\n");
    printf("======================\n");
    printf("Array size: %d elements (%.2f MB)\n", N, bytes / (1024.0f * 1024.0f));
    
    // Allocate host memory
    float *h_in = (float *)malloc(bytes);
    float *h_out = (float *)malloc(bytes);
    float *h_ref = (float *)malloc(bytes);
    
    if (!h_in || !h_out || !h_ref) {
        printf("Failed to allocate host memory\n");
        return 1;
    }
    
    // Initialize input with random values in a reasonable range
    srand(42); // Fixed seed for reproducibility
    for (int i = 0; i < N; i++) {
        // Range: -10 to 10 (sigmoid is most interesting in this range)
        h_in[i] = (float)(rand() % 20000 - 10000) / 1000.0f;
    }
    
    // Allocate device memory
    float *d_in, *d_out;
    cudaError_t err;
    
    err = cudaMalloc(&d_in, bytes);
    if (err != cudaSuccess) {
        printf("Failed to allocate device input memory: %s\n", cudaGetErrorString(err));
        return 1;
    }
    
    err = cudaMalloc(&d_out, bytes);
    if (err != cudaSuccess) {
        printf("Failed to allocate device output memory: %s\n", cudaGetErrorString(err));
        cudaFree(d_in);
        return 1;
    }
    
    // Copy input to device
    err = cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        printf("Failed to copy input to device: %s\n", cudaGetErrorString(err));
        cudaFree(d_in);
        cudaFree(d_out);
        return 1;
    }
    
    // Warmup runs
    const int warmup_runs = 10;
    printf("Warming up (%d runs)...\n", warmup_runs);
    for (int i = 0; i < warmup_runs; i++) {
        solve(d_in, d_out, N);
    }
    
    // Synchronize before timing
    cudaDeviceSynchronize();
    
    // Create CUDA events for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    // Benchmark with multiple runs
    const int num_runs = 100;
    printf("Running benchmark (%d iterations)...\n", num_runs);
    
    cudaEventRecord(start);
    for (int i = 0; i < num_runs; i++) {
        solve(d_in, d_out, N);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float total_ms = 0.0f;
    cudaEventElapsedTime(&total_ms, start, stop);
    float avg_ms = total_ms / num_runs;
    
    // Calculate bandwidth
    // Read: N floats (input)
    // Write: N floats (output)
    // Total: 2 * N * sizeof(float) bytes
    double total_bytes = 2.0 * N * sizeof(float);
    double bandwidth_gb_s = (total_bytes / (avg_ms / 1000.0)) / 1e9;
    
    // Calculate throughput (elements per second)
    double throughput = N / (avg_ms / 1000.0);
    
    printf("\nPerformance Results:\n");
    printf("-------------------\n");
    printf("Average time: %.4f ms\n", avg_ms);
    printf("Total time (%d runs): %.4f ms\n", num_runs, total_ms);
    printf("Throughput: %.2f million elements/sec\n", throughput / 1e6);
    printf("Bandwidth: %.2f GB/s\n", bandwidth_gb_s);
    
    // Cleanup
    cudaFree(d_in);
    cudaFree(d_out);
    free(h_in);
    free(h_out);
    free(h_ref);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    return 0;
}

