#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

#define WARP_SIZE 32
#define TILE_SIZE 32

template <typename T, typename U> __host__ __device__ __forceinline__ auto CDIV(T a, U b) { return (a + b - 1) / b; }
template <typename T>
__forceinline__ __host__ __device__ auto OFFSET(T row, T col, T ld)
{
    return row * ld + col;
}

#define CHECK_CUDA(func)                                                                                              \
    {                                                                                                                 \
        cudaError_t status = (func);                                                                                  \
        if (status != cudaSuccess) {                                                                                  \
            printf("CUDA API failed at line %d with error: %s (%d)\n", __LINE__, cudaGetErrorString(status), status); \
            exit(EXIT_FAILURE);                                                                                       \
        }                                                                                                             \
    }

// simple version
__global__ void matrix_multiplication_kernel(const float *__restrict__ A,
                                             const float *__restrict__ B,
                                             float *__restrict__ C,
                                             const int M,
                                             const int N,
                                             const int K)
{
    const auto tx = threadIdx.x + blockDim.x * blockIdx.x;
    const auto ty = threadIdx.y + blockDim.y * blockIdx.y;
    if (tx < N && ty < M) {
        float c = 0;
#pragma unroll
        for (int k = 0; k < K; ++k) {
            c += A[ty * K + k] * B[k * N + tx];
        }
        C[ty * N + tx] = c;
    }
}

// A: M x K, B: K x N, C: M x N
extern "C" void solve(const float *A, const float *B, float *C, int M, int N, int K)
{
    dim3 threadsPerBlock(TILE_SIZE, TILE_SIZE);
    dim3 blocksPerGrid(CDIV(N, TILE_SIZE), CDIV(M, TILE_SIZE));

    matrix_multiplication_kernel<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, M, N, K);
    cudaDeviceSynchronize();
}

int main(int argc, char **argv)
{
    int M = 4096;
    int N = 4096;
    int K = 4096;

    printf("Benchmarking SGEMM with M=%d, N=%d, K=%d\n", M, N, K);

    size_t size_A = (size_t)M * K * sizeof(float);
    size_t size_B = (size_t)K * N * sizeof(float);
    size_t size_C = (size_t)M * N * sizeof(float);

    float *h_A = (float *)malloc(size_A);
    float *h_B = (float *)malloc(size_B);

    if (!h_A || !h_B) {
        fprintf(stderr, "Host memory allocation failed\n");
        return EXIT_FAILURE;
    }

    // Initialize with random values
    for (size_t i = 0; i < (size_t)M * K; ++i)
        h_A[i] = (float)(rand() % 100) / 100.0f;
    for (size_t i = 0; i < (size_t)K * N; ++i)
        h_B[i] = (float)(rand() % 100) / 100.0f;

    float *d_A, *d_B, *d_C;
    CHECK_CUDA(cudaMalloc(&d_A, size_A));
    CHECK_CUDA(cudaMalloc(&d_B, size_B));
    CHECK_CUDA(cudaMalloc(&d_C, size_C));

    CHECK_CUDA(cudaMemcpy(d_A, h_A, size_A, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B, h_B, size_B, cudaMemcpyHostToDevice));

    // Warmup
    solve(d_A, d_B, d_C, M, N, K);
    CHECK_CUDA(cudaDeviceSynchronize());

    // Benchmark
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    int num_runs = 100;
    CHECK_CUDA(cudaEventRecord(start));
    for (int i = 0; i < num_runs; ++i) {
        solve(d_A, d_B, d_C, M, N, K);
    }
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float milliseconds = 0;
    CHECK_CUDA(cudaEventElapsedTime(&milliseconds, start, stop));

    float avg_time_ms = milliseconds / num_runs;
    printf("Average Runtime: %f ms\n", avg_time_ms);

    // Calculate TFLOPS
    // FLOPs = 2 * M * N * K
    double flops  = 2.0 * (double)M * (double)N * (double)K;
    double tflops = (flops * 1e-12) / (avg_time_ms * 1e-3);
    printf("Compute Performance: %f TFLOPS\n", tflops);

    // Effective Memory Bandwidth (R+W)
    // 2 reads (A, B) + 1 write (C)
    double total_bytes  = sizeof(float) * ((double)M * K + (double)K * N + (double)M * N);
    double bandwidth_gb = (total_bytes * 1e-9) / (avg_time_ms * 1e-3);
    printf("Memory Bandwidth: %f GB/s\n", bandwidth_gb);

    free(h_A);
    free(h_B);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return 0;
}

/**
Benchmarking cuBLAS SGEMM with M=2048, N=4096, K=4096
Average Runtime: 44.334274 ms
Compute Performance: 1.550030 TFLOPS
Memory Bandwidth: 3.027403 GB/s
 */