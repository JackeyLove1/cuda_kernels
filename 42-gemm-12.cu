#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <cmath>
#include <cassert>

#define WARP_SIZE 32
#define TILE 32

template <typename T, typename U> __host__ __device__ __forceinline__ auto CDIV(T a, U b) { return (a + b - 1) / b; }

#define CHECK_CUDA(func)                                                                                              \
    {                                                                                                                 \
        cudaError_t status = (func);                                                                                  \
        if (status != cudaSuccess) {                                                                                  \
            printf("CUDA API failed at line %d with error: %s (%d)\n", __LINE__, cudaGetErrorString(status), status); \
            exit(EXIT_FAILURE);                                                                                       \
        }                                                                                                             \
    }


template <const int BM=32, const int BN=32, const int BK=32, const int TM=8>
__global__ void matrix_multiplication_kernel(const float * A,
                                             const float * B,
                                             float * C,
                                             const int M,
                                             const int N,
                                             const int K)
{
    assert(BM * BK == blockDim.x);
    assert(BK * BN == blockDim.x);
    assert(BM % TM == 0);


    const auto cCol = blockIdx.x;
    const auto cRow = blockIdx.y;

    __shared__ __align__(128) float As[BM][BK];
    __shared__ __align__(128) float Bs[BK][BN];

    // advance the pointer to block tile
    A += cRow * BM * K;
    B += cCol * BN;
    C += cRow * BM * N + cCol * BN;


    float threadResults[TM] = {0.f};

    // copy thread location
    const auto innerACol = threadIdx.x % BK;
    const auto innerARow = threadIdx.x / BK;
    const auto innerBCol = threadIdx.x % BN;
    const auto innerBRow = threadIdx.x / BN;

    // compute thread location
    const auto threadRow = threadIdx.x / BN;
    const auto threadCol = threadIdx.x % BN;

    // traverse the K to calculate the C block Tile
    // 1. copy the A and B to share memory
    // 2. copy B warp tile to register
    // 3. traverse As to multiply B resigter to threadResult
    // 4. write back threadResult to C
    for (int bk = 0; bk < K; bk += BK) {
        As[innerARow * BK + innerACol] = A[innerARow * K + innerACol];
        Bs[innerBRow * BN + innerBCol] = B[innerBRow * N + innerBCol];
        __syncthreads();

        // Advanced A and B point
        A += BK;
        B += BK * N;

        for (int dotk = 0; dotk < BK; ++dotk) {
            float tmpB = Bs[dotk * BK + threadCol];
            for (int resIdx = 0; resIdx < TM; ++resIdx) {
                threadResults[resIdx] = As[ (threadCol * TM + resIdx) * BK + dotk] * tmpB;
            }
        }
        __syncthreads();



    }

}

// A: M x K, B: K x N, C: M x N
extern "C" void solve(const float *A, const float *B, float *C, int M, int N, int K)
{
    constexpr int BS = 32;

    dim3 threadsPerBlock(BS, BS); // 1024 线程覆盖 32x32 输出
    dim3 blocksPerGrid(CDIV(N, BS), CDIV(M, BS));

    matrix_multiplication_kernel<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, M, N, K);
    cudaDeviceSynchronize();
}

void verify_result(float *h_A, float *h_B, float *h_C, int M, int N, int K) {
    printf("Verifying result...\n");
    float max_error = 0.0f;
    int error_count = 0;
    // 随机检查 1000 个点以节省 CPU 时间
    for (int i = 0; i < 1000; ++i) {
        int row = rand() % M;
        int col = rand() % N;
        double sum = 0.0; // 使用 double 保证基准计算精度
        for (int k = 0; k < K; ++k) {
            sum += (double)h_A[row * K + k] * (double)h_B[k * N + col];
        }
        float diff = std::abs(h_C[row * N + col] - (float)sum);
        if (diff > max_error) max_error = diff;

        // 设定容差，对于大规模累加，浮点数误差是正常的
        if (diff > 1e-2f) {
            if (error_count < 5) {
                printf("Error at (%d, %d): GPU=%f, CPU=%f, Diff=%f\n", row, col, h_C[row * N + col], (float)sum, diff);
            }
            error_count++;
        }
    }
    if (error_count == 0) {
        printf("Verification PASSED (checked 1000 random elements). Max Error: %e\n", max_error);
    } else {
        printf("Verification FAILED. Total errors: %d, Max Error: %e\n", error_count, max_error);
    }
}

int main(int argc, char **argv)
{
    int M = 2048;
    int N = 1024;
    int K = 4096;

    printf("Benchmarking cuBLAS SGEMM with M=%d, N=%d, K=%d\n", M, N, K);

    size_t size_A = (size_t)M * K * sizeof(float);
    size_t size_B = (size_t)K * N * sizeof(float);
    size_t size_C = (size_t)M * N * sizeof(float);

    float *h_A = (float *)malloc(size_A);
    float *h_B = (float *)malloc(size_B);
    float *h_C = (float *)malloc(size_C); // 分配 h_C 用于保存 GPU 结果

    if (!h_A || !h_B || !h_C) {
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

    // --- 验证准确性 ---
    CHECK_CUDA(cudaMemcpy(h_C, d_C, size_C, cudaMemcpyDeviceToHost));
    verify_result(h_A, h_B, h_C, M, N, K);

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
Average Runtime: 7.286674 ms
Compute Performance: 2.357711 TFLOPS
Memory Bandwidth: 8.058581 GB/s


 */