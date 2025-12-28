#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

#define WARP_SIZE 32
#define TILE      32

template <typename T, typename U> __host__ __device__ __forceinline__ auto CDIV(T a, U b) { return (a + b - 1) / b; }

#define CHECK_CUDA(func)                                                                                              \
    {                                                                                                                 \
        cudaError_t status = (func);                                                                                  \
        if (status != cudaSuccess) {                                                                                  \
            printf("CUDA API failed at line %d with error: %s (%d)\n", __LINE__, cudaGetErrorString(status), status); \
            exit(EXIT_FAILURE);                                                                                       \
        }                                                                                                             \
    }

constexpr static int BLOCK_M     = 128;
constexpr static int BLOCK_N     = 128;
constexpr static int THREAD_NUMS = 256;
constexpr static int BLOCK_K     = 16;

namespace gemm {
template <const int BM, const int BN, const int BK, const int rowStrideA, const int rowStrideB>
__device__ void loadFromGmem(int          N,
                             int          K,
                             const float *A,
                             const float *B,
                             float       *As,
                             float       *Bs,
                             int          innerRowA,
                             int          innerColA,
                             int          innerRowB,
                             int          innerColB)
{
    float4 tmp;
    for (auto offset = 0; offset < BM; offset += rowStrideA) {
        tmp = reinterpret_cast<const float4 *>(&A[(innerRowA + offset) * K + innerColA * 4])[0];
        // traverse As matrix to store
        As[(innerColA * 4 + 0) * BM + innerRowA + offset] = tmp.x;
        As[(innerColA * 4 + 1) * BM + innerRowA + offset] = tmp.y;
        As[(innerColA * 4 + 2) * BM + innerRowA + offset] = tmp.z;
        As[(innerColA * 4 + 3) * BM + innerRowA + offset] = tmp.w;
    }

    for (auto offset = 0; offset < BK; offset += rowStrideB) {
        tmp = reinterpret_cast<const float4 *>(&B[(innerRowB + offset) * N + innerColB * 4])[0];
        reinterpret_cast<float4 *>(&Bs[(innerRowB + offset) * BN + innerColB * 4])[0] = tmp;
    }
}

template <const int BM,
          const int BN,
          const int BK,
          const int WM,
          const int WN,
          const int WMITER,
          const int WNITER,
          const int WSUBM,
          const int WSUBN,
          const int TM,
          const int TN>
__device__ void processFromSmem(float       *regM,
                                float       *regN,
                                float       *threadResults,
                                const float *As,
                                const float *Bs,
                                const uint   warpRow,
                                const uint   warpCol,
                                const uint   threadRowInWarp,
                                const uint   threadColInWarp)
{
    for (auto dotk = 0; dotk < BK; ++dotk) {
        for (auto wSubRowIdx = 0; wSubRowIdx < WMITER; ++wSubRowIdx) {
            for (auto i = 0; i < TM; ++i) {
                regM[wSubRowIdx * TM + i] =
                    As[dotk * BM + warpRow * WM + wSubRowIdx * WSUBM + threadRowInWarp * TM + i];
            }
        }

        for (auto wSubColIdx = 0; wSubColIdx < WNITER; ++wSubColIdx) {
            for (auto i = 0; i < TN; ++i) {
                regN[wSubColIdx * TN + i] =
                    Bs[dotk * BN + warpCol * WN + wSubColIdx * WSUBN + threadColInWarp * TN + i];
            }
        }

        // execute warp matmul
        for (auto wSubRowIdx = 0; wSubRowIdx < WMITER; ++wSubRowIdx) {
            for (auto wSubColIdx = 0; wSubColIdx < WNITER; ++wSubColIdx) {
                for (auto resIdxM = 0; resIdxM < TM; ++resIdxM) {
                    for (auto resIdxN = 0; resIdxN < TN; ++resIdxN) {
                        threadResults[(wSubRowIdx * TM + resIdxM) * (WNITER * TN) + (wSubColIdx * TN) + resIdxN] +=
                            regM[wSubRowIdx * TM + resIdxM] * regN[wSubColIdx * TN + resIdxN];
                    }
                }
            }
        }
    }
}
} // namespace gemm

template <const int BM,
          const int BN,
          const int BK,
          const int WM,
          const int WN,
          const int WNITER,
          const int TM,
          const int TN,
          const int NUM_THREADS>
__global__ __launch_bounds__(
    NUM_THREADS) void matrix_multiplication_kernel(float *A, float *B, float *C, const int M, const int N, const int K)
{
    assert(NUM_THREADS == blockDim.x);

    const auto cRow = blockIdx.y;
    const auto cCol = blockIdx.x;

    // warp tile
    const auto warpIdx = threadIdx.x / WARP_SIZE;
    const auto warpCol = warpIdx % (BN / WN);
    const auto warpRow = warpIdx / (BN / WN);

    constexpr auto WMITER = 1;
    constexpr auto WSUBM  = WM / WMITER;
    constexpr auto WSUBN  = WN / WNITER;

    // thread tile
    const auto threadIdxInWarp = threadIdx.x % WARP_SIZE;
    const auto threadRowInWarp = threadIdxInWarp / (WN / TN);
    const auto threadColInWarp = threadIdxInWarp % (WN / TN);

    // shared memory
    __shared__ __align__(128) float As[BM * BK];
    __shared__ __align__(128) float Bs[BK * BN];

    // advanced the point
    A += cRow * BM * K;
    B += cCol * BN;
    C += (cRow * BM + warpRow * WM) * N + cCol * BN + warpCol * WN;

    // copy thread index
    const auto     innerRowA  = threadIdx.x / (BK / 4);
    const auto     innerColA  = threadIdx.x % (BK / 4);
    constexpr auto rowStrideA = NUM_THREADS / (BK / 4);
    const auto     innerRowB  = threadIdx.x / (BN / 4);
    const auto     innerColB  = threadIdx.x % (BN / 4);
    constexpr auto rowStrideB = NUM_THREADS / (BN / 4);

    // register cache
    float threadResults[WMITER * TM * WNITER * TN] = {0.f};
    float regM[WMITER * TM]                        = {0.f};
    float regN[WNITER * TN]                        = {0.f};

    for (auto bkIdx = 0; bkIdx < K; bkIdx += BK) {
        gemm::loadFromGmem<BM, BN, BK, rowStrideA, rowStrideB>(
            N, K, A, B, As, Bs, innerRowA, innerColA, innerRowB, innerColB);
        __syncthreads();

        gemm::processFromSmem<BM, BN, BK, WM, WN, WMITER, WNITER, WSUBM, WSUBN, TM, TN>(
            regM, regN, threadResults, As, Bs, warpRow, warpCol, threadRowInWarp, threadColInWarp);
        A += BK;
        B += BK * N;
        __syncthreads();
    }

    for (auto wSubRowIdx = 0; wSubRowIdx < WMITER; ++wSubRowIdx) {
        for (auto wSubColIdx = 0; wSubColIdx < WNITER; ++wSubColIdx) {
            float *C_interim = C + (wSubRowIdx * WSUBM) * N + wSubColIdx * WSUBN;
            for (auto resIdxM = 0; resIdxM < TM; ++resIdxM) {
                for (auto resIdxN = 0; resIdxN < TN; resIdxN += 4) {
                    const int i = (wSubRowIdx * TM + resIdxM) * (WNITER * TN) + wSubColIdx * TN + resIdxN;
                    float4 tmp;
                    tmp.x = threadResults[i + 0];
                    tmp.y = threadResults[i + 1];
                    tmp.z = threadResults[i + 2];
                    tmp.w = threadResults[i + 3];
                    reinterpret_cast<float4 *>(
                        &C_interim[(threadRowInWarp * TM + resIdxM) * N + threadColInWarp * TN + resIdxN])[0] = tmp;
                }
            }
        }
    }
}


// A: M x K, B: K x N, C: M x N
extern "C" void solve(float *A, float *B, float *C, int M, int N, int K)
{

    dim3 threadsPerBlock(THREAD_NUMS);
    dim3 blocksPerGrid(CDIV(N, BLOCK_N), CDIV(M, BLOCK_M));

    matrix_multiplication_kernel<BLOCK_M, BLOCK_N, BLOCK_K, 64, 32, 1, 8, 8, THREAD_NUMS><<<blocksPerGrid, threadsPerBlock>>>(A, B, C, M, N, K);
    cudaDeviceSynchronize();
}

void verify_result(float *h_A, float *h_B, float *h_C, int M, int N, int K)
{
    printf("Verifying result...\n");
    float max_error   = 0.0f;
    int   error_count = 0;
    // 随机检查 1000 个点以节省 CPU 时间
    for (int i = 0; i < 1000; ++i) {
        int    row = rand() % M;
        int    col = rand() % N;
        double sum = 0.0; // 使用 double 保证基准计算精度
        for (int k = 0; k < K; ++k) {
            sum += (double)h_A[row * K + k] * (double)h_B[k * N + col];
        }
        float diff = std::abs(h_C[row * N + col] - (float)sum);
        if (diff > max_error)
            max_error = diff;

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
    }
    else {
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
Benchmarking cuBLAS SGEMM with M=2048, N=1024, K=4096
Verifying result...
Verification PASSED (checked 1000 random elements). Max Error: 3.112793e-03
Average Runtime: 1.009092 ms
Compute Performance: 17.025074 TFLOPS
Memory Bandwidth: 58.191170 GB/s
 */