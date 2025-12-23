#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

#define WARP_SIZE 32
#define TILE_SIZE 16
#define BM 128
#define BN 128
#define BK 8
#define TM 8
#define TN 8

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

// Optimition 4: Warp Tile Swizzle
// Re-map thread indices to improve Warp Tile shape (locality)
template<int BLOCK_SIZE, int THREAD_TILE_M, int THREAD_TILE_N>
__global__ void matrix_multiplication_kernel(const float *__restrict__ A,
                                             const float *__restrict__ B,
                                             float *__restrict__ C,
                                             const int M,
                                             const int N,
                                             const int K)
{
    // As needs to be [BM][BK] = [BLOCK_SIZE * THREAD_TILE_M][BLOCK_SIZE]
    // Bs needs to be [BK][BN] = [BLOCK_SIZE][BLOCK_SIZE * THREAD_TILE_N]
    // Added padding to As to avoid bank conflicts (Stride 16 -> 17)
    __shared__ __align__(16) float As[BLOCK_SIZE * THREAD_TILE_M][BLOCK_SIZE + 1];
    __shared__ __align__(16) float Bs[BLOCK_SIZE][BLOCK_SIZE * THREAD_TILE_N];

    const auto bx = blockIdx.x;
    const auto by = blockIdx.y;

    // Warp Tile Swizzle Logic
    const int tid = threadIdx.y * blockDim.x + threadIdx.x;
    const int warp_id = tid / 32;
    const int lane_id = tid % 32;

    // Plan: 256 threads -> 8 warps.
    // Block Tile: 128x128 (MxN)
    // New Plan: 8x4 threads per warp (8 threads M, 4 threads N).
    // Warp Tile: (8*8)x(4*8) = 64x32.
    // Warps Grid: 2x4 (2 warps M, 4 warps N).
    // Total: (2*8)x(4*4) = 16x16 threads.

    const int WARP_COLS = 4;
    const int THREAD_ROWS = 8;
    const int THREAD_COLS = 4;

    const int warp_row = warp_id / WARP_COLS;
    const int warp_col = warp_id % WARP_COLS;

    const int thread_row = lane_id / THREAD_COLS;
    const int thread_col = lane_id % THREAD_COLS;

    // Swizzled ID
    const int ty = warp_row * THREAD_ROWS + thread_row;
    const int tx = warp_col * THREAD_COLS + thread_col;


    // Block handles BM x BN tile.
    // BM = BLOCK_SIZE * THREAD_TILE_M
    // BN = BLOCK_SIZE * THREAD_TILE_N
    const auto row_base = by * (BLOCK_SIZE * THREAD_TILE_M) + ty * THREAD_TILE_M;
    const auto col_base = bx * (BLOCK_SIZE * THREAD_TILE_N) + tx * THREAD_TILE_N;

    // register tiles for accumulation
    float sum[THREAD_TILE_M][THREAD_TILE_N] = {0.f};

    for (int t = 0; t < CDIV(K, BLOCK_SIZE); ++t) {
        for (int i = 0; i < THREAD_TILE_M; ++i) {
            const int row = row_base + i;
            const int a_col = t * BLOCK_SIZE + tx;
            // Access As as [row_in_tile][col_in_tile]
            // row_in_tile = ty * THREAD_TILE_M + i
            As[ty * THREAD_TILE_M + i][tx] = (row < M && a_col < K) ?
                A[row * K + a_col] : 0.f;
        }
        for (int j = 0; j < THREAD_TILE_N; ++j) {
            const int col = col_base + j;
            // Load B tile of size [BLOCK_SIZE][BN]
            // We need to load Bs[row_in_tile][col_in_tile]
            // row_in_tile corresponds to K dimension (0..BLOCK_SIZE-1)
            // col_in_tile corresponds to N dimension (0..BN-1)
            // Each thread loads a part of Bs.
            // There are BLOCK_SIZE*BLOCK_SIZE threads (256).
            // Bs size is BLOCK_SIZE * BN = 16 * 128 = 2048.
            // Each thread loads 2048/256 = 8 elements.
            // We use loop j (0..7) for this.

            // The mapping used in assignment: Bs[ty][tx * THREAD_TILE_N + j]
            // This implies:
            //   row_in_tile = ty  (0..15)
            //   col_in_tile = tx * THREAD_TILE_N + j  (0..127)

            // Global row index for B:
            //   t * BLOCK_SIZE + row_in_tile
            const int b_row = t * BLOCK_SIZE + ty;

            Bs[ty][tx * THREAD_TILE_N + j] = (b_row < K && col < N) ?
                B[b_row * N + col] : 0.f;
        }

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < BLOCK_SIZE; ++k) {
            #pragma unroll
            for (int i = 0; i < THREAD_TILE_M; ++i) {
                #pragma unroll
                for (int j = 0; j < THREAD_TILE_N; ++j) {
                    sum[i][j] += As[ty*THREAD_TILE_M + i][k] * Bs[k][tx*THREAD_TILE_N + j];
                }
            }
        }

        __syncthreads();
    }

#pragma unroll
    for (int i = 0; i < THREAD_TILE_M; ++i) {
        #pragma unroll
        for (int j = 0; j < THREAD_TILE_N; ++j) {
            const int row = row_base + i;
            const int col = col_base + j;
            if (row < M && col < N) {
                C[row * N + col] = sum[i][j];
            }
        }
    }

}

// A: M x K, B: K x N, C: M x N
extern "C" void solve(const float *A, const float *B, float *C, int M, int N, int K)
{
    dim3 threadsPerBlock(TILE_SIZE, TILE_SIZE);
    // Block computes BM x BN = (TILE_SIZE * TM) x (TILE_SIZE * TN) = 128 x 128
    dim3 blocksPerGrid(CDIV(N, (TILE_SIZE * TN)), CDIV(M, (TILE_SIZE * TM)));

    matrix_multiplication_kernel<TILE_SIZE, TM, TN><<<blocksPerGrid, threadsPerBlock>>>(A, B, C, M, N, K);
    cudaDeviceSynchronize();
}

void verify_result(float *h_A, float *h_B, float *h_C, int M, int N, int K) {
    float max_error = 0.0f;
    int error_count = 0;
    // Check 1000 random elements
    for (int i = 0; i < 1000; ++i) {
        int row = rand() % M;
        int col = rand() % N;
        float sum = 0.0f;
        for (int k = 0; k < K; ++k) {
            sum += h_A[row * K + k] * h_B[k * N + col];
        }
        float diff = std::abs(h_C[row * N + col] - sum);
        if (diff > max_error) max_error = diff;
        float ref = std::abs(sum);
        // Relative error tolerance for large sums
        if (diff > 1e-2f + 1e-3f * ref) {
            if (error_count < 10)
                printf("Error at (%d, %d): GPU=%f, CPU=%f, Diff=%f\n", row, col, h_C[row * N + col], sum, diff);
            error_count++;
        }
    }
    if (error_count == 0) {
        printf("Verification PASSED (checked 1000 random elements). Max Error: %f\n", max_error);
    } else {
        printf("Verification FAILED. Total errors found in sample: %d\n", error_count);
    }
}

int main(int argc, char **argv)
{
    int M = 2048;
    int N = 4096;
    int K = 4096;

    printf("Benchmarking cuBLAS SGEMM with M=%d, N=%d, K=%d\n", M, N, K);

    size_t size_A = (size_t)M * K * sizeof(float);
    size_t size_B = (size_t)K * N * sizeof(float);
    size_t size_C = (size_t)M * N * sizeof(float);

    float *h_A = (float *)malloc(size_A);
    float *h_B = (float *)malloc(size_B);
    float *h_C = (float *)malloc(size_C);

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

    // Verify result
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
    free(h_C);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return 0;
}

/**
Benchmarking cuBLAS SGEMM with M=2048, N=4096, K=4096
Verification PASSED (checked 1000 random elements). Max Error: 0.000183
Average Runtime: 4.741945 ms
Compute Performance: 14.491833 TFLOPS
Memory Bandwidth: 28.304360 GB/s
 **/