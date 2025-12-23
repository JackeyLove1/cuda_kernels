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

// Optimition 5: Double Buffering
// Use 2 sets of Shared Memory to overlap Main Loop Global Memory Load and Math
template<int BLOCK_SIZE, int THREAD_TILE_M, int THREAD_TILE_N>
__global__ void matrix_multiplication_kernel(const float *__restrict__ A,
                                             const float *__restrict__ B,
                                             float *__restrict__ C,
                                             const int M,
                                             const int N,
                                             const int K)
{
    // Double Buffering: [2]
    __shared__ __align__(16) float As[2][BLOCK_SIZE * THREAD_TILE_M][BLOCK_SIZE + 1];
    __shared__ __align__(16) float Bs[2][BLOCK_SIZE][BLOCK_SIZE * THREAD_TILE_N];

    const auto bx = blockIdx.x;
    const auto by = blockIdx.y;

    // Warp Tile Swizzle Logic
    const int tid = threadIdx.y * blockDim.x + threadIdx.x;
    const int warp_id = tid / 32;
    const int lane_id = tid % 32;

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
    const auto row_base = by * (BLOCK_SIZE * THREAD_TILE_M) + ty * THREAD_TILE_M;
    const auto col_base = bx * (BLOCK_SIZE * THREAD_TILE_N) + tx * THREAD_TILE_N;

    // Accumulators
    float sum[THREAD_TILE_M][THREAD_TILE_N] = {0.f};

    // Registers for prefetching global memory
    float ldg_a_reg[THREAD_TILE_M];
    float ldg_b_reg[THREAD_TILE_N];

    const int num_tiles = CDIV(K, BLOCK_SIZE);

    // Prologue: Load Tile 0
    {
        int t = 0;
        // Load A
        for (int i = 0; i < THREAD_TILE_M; ++i) {
            const int row = row_base + i;
            const int a_col = t * BLOCK_SIZE + tx;
            ldg_a_reg[i] = (row < M && a_col < K) ? A[row * K + a_col] : 0.f;
        }
        // Load B
        for (int j = 0; j < THREAD_TILE_N; ++j) {
            const int col = col_base + j;
            const int b_row = t * BLOCK_SIZE + ty;
            ldg_b_reg[j] = (b_row < K && col < N) ? B[b_row * N + col] : 0.f;
        }
        
        // Write to Shared Memory Buffer 0
        for (int i = 0; i < THREAD_TILE_M; ++i) {
            As[0][ty * THREAD_TILE_M + i][tx] = ldg_a_reg[i];
        }
        for (int j = 0; j < THREAD_TILE_N; ++j) {
            Bs[0][ty][tx * THREAD_TILE_N + j] = ldg_b_reg[j];
        }
    }

    __syncthreads();

    // Main Loop with Unrolling to constant-fold shared memory indices
    int t = 0;
    while (t < num_tiles) {
        
        // Iteration t (Even): Compute from Buf0, Load t+1, Store to Buf1
        {
            // 1. Prefetch Next Tile (t+1) to Registers
            if (t < num_tiles - 1) {
                int next_t = t + 1;
                // Load A
                for (int i = 0; i < THREAD_TILE_M; ++i) {
                    const int row = row_base + i;
                    const int a_col = next_t * BLOCK_SIZE + tx;
                    ldg_a_reg[i] = (row < M && a_col < K) ? A[row * K + a_col] : 0.f;
                }
                // Load B
                for (int j = 0; j < THREAD_TILE_N; ++j) {
                    const int col = col_base + j;
                    const int b_row = next_t * BLOCK_SIZE + ty;
                    ldg_b_reg[j] = (b_row < K && col < N) ? B[b_row * N + col] : 0.f;
                }
            }

            // 2. Compute Current Tile (t) using Shared Memory Buffer 0
            #pragma unroll
            for (int k = 0; k < BLOCK_SIZE; ++k) {
                #pragma unroll
                for (int i = 0; i < THREAD_TILE_M; ++i) {
                    #pragma unroll
                    for (int j = 0; j < THREAD_TILE_N; ++j) {
                        sum[i][j] += As[0][ty * THREAD_TILE_M + i][k] * Bs[0][k][tx * THREAD_TILE_N + j];
                    }
                }
            }

            // 3. Store Next Tile to Shared Memory Buffer 1
            if (t < num_tiles - 1) {
                for (int i = 0; i < THREAD_TILE_M; ++i) {
                    As[1][ty * THREAD_TILE_M + i][tx] = ldg_a_reg[i];
                }
                for (int j = 0; j < THREAD_TILE_N; ++j) {
                    Bs[1][ty][tx * THREAD_TILE_N + j] = ldg_b_reg[j];
                }
            }

            __syncthreads();
        }
        
        t++;
        if (t >= num_tiles) break;

        // Iteration t (Odd): Compute from Buf1, Load t+1, Store to Buf0
        {
             // 1. Prefetch Next Tile (t+1) to Registers
            if (t < num_tiles - 1) {
                int next_t = t + 1;
                // Load A
                for (int i = 0; i < THREAD_TILE_M; ++i) {
                    const int row = row_base + i;
                    const int a_col = next_t * BLOCK_SIZE + tx;
                    ldg_a_reg[i] = (row < M && a_col < K) ? A[row * K + a_col] : 0.f;
                }
                // Load B
                for (int j = 0; j < THREAD_TILE_N; ++j) {
                    const int col = col_base + j;
                    const int b_row = next_t * BLOCK_SIZE + ty;
                    ldg_b_reg[j] = (b_row < K && col < N) ? B[b_row * N + col] : 0.f;
                }
            }

            // 2. Compute Current Tile (t) using Shared Memory Buffer 1
            #pragma unroll
            for (int k = 0; k < BLOCK_SIZE; ++k) {
                #pragma unroll
                for (int i = 0; i < THREAD_TILE_M; ++i) {
                    #pragma unroll
                    for (int j = 0; j < THREAD_TILE_N; ++j) {
                        sum[i][j] += As[1][ty * THREAD_TILE_M + i][k] * Bs[1][k][tx * THREAD_TILE_N + j];
                    }
                }
            }

            // 3. Store Next Tile to Shared Memory Buffer 0
            if (t < num_tiles - 1) {
                for (int i = 0; i < THREAD_TILE_M; ++i) {
                    As[0][ty * THREAD_TILE_M + i][tx] = ldg_a_reg[i];
                }
                for (int j = 0; j < THREAD_TILE_N; ++j) {
                    Bs[0][ty][tx * THREAD_TILE_N + j] = ldg_b_reg[j];
                }
            }

            __syncthreads();
        }
        t++;
    }

    // Write Result
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
Verification PASSED (checked 1000 random elements). Max Error: 0.000183
Average Runtime: 5.287733 ms
Compute Performance: 12.996018 TFLOPS
Memory Bandwidth: 25.382849 GB/s
 **/