#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <cmath>

#define CHECK_CUDA(func) \
    { \
        cudaError_t status = (func); \
        if (status != cudaSuccess) { \
            printf("CUDA API failed at line %d with error: %s (%d)\n", __LINE__, cudaGetErrorString(status), status); \
            exit(EXIT_FAILURE); \
        } \
    }

template <typename T, typename U> __host__ __device__ __forceinline__ auto CDIV(T a, U b) { return (a + b - 1) / b; }

// Tunable parameters for RTX 3080
// Block Tile: 128x128
// K Tile: 8
// Thread Tile: 8x8
// Threads: 16x16 = 256
const int BM = 128;
const int BN = 128;
const int BK = 8;
const int TM = 8;
const int TN = 8;

__global__ void sgemm_register_tiled(const float * __restrict__ A, const float * __restrict__ B, float * __restrict__ C, int M, int N, int K) {
    // Thread index
    const int tx = threadIdx.x; // 0..255
    
    // Block index
    const int bx = blockIdx.x;
    const int by = blockIdx.y;

    // Calculate thread's position in the block
    // We map 256 threads to a logical grid of (BM/TM) x (BN/TN) = 16 x 16 tile-workers
    // threadIdx.x goes 0..255.
    // row within block (in units of threads) = tx / (BN/TN) = tx / 16
    // col within block (in units of threads) = tx % (BN/TN) = tx % 16
    const int thread_row = tx / (BN / TN);
    const int thread_col = tx % (BN / TN);

    // Global memory pointers for this block
    // A: move to row (by * BM)
    // B: move to col (bx * BN)
    const float *A_ptr = A + by * BM * K;
    const float *B_ptr = B + bx * BN;
    float *C_ptr = C + (by * BM + thread_row * TM) * N + (bx * BN + thread_col * TN);

    // Shared memory
    __shared__ float As[BK][BM];
    __shared__ float Bs[BK][BN];

    // Registers for accumulation
    float accum[TM][TN] = {0.0f};

    // Registers for loading from SMEM
    float reg_a[TM];
    float reg_b[TN];

    // Load Global -> Shared stride
    // Threads: 256. Elements to load: BM*BK = 128*8 = 1024.
    // Each thread loads 1024/256 = 4 elements.
    // We need to carefully map threads to load A and B.
    
    // For A (row-major): M rows, K cols. 
    // We want to load a [BM x BK] tile from A (transposed effectively if we think about K as inner dim)
    // Actually, A is M x K. Tile is [by*BM : by*BM+BM, k : k+BK].
    // Dimensions of As tile: BM rows, BK cols.
    // Wait, standard is As[BM][BK] if we want A[row][k]. 
    // To minimize bank conflicts in broadcasting, we might want As[BK][BM] (transposed) or standard.
    // Let's stick to standard layout for simplicity first: As[BM][BK]? 
    // No, standard optimization often transposes A in shared mem to allow vectorized loads or better stride.
    // Let's stick to simplest correct logic:
    // A tile: [BM rows x BK cols].
    // B tile: [BK rows x BN cols].
    
    // Let's use 1D array for SMEM to avoid confusion or [BK][BM] / [BM][BK] mismatches.
    // But 2D is easier to read.
    // If we load A into As[BM][BK] (row-major), and we iterate k=0..BK-1.
    // In inner loop, we need A[thread_row*TM + i][k].
    // This requires loading from different rows. 
    // If we transpose A in SMEM -> As[BK][BM], then A[..][k] becomes As[k][..], which is contiguous.
    // Let's use As[BK][BM] (transposed A) and Bs[BK][BN] (normal B).

    // A_ptr points to A[by*BM][0].
    // We need to load A[row][k] -> As[k][row].
    
    // Threads loading A: 256 threads. 128*8 = 1024 elements.
    // Each thread loads 4 elements.
    // Stride = 256.
    // Inner loop for loading
    int a_inner_row = tx / BK; // 0..31
    int a_inner_col = tx % BK; // 0..7
    int a_stride = 256 / BK; // 32 rows per step
    
    // Threads loading B: 256 threads. 8*128 = 1024 elements.
    // B tile is BK rows x BN cols.
    // Load B[row][col] -> Bs[row][col].
    int b_inner_row = tx / BN; // 0..1 (since 256/128=2)
    int b_inner_col = tx % BN; // 0..127
    int b_stride = 256 / BN; // 2 rows per step

    for (int k = 0; k < K; k += BK) {
        // 1. Load Global -> Shared
        
        // Load A (transposed: A[row][col] -> As[col][row])
        // We load 4 elements per thread.
        // Mapping: flat index i (0..1023). 
        // row = i / BK, col = i % BK.
        // A is M x K. A_ptr is at start of block row.
        // We want A[row][k_offset + col].
        // But wait, K is large, BK is small.
        
        // Simpler load scheme: 
        // Thread i loads A_tile[i], A_tile[i+256], A_tile[i+512], A_tile[i+768].
        // A_tile is BM x BK.
        // We want As[col][row] = A[global_row][global_col].
        // Let's map thread to (row, col) of the tile.
        // 1024 elements.
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
             int idx = tx + i * 256;
             // For A: Tile is BMxBK.
             int row = idx / BK; // 0..127
             int col = idx % BK; // 0..7
             // Load A[global_row][global_k]
             // A_ptr points to A[by*BM][k_current_base]
             // Actually A_ptr initialized to A[by*BM][0]. 
             // We need to adjust pointer.
             // Let's just use raw indices.
             int global_row = by * BM + row;
             int global_col = k + col;
             float val = (global_row < M && global_col < K) ? A[global_row * K + global_col] : 0.0f;
             // Transposed store
             As[col][row] = val;
        }

        // Load B: Tile is BKxBN.
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
             int idx = tx + i * 256;
             int row = idx / BN; // 0..7
             int col = idx % BN; // 0..127
             
             int global_row = k + row;
             int global_col = bx * BN + col;
             float val = (global_row < K && global_col < N) ? B[global_row * N + global_col] : 0.0f;
             Bs[row][col] = val;
        }

        __syncthreads();

        // 2. Compute (Register Tiling)
        #pragma unroll
        for (int dot_k = 0; dot_k < BK; ++dot_k) {
            // Load A row fragment from SMEM to Register
            // We need A elements corresponding to our thread tile rows.
            // My thread computes C[thread_row*TM ...][thread_col*TN ...]
            // I need A[...][dot_k] and B[dot_k][...]
            // Since As is transposed (As[k][row]), we access As[dot_k][row].
            // We need 8 values: As[dot_k][thread_row*TM + 0..7]
            #pragma unroll
            for (int i = 0; i < TM; ++i) {
                reg_a[i] = As[dot_k][thread_row * TM + i];
            }
            
            // Load B col fragment
            // Bs[dot_k][thread_col*TN + 0..7]
            #pragma unroll
            for (int i = 0; i < TN; ++i) {
                reg_b[i] = Bs[dot_k][thread_col * TN + i];
            }

            // Outer Product
            #pragma unroll
            for (int r = 0; r < TM; ++r) {
                #pragma unroll
                for (int c = 0; c < TN; ++c) {
                    accum[r][c] += reg_a[r] * reg_b[c];
                }
            }
        }

        __syncthreads();
    }

    // 3. Store Result
    for (int r = 0; r < TM; ++r) {
        for (int c = 0; c < TN; ++c) {
            int global_row = by * BM + thread_row * TM + r;
            int global_col = bx * BN + thread_col * TN + c;
            if (global_row < M && global_col < N) {
                C[global_row * N + global_col] = accum[r][c];
            }
        }
    }
}

extern "C" void solve(const float *A, const float *B, float *C, int M, int N, int K) {
    dim3 block(256);
    dim3 grid(CDIV(N, BN), CDIV(M, BM));
    sgemm_register_tiled<<<grid, block>>>(A, B, C, M, N, K);
}

// Reuse the verify and main from the other file by compiling together or just copy-pasting basics.
// To make it standalone and easy to run, I'll copy the minimal main logic.

void verify_result(float *h_A, float *h_B, float *h_C, int M, int N, int K) {
    printf("Verifying result...\n");
    float max_error = 0.0f;
    int error_count = 0;
    for (int i = 0; i < 1000; ++i) {
        int row = rand() % M;
        int col = rand() % N;
        double sum = 0.0;
        for (int k = 0; k < K; ++k) {
            sum += (double)h_A[row * K + k] * (double)h_B[k * N + col];
        }
        float diff = std::abs(h_C[row * N + col] - (float)sum);
        if (diff > max_error) max_error = diff;
        if (diff > 1e-2f) {
            if (error_count < 5) printf("Error: CPU=%f, GPU=%f\n", sum, h_C[row * N + col]);
            error_count++;
        }
    }
    printf("Verification %s. Max Error: %e\n", error_count == 0 ? "PASSED" : "FAILED", max_error);
}

int main(int argc, char **argv) {
    int M = 2048;
    int N = 1024;
    int K = 4096;
    size_t size_A = M * K * sizeof(float);
    size_t size_B = K * N * sizeof(float);
    size_t size_C = M * N * sizeof(float);

    float *h_A = (float *)malloc(size_A);
    float *h_B = (float *)malloc(size_B);
    float *h_C = (float *)malloc(size_C);

    for (int i = 0; i < M * K; ++i) h_A[i] = (float)(rand() % 100) / 100.0f;
    for (int i = 0; i < K * N; ++i) h_B[i] = (float)(rand() % 100) / 100.0f;

    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, size_A);
    cudaMalloc(&d_B, size_B);
    cudaMalloc(&d_C, size_C);

    cudaMemcpy(d_A, h_A, size_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size_B, cudaMemcpyHostToDevice);

    solve(d_A, d_B, d_C, M, N, K); // Warmup
    cudaDeviceSynchronize();

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    for (int i = 0; i < 20; ++i) {
        solve(d_A, d_B, d_C, M, N, K);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    float avg_time = milliseconds / 20.0f;
    
    printf("Avg Time: %f ms\n", avg_time);
    double flops = 2.0 * M * N * K;
    printf("TFLOPS: %f\n", (flops * 1e-12) / (avg_time * 1e-3));

    cudaMemcpy(h_C, d_C, size_C, cudaMemcpyDeviceToHost);
    verify_result(h_A, h_B, h_C, M, N, K);

    return 0;
}

/**
Avg Time: 1.232230 ms
TFLOPS: 13.942091
Verifying result...
Verification PASSED. Max Error: 3.112793e-03
 */