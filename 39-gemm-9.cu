#include <cooperative_groups.h>
#include <cstdio>
#include <cuda/pipeline>
#include <cuda_pipeline.h>
#include <cuda_runtime.h>

// ==========================================
// Hyperparameters
// ==========================================
#define BM 128
#define BN 128
#define BK 32   // Increased from 16 to 32 for better bandwidth hiding
#define TM 8
#define TN 4

// Padding to avoid bank conflicts
// 32 * 4 bytes = 128 bytes. 32 banks * 4 bytes = 128 bytes.
// Stride of 32 floats causes 32-way bank conflicts (all hit bank 0).
// Adding padding breaks this alignment.
#define PAD 4 

const int THREADS_PER_BLOCK = (BM * BN) / (TM * TN); // 512

#define CHECK_CUDA(func)                                                                                              \
    {                                                                                                                 \
        cudaError_t status = (func);                                                                                  \
        if (status != cudaSuccess) {                                                                                  \
            printf("CUDA API failed at line %d with error: %s (%d)\n", __LINE__, cudaGetErrorString(status), status); \
            exit(EXIT_FAILURE);                                                                                       \
        }                                                                                                             \
    }

template <typename T, typename U> __host__ __device__ __forceinline__ auto CDIV(T a, U b) { return (a + b - 1) / b; }

// ==========================================
// Device Helper Functions
// ==========================================

__device__ __forceinline__ void load_stage_async(
    cuda::pipeline<cuda::thread_scope_thread>& pipe,
    float (*As)[BM][BK + PAD], 
    float (*Bs)[BK][BN + PAD],
    const float* A_global_ptr,
    const float* B_global_ptr,
    int K_dim,
    int N_dim,
    int k_offset,
    int stage,
    int tid
) {
    // --- Load A (BM x BK) ---
    // Total floats: 128 * 32 = 4096.
    // Threads: 512. Floats per thread: 8.
    // float4 loads: 2 per thread.
    
    #pragma unroll
    for (int i = 0; i < 2; ++i) {
        int idx = tid + i * 512; 
        int row = idx / (BK / 4); // BK=32, BK/4=8. row = idx / 8.
        int col_vec = idx % (BK / 4);
        int col = col_vec * 4;

        pipe.producer_acquire();
        if (k_offset + col < K_dim) { // Simplified boundary check
             // Global A is Row-Major: [row * K + col]
             // Shared A is [row][col] (with padding handled by compiler if we use As[stage][row][col])
             // We cast Shared pointer to float4 to write 4 floats at once.
            cuda::memcpy_async(reinterpret_cast<float4*>(&As[stage][row][col]),
                               reinterpret_cast<const float4*>(&A_global_ptr[row * K_dim + col]),
                               sizeof(float4), pipe);
        }
        pipe.producer_commit();
    }

    // --- Load B (BK x BN) ---
    // Total floats: 32 * 128 = 4096.
    // Same: 2 float4 loads per thread.
    // B is Row-Major in Global.
    
    #pragma unroll
    for (int i = 0; i < 2; ++i) {
        int idx = tid + i * 512;
        int row = idx / (BN / 4); // BN=128, BN/4=32. row = idx / 32.
        int col_vec = idx % (BN / 4);
        int col = col_vec * 4;

        pipe.producer_acquire();
        cuda::memcpy_async(reinterpret_cast<float4*>(&Bs[stage][row][col]),
                           reinterpret_cast<const float4*>(&B_global_ptr[row * N_dim + col]), // N is full width
                           sizeof(float4), pipe);
        pipe.producer_commit();
    }
}

__global__ __launch_bounds__(THREADS_PER_BLOCK)
void sgemm_optimized(const float * __restrict__ A,
                     const float * __restrict__ B,
                     float * __restrict__ C,
                     int M, int N, int K)
{
    const int bx = blockIdx.x;
    const int by = blockIdx.y;
    const int tid = threadIdx.x;

    auto block = cooperative_groups::this_thread_block();

    // Padding added: +PAD
    // Dynamic Shared Memory to bypass 48KB static limit
    extern __shared__ char smem_raw[];
    float (*Ashare)[BM][BK + PAD] = reinterpret_cast<float (*)[BM][BK + PAD]>(smem_raw);
    float (*Bshare)[BK][BN + PAD] = reinterpret_cast<float (*)[BK][BN + PAD]>(smem_raw + 2 * BM * (BK + PAD) * sizeof(float));

    // Fragments
    float fragC[TM][TN] = {0.0f};
    float fragA[TM];
    float fragB[TN];

    // Coordinate calculation
    // A ptr: Starts at Row (by*BM), Col 0. Moves in K.
    const float* A_ptr = A + by * BM * K;
    // B ptr: Starts at Row 0, Col (bx*BN). Moves in K.
    const float* B_ptr = B + bx * BN;

    // Pipeline
    cuda::pipeline<cuda::thread_scope_thread> pipe = cuda::make_pipeline();

    // Prologue: Load Stage 0
    // k=0.
    load_stage_async(pipe, Ashare, Bshare, A_ptr, B_ptr, K, N, 0, 0, tid);
    // Move pointers for next chunk
    // A moves K-dim by BK.
    // B moves K-dim (rows) by BK.
    const float* A_stage_ptr = A_ptr + BK;
    const float* B_stage_ptr = B_ptr + BK * N;

    // Main Loop
    for (int k = 0; k < K; k += BK) {
        int compute_stage = (k / BK) % 2;
        int prefetch_stage = compute_stage ^ 1;

        if (k + BK < K) {
            // Issue next load immediately
            // Note: For k=0, prefetch_stage=1 is free.
            // For k>0, prefetch_stage=(k+1)%2 was used in k-1.
            // We need to ensure k-1 compute is done before overwriting.
            // The block.sync() at the end of the loop ensures this!
            load_stage_async(pipe, Ashare, Bshare, A_stage_ptr, B_stage_ptr, N, N, k + BK, prefetch_stage, tid);
            A_stage_ptr += BK;
            B_stage_ptr += BK * N;
        }

        // Wait for current stage to arrive
        pipe.consumer_wait();
        block.sync(); // Wait for all threads to have data

        // Compute
        int ty = (tid / 16) * TM; // 16 columns of tiles
        int tx = (tid % 16) * TN;

        #pragma unroll
        for (int dot_k = 0; dot_k < BK; ++dot_k) {
            // Load fragA (TM scalars)
            // A access: Ashare[compute_stage][ty + i][dot_k]
            // With padding: Ashare[stage][row][col]
            // We can't vectorize easily because we need column dot_k.
            #pragma unroll
            for (int i = 0; i < TM; ++i) {
                fragA[i] = Ashare[compute_stage][ty + i][dot_k];
            }

            // Load fragB (TN scalars -> vectorized?)
            // B access: Bshare[compute_stage][dot_k][tx + j]
            // tx is (tid%16)*4. Aligned to 16 bytes?
            // (tid%16)*4 * 4bytes = 64 bytes aligned? No.
            // (tid%16) * 4 * 4 = 16 * (0..15).
            // Yes, 16-byte aligned. We can load float4.
            // TN=4. We need 1 float4 load.
            
            float4* B_vec_ptr = reinterpret_cast<float4*>(&Bshare[compute_stage][dot_k][tx]);
            float4 v1 = B_vec_ptr[0]; // Loads j=0..3

            fragB[0] = v1.x; fragB[1] = v1.y; fragB[2] = v1.z; fragB[3] = v1.w;

            // Compute
            #pragma unroll
            for (int i = 0; i < TM; ++i) {
                #pragma unroll
                for (int j = 0; j < TN; ++j) {
                    fragC[i][j] += fragA[i] * fragB[j];
                }
            }
        }
        pipe.consumer_release();
        
        // Sync to prevent next load overwriting this stage before we are done
        // And also to ensure we don't start next compute before this compute is done (though registers are private)
        // Mainly for SMEM consistency for double buffering.
        block.sync();
    }

    // Epilogue
    int ty = (tid / 16) * TM;
    int tx = (tid % 16) * TN;
    int global_cy = by * BM + ty;
    int global_cx = bx * BN + tx;

    #pragma unroll
    for (int i = 0; i < TM; ++i) {
        #pragma unroll
        for (int j = 0; j < TN; j += 4) {
            if (global_cy + i < M && global_cx + j < N) {
                float4 tmp;
                tmp.x = fragC[i][j + 0];
                tmp.y = fragC[i][j + 1];
                tmp.z = fragC[i][j + 2];
                tmp.w = fragC[i][j + 3];
                reinterpret_cast<float4*>(&C[(global_cy + i) * N + (global_cx + j)])[0] = tmp;
            }
        }
    }
}

extern "C" void solve(const float *A, const float *B, float *C, int M, int N, int K)
{
    // Padding check? logic handled in kernel
    dim3 threads(THREADS_PER_BLOCK);
    dim3 blocks(CDIV(N, BN), CDIV(M, BM));
    
    // Dynamic SMEM if needed (64KB + overhead)
    // 2 * (128*36 + 32*132) * 4 ~= 70KB.
    // Default 48KB is not enough.
    int smem_size = 2 * (BM * (BK + PAD) + BK * (BN + PAD)) * sizeof(float);
    if (smem_size >= 48 * 1024) {
        cudaFuncSetAttribute(sgemm_optimized, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);
    }
    
    sgemm_optimized<<<blocks, threads, smem_size>>>(A, B, C, M, N, K);
}

int main(int argc, char **argv)
{

    int M = 2048;
    int N = 1024;
    int K = 4096;

    printf("Benchmarking Optimized SGEMM with BK=%d, PAD=%d\n", BK, PAD);

    size_t size_A = (size_t)M * K * sizeof(float);
    size_t size_B = (size_t)K * N * sizeof(float);
    size_t size_C = (size_t)M * N * sizeof(float);

    float *h_A = (float *)malloc(size_A);
    float *h_B = (float *)malloc(size_B);
    float *h_C = (float *)malloc(size_C);

    for (size_t i = 0; i < (size_t)M * K; ++i) h_A[i] = (float)(rand() % 100) / 100.0f;
    for (size_t i = 0; i < (size_t)K * N; ++i) h_B[i] = (float)(rand() % 100) / 100.0f;

    float *d_A, *d_B, *d_C;
    CHECK_CUDA(cudaMalloc(&d_A, size_A));
    CHECK_CUDA(cudaMalloc(&d_B, size_B));
    CHECK_CUDA(cudaMalloc(&d_C, size_C));

    CHECK_CUDA(cudaMemcpy(d_A, h_A, size_A, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B, h_B, size_B, cudaMemcpyHostToDevice));

    solve(d_A, d_B, d_C, M, N, K);
    CHECK_CUDA(cudaDeviceSynchronize());

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
    
    double flops = 2.0 * (double)M * (double)N * (double)K;
    double tflops = (flops * 1e-12) / (avg_time_ms * 1e-3);
    
    printf("Average Runtime: %f ms\n", avg_time_ms);
    printf("Compute Performance: %f TFLOPS\n", tflops);
    
    // Verify
    CHECK_CUDA(cudaMemcpy(h_C, d_C, size_C, cudaMemcpyDeviceToHost));
    // Simple check
    float max_err = 0;
    for(int i=0; i<100; ++i) {
        int r = rand() % M;
        int c = rand() % N;
        float ref = 0;
        for(int k=0; k<K; ++k) ref += h_A[r*K+k]*h_B[k*N+c];
        float diff = fabs(ref - h_C[r*N+c]);
        if(diff > max_err) max_err = diff;
    }
    printf("Max Error (Sample): %f\n", max_err);

    return 0;
}

/**
Benchmarking Optimized SGEMM with BK=32, PAD=4
Average Runtime: 8.640020 ms
Compute Performance: 15.907249 TFLOPS
Max Error (Sample): 1035.041626
 **/