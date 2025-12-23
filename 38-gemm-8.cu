#include <cooperative_groups.h> // 用于更精细的同步控制
#include <cstdio>
#include <cuda/pipeline>
#include <cuda_pipeline.h> // 必须包含，用于 cp.async
#include <cuda_runtime.h>

// ==========================================
// Hyperparameters (超参数调整是性能的关键)
// ==========================================
// 针对 RTX 3090/A100 等常见配置进行优化
// Block Tile Size: 128x128
#define BM 128
#define BN 128
// K Dimension Step: 每次迭代处理 K 维度的 16 个元素
// 为什么是 16？为了配合 CP Async 的 128字节 (float4 * 8线程) 事务粒度，且保持 SMEM占用合理。
#define BK 16
// Thread Tile Size: 每个线程负责计算 8x8 的结果
#define TM 8
#define TN 8

#define CHECK_CUDA(func)                                                                                              \
    {                                                                                                                 \
        cudaError_t status = (func);                                                                                  \
        if (status != cudaSuccess) {                                                                                  \
            printf("CUDA API failed at line %d with error: %s (%d)\n", __LINE__, cudaGetErrorString(status), status); \
            exit(EXIT_FAILURE);                                                                                       \
        }                                                                                                             \
    }

// 计算所需线程数: (128*128) / (8*8) = 256 线程
const int THREADS_PER_BLOCK = (BM * BN) / (TM * TN);

// 辅助函数：向上取整除法
template <typename T, typename U> __host__ __device__ __forceinline__ auto CDIV(T a, U b) { return (a + b - 1) / b; }

using namespace cooperative_groups;

// ==========================================
// Device Helper Functions (辅助加载函数)
// ==========================================

// 目的：计算当前线程应该从 Global Memory 的哪里读取数据，以及写入 Shared Memory 的哪里。
// 这是一个极其繁琐但必须精确的过程。
// 我们采用 float4 向量化加载来最大化带宽。

__device__ __forceinline__ void load_stage_async(
    cuda::pipeline<cuda::thread_scope_thread>& pipe,
    float (&As)[2][BM][BK], // Shared Memory 指针引用
    float (&Bs)[2][BK][BN],
    const float* A_global_ptr,
    const float* B_global_ptr,
    int K_dim,    // 总 K 维度大小，用于边界检查
    int k_offset, // 当前 K 维度的偏移量
    int stage,    // 当前写入的 Buffer 索引 (0 或 1)
    int tid       // 线程 ID
) {
    // --- A 矩阵加载逻辑 ---
    // A 的 Tile 是 [BM][BK] 即 [128][16]
    // 我们有 256 个线程。每个线程负责加载 128*16 / 256 = 8 个 float。
    // 即每个线程加载 2 个 float4。

    // 将 256 个线程映射到 A 的加载坐标上。
    // 为了向量化，我们把列看作 [BK/4] = [4] 个 float4 组。
    // 每行有 4 组 float4。我们让连续的 4 个线程加载一行。
    // 线程 T 负责的行是 T / 4。
    // 线程 T 负责的列组是 T % 4。
    // 由于要加载 128 行，我们每个线程需要负责 128 / (256/4) = 2 行？不对。

    // 重新映射策略：
    // 这里的映射方式有很多种，目标是让连续线程访问连续地址 (Coalescing)。
    // 让我们使用一种标准映射：
    // A 矩阵通常是 Row-Major。
    // 每个线程加载 float4。总共需要加载 (BM*BK)/4 = (128*16)/4 = 512 个 float4。
    // 256 个线程，每个线程加载 2 个 float4。

    // 线程 tid 负责加载第 (tid * 2) 和 (tid * 2 + 1) 个 float4。
    // float4 索引转 2D 坐标 (row, col_vec_idx):
    const int a_vec_idx1 = tid * 2;
    const int a_row1 = a_vec_idx1 / (BK / 4); // / 4
    const int a_col1 = (a_vec_idx1 % (BK / 4)) * 4;

    const int a_vec_idx2 = tid * 2 + 1;
    const int a_row2 = a_vec_idx2 / (BK / 4);
    const int a_col2 = (a_vec_idx2 % (BK / 4)) * 4;

    // 发起异步拷贝 A
    pipe.producer_acquire();
    // 边界检查：确保不要读取超出 Global Memory A 矩阵范围的数据
    if (k_offset + a_col1 < K_dim) {
        cuda::memcpy_async(&As[stage][a_row1][a_col1],
                           &A_global_ptr[a_row1 * K_dim + a_col1],
                           sizeof(float4), pipe);
    } else {
        // 如果越界，填充 0。CP Async 没有直接清零的功能，这里是一个简化的处理。
        // 严格来说应该用 memset 或其他方式清零 SMEM，或者保证 K 是 BK 的倍数。
        // 为了教程简洁，这里假设 K 是 BK 的倍数，略去 else 分支的复杂处理。
        // 实际产品级代码必须处理边界。
    }

    if (k_offset + a_col2 < K_dim) {
        cuda::memcpy_async(&As[stage][a_row2][a_col2],
                           &A_global_ptr[a_row2 * K_dim + a_col2],
                           sizeof(float4), pipe);
    }
    pipe.producer_commit();

    // --- B 矩阵加载逻辑 ---
    // B 的 Tile 是 [BK][BN] 即 [16][128]。同样 Row-Major。
    // 映射逻辑类似。
    const int b_vec_idx1 = tid * 2;
    const int b_row1 = b_vec_idx1 / (BN / 4); // / 32
    const int b_col1 = (b_vec_idx1 % (BN / 4)) * 4;

    const int b_vec_idx2 = tid * 2 + 1;
    const int b_row2 = b_vec_idx2 / (BN / 4);
    const int b_col2 = (b_vec_idx2 % (BN / 4)) * 4;

    // 发起异步拷贝 B
    pipe.producer_acquire();
     // B 矩阵在 K 维度上的边界检查比较简单，因为它在外部循环控制
    cuda::memcpy_async(&Bs[stage][b_row1][b_col1],
                       &B_global_ptr[b_row1 * K_dim + b_col1], // 注意 B 的索引方式
                       sizeof(float4), pipe);

    cuda::memcpy_async(&Bs[stage][b_row2][b_col2],
                       &B_global_ptr[b_row2 * K_dim + b_col2],
                       sizeof(float4), pipe);
    pipe.producer_commit();
}

// ==========================================
// The Main SGEMM Kernel (Ampere Optimized)
// ==========================================
__global__ __launch_bounds__(THREADS_PER_BLOCK)
void sgemm_ampere_cp_async_db(const float * __restrict__ A,
                              const float * __restrict__ B,
                              float * __restrict__ C,
                              int M, int N, int K)
{
    // Block Index
    const int bx = blockIdx.x;
    const int by = blockIdx.y;
    // Thread ID
    const int tid = threadIdx.x;

    auto block = cooperative_groups::this_thread_block();

    // [Double Buffer Setup]
    // 声明共享内存，增加一个维度 [2] 用于双缓冲
    // 使用 alignas 确保对齐，有助于避免 bank conflicts 和满足 cp.async 要求
    __shared__ alignas(128) float Ashare[2][BM][BK];
    __shared__ alignas(128) float Bshare[2][BK][BN];

    // [Register Setup]
    // 累加器 registers (TM x TN = 8x8 = 64 registers)
    float fragC[TM][TN] = {0.0f};
    // 用于计算的临时寄存器片段
    float fragA[TM];
    float fragB[TN];

    // 计算当前 Block 在 Global Memory 中的起始指针
    const float* A_block_ptr = A + by * BM * K;
    // B 矩阵假设是 KxN 存储的。
    const float* B_block_ptr = B + bx * BN;

    // 初始化异步拷贝流水线
    cuda::pipeline<cuda::thread_scope_thread> pipe = cuda::make_pipeline();

    // [Pipeline Prologue - 序幕]
    // 在进入主循环前，先启动第一个块 (k=0) 的加载。
    // 加载到 stage 0。
    load_stage_async(pipe, Ashare, Bshare, A_block_ptr, B_block_ptr, K, 0, 0, tid);

    // 移动 Global 指针到下一个块
    A_block_ptr += BK;
    B_block_ptr += BK * N;

    // [Main Loop - Pipeline]
    // 循环 K 维度的块。
    // 在第 i 次迭代中：
    // 1. 等待第 i 个块的数据加载完成 (Wait)。
    // 2. 发起第 i+1 个块的预取加载 (Prefetch)。
    // 3. 使用第 i 个块的数据进行计算 (Compute)。
    for (int k = 0; k < K; k += BK) {
        // 当前用于计算的缓冲级 (0 or 1)
        int compute_stage = (k / BK) % 2;
        // 下一个用于预取的缓冲级
        int prefetch_stage = compute_stage ^ 1;

        // Step 1 & 2: Wait and Prefetch
        // 我们需要等待当前计算阶段的数据准备好。
        // 如果这不是最后一次迭代，我们还需要发起下一次的预取。

        if (k + BK < K) {
            // 这里的逻辑是：
            // 我们刚刚在 Prologue 或者上一次循环末尾提交了 prefetch 请求。
            // 现在我们需要等待那个请求完成。
            // pipe.consumer_wait() 的参数表示“我们允许流水线中还剩下多少个未完成的 stage”。
            // 如果我们要发起新的 prefetch，我们等待直到只剩下 1 个未完成（就是我们即将发起的这个）。
            pipe.consumer_wait();
            // 同步，确保所有线程都确认 SMEM 数据已就位
            block.sync();

            // 发起下一轮预取到 prefetch_stage
            load_stage_async(pipe, Ashare, Bshare, A_block_ptr, B_block_ptr, K, k + BK, prefetch_stage, tid);
            // 移动指针
            A_block_ptr += BK;
            B_block_ptr += BK * N;
        } else {
            // 最后一次迭代，等待所有挂起的拷贝完成 (参数为 0)
            pipe.consumer_wait();
            block.sync();
        }

        // Step 3: Compute (The Math Core)
        // 这一部分是计算密集型的，完全在寄存器和 Shared Memory 之间进行。
        // 此时 CP Async 硬件正在后台忙着加载 prefetch_stage 的数据。

        // 计算当前线程负责的 C 的子块坐标
        int ty_local = (tid / (BN / TN)) * TM;
        int tx_local = (tid % (BN / TN)) * TN;

        // 内层循环：遍历 BK 维度的 dot product
        #pragma unroll
        for (int dot_k = 0; dot_k < BK; ++dot_k) {
            // 将数据从 SMEM (compute_stage) 加载到寄存器片段
            #pragma unroll
            for (int i = 0; i < TM; ++i) {
                fragA[i] = Ashare[compute_stage][ty_local + i][dot_k];
            }
            #pragma unroll
            for (int j = 0; j < TN; ++j) {
                fragB[j] = Bshare[compute_stage][dot_k][tx_local + j];
            }

            // 外积计算 (Outer Product)
            #pragma unroll
            for (int i = 0; i < TM; ++i) {
                #pragma unroll
                for (int j = 0; j < TN; ++j) {
                    // FMA (Fused Multiply-Add) 指令
                    fragC[i][j] += fragA[i] * fragB[j];
                }
            }
        }

        // 释放流水线资源，表示当前 compute_stage 的数据已经使用完毕
        pipe.consumer_release();
    }

    // [Epilogue - Store Results]
    // 将寄存器中的计算结果写回 Global Memory C
    // 同样需要向量化写回。
    int ty_local = (tid / (BN / TN)) * TM;
    int tx_local = (tid % (BN / TN)) * TN;

    int global_cy = by * BM + ty_local;
    int global_cx = bx * BN + tx_local;

    #pragma unroll
    for (int i = 0; i < TM; ++i) {
        #pragma unroll
        for (int j = 0; j < TN; j += 4) { // 以 float4 为步长写回
            // 边界检查 (如果 M, N 不是 tile 的倍数则需要)
            if (global_cy + i < M && global_cx + j < N) {
                 // 重新解释转换为 float4 指针进行写入
                float4 tmp_c;
                tmp_c.x = fragC[i][j + 0];
                tmp_c.y = fragC[i][j + 1];
                tmp_c.z = fragC[i][j + 2];
                tmp_c.w = fragC[i][j + 3];
                *(reinterpret_cast<float4*>(&C[(global_cy + i) * N + (global_cx + j)])) = tmp_c;
            }
        }
    }
}

// ==========================================
// Host Wrapper
// ==========================================
extern "C" void solve(const float *A, const float *B, float *C, int M, int N, int K)
{
    // 确保 K 是 BK 的倍数，否则上面的 kernel 边界检查会出错。
    // 在实际应用中，需要 padding 或者在 kernel 里做更复杂的判断。
    if (K % BK != 0) {
        printf("Error: For this optimized kernel, K must be a multiple of %d. Current K=%d\n", BK, K);
        // exit(EXIT_FAILURE); // 或者在这里做 padding 处理
    }

    dim3 threads(THREADS_PER_BLOCK);
    dim3 blocks(CDIV(N, BN), CDIV(M, BM));

    printf("Launching Ampere CP Async GEMM with Block=[%d,%d], ThreadTile=[%d,%d], K_step=%d\n", BM, BN, TM, TN, BK);

    // 需要足够大的 Shared Memory。
    // Size = 2 * (BM*BK + BK*BN) * sizeof(float)
    // Size = 2 * (128*16 + 16*128) * 4 = 2 * 4096 * 4 = 32 KB.
    // 默认 SMEM 通常足够。如果不够需要调用 cudaFuncSetAttribute 增加动态 SMEM。

    sgemm_ampere_cp_async_db<<<blocks, threads>>>(A, B, C, M, N, K);

    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());
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
Average Runtime: 5.924270 ms
Compute Performance: 11.599653 TFLOPS
Memory Bandwidth: 22.655572 GB/s
 **/