#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <mma.h> // 必须包含这个头文件来使用 Tensor Core WMMA
#include <iostream>
#include <vector>
#include <random>

using namespace nvcuda; // WMMA 位于 nvcuda 命名空间

#define CHECK_CUDA(call)                                                 \
    do {                                                                 \
        cudaError_t status_ = call;                                      \
        if (status_ != cudaSuccess) {                                    \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__,       \
                    __LINE__, cudaGetErrorString(status_));              \
            exit(EXIT_FAILURE);                                          \
        }                                                                \
    } while (0)

// -----------------------------------------------------------------------------
// WGEMM Tensor Core Kernel: FP16 Activation x INT8 Weight -> FP16 Output
// 使用 WMMA (Warp Matrix Multiply and Accumulate)
// -----------------------------------------------------------------------------

// WMMA 常用的尺寸为 16x16x16
const int WMMA_M = 16;
const int WMMA_N = 16;
const int WMMA_K = 16;

__global__ void wgemm_wmma_kernel(
    const half* __restrict__ A,
    const int8_t* __restrict__ B,
    const half* __restrict__ scales,
    half* __restrict__ C,
    int M, int N, int K) 
{
    // 每个 Block 包含 4x2 个 Warp (blockDim.x=128, blockDim.y=2)
    int warpIdxX = threadIdx.x / 32;
    int warpIdxY = threadIdx.y;
    int laneIdx = threadIdx.x % 32;
    
    int warpsInX = blockDim.x / 32;
    int warpsInY = blockDim.y;

    int warpX = (blockIdx.x * warpsInX + warpIdxX);
    int warpY = (blockIdx.y * warpsInY + warpIdxY);

    int c_row = warpY * WMMA_M;
    int c_col = warpX * WMMA_N;

    // 声明共享内存用于去量化后的权重
    // 每个 Warp 需要 16x16 = 256 个 half
    // Block 中总共有 warpsInX * warpsInY = 8 个 Warp
    __shared__ half shmem_dequant_B[8 * WMMA_K * WMMA_N]; 
    half* my_dequant_B = &shmem_dequant_B[(warpIdxY * warpsInX + warpIdxX) * WMMA_K * WMMA_N];

    // 声明 WMMA 片段
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> b_frag; 
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, half> acc_frag;

    // 初始化累加器为 0
    wmma::fill_fragment(acc_frag, __float2half(0.0f));

    // 沿 K 维度循环
    for (int k = 0; k < K; k += WMMA_K) {
        // 1. 加载 A 矩阵到 A 片段
        if (c_row < M && k < K) {
            wmma::load_matrix_sync(a_frag, A + c_row * K + k, K);
        } else {
            wmma::fill_fragment(a_frag, __float2half(0.0f));
        }

        // 2. 去量化 B 矩阵并加载到 B 片段
        // 每个 Warp 中的线程协作完成去量化
        for (int idx = laneIdx; idx < WMMA_K * WMMA_N; idx += 32) {
            int i = idx / WMMA_N;
            int j = idx % WMMA_N;
            int b_r = k + i;
            int b_c = c_col + j;
            if (b_r < K && b_c < N) {
                float s = __half2float(scales[b_c]);
                float val = (float)B[b_r * N + b_c];
                my_dequant_B[idx] = __float2half(val * s);
            } else {
                my_dequant_B[idx] = __float2half(0.0f);
            }
        }
        
        // WMMA 指令是 Warp 级别的，确保数据在加载前已准备好
        // 虽然在同一个 Warp 内，但 wmma::load_matrix_sync 本身会同步 Warp
        wmma::load_matrix_sync(b_frag, my_dequant_B, WMMA_N);

        // 3. 执行核心指令：mma_sync
        wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
    }

    // 4. 将结果写回全局内存 C
    if (c_row < M && c_col < N) {
        wmma::store_matrix_sync(C + c_row * N + c_col, acc_frag, N, wmma::mem_row_major);
    }
}

// -----------------------------------------------------------------------------
// Host Verification & Main (保持不变但调整参数)
// -----------------------------------------------------------------------------

void verify_wgemm(const half* h_A, const int8_t* h_B, const half* h_scales, const half* h_C, int M, int N, int K) {
    std::cout << "Verifying WMMA results..." << std::endl;
    float max_diff = 0.0f;
    for (int i = 0; i < std::min(M, 16); ++i) {
        for (int j = 0; j < std::min(N, 16); ++j) {
            float gpu_val = __half2float(h_C[i * N + j]);
            float cpu_val = 0.0f;
            float s = __half2float(h_scales[j]);
            for (int k = 0; k < K; ++k) {
                cpu_val += __half2float(h_A[i * K + k]) * ((float)h_B[k * N + j] * s);
            }
            max_diff = std::max(max_diff, std::abs(gpu_val - cpu_val));
        }
    }
    std::cout << "Max Diff: " << max_diff << (max_diff < 0.5f ? " (PASSED)" : " (FAILED)") << std::endl;
}

int main() {
    // 为了满足 WMMA 16x16x16，将尺寸设为 16 的倍数
    int M = 512, N = 512, K = 1024;
    std::cout << "WGEMM Tensor Core Demo (M=" << M << ", N=" << N << ", K=" << K << ")" << std::endl;

    std::vector<half> h_A(M * K);
    std::vector<int8_t> h_B(K * N);
    std::vector<half> h_scales(N);
    std::vector<half> h_C(M * N);

    for (auto& val : h_A) val = __float2half((float)(rand() % 100) / 100.0f - 0.5f);
    for (auto& val : h_B) val = (int8_t)(rand() % 10 - 5);
    for (auto& val : h_scales) val = __float2half(0.01f);

    half *d_A, *d_C, *d_scales;
    int8_t *d_B;
    CHECK_CUDA(cudaMalloc(&d_A, M * K * sizeof(half)));
    CHECK_CUDA(cudaMalloc(&d_B, K * N * sizeof(int8_t)));
    CHECK_CUDA(cudaMalloc(&d_scales, N * sizeof(half)));
    CHECK_CUDA(cudaMalloc(&d_C, M * N * sizeof(half)));

    CHECK_CUDA(cudaMemcpy(d_A, h_A.data(), M * K * sizeof(half), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B, h_B.data(), K * N * sizeof(int8_t), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_scales, h_scales.data(), N * sizeof(half), cudaMemcpyHostToDevice));

    // WMMA 配置：每个 Block 包含 4x2 个 Warp
    dim3 block(32 * 4, 2); 
    dim3 grid((N + (WMMA_N * 4) - 1) / (WMMA_N * 4), (M + (WMMA_M * 2) - 1) / (WMMA_M * 2));

    wgemm_wmma_kernel<<<grid, block>>>(d_A, d_B, d_scales, d_C, M, N, K);
    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaMemcpy(h_C.data(), d_C, M * N * sizeof(half), cudaMemcpyDeviceToHost));
    verify_wgemm(h_A.data(), h_B.data(), h_scales.data(), h_C.data(), M, N, K);

    // Benchmark
    cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);
    cudaEventRecord(start);
    for(int i=0; i<100; ++i) wgemm_wmma_kernel<<<grid, block>>>(d_A, d_B, d_scales, d_C, M, N, K);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms; cudaEventElapsedTime(&ms, start, stop);
    std::cout << "Avg time: " << ms/100.0f << " ms" << std::endl;

    return 0;
}