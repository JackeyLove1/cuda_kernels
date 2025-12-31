// Standard Library includes
#include <iostream>
#include <sstream>
#include <vector>
#include <cublas_v2.h>
#include "helper.h"
#include "cutlass/gemm/device/gemm.h"

using half_t = cutlass::half_t;

// 定义优化后的 Gemm 结构
using ColumnMajor = cutlass::layout::ColumnMajor;
using CutlassGemm = cutlass::gemm::device::Gemm<
  float,                // ElementA (FP32)
  ColumnMajor,          // LayoutA
  float,                // ElementB (FP32)
  ColumnMajor,          // LayoutB
  float,                // ElementC (FP32)
  ColumnMajor,          // LayoutC
  float,                // ElementAccumulator (通常 FP32)
  cutlass::arch::OpClassTensorOp, // 关键点：使用 Tensor Cores (TF32)
  cutlass::arch::Sm80,  // 针对 Ampere 架构 (Sm80)
  cutlass::gemm::GemmShape<128, 128, 16>, // Threadblock Shape (TF32 建议 K=16)
  cutlass::gemm::GemmShape<64, 64, 16>,   // Warp Shape
  cutlass::gemm::GemmShape<16, 8, 8>,     // Instruction Shape (TF32 格式)
  cutlass::epilogue::thread::LinearCombination<float, 1, float, float>,
  cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
  3,                    // kStages: Ampere 架构建议 3-5
  4,                    // AlignmentA (FP32 通常选 4)
  4                     // AlignmentB
>;

cudaError_t CutlassSgemmNN(
  int M,
  int N,
  int K,
  float alpha,
  float const *A,
  int lda,
  float const *B,
  int ldb,
  float beta,
  float *C,
  int ldc) {
    
  CutlassGemm gemm_operator;
  CutlassGemm::Arguments args({M, N, K},  // Gemm Problem dimensions
                              {A, lda},    // Tensor-ref for source matrix A
                              {B, ldb},    // Tensor-ref for source matrix B
                              {C, ldc},    // Tensor-ref for source matrix C
                              {C, ldc},    // Tensor-ref for destination matrix D
                              {alpha, beta}); // Scalars used in the Epilogue

  // 检查是否需要 workspace
  size_t workspace_size = CutlassGemm::get_workspace_size(args);
  if (workspace_size > 0) {
      // 如果需要 workspace，这里简单处理，实际 benchmark 时应该在循环外分配
      static void* workspace = nullptr;
      static size_t current_workspace_size = 0;
      if (current_workspace_size < workspace_size) {
          if (workspace) cudaFree(workspace);
          CUDA_CHECK(cudaMalloc(&workspace, workspace_size));
          current_workspace_size = workspace_size;
      }
      cutlass::Status status = gemm_operator(args, workspace);
      if (status != cutlass::Status::kSuccess) return cudaErrorUnknown;
  } else {
      cutlass::Status status = gemm_operator(args);
      if (status != cutlass::Status::kSuccess) {
          // std::cerr << "CUTLASS error: " << cutlassGetStatusString(status) << std::endl;
          return cudaErrorUnknown;
      }
  }

  return cudaSuccess;
}

int main(int argc, char **argv)
{
    int M = 4096;
    int N = 4096;
    int K = 4096;

    float alpha = 1.0f;
    float beta = 0.0f;

    printf("Benchmarking CUTLASS vs cuBLAS SGEMM with M=%d, N=%d, K=%d\n", M, N, K);

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
    for (size_t i = 0; i < (size_t)M * K; ++i) h_A[i] = (float)(rand() % 100) / 100.0f;
    for (size_t i = 0; i < (size_t)K * N; ++i) h_B[i] = (float)(rand() % 100) / 100.0f;

    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, size_A));
    CUDA_CHECK(cudaMalloc(&d_B, size_B));
    CUDA_CHECK(cudaMalloc(&d_C, size_C));

    CUDA_CHECK(cudaMemcpy(d_A, h_A, size_A, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, size_B, cudaMemcpyHostToDevice));

    // Leading dimensions for Column-Major
    int lda = M;
    int ldb = K;
    int ldc = M;

    // --- cuBLAS Benchmark ---
    cublasHandle_t handle;
    cublasCreate(&handle);
    
    // Set math mode to Pedantic to ensure we are using CUDA Cores (SimT)
    cublasSetMathMode(handle, CUBLAS_PEDANTIC_MATH);
    
    // Warmup cuBLAS
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, M, N, K, &alpha, d_A, lda, d_B, ldb, &beta, d_C, ldc);
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    int num_runs = 50;
    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < num_runs; ++i) {
        cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, M, N, K, &alpha, d_A, lda, d_B, ldb, &beta, d_C, ldc);
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms_cublas = 0;
    CUDA_CHECK(cudaEventElapsedTime(&ms_cublas, start, stop));
    float avg_ms_cublas = ms_cublas / num_runs;

    // --- CUTLASS Benchmark ---
    // Warmup CUTLASS
    cudaError_t err = CutlassSgemmNN(M, N, K, alpha, d_A, lda, d_B, ldb, beta, d_C, ldc);
    if (err != cudaSuccess) {
        printf("CUTLASS warmup failed!\n");
        return EXIT_FAILURE;
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < num_runs; ++i) {
        CutlassSgemmNN(M, N, K, alpha, d_A, lda, d_B, ldb, beta, d_C, ldc);
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms_cutlass = 0;
    CUDA_CHECK(cudaEventElapsedTime(&ms_cutlass, start, stop));
    float avg_ms_cutlass = ms_cutlass / num_runs;

    // Performance calculation
    double flops = 2.0 * (double)M * (double)N * (double)K;
    double tflops_cublas = (flops * 1e-12) / (avg_ms_cublas * 1e-3);
    double tflops_cutlass = (flops * 1e-12) / (avg_ms_cutlass * 1e-3);

    printf("\ncuBLAS Performance: %.2f TFLOPS (Avg Time: %.3f ms)\n", tflops_cublas, avg_ms_cublas);
    printf("CUTLASS Performance: %.2f TFLOPS (Avg Time: %.3f ms)\n", tflops_cutlass, avg_ms_cutlass);
    printf("Ratio (CUTLASS/cuBLAS): %.2f%%\n", (tflops_cutlass / tflops_cublas) * 100.0);

    cublasDestroy(handle);
    free(h_A); free(h_B); free(h_C);
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    cudaEventDestroy(start); cudaEventDestroy(stop);

    return 0;
}
