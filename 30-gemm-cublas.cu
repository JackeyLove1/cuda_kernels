#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>

// Error checking macros
#define CHECK_CUDA(func) \
{ \
    cudaError_t status = (func); \
    if (status != cudaSuccess) { \
        printf("CUDA API failed at line %d with error: %s (%d)\n", \
               __LINE__, cudaGetErrorString(status), status); \
        exit(EXIT_FAILURE); \
    } \
}

#define CHECK_CUBLAS(func) \
{ \
    cublasStatus_t status = (func); \
    if (status != CUBLAS_STATUS_SUCCESS) { \
        printf("CUBLAS API failed at line %d with error: %d\n", \
               __LINE__, status); \
        exit(EXIT_FAILURE); \
    } \
}

// Global handle to avoid creation overhead during benchmarking loop
static cublasHandle_t handle = nullptr;

// A: M x K, B: K x N, C: M x N (Row Major inputs)
// We use cuBLAS which expects Column Major.
// To perform C = A * B in row major, we compute C^T = B^T * A^T in column major.
// - B^T in col major has same memory layout as B in row major.
// - A^T in col major has same memory layout as A in row major.
// - C^T in col major has same memory layout as C in row major.
// So we call cublasSgemm with:
//   op(A) = N (no transpose), op(B) = N
//   m = N, n = M, k = K
//   A = B_ptr, lda = N
//   B = A_ptr, ldb = K
//   C = C_ptr, ldc = N
extern "C" void solve(const float* A, const float* B, float* C, int M, int N, int K) {
    if (handle == nullptr) {
        CHECK_CUBLAS(cublasCreate(&handle));
    }

    float alpha = 1.0f;
    float beta = 0.0f;

    // cublasSgemm(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc)
    // We want C^T (NxM) = B^T (NxK) * A^T (KxM)
    CHECK_CUBLAS(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                             N, M, K,
                             &alpha,
                             B, N,  // First matrix is B (interpreted as B^T)
                             A, K,  // Second matrix is A (interpreted as A^T)
                             &beta,
                             C, N)); // Result is C (interpreted as C^T)
}

int main(int argc, char **argv) {
    int M = 2048;
    int N = 4096;
    int K = 4096;
    
    printf("Benchmarking cuBLAS SGEMM with M=%d, N=%d, K=%d\n", M, N, K);

    size_t size_A = (size_t)M * K * sizeof(float);
    size_t size_B = (size_t)K * N * sizeof(float);
    size_t size_C = (size_t)M * N * sizeof(float);

    float *h_A = (float*)malloc(size_A);
    float *h_B = (float*)malloc(size_B);
    
    if (!h_A || !h_B) {
        fprintf(stderr, "Host memory allocation failed\n");
        return EXIT_FAILURE;
    }

    // Initialize with random values
    for(size_t i=0; i<(size_t)M*K; ++i) h_A[i] = (float)(rand() % 100) / 100.0f;
    for(size_t i=0; i<(size_t)K*N; ++i) h_B[i] = (float)(rand() % 100) / 100.0f;

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
    for(int i=0; i<num_runs; ++i) {
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
    double flops = 2.0 * (double)M * (double)N * (double)K;
    double tflops = (flops * 1e-12) / (avg_time_ms * 1e-3);
    printf("Compute Performance: %f TFLOPS\n", tflops);
    
    // Effective Memory Bandwidth (R+W)
    // 2 reads (A, B) + 1 write (C)
    double total_bytes = sizeof(float) * ((double)M*K + (double)K*N + (double)M*N);
    double bandwidth_gb = (total_bytes * 1e-9) / (avg_time_ms * 1e-3);
    printf("Memory Bandwidth: %f GB/s\n", bandwidth_gb);

    free(h_A); free(h_B);
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    cudaEventDestroy(start); cudaEventDestroy(stop);
    if (handle) cublasDestroy(handle);

    return 0;
}

/**
Benchmarking cuBLAS SGEMM with M=2048, N=4096, K=4096
Average Runtime: 3.142165 ms
Compute Performance: 21.870106 TFLOPS
Memory Bandwidth: 42.715052 GB/s
 */