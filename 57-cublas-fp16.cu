#include <iostream>
#include <vector>
#include <cstdlib>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cuda_fp16.h>

void checkCuda(cudaError_t status, const char *msg) {
    if (status != cudaSuccess) {
        std::cerr << "CUDA Error: " << msg << " - " << cudaGetErrorString(status) << std::endl;
        exit(EXIT_FAILURE);
    }
}

void checkCublas(cublasStatus_t status, const char *msg) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        std::cerr << "cuBLAS Error: " << msg << std::endl;
        exit(EXIT_FAILURE);
    }
}

int main(int argc, char** argv) {
    // Default sizes matching the previous example
    int m = 5120;
    int n = 5120;
    int k = 4096;

    if (argc >= 2) m = atoi(argv[1]);
    if (argc >= 3) n = atoi(argv[2]);
    if (argc >= 4) k = atoi(argv[3]);

    std::cout << "Running cuBLAS HGEMM (FP16)..." << std::endl;
    std::cout << "M = " << m << ", N = " << n << ", K = " << k << std::endl;

    // Allocate host memory
    size_t size_A = m * k;
    size_t size_B = k * n;
    size_t size_C = m * n;

    std::vector<__half> h_A(size_A);
    std::vector<__half> h_B(size_B);
    std::vector<__half> h_C(size_C);

    // Initialize with random data
    // Note: On host, we can cast float to __half if CUDA toolkit supports it, 
    // or just fill with raw bytes for performance testing purposes.
    // Using a simple pattern to avoid NaN/Inf issues just in case.
    for (size_t i = 0; i < size_A; ++i) h_A[i] = __float2half_rn(static_cast<float>(rand()) / RAND_MAX);
    for (size_t i = 0; i < size_B; ++i) h_B[i] = __float2half_rn(static_cast<float>(rand()) / RAND_MAX);
    
    // Allocate device memory
    __half *d_A, *d_B, *d_C;
    checkCuda(cudaMalloc(&d_A, size_A * sizeof(__half)), "Alloc A");
    checkCuda(cudaMalloc(&d_B, size_B * sizeof(__half)), "Alloc B");
    checkCuda(cudaMalloc(&d_C, size_C * sizeof(__half)), "Alloc C");

    checkCuda(cudaMemcpy(d_A, h_A.data(), size_A * sizeof(__half), cudaMemcpyHostToDevice), "Copy A");
    checkCuda(cudaMemcpy(d_B, h_B.data(), size_B * sizeof(__half), cudaMemcpyHostToDevice), "Copy B");
    checkCuda(cudaMemset(d_C, 0, size_C * sizeof(__half)), "Clear C");

    // cuBLAS setup
    cublasHandle_t handle;
    checkCublas(cublasCreate(&handle), "Create handle");

    // Set math mode to allow Tensor Cores
    // CUBLAS_DEFAULT_MATH might not enable Tensor Cores for some precisions on some GPUs
    // CUBLAS_TENSOR_OP_MATH allows Tensor Cores
    checkCublas(cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH), "Set math mode");

    const __half alpha = __float2half(1.0f);
    const __half beta = __float2half(0.0f);

    // Operation: C = A^T * B
    // A is KxM in memory (because we transpose it logically to be MxK)
    // B is KxN in memory (logical KxN)
    // C is MxN
    // Leading dimensions (column major):
    // A (KxM) -> ldA = K
    // B (KxN) -> ldB = K
    // C (MxN) -> ldC = M
    
    int lda = k;
    int ldb = k;
    int ldc = m;

    // Warmup
    checkCublas(cublasHgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N,
                            m, n, k,
                            &alpha,
                            d_A, lda,
                            d_B, ldb,
                            &beta,
                            d_C, ldc), "Warmup Hgemm");
    checkCuda(cudaDeviceSynchronize(), "Warmup sync");

    // Timing
    int iterations = 100;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    for (int i = 0; i < iterations; ++i) {
        cublasHgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N,
                    m, n, k,
                    &alpha,
                    d_A, lda,
                    d_B, ldb,
                    &beta,
                    d_C, ldc);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    float avg_time_ms = milliseconds / iterations;

    // GFLOPS calculation
    double flops = 2.0 * static_cast<double>(m) * n * k;
    double gflops = (flops * 1e-9) / (avg_time_ms * 1e-3);

    std::cout << "------------------------------------------------" << std::endl;
    std::cout << "cuBLAS HGEMM Performance Results:" << std::endl;
    std::cout << "Time: " << avg_time_ms << " ms" << std::endl;
    std::cout << "Throughput: " << gflops << " GFLOPS" << std::endl;
    std::cout << "------------------------------------------------" << std::endl;

    // Cleanup
    cublasDestroy(handle);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}

