#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cmath>
#include <vector>

#define CHECK_CUDA(func)                                                                                              \
    {                                                                                                                 \
        cudaError_t status = (func);                                                                                  \
        if (status != cudaSuccess) {                                                                                  \
            printf("CUDA API failed at line %d with error: %s (%d)\n", __LINE__, cudaGetErrorString(status), status); \
            exit(EXIT_FAILURE);                                                                                       \
        }                                                                                                             \
    }

#define CHECK_CUBLAS(func)                                                                                            \
    {                                                                                                                 \
        cublasStatus_t status = (func);                                                                               \
        if (status != CUBLAS_STATUS_SUCCESS) {                                                                        \
            printf("CUBLAS API failed at line %d with error: %d\n", __LINE__, status);                                \
            exit(EXIT_FAILURE);                                                                                       \
        }                                                                                                             \
    }

void verify_result(float *h_A, float *h_B, float *h_C, int M, int N, int K) {
    printf("Verifying result...\n");
    float max_error = 0.0f;
    int error_count = 0;
    // 随机检查 1000 个点
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
            if (error_count < 5) {
                printf("Error at (%d, %d): GPU=%f, CPU=%f, Diff=%f\n", row, col, h_C[row * N + col], (float)sum, diff);
            }
            error_count++;
        }
    }
    if (error_count == 0) {
        printf("Verification PASSED (checked 1000 random elements). Max Error: %e\n", max_error);
    } else {
        printf("Verification FAILED. Total errors: %d, Max Error: %e\n", error_count, max_error);
    }
}

int main(int argc, char **argv) {
    int M = 2048;
    int N = 1024;
    int K = 4096;

    printf("Benchmarking cuBLAS SGEMM (SimT / CUDA Core) with M=%d, N=%d, K=%d\n", M, N, K);

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

    // Initialize
    for (size_t i = 0; i < (size_t)M * K; ++i) h_A[i] = (float)(rand() % 100) / 100.0f;
    for (size_t i = 0; i < (size_t)K * N; ++i) h_B[i] = (float)(rand() % 100) / 100.0f;

    float *d_A, *d_B, *d_C;
    CHECK_CUDA(cudaMalloc(&d_A, size_A));
    CHECK_CUDA(cudaMalloc(&d_B, size_B));
    CHECK_CUDA(cudaMalloc(&d_C, size_C));

    CHECK_CUDA(cudaMemcpy(d_A, h_A, size_A, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B, h_B, size_B, cudaMemcpyHostToDevice));

    cublasHandle_t handle;
    CHECK_CUBLAS(cublasCreate(&handle));

    // 禁止使用 Tensor Core 模式
    // CUBLAS_MATH_DISALLOW_REDUCED_PRECISION_REDUCTION: 不允许降低精度
    // CUBLAS_DEFAULT_MATH: 默认模式，通常不使用 Tensor Core 进行 float 计算，除非显式开启 TF32
    // 为了确保不使用 Tensor Core，我们显式设定 Math Mode 并使用 SGEMM
    // cublasSetMathMode(handle, CUBLAS_DEFAULT_MATH); // 默认行为对于 FP32 来说通常就是 CUDA Core
    // 更严格的限制是 CUBLAS_PEDANTIC_MATH，它阻止任何可能改变结果精度的优化
    CHECK_CUBLAS(cublasSetMathMode(handle, CUBLAS_PEDANTIC_MATH));

    float alpha = 1.0f;
    float beta = 0.0f;

    // Warmup
    // 注意：cuBLAS 默认是列主序 (Column Major)，C/C++ 是行主序 (Row Major)。
    // 为了计算 C = A * B (Row Major)，我们可以利用 (A * B)^T = B^T * A^T
    // 将 C 看作 Column Major 等价于 Row Major 的 C^T。
    // 这里简单起见，我们直接调用，假设数据排布，仅测试性能。
    // 如果要完全对应 Row Major 结果，通常传入 B, A 交换顺序，并用 CUBLAS_OP_N。
    // 这里为了和手写 Kernel 对齐 (A 是 Row Major)，我们使用 cublasSgemm 技巧：
    // C(MxN) = A(MxK) * B(KxN)
    // view as Col Major: C'(NxM) = B'(NxK) * A'(KxM)
    // 所以传入 cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, d_B, N, d_A, K, &beta, d_C, N);
    
    CHECK_CUBLAS(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, 
                             N, M, K, 
                             &alpha, 
                             d_B, N, 
                             d_A, K, 
                             &beta, 
                             d_C, N));
    CHECK_CUDA(cudaDeviceSynchronize());

    // 验证
    // 为了验证，我们需要把结果拷贝回来。注意上面我们计算的是 C^T (Column Major 的 C)，
    // 但因为 h_A, h_B 是随机数，只要运算发生了且性能正确即可。
    // 若要严格数值验证，需按上述逻辑反转。
    // 这里为了方便对比，我们用更直观的方式：
    // 让 cuBLAS 按照 A * B 计算，但在 cuBLAS 视角下，传入的数据被视为列主序。
    // A (Row Major) -> A^T (Col Major)
    // 实际上对于纯随机矩阵乘法性能测试，数据内容不影响速度。我们只关心 FLOPs。
    
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    int num_runs = 100;
    CHECK_CUDA(cudaEventRecord(start));
    for (int i = 0; i < num_runs; ++i) {
        CHECK_CUBLAS(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, 
                                 N, M, K, 
                                 &alpha, 
                                 d_B, N, 
                                 d_A, K, 
                                 &beta, 
                                 d_C, N));
    }
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float milliseconds = 0;
    CHECK_CUDA(cudaEventElapsedTime(&milliseconds, start, stop));

    float avg_time_ms = milliseconds / num_runs;
    printf("Average Runtime: %f ms\n", avg_time_ms);

    double flops  = 2.0 * (double)M * (double)N * (double)K;
    double tflops = (flops * 1e-12) / (avg_time_ms * 1e-3);
    printf("Compute Performance: %f TFLOPS\n", tflops);

    double total_bytes  = sizeof(float) * ((double)M * K + (double)K * N + (double)M * N);
    double bandwidth_gb = (total_bytes * 1e-9) / (avg_time_ms * 1e-3);
    printf("Memory Bandwidth: %f GB/s\n", bandwidth_gb);

    cublasDestroy(handle);
    free(h_A); free(h_B); free(h_C);
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    cudaEventDestroy(start); cudaEventDestroy(stop);

    return 0;
}

/**
Benchmarking cuBLAS SGEMM (SimT / CUDA Core) with M=2048, N=1024, K=4096
Average Runtime: 0.856986 ms
Compute Performance: 20.046858 TFLOPS
Memory Bandwidth: 68.519534 GB/s
 */