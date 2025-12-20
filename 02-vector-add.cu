#include <cuda_runtime.h>
#include <stdio.h>
#include <cmath>
#include <iostream>

// 错误检查宏
#define CHECK_CUDA(call) \
do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        printf("CUDA Error at %s:%d - %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(1); \
    } \
} while(0)

// 标量加法内核 - 处理剩余元素
__global__ void vector_add_scalar(const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C, int N, int offset) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x + offset;
    if (idx < N) {
        C[idx] = A[idx] + B[idx];
    }
}

// 优化后的 FLOAT4 向量化内核
__global__ void vector_add_float4(const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C, int N) {
    // 计算 float4 向量的索引
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // 每个线程处理 4 个 float 元素
    // 注意：输入指针 A, B, C 必须是 16 字节对齐的 (cudaMalloc 保证了这一点)
    if (idx < N / 4) {
        // 使用 reinterpret_cast 将 float* 转换为 float4* 进行加载
        float4 a4 = reinterpret_cast<const float4*>(A)[idx];
        float4 b4 = reinterpret_cast<const float4*>(B)[idx];
        
        float4 c4;
        c4.x = a4.x + b4.x;
        c4.y = a4.y + b4.y;
        c4.z = a4.z + b4.z;
        c4.w = a4.w + b4.w;
        
        reinterpret_cast<float4*>(C)[idx] = c4;
    }
}

// 校验函数
void verify_result(const float* A, const float* B, const float* C, int N) {
    bool passed = true;
    double max_diff = 0.0;
    
    for (int i = 0; i < N; ++i) {
        float expected = A[i] + B[i];
        float diff = std::abs(C[i] - expected);
        if (diff > max_diff) max_diff = diff;
        
        if (diff > 1e-5) {
            if (passed) { // 只打印第一个错误
                printf("Verification FAILED at index %d: A=%f, B=%f, Expected=%f, Got=%f\n", 
                       i, A[i], B[i], expected, C[i]);
            }
            passed = false;
        }
    }
    
    if (passed) {
        printf("Verification PASSED! Max diff: %e\n", max_diff);
    } else {
        printf("Verification FAILED! Max diff: %e\n", max_diff);
    }
}

int main() {
    // 设置设备
    int deviceId = 0;
    cudaSetDevice(deviceId);
    
    // 设置数据规模 (使用非4倍数测试边界条件)
    int N = 10000003; 
    size_t size = N * sizeof(float);
    
    printf("Vector Add Optimization Test\n");
    printf("Vector size: %d elements\n", N);
    
    // 分配主机内存
    float *h_A = (float*)malloc(size);
    float *h_B = (float*)malloc(size);
    float *h_C = (float*)malloc(size);
    
    // 初始化数据
    for (int i = 0; i < N; ++i) {
        h_A[i] = static_cast<float>(i) * 0.1f;
        h_B[i] = static_cast<float>(i) * 0.2f;
    }
    
    // 分配设备内存
    float *d_A, *d_B, *d_C;
    CHECK_CUDA(cudaMalloc(&d_A, size));
    CHECK_CUDA(cudaMalloc(&d_B, size));
    CHECK_CUDA(cudaMalloc(&d_C, size));
    
    // 拷贝数据到设备
    CHECK_CUDA(cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice));
    
    // 计算网格维度
    int blockSize = 256;
    // float4 向量的数量
    int numVecs = N / 4;
    int gridSize = (numVecs + blockSize - 1) / blockSize;
    
    // 创建 CUDA 事件用于计时
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    // 开始计时
    cudaEventRecord(start);
    
    // 1. 启动 float4 优化内核处理主要部分
    if (numVecs > 0) {
        vector_add_float4<<<gridSize, blockSize>>>(d_A, d_B, d_C, N);
    }
    
    // 2. 处理剩余元素 (尾部)
    int remainder = N % 4;
    if (remainder > 0) {
        vector_add_scalar<<<1, remainder>>>(d_A, d_B, d_C, N, N - remainder);
    }
    
    // 停止计时
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    CHECK_CUDA(cudaGetLastError());
    
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    
    // 拷贝结果回主机
    CHECK_CUDA(cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost));
    
    // 计算性能指标
    double bandwidth = (3.0 * size) / (milliseconds / 1000.0) / 1e9; // GB/s
    printf("Time: %.3f ms\n", milliseconds);
    printf("Effective Bandwidth: %.2f GB/s\n", bandwidth);
    
    // 验证结果
    verify_result(h_A, h_B, h_C, N);
    
    // 清理资源
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    return 0;
}
