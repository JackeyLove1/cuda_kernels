#include <iostream>
#include <vector>
#include <cmath>
#include <cstdio>

// Cutlass includes
#include "cutlass/cutlass.h"
#include "cutlass/array.h"
#include "cutlass/layout/matrix.h"
#include "cutlass/util/host_tensor.h"

#define CHECK_CUDA(call) \
do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        printf("CUDA Error at %s:%d - %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(1); \
    } \
} while(0)

// 使用 Cutlass Array 的向量化内核
// N 是向量长度 (例如 4)
// 使用 Cutlass Array 可以保证生成的代码使用了对应的向量指令 (如 LD.128/ST.128)
// 并且相比于 float4，它支持任意长度的向量化 (前提是满足对齐要求)
template <int N>
__global__ void vector_add_cutlass_kernel(
    float const* A,
    float const* B,
    float* C,
    int num_elements) {

    // 定义 VectorType 为 cutlass::Array<float, N>
    using VectorType = cutlass::Array<float, N>;

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int vector_idx = idx; 

    // 边界检查：确保我们没有越界访问
    // 注意：输入指针必须是 sizeof(VectorType) 对齐的
    if (vector_idx * N < num_elements) {
        // 使用 reinterpret_cast 将 float* 转换为 VectorType*
        VectorType const* A_vec = reinterpret_cast<VectorType const*>(A);
        VectorType const* B_vec = reinterpret_cast<VectorType const*>(B);
        VectorType* C_vec = reinterpret_cast<VectorType*>(C);

        // 向量化加载
        VectorType a = A_vec[vector_idx];
        VectorType b = B_vec[vector_idx];
        VectorType c;

        // 执行加法
        // CUTLASS_PRAGMA_UNROLL 是 Cutlass 提供的宏，用于强制展开循环
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < N; ++i) {
            c[i] = a[i] + b[i];
        }

        // 向量化存储
        C_vec[vector_idx] = c;
    }
}

// 处理剩余元素的标量内核
// 当数据大小不是向量大小的倍数时使用
__global__ void vector_add_scalar_kernel(
    float const* A,
    float const* B,
    float* C,
    int num_elements,
    int offset) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x + offset;
    
    if (idx < num_elements) {
        C[idx] = A[idx] + B[idx];
    }
}

int main() {
    // 设置设备
    int deviceId = 0;
    cudaSetDevice(deviceId);

    // 数据规模
    int N = 10000003; 
    size_t size = N * sizeof(float);

    // 使用 Cutlass HostTensor 管理内存
    // HostTensor 是 Cutlass 工具库中非常有用的类，它可以同时管理 Host 和 Device 内存
    // 并且提供了简便的同步方法 (sync_device, sync_host)
    // 这里我们定义维度为 {1, N} 的张量，即一个长度为 N 的向量
    cutlass::HostTensor<float, cutlass::layout::RowMajor> tensor_A({1, N});
    cutlass::HostTensor<float, cutlass::layout::RowMajor> tensor_B({1, N});
    cutlass::HostTensor<float, cutlass::layout::RowMajor> tensor_C({1, N});

    // 初始化数据 (在 Host 端)
    // host_data() 返回主机内存指针
    float* h_A = tensor_A.host_data();
    float* h_B = tensor_B.host_data();

    for (int i = 0; i < N; ++i) {
        h_A[i] = static_cast<float>(i) * 0.1f;
        h_B[i] = static_cast<float>(i) * 0.2f;
    }

    // 将数据从 Host 同步到 Device
    // HostTensor 会自动处理内存拷贝
    tensor_A.sync_device();
    tensor_B.sync_device();

    // 配置 Kernel 参数
    const int kVectorSize = 4; // 每个线程处理 4 个 float，类似于 float4，但更灵活
    int block_size = 256;
    int num_vectors = N / kVectorSize;
    int grid_size = (num_vectors + block_size - 1) / block_size;

    printf("Vector Add using Cutlass utilities (HostTensor & Array)\n");
    printf("Vector size: %d elements\n", N);
    printf("Kernel configuration: Grid=%d, Block=%d, VectorSize=%d\n", grid_size, block_size, kVectorSize);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    // 1. 启动 Cutlass 风格的向量化内核处理主要部分
    if (num_vectors > 0) {
        vector_add_cutlass_kernel<kVectorSize><<<grid_size, block_size>>>(
            tensor_A.device_data(),
            tensor_B.device_data(),
            tensor_C.device_data(),
            N
        );
    }

    // 2. 处理剩余元素 (尾部)
    int remainder = N % kVectorSize;
    if (remainder > 0) {
        vector_add_scalar_kernel<<<1, remainder>>>(
            tensor_A.device_data(),
            tensor_B.device_data(),
            tensor_C.device_data(),
            N,
            N - remainder
        );
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    CHECK_CUDA(cudaGetLastError());

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    // 计算性能指标
    double bandwidth = (3.0 * size) / (milliseconds / 1000.0) / 1e9; // GB/s
    printf("Time: %.3f ms\n", milliseconds);
    printf("Effective Bandwidth: %.2f GB/s\n", bandwidth);

    // 将结果从 Device 同步回 Host
    tensor_C.sync_host();

    // 验证结果
    bool passed = true;
    double max_diff = 0.0;
    
    const float* h_C = tensor_C.host_data();

    for (int i = 0; i < N; ++i) {
        float expected = h_A[i] + h_B[i];
        float diff = std::abs(h_C[i] - expected);
        if (diff > max_diff) max_diff = diff;
        
        if (diff > 1e-5) {
            if (passed) { // 只打印第一个错误
                printf("Verification FAILED at index %d: A=%f, B=%f, Expected=%f, Got=%f\n", 
                       i, h_A[i], h_B[i], expected, h_C[i]);
            }
            passed = false;
        }
    }

    if (passed) {
        printf("Verification PASSED! Max diff: %e\n", max_diff);
    } else {
        printf("Verification FAILED! Max diff: %e\n", max_diff);
    }

    // 清理资源
    // HostTensor 析构函数会自动释放内存，不需要手动 cudaFree
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
