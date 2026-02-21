// cute_tensor_print.cu
// 编译: nvcc -std=c++17 -I /path/to/cutlass/include cute_tensor_print.cu -o cute_tensor_print

#include <cuda_runtime.h>
#include <stdio.h>

// CuTe 核心头文件
#include "cute/tensor.hpp"
#include "cute/layout.hpp"

using namespace cute;

// ============================================================
// 1. 在 GPU kernel 中打印 Tensor（用于寄存器 Tensor）
// ============================================================
__global__ void kernel_print_register_tensor() {
    // 只让 thread 0 打印，避免输出混乱
    if (threadIdx.x != 0 || blockIdx.x != 0) return;

    printf("\n===== [GPU] Register Tensor =====\n");

    // --- 创建 1D 寄存器 Tensor ---
    auto tensor_1d = make_tensor<float>(make_shape(Int<8>{}));
    // 填充数据
    for (int i = 0; i < size(tensor_1d); ++i) {
        tensor_1d(i) = float(i) * 1.1f;
    }
    // 打印
    printf("[1D tensor, shape=8]\n");
    for (int i = 0; i < size(tensor_1d); ++i) {
        printf("  tensor_1d(%d) = %.2f\n", i, tensor_1d(i));
    }

    // --- 创建 2D 寄存器 Tensor (4x4) ---
    auto tensor_2d = make_tensor<float>(make_shape(Int<4>{}, Int<4>{}));
    for (int i = 0; i < size<0>(tensor_2d); ++i) {
        for (int j = 0; j < size<1>(tensor_2d); ++j) {
            tensor_2d(i, j) = float(i * 4 + j);
        }
    }
    printf("\n[2D tensor, shape=(4,4)]\n");
    for (int i = 0; i < size<0>(tensor_2d); ++i) {
        printf("  row %d: ", i);
        for (int j = 0; j < size<1>(tensor_2d); ++j) {
            printf("%6.1f ", tensor_2d(i, j));
        }
        printf("\n");
    }

    // --- 打印 Layout 信息 ---
    printf("\n[Layout info]\n");
    printf("  rank        = %d\n", (int)decltype(tensor_2d)::layout_type::rank);
    printf("  size<0>     = %d\n", (int)size<0>(tensor_2d));
    printf("  size<1>     = %d\n", (int)size<1>(tensor_2d));
    printf("  total size  = %d\n", (int)size(tensor_2d));
}

// ============================================================
// 2. 在 GPU kernel 中打印全局内存 Tensor
// ============================================================
__global__ void kernel_print_gmem_tensor(float* ptr, int M, int N) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;

    printf("\n===== [GPU] Global Memory Tensor =====\n");

    // 行主序 layout: shape=(M,N), stride=(N,1)
    auto layout = make_layout(make_shape(M, N), make_stride(N, 1));
    auto tensor  = make_tensor(make_gmem_ptr(ptr), layout);

    printf("[gmem tensor, shape=(%d,%d), stride=(row-major)]\n", M, N);
    for (int i = 0; i < size<0>(tensor); ++i) {
        printf("  row %d: ", i);
        for (int j = 0; j < size<1>(tensor); ++j) {
            printf("%6.2f ", tensor(i, j));
        }
        printf("\n");
    }

    // 演示切片：取第 0 行
    auto row0 = tensor(0, _);
    printf("\n[Slice: row 0]\n  ");
    for (int j = 0; j < size(row0); ++j) {
        printf("%6.2f ", row0(j));
    }
    printf("\n");

    // 演示切片：取第 1 列
    auto col1 = tensor(_, 1);
    printf("\n[Slice: col 1]\n  ");
    for (int i = 0; i < size(col1); ++i) {
        printf("%6.2f ", col1(i));
    }
    printf("\n");
}

// ============================================================
// 3. 在 GPU kernel 中打印共享内存 Tensor
// ============================================================
__global__ void kernel_print_smem_tensor() {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;

    __shared__ float smem[32];

    // 初始化共享内存
    for (int i = 0; i < 32; ++i) smem[i] = float(i) * 0.5f;

    printf("\n===== [GPU] Shared Memory Tensor =====\n");

    // 把共享内存包装成 2D Tensor (4x8)，行主序
    auto tensor = make_tensor(make_smem_ptr(smem),
                              make_layout(make_shape(4, 8),
                                          make_stride(8, 1)));

    printf("[smem tensor, shape=(4,8)]\n");
    for (int i = 0; i < size<0>(tensor); ++i) {
        printf("  row %d: ", i);
        for (int j = 0; j < size<1>(tensor); ++j) {
            printf("%5.1f ", tensor(i, j));
        }
        printf("\n");
    }
}

// ============================================================
// 4. CPU 端打印 Layout 结构（无需 GPU）
// ============================================================
void cpu_print_layout_info() {
    printf("\n===== [CPU] Layout Info =====\n");

    // 行主序 (4,8) Layout
    auto layout_rm = make_layout(make_shape(4, 8), make_stride(8, 1));
    printf("[Row-Major Layout (4,8)]\n");
    printf("  shape  = (%d, %d)\n", (int)shape<0>(layout_rm), (int)shape<1>(layout_rm));
    printf("  stride = (%d, %d)\n", (int)stride<0>(layout_rm), (int)stride<1>(layout_rm));
    // 线性地址计算验证
    printf("  layout(1,2) -> linear index = %d\n", (int)layout_rm(1, 2)); // 应为 1*8+2=10

    // 列主序 (4,8) Layout
    auto layout_cm = make_layout(make_shape(4, 8), make_stride(1, 4));
    printf("\n[Col-Major Layout (4,8)]\n");
    printf("  shape  = (%d, %d)\n", (int)shape<0>(layout_cm), (int)shape<1>(layout_cm));
    printf("  stride = (%d, %d)\n", (int)stride<0>(layout_cm), (int)stride<1>(layout_cm));
    printf("  layout(1,2) -> linear index = %d\n", (int)layout_cm(1, 2)); // 应为 1*1+2*4=9

    // 使用 cute::print 打印 Layout（调试利器）
    printf("\n[cute::print output]\n  ");
    print(layout_rm);
    printf("\n  ");
    print(layout_cm);
    printf("\n");
}

// ============================================================
// main
// ============================================================
int main() {
    cpu_print_layout_info();

    // 准备全局内存数据
    const int M = 3, N = 5;
    float h_data[M * N];
    for (int i = 0; i < M * N; ++i) h_data[i] = float(i) + 0.5f;

    float* d_data;
    cudaMalloc(&d_data, M * N * sizeof(float));
    cudaMemcpy(d_data, h_data, M * N * sizeof(float), cudaMemcpyHostToDevice);

    // 启动 kernels（单 block，单 thread 打印）
    kernel_print_register_tensor<<<1, 1>>>();
    cudaDeviceSynchronize();

    kernel_print_gmem_tensor<<<1, 1>>>(d_data, M, N);
    cudaDeviceSynchronize();

    kernel_print_smem_tensor<<<1, 1>>>();
    cudaDeviceSynchronize();

    cudaFree(d_data);

    printf("\nDone.\n");
    return 0;
}