#include <algorithm>
#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <iostream>
#include <time.h>
#include <cuda/barrier>
#include <cooperative_groups.h>
#include <cuda/pipeline>

using namespace std;
#define N          32 * 1024 * 1024
#define kBlockSize 256
#define kPackSize  8  // 增加到 8，即 128 bit 加载 (8 * 16 bit)

// vector inner product

template <typename T, size_t pack_size> struct alignas(sizeof(T) * pack_size) Pack
{
    T elem[pack_size];
};

template <typename T, int32_t pack_size> __device__ __inline__ void AtomicAdd(Pack<T, pack_size> *address, T val)
{
#pragma unroll
    for (int i = 0; i < pack_size; ++i) {
        atomicAdd(reinterpret_cast<T *>(address) + i, static_cast<T>(val));
    }
}

template <> __device__ __inline__ void AtomicAdd<half, 2>(Pack<half, 2> *address, half val)
{
    half2 h2_val;
    h2_val.x = static_cast<half>(val);
    h2_val.y = static_cast<half>(val);
    atomicAdd(reinterpret_cast<half2 *>(address), h2_val);
}

// 针对 pack_size=8 的特化：使用 half2 进行原子加
template <> __device__ __inline__ void AtomicAdd<half, 8>(Pack<half, 8> *address, half val)
{
    half2 h2_val;
    h2_val.x = static_cast<half>(val);
    h2_val.y = static_cast<half>(val);
    half2 *addr_h2 = reinterpret_cast<half2 *>(address);
    // 展开 atomicAdd，每次处理 2 个 half (1 个 half2)
    // 注意：这里是对 output 的每个元素都加上了 val，保持了原代码的逻辑
    atomicAdd(addr_h2, h2_val);
    atomicAdd(addr_h2 + 1, h2_val);
    atomicAdd(addr_h2 + 2, h2_val);
    atomicAdd(addr_h2 + 3, h2_val);
}

template <typename T, int32_t pack_size>
__global__ void dot(Pack<T, pack_size> *a, Pack<T, pack_size> *b, Pack<T, pack_size> *c, int n)
{
    const int nStep = gridDim.x * blockDim.x;
    T         temp  = 0.0;
    int       gid   = blockIdx.x * blockDim.x + threadIdx.x;
    while (gid < n / pack_size) {
        for (int i = 0; i < pack_size; i++) {
            temp = temp + a[gid].elem[i] * b[gid].elem[i];
        }
        gid += nStep;
    }
    AtomicAdd<T, pack_size>(c, temp);
}

// 特化 dot kernel 以利用 half2 指令和 ILP
template <>
__global__ void dot<half, 8>(Pack<half, 8> *a, Pack<half, 8> *b, Pack<half, 8> *c, int n)
{
    const int nStep = gridDim.x * blockDim.x;
    
    // 使用 4 个 accumulator 来增加指令级并行 (ILP)
    // 从而掩盖 FMA 指令的延迟
    half2 sum2_0 = __float2half2_rn(0.0f);
    half2 sum2_1 = __float2half2_rn(0.0f);
    half2 sum2_2 = __float2half2_rn(0.0f);
    half2 sum2_3 = __float2half2_rn(0.0f);

    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    
    while (gid < n / 8) {
        // 将 Pack 中的 8 个 half 解析为 4 个 half2
        // Pack<half, 8> 是 16 字节对齐的，所以 elem 是 16 字节对齐
        // 可以安全地转换为 half2* (需要 4 字节对齐)
        half2 *a_ptr = reinterpret_cast<half2*>(a[gid].elem);
        half2 *b_ptr = reinterpret_cast<half2*>(b[gid].elem);

        // 利用 HFMA2 指令进行向量化计算
        sum2_0 = __hfma2(a_ptr[0], b_ptr[0], sum2_0);
        sum2_1 = __hfma2(a_ptr[1], b_ptr[1], sum2_1);
        sum2_2 = __hfma2(a_ptr[2], b_ptr[2], sum2_2);
        sum2_3 = __hfma2(a_ptr[3], b_ptr[3], sum2_3);

        gid += nStep;
    }
    
    // 归约 4 个 accumulator
    half2 res2 = __hadd2(__hadd2(sum2_0, sum2_1), __hadd2(sum2_2, sum2_3));
    half res = res2.x + res2.y;
    
    AtomicAdd<half, 8>(c, res);
}

int main()
{
    half *x_host = (half *)malloc(N * sizeof(half));
    half *x_device;
    cudaMalloc((void **)&x_device, N * sizeof(half));
    for (int i = 0; i < N; i++)
        x_host[i] = 0.1;
    cudaMemcpy(x_device, x_host, N * sizeof(half), cudaMemcpyHostToDevice);
    Pack<half, kPackSize> *x_pack = reinterpret_cast<Pack<half, kPackSize> *>(x_device);

    half *y_host = (half *)malloc(N * sizeof(half));
    half *y_device;
    cudaMalloc((void **)&y_device, N * sizeof(half));
    for (int i = 0; i < N; i++)
        y_host[i] = 0.1;
    cudaMemcpy(y_device, y_host, N * sizeof(half), cudaMemcpyHostToDevice);
    Pack<half, kPackSize> *y_pack = reinterpret_cast<Pack<half, kPackSize> *>(y_device);

    // output buffer size matches pack size
    half *output_host = (half *)malloc(kPackSize * sizeof(half));
    half *output_device;
    cudaMalloc((void **)&output_device, kPackSize * sizeof(half));
    cudaMemset(output_device, 0, sizeof(half) * kPackSize);
    Pack<half, kPackSize> *output_pack = reinterpret_cast<Pack<half, kPackSize> *>(output_device);

    int32_t block_num = (N + kBlockSize - 1) / kBlockSize;
    dim3    grid(block_num, 1);
    dim3    block(kBlockSize, 1);
    
    // 调用特化的 kernel
    dot<half, kPackSize><<<grid, block>>>(x_pack, y_pack, output_pack, N);
    
    cudaMemcpy(output_host, output_device, kPackSize * sizeof(half), cudaMemcpyDeviceToHost);
    printf("%.6f\n", static_cast<double>(output_host[0]));
    
    free(x_host);
    free(y_host);
    free(output_host);
    cudaFree(x_device);
    cudaFree(y_device);
    cudaFree(output_device);


}
