#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <vector>

#define FULL_MASK 0xFFFFFFFF

template <typename T, typename K> __host__ __device__ __forceinline__ auto CEIL(T a, K b)
{
    static_assert(std::is_arithmetic_v<T> && std::is_arithmetic_v<K>, "T and K must be integral");
    return (a + b - 1) / b;
}

__global__ void sumKernel(const float *__restrict__ input, float *__restrict__ output, const int N)
{
    // 1. load float4 data and add
    // 2. reduce sum from wrap
    // 3. load wrap sum to shared mem
    // 4. use wrap shuffle to add wrap sum to block sum
    // 5. atomic add block sum to output
    __shared__ float smem[32];
    const int        tid       = threadIdx.x;
    const int        gid       = blockDim.x * blockIdx.x + threadIdx.x;
    const int        wrapId    = tid >> 5;
    const int        laneId    = tid & (warpSize - 1);
    const int        baseIdx   = gid * 4;
    const int        remaining = N - baseIdx;

    float val = 0.0f;
    if (remaining >= 4) {
        const float4 *input_float4 = reinterpret_cast<const float4 *>(input);
        float4        v4           = input_float4[gid];
        val                        = v4.x + v4.y + v4.z + v4.w;
    }
    else if (remaining > 0) {
#pragma unroll
        for (int i = 0; i < remaining; i++) {
            val += input[baseIdx + i];
        }
    }

    // wrap reduce sum
    const unsigned mask = __activemask();
#pragma unroll
    for (int stride = 16; stride > 0; stride >>= 1) {
        val += __shfl_down_sync(mask, val, stride);
    }

    if (laneId == 0)
        smem[wrapId] = val; // wrapId is 0 ~ MIN(CEIL(blockDim.x, 32), 5)
    __syncthreads();

    // block reduce sum
    auto wrapNum = CEIL(blockDim.x, 32);
    if (wrapId == 0) {
        val = (laneId < wrapNum) ? smem[laneId] : 0.0f; // laneId is 0 ~ 31
#pragma unroll
        for (int stride = 16; stride > 0; stride >>= 1) {
            val += __shfl_down_sync(FULL_MASK, val, stride);
        }
        if (laneId == 0)
            atomicAdd(output, val);
    }
}


// input, output are device pointers
extern "C" void solve(const float *input, float *output, int N)
{
    constexpr int threadsPerBlock = 512;
    constexpr int maxBlocks       = 65535;
    int           blocksPerGrid   = std::min(maxBlocks, CEIL(N, threadsPerBlock * 4)); // float4进行优化
    sumKernel<<<blocksPerGrid, threadsPerBlock>>>(input, output, N);
    cudaDeviceSynchronize();
}
