#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <type_traits>


#define WARP_SIZE 32
#define FULL_MASK 0xFFFFFFFF
#define M_LOG2E_F 1.44269504089f

#define FLOAT4(value) (reinterpret_cast<float4 *>(&(value))[0])

template <typename T, typename U> __host__ __device__ __forceinline__ auto CDIV(T a, U b) { return (a + b - 1) / b; }

template <typename T> __forceinline__ __device__ auto WarpReduceSum(T value)
{
#pragma unroll
    for (int offsets = WARP_SIZE / 2; offsets > 0; offsets >>= 1) {
        value += __shfl_down_sync(FULL_MASK, value, offsets);
    }
    return value;
}

template<typename T>
__forceinline__ __device__ auto silu(T value){
    if constexpr (std::is_same_v<T, float>) {
        return 1.0f * value / (1.0f + __expf(-value));
    } else if constexpr (std::is_same_v<T, float2>) {
        const auto v1 = value.x;
        const auto v2 = value.y;
        float2 result;
        result.x = silu(v1);
        result.y = silu(v2);
        return result;
    } else if constexpr (std::is_same_v<T, float4>) {
        const auto v1 = value.x;
        const auto v2 = value.y;
        const auto v3 = value.z;
        const auto v4 = value.w;
        float4 result;
        result.x = silu(v1);
        result.y = silu(v2);
        result.z = silu(v3);
        result.w = silu(v4);
        return result;
    }
}

__global__ void sigmoid_kernel(const float *__restrict__ input, float *__restrict__ output, const int N)
{
    auto       id     = blockDim.x * blockIdx.x + threadIdx.x;
    const auto stride = gridDim.x * blockDim.x;
    for (; id * 4 < N; id += stride) {
        const auto baseIdx   = id * 4;
        const auto remaining = N - baseIdx;
        if (remaining >= 4) {
            const auto a = reinterpret_cast<const float4 *>(input)[id];
            const auto b = silu(a);
            reinterpret_cast<float4*>(output)[id] = b;
        } else {
#pragma unroll
            for (auto j = baseIdx; j < N; ++j) {
                output[j] = silu(input[j]);
            }
        }
    }
}

// input, output are device pointers
extern "C" void solve(const float *input, float *output, int N)
{
    constexpr int threadsPerBlock = 256;
    constexpr int maxBlocks       = 2048;
    int           blocksPerGrid   = std::min((N + threadsPerBlock - 1) / threadsPerBlock, maxBlocks);

    sigmoid_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, output, N);
    cudaDeviceSynchronize();
}

