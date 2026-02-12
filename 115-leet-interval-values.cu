#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cub/cub.cuh>
#include <cuda/pipeline>
#include <numeric>
#include <random>
#include <type_traits>
#include <vector>

#define CEIL(a, b) (((a) + (b) - 1) / (b))

template <typename>
struct dependent_false : std::false_type {};

template <typename T>
__forceinline__ __device__ auto LOAD4(const T* ptr) {
  if constexpr (std::is_same_v<T, int>) {
    return reinterpret_cast<const int4*>(ptr);
  } else if constexpr (std::is_same_v<T, float>) {
    return reinterpret_cast<const float4*>(ptr);
  } else if constexpr (std::is_same_v<T, double>) {
    return reinterpret_cast<const double4*>(ptr);
  } else if constexpr (std::is_same_v<T, short>) {
    return reinterpret_cast<const short4*>(ptr);
  } else {
    static_assert(dependent_false<T>::value, "Unsupported type for LOAD4");
    return nullptr;
  }
}

template <typename T>
__forceinline__ __device__ auto STORE4(T* ptr) {
  if constexpr (std::is_same_v<T, int>) {
    return reinterpret_cast<int4*>(ptr);
  } else if constexpr (std::is_same_v<T, float>) {
    return reinterpret_cast<float4*>(ptr);
  } else if constexpr (std::is_same_v<T, double>) {
    return reinterpret_cast<double4*>(ptr);
  } else if constexpr (std::is_same_v<T, short>) {
    return reinterpret_cast<short4*>(ptr);
  } else {
    static_assert(dependent_false<T>::value, "Unsupported type for STORE4");
    return nullptr;
  }
}

__forceinline__ __device__ __host__ float4 operator+(const float4 a, const float4 b) {
  return make_float4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w);
}

static constexpr unsigned int FULL_MASK = 0xffffffff;

template <typename T>
__forceinline__ __device__ T WarpReduceSum(T value) {
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    value += __shfl_down_sync(FULL_MASK, value, offset);
  }
  return value;
}

static constexpr int PER_THREAD_WORK_ITEMS = 1;
static constexpr int WARP_SIZE = 32;
static constexpr int MAX_BLOCKS = 2048;
static constexpr int THREADS_PER_BLOCK = 256;

/**
Write a GPU program that interleaves two arrays of 32-bit floating point numbers. Given two input
arrays A and B, each of length N, produce an output array of length 2N where elements alternate
between the two inputs:[A[0],B[0],A[1],B[1],A[2],B[2],…]
 */
__global__ void interleave_kernel(const float* __restrict__ A, const float* __restrict__ B,
                                  float* __restrict__ output, const int N) {
  const int tid = threadIdx.x;
  const int gid = blockIdx.x * blockDim.x + tid;
  const int stride = gridDim.x * blockDim.x;
  const int total = N;
  const int vec_size = total >> 2;
  const int vec_tail = total & 3;
  const float4* a4 = LOAD4(A);
  const float4* b4 = LOAD4(B);
  float4* output4 = STORE4(output);
  for (int i = gid; i < vec_size; i += stride) {
    const float4 a = a4[i];
    const float4 b = b4[i];
    const float4 result1 = make_float4(a.x, b.x, a.y, b.y);
    const float4 result2 = make_float4(a.z, b.z, a.w, b.w);
    output4[i * 2] = result1;
    output4[i * 2 + 1] = result2;
  }

  if (vec_tail && gid < vec_tail) {
    const int base = (vec_size << 2) + gid;
    const float a = A[base];
    const float b = B[base];
    output[base * 2] = a;
    output[base * 2 + 1] = b;
  }
}

// A, B, output are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* A, const float* B, float* output, int N) {
  int threadsPerBlock = THREADS_PER_BLOCK;
  int blocksPerGrid = std::min(CEIL(N, threadsPerBlock * 4), MAX_BLOCKS);

  interleave_kernel<<<blocksPerGrid, threadsPerBlock>>>(A, B, output, N);
  cudaDeviceSynchronize();
}