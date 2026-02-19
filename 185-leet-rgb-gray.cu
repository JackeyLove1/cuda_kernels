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
static constexpr int MAX_BLOCKS = 4096;
static constexpr int THREADS_PER_BLOCK = 1024;

/**
Implement a GPU program that converts an RGB image to grayscale on the GPU. Given an input RGB image
represented as a 1D array of 32-bit floating point values, compute the corresponding grayscale image
using the standard RGB to grayscale conversion formula. The conversion formula is: gray = 0.299 × R
+ 0.587 × G + 0.114 × B The input array input contains height × width × 3 elements, where the RGB
values for each pixel are stored consecutively (R, G, B, R, G, B,…). The output array output should
contain height × width grayscale values.
 */

__forceinline__ __device__ float op(float r, float g, float b) {
  return fmaf(0.299f, r, fmaf(0.587f, g, 0.114f * b));
}

__global__ void rgb_to_grayscale_kernel(const float* __restrict__ input, float* __restrict__ output,
                                        const int width, const int height) {
  const int tid = threadIdx.x;
  const int gid = blockIdx.x * blockDim.x + tid;
  const int stride = gridDim.x * blockDim.x;
  const int total = width * height;  // RGB
  const int vec_size = total >> 2;
  const int vec_tail = total & 3;
  // one thread process 4 items float3 * 4 for alignment
  for (int i = gid; i < vec_size; i += stride) {
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      const auto base = (i * 4 + j) * 3;
      auto r = __ldg(input + base + 0);
      auto g = __ldg(input + base + 1);
      auto b = __ldg(input + base + 2);
      output[i * 4 + j] = op(r, g, b);
    }
  }
  if (vec_tail > 0 && gid < vec_tail) {
    const auto base = ((vec_size << 2) + gid) * 3;
    auto r = __ldg(input + base + 0);
    auto g = __ldg(input + base + 1);
    auto b = __ldg(input + base + 2);
    output[(vec_size << 2) + gid] = op(r, g, b);
  }
}

// input, output are device pointers
extern "C" void solve(const float* input, float* output, int width, int height) {
  int total_pixels = width * height;
  int threadsPerBlock = THREADS_PER_BLOCK;
  int blocksPerGrid = std::min(CEIL(total_pixels, threadsPerBlock), MAX_BLOCKS);

  rgb_to_grayscale_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, output, width, height);
  cudaDeviceSynchronize();
}

int main() {

}