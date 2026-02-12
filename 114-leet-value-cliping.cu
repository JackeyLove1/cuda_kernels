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

__device__ __forceinline__ float clamp(float x, float low, float high) {
  return fminf(fmaxf(x, low), high);
}

template <typename T>
__forceinline__ __device__ T op(T val, float lo, float hi) {
  return clamp(val, lo, hi);
}

__global__ void kernel(const float* __restrict__ input, float* __restrict__ output, float lo,
                       float hi, int N) {
  const int tid = threadIdx.x;
  const int gid = blockIdx.x * blockDim.x + tid;
  const int stride = gridDim.x * blockDim.x;
  const int total = N;
  const int vec_size = total >> 2;
  const int vec_tail = total & 3;
  const float4* input4 = LOAD4(input);
  float4* output4 = STORE4(output);
  for (int i = gid; i < vec_size; i += stride) {
    const float4 value = input4[i];
    const float4 result = make_float4(op(value.x, lo, hi), op(value.y, lo, hi), op(value.z, lo, hi),
                                      op(value.w, lo, hi));
    output4[i] = result;
  }
  if (vec_tail && gid < vec_tail) {
    const int base = (vec_size << 2) + gid;
    const float value = input[base];
    output[base] = op(value, lo, hi);
  }
}

// input, output are device pointers
extern "C" void solve(const float* input, float* output, float lo, float hi, int N) {
  constexpr int threadsPerBlock = THREADS_PER_BLOCK;
  int blocksPerGrid = std::min(CEIL(N, threadsPerBlock * 4), MAX_BLOCKS);
  kernel<<<blocksPerGrid, threadsPerBlock>>>(input, output, lo, hi, N);
  cudaDeviceSynchronize();
}
