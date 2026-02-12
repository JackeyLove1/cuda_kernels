#include <cuda.h>
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cub/cub.cuh>
#include <cuda/pipeline>
#include <random>
#include <type_traits>

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
static constexpr int MAX_BLOCKS = 1024;
static constexpr int THREADS_PER_BLOCK = 512;
/**
Write a GPU program that counts the number of elements with the integer value k in an array of
32-bit integers. The program should count the number of elements with k in an array. You are given
an input array input of length N and integer k.
 */
__global__ void count_equal_kernel(const int* __restrict__ input, int* __restrict__ output,
                                   const int total, const int K) {
  __shared__ int shared_count[THREADS_PER_BLOCK];
  const int tid = threadIdx.x;
  const int gid = blockIdx.x * THREADS_PER_BLOCK + tid;
  const int stride = gridDim.x * THREADS_PER_BLOCK * PER_THREAD_WORK_ITEMS;
  const int vec4_size = total / 4;
  const int vec_tail = total % 4;
  const int4* input4 = LOAD4(input);

  int thread_count = 0;
  for (int i = gid * PER_THREAD_WORK_ITEMS; i < vec4_size; i += stride) {
#pragma unroll
    for (int j = 0; j < PER_THREAD_WORK_ITEMS; ++j) {
      const int idx4 = i + j;
      if (idx4 < vec4_size) {
        const int4 value = input4[idx4];
        thread_count += (value.x == K) + (value.y == K) + (value.z == K) + (value.w == K);
      }
    }
  }
  if (vec_tail && gid < vec_tail) {
    const int base = (vec4_size * 4) + gid;
    thread_count += (input[base] == K);
  }
  shared_count[tid] = thread_count;
  __syncthreads();

  const int warp_id = tid >> 5;
  const int lane_id = tid & 31;
  const int warp_count = WarpReduceSum(thread_count);
  if (lane_id == 0) {
    shared_count[warp_id] = warp_count;
  }
  __syncthreads();

  if (warp_id == 0) {
    constexpr int WARP_COUNT = THREADS_PER_BLOCK / WARP_SIZE;
    const auto lane_value = (lane_id < WARP_COUNT) ? shared_count[lane_id] : 0;
    const int block_count = WarpReduceSum(lane_value);
    if (tid == 0) {
      atomicAdd(output, block_count);
    }
  }
}

// input, output are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const int* input, int* output, int N, int M, int K) {
  const int total = N * M;
  int threadsPerBlock = THREADS_PER_BLOCK;
  int blocksPerGrid =
      std::min(CEIL(total, threadsPerBlock * 4 * PER_THREAD_WORK_ITEMS), MAX_BLOCKS);

  count_equal_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, output, total, K);
  cudaDeviceSynchronize();
}
