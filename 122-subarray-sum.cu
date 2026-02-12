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

static inline __device__ float atomicMax(float* addr, float value) {
  float old = *addr, assumed;
  if (old >= value) return old;
  do {
    assumed = old;
    old = atomicCAS((unsigned int*)addr, __float_as_int(assumed), __float_as_int(value));

  } while (old != assumed);

  return old;
}

template <typename T>
struct ReduceOp {
  static constexpr char op_id = 0;
};

template <typename T>
struct SumOp : public ReduceOp<T> {
  static constexpr char op_id = 1;

  __forceinline__ __device__ static T apply(const T& a, const T& b) { return a + b; }

  __forceinline__ __device__ static constexpr auto identity() { return T{0}; }
};

template <typename T>
struct MaxOp : public ReduceOp<T> {
  static constexpr char op_id = 2;

  __forceinline__ __device__ static T apply(const T& a, const T& b) {
    if constexpr (std::is_same_v<T, float>) {
      return fmaxf(a, b);
    } else {
      return max(a, b);
    }
  }

  __forceinline__ __device__ static constexpr auto identity() {
    if constexpr (std::is_same_v<T, float>) {
      return -INFINITY;
    } else {
      return -HUGE_VAL;
    }
  }
};

static constexpr unsigned int FULL_MASK = 0xffffffff;

template <typename T, typename Op>
__forceinline__ __device__ T WarpReduceOp(T value) {
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    value = Op::apply(value, __shfl_down_sync(FULL_MASK, value, offset));
  }
  return value;
}

template <typename T, typename Op>
inline __device__ T BlockReduceOp(T value) {
  const auto tid = threadIdx.x;
  const auto lane_id = tid & 31;
  const auto warp_id = tid >> 5;

  // warp reduce
  value = WarpReduceOp<T, Op>(value);

  __shared__ T shared_data[32];
  if (lane_id == 0) {
    shared_data[warp_id] = value;
  }
  __syncthreads();

  // block reduce
  if (warp_id == 0) {
    const int nums_warps = blockDim.x >> 5;
    const T lane_value = (lane_id < nums_warps) ? shared_data[lane_id] : Op::identity();
    value = WarpReduceOp<T, Op>(lane_value);
  }
  return value;
}

enum class ReduceOpType {
  SUM = 1,
  MAX = 2,
};

template <typename T>
__forceinline__ __device__ T BlockReduceDynamic(T value, ReduceOpType op_type) {
  switch (op_type) {
    case ReduceOpType::SUM:
      return BlockReduceOp<T, SumOp<T>>(value);
    case ReduceOpType::MAX:
      return BlockReduceOp<T, MaxOp<T>>(value);
    default:
      // Runtime-dispatched op type: keep a runtime fallback instead of
      // compile-time static_assert in a potentially unreachable branch.
      return value;
  }
}

static constexpr int PER_THREAD_WORK_ITEMS = 1;
static constexpr int WARP_SIZE = 32;
static constexpr int MAX_BLOCKS = 4096;
static constexpr int THREADS_PER_BLOCK = 1024;

/**
Implement a program that computes the sum of a subarray of 32-bit integers. You are given an input
array input of length N, and two indices S and E. S and E are inclusive, 0-based start and end
indices— compute the sum of input[S...E].
 */

void __global__ subarray_sum_kernel(const int* __restrict__ input, int* __restrict__ output,
                                    const int S, const int E) {
  const int tid = threadIdx.x;
  const int gid = THREADS_PER_BLOCK * blockIdx.x + tid;
  const int stride = THREADS_PER_BLOCK * gridDim.x;
  int thread_sum = 0;

  // Use a single half-open range [start, end) to keep boundaries consistent.
  const int start = S;
  const int end = E + 1;
  const int prefix_end = min(end, (start + 3) & ~3);
  const int vec_end = end - ((end - prefix_end) & 3);

  for (int idx = start + gid; idx < prefix_end; idx += stride) {
    thread_sum += input[idx];
  }

  const int vec_size = (vec_end - prefix_end) >> 2;
  const int4* in4 = LOAD4(input + prefix_end);
  for (int idx = gid; idx < vec_size; idx += stride) {
    const auto d = in4[idx];
    thread_sum += d.x + d.y + d.z + d.w;
  }

  for (int idx = vec_end + gid; idx < end; idx += stride) {
    thread_sum += input[idx];
  }

  thread_sum = BlockReduceDynamic<int>(thread_sum, ReduceOpType::SUM);
  if (tid == 0) {
    atomicAdd(output, thread_sum);
  }
}

// A, B, and C are device pointers
extern "C" void solve(const int* input, int* output, int N, int S, int E) {
  cudaMemset(output, 0, sizeof(int));

  if (N <= 0 || S < 0 || E < 0 || S >= N || E >= N || S > E) {
    cudaDeviceSynchronize();
    return;
  }

  const int range_len = E - S + 1;
  const auto threadsPerBlock = THREADS_PER_BLOCK;
  const auto blocksPerGrid = std::min(CEIL(range_len, threadsPerBlock), MAX_BLOCKS);

  subarray_sum_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, output, S, E);
  cudaDeviceSynchronize();
}
