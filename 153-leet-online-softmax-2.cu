#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/sort.h>

#include <algorithm>
#include <cfloat>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cub/cub.cuh>
#include <cuda/pipeline>
#include <limits>
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
      return (a > b) ? a : b;
    }
  }

  __forceinline__ __device__ static constexpr auto identity() {
    if constexpr (std::is_same_v<T, float>) {
      return -INFINITY;
    } else if constexpr (std::is_same_v<T, half>) {
      return -65504;
    } else {
      return std::numeric_limits<T>::lowest();
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

[[maybe_unused]] static constexpr int CFACTOR = 1;
[[maybe_unused]] static constexpr int WARP_SIZE = 32;
[[maybe_unused]] static constexpr int MAX_BLOCKS = 4096;
[[maybe_unused]] static constexpr int THREADS_PER_BLOCK = 256;
[[maybe_unused]] static constexpr float FLOAT_MAX = INFINITY;
/**
Write a program that compute s the softmax function for an array of 32-bit floating-point numbers on
a GPU. The softmax function is defined as follows: For an input array x of length n, the softmax of
x, denoted σ(x), is an array of length n where the i-th element is: σ(x)i= e xi ∑j=1ne xj Your
solution should handle potential overflow issues by using the"max trick". Subtract the maximum value
of the input array from each element before exponentiation.
 */

struct __align__(8) stats {
  float max_val;
  float sum_val;
};

__device__ __forceinline__ stats default_stats() { return stats{FLOAT_MAX, 0.0f}; }

__device__ __forceinline__ stats combine_state(const stats a, const stats b) {
  if (a.max_val == FLOAT_MAX) return b;
  if (b.max_val == FLOAT_MAX) return a;
  float max_val = fmaxf(a.max_val, b.max_val);
  float a_sum = __expf(a.max_val - max_val) * a.sum_val;
  float b_sum = __expf(b.max_val - max_val) * b.sum_val;
  float sum_val = a_sum + b_sum;
  return stats{max_val, sum_val};
}

__device__ __forceinline__ stats WarpReduceStats(stats value) {
  for (int offset = 16; offset > 0; offset >>= 1) {
    float other_max = __shfl_down_sync(FULL_MASK, value.max_val, offset);
    float other_sum = __shfl_down_sync(FULL_MASK, value.sum_val, offset);
    value = combine_state(value, {other_max, other_sum});
  }
  return value;
}

__device__ inline stats BlockReduceStats(stats value) {
  __shared__ stats shared_data[32];
  value = WarpReduceStats(value);
  const int lane_id = threadIdx.x & 31;
  const int warp_id = threadIdx.x >> 5;

  if (lane_id == 0) {
    shared_data[warp_id] = value;
  }
  __syncthreads();

  if (warp_id == 0) {
    const int nums_warps = blockDim.x >> 5;
    const stats lane_value = (lane_id < nums_warps) ? shared_data[lane_id] : default_stats();
    value = WarpReduceStats(lane_value);
  }
  return value;
}

// stage1
__global__ void compute_block_state(const float* __restrict__ input,
                                    stats* __restrict__ block_state, const int N) {
  const int tid = threadIdx.x;
  const int gid = blockIdx.x * blockDim.x + tid;
  const int stride = blockDim.x * gridDim.x;
  stats value = default_stats();
  for (int i = gid; i < N; i += stride) {
    value = combine_state(value, {input[i], 1.0f});
  }
  value = BlockReduceStats(value);
  if (tid == 0) {
    block_state[blockIdx.x] = value;
  }
}

// stage2

__global__ void reduce_block_stats(stats* __restrict__ block_state, float* global_max,
                                   float* global_sum, const int num_blocks) {
  const int tid = threadIdx.x;
  auto value = (tid < num_blocks) ? block_state[tid] : default_stats();
  value = BlockReduceStats(value);
  if (tid == 0) {
    *global_max = value.max_val;
    *global_sum = value.sum_val;
  }
}

// stage3
__global__ void softmax_kernel(const float* input, float* output, float* global_max,
                               float* global_sum, const int N) {
  const int tid = threadIdx.x;
  const int gid = blockIdx.x * blockDim.x + tid;
  const int stride = blockDim.x * gridDim.x;
  const auto max_val = *global_max;
  const auto sum_val = *global_sum;
  const auto sum_inv = 1.0f / sum_val;
  for (int i = gid; i < N; i += stride) {
    output[i] = __expf(input[i] - max_val) * sum_inv;
  }
}

// input, output are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* input, float* output, int N) {
  const int threadsPerBlock = THREADS_PER_BLOCK;
  const int blocksPerGrid = CEIL(N, THREADS_PER_BLOCK);
  stats* block_states;
  float* global_max;
  float* global_sum;

  cudaMalloc(&block_states, blocksPerGrid * sizeof(stats));
  cudaMalloc(&global_max, sizeof(float));
  cudaMalloc(&global_sum, sizeof(float));

  compute_block_state<<<blocksPerGrid, threadsPerBlock>>>(input, block_states, N);
  reduce_block_stats<<<1, THREADS_PER_BLOCK>>>(block_states, global_max, global_sum, blocksPerGrid);
  softmax_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, output, global_max, global_sum, N);
  cudaDeviceSynchronize();
}
