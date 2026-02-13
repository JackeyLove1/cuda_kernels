#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/sort.h>

#include <algorithm>
#include <climits>
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

static constexpr int CFACTOR = 1;
static constexpr int WARP_SIZE = 32;
static constexpr int MAX_BLOCKS = 4096;
[[maybe_unused]] static constexpr int THREADS_PER_BLOCK = 512;

/**
Implement a GPU program that computes the Rotary Positional Embedding (RoPE) for a batch of query
vectors. RoPE is a method for encoding positional information in transformer models by rotating the
query and key vectors using precomputed cosine and sine components. Mathematically, given a query
vector x and corresponding cosine and sine vectors, the operation is defined as:
RoPE(x)=x⊙cos+rotate\_ half(x)⊙sin
Where ⊙ denotes element-wise multiplication. The rotate\_ half(x) operation swaps the first and
second halves of the vector and negates the first half. For a vector of dimension d: rotate\_
half([x₁,···,x₄/2,x₄/2+1,···,x₄])=-x₄/2+1,···
 */

// Q, cos, sin, output are device pointers
/**
• Q, cos, and sin have identical dimensions
• D % 2 == 0
• 1 ≤ M, D ≤ 10,000
• Performance is measured with D = 128, M = 1,048,576
 */
__global__ void rope_kernel(float* __restrict__ Q, float* __restrict__ cos, float* __restrict__ sin,
                            float* __restrict__ output, int M, int D) {
  const auto pair_idx = threadIdx.x;
  const auto pos = blockIdx.x;
  const auto half_d = D / 2;
  const int row_base = pos * D;

  // Pairwise compute: each thread handles one (low, high) pair to avoid
  // loading Q[low]/Q[high] twice by two different threads.
  for (int j = pair_idx; j < half_d; j += blockDim.x) {
    const int low = row_base + j;
    const int high = low + half_d;

    const float x_low = Q[low];
    const float x_high = Q[high];

    const float cos_low = cos[low];
    const float sin_low = sin[low];
    const float cos_high = cos[high];
    const float sin_high = sin[high];

    output[low] = x_low * cos_low - x_high * sin_low;
    output[high] = x_high * cos_high + x_low * sin_high;
  }
}

template <int ROWS_PER_BLOCK>
__global__ void rope_kernel_d128(float* __restrict__ Q, float* __restrict__ cos, float* __restrict__ sin,
                                 float* __restrict__ output, int M) {
  constexpr int D128 = 128;
  constexpr int HALF_D128 = 64;
  constexpr int VEC = 4;
  constexpr int PAIRS_PER_ROW = HALF_D128;
  constexpr int THREADS_PER_ROW = PAIRS_PER_ROW / VEC;  // 16 threads per row

  const int tid = threadIdx.x;
  const int row_in_block = tid / THREADS_PER_ROW;
  const int lane = tid % THREADS_PER_ROW;
  const int row = blockIdx.x * ROWS_PER_BLOCK + row_in_block;
  if (row >= M) return;

  const int base = row * D128;
  const int low = base + lane * VEC;
  const int high = low + HALF_D128;

  const float4 q_low = *reinterpret_cast<const float4*>(Q + low);
  const float4 q_high = *reinterpret_cast<const float4*>(Q + high);

  const float4 cos_low = *reinterpret_cast<const float4*>(cos + low);
  const float4 sin_low = *reinterpret_cast<const float4*>(sin + low);
  const float4 cos_high = *reinterpret_cast<const float4*>(cos + high);
  const float4 sin_high = *reinterpret_cast<const float4*>(sin + high);

  float4 out_low;
  out_low.x = q_low.x * cos_low.x - q_high.x * sin_low.x;
  out_low.y = q_low.y * cos_low.y - q_high.y * sin_low.y;
  out_low.z = q_low.z * cos_low.z - q_high.z * sin_low.z;
  out_low.w = q_low.w * cos_low.w - q_high.w * sin_low.w;

  float4 out_high;
  out_high.x = q_high.x * cos_high.x + q_low.x * sin_high.x;
  out_high.y = q_high.y * cos_high.y + q_low.y * sin_high.y;
  out_high.z = q_high.z * cos_high.z + q_low.z * sin_high.z;
  out_high.w = q_high.w * cos_high.w + q_low.w * sin_high.w;

  *reinterpret_cast<float4*>(output + low) = out_low;
  *reinterpret_cast<float4*>(output + high) = out_high;
}

extern "C" void solve(float* Q, float* cos, float* sin, float* output, int M, int D) {
  if (D == 128) {
    constexpr int ROWS_PER_BLOCK = 8;
    constexpr int THREADS = ROWS_PER_BLOCK * 16;
    const int blocks_per_grid = CEIL(M, ROWS_PER_BLOCK);
    rope_kernel_d128<ROWS_PER_BLOCK><<<blocks_per_grid, THREADS>>>(Q, cos, sin, output, M);
  } else {
    const int threads_per_block = std::min(THREADS_PER_BLOCK, D / 2);
    const int blocks_per_grid = M;
    rope_kernel<<<blocks_per_grid, threads_per_block>>>(Q, cos, sin, output, M, D);
  }
  cudaDeviceSynchronize();
}
