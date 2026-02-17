#include <cooperative_groups.h>
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
#include <cstdint>
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

__forceinline__ __device__ __host__ float2 operator+(const float2 a, const float2 b) {
  return make_float2(a.x + b.x, a.y + b.y);
}

[[maybe_unused]] static inline __device__ float atomicMax(float* addr, float value) {
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
static constexpr int MATMUL_TILE = 16;
[[maybe_unused]] static constexpr float DEFAULT_ALIBI_ALPHA = 1.0f;

#define CUDA_CHECK(call)                                               \
  do {                                                                 \
    cudaError_t _status = (call);                                      \
    if (_status != cudaSuccess) {                                      \
      fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
              cudaGetErrorString(_status));                            \
      return;                                                          \
    }                                                                  \
  } while (0)

/**
1. Standard Softmax Attention
Given query matrix Q, key matrix K, and value matrix V, each position i attends to all positions j
using a softmax-weighted sum:scoreᵢⱼ= Qᵢ⋅Kⱼ/✔d outputᵢ= ∑ⱼ∈Msoftmax(scoreᵢⱼ*)ⱼ⋅Vⱼ In other words,
each query computes similarity with all keys, applies a softmax to get attention weights, and then
computes a weighted sum of values.
2. Sliding Window Self-Attention
Sliding Window Attention modifies standard attention by restricting each query to attend only to a
local window around its position. ·For each position i, only consider the keys and values within a
window of size window size around i(positions [i-window size, i+window size]) ·Compute similarity
scores between Qᵢ and the keys in this window:scoreᵢⱼ= Qᵢ⋅Kⱼ/✔d ·Apply softmax over these local
scores to obtain attention weights. ·Use the weights to compute a weighted average of the values in
the same window: outputᵢ= ∑ⱼ∈[i-window size,i+window size]softmax(scoreᵢⱼ*)ⱼ⋅Vⱼ
 */

// A @ B.T , A[M, N] B[K, N] C[M, K]
// threadPerBlock(TILE, TILE) blockPerGrid(CEIL(K, TILE), CEIL(M, TILE))
template <int TILE>
__global__ void matmul_kernel(const float* __restrict__ A, const float* __restrict__ B,
                              float* __restrict__ c, const int M, const int N, const int K,
                              const float scale) {
  const int tx = threadIdx.x, ty = threadIdx.y;
  const int row = blockIdx.y * TILE + ty;
  const int col = blockIdx.x * TILE + tx;
  float sum{0.0f};
  const auto num_tiles = CEIL(N, TILE);
  __shared__ __align__(16) float sh_A[2][TILE][TILE + 1];
  __shared__ __align__(16) float sh_B[2][TILE][TILE + 1];

  cuda::pipeline<cuda::thread_scope_thread> pipe = cuda::make_pipeline();

  pipe.producer_acquire();
  if (row < M && tx < N) {
    cuda::memcpy_async(&sh_A[0][ty][tx], &A[row * N + tx], sizeof(float), pipe);
  } else {
    sh_A[0][ty][tx] = 0.0f;
  }
  if (col < K && ty < N) {
    // B is [K, N], while this kernel computes A @ B.T.
    // So B.T[ty, col] maps to B[col, ty].
    cuda::memcpy_async(&sh_B[0][ty][tx], &B[col * N + ty], sizeof(float), pipe);
  } else {
    sh_B[0][ty][tx] = 0.0f;
  }
  pipe.producer_commit();

  for (int k = 0; k < num_tiles; ++k) {
    if (k + 1 < num_tiles) {
      pipe.producer_acquire();
      auto next_tile = k + 1;
      auto write_stage = next_tile % 2;
      const auto a_col = next_tile * TILE + tx;
      const auto b_col = next_tile * TILE + ty;

      if (row < M && a_col < N) {
        cuda::memcpy_async(&sh_A[write_stage][ty][tx], &A[row * N + a_col], sizeof(float), pipe);
      } else {
        sh_A[write_stage][ty][tx] = 0.0f;
      }
      if (col < K && b_col < N) {
        cuda::memcpy_async(&sh_B[write_stage][ty][tx], &B[col * N + b_col], sizeof(float), pipe);
      } else {
        sh_B[write_stage][ty][tx] = 0.0f;
      }
      pipe.producer_commit();
    }

    pipe.consumer_wait();
    __syncthreads();

    auto read_stage = k % 2;
#pragma unroll
    for (int kk = 0; kk < TILE; ++kk) {
      sum += sh_A[read_stage][ty][kk] * sh_B[read_stage][kk][tx];
    }
    __syncthreads();
    pipe.consumer_release();
  }

  if (row < M && col < K) {
    c[row * K + col] = sum / scale;
  }
}

__global__ __launch_bounds__(256) void row_softmax_kernel(float* __restrict__ A, const int M,
                                                          const int N) {
  const int tid = threadIdx.x;
  const int row = blockIdx.x;
  const int row_offset = row * N;
  const auto nthreads = blockDim.x;
  float max_val = -INFINITY;
  float sum_val = 0.0f;
  __shared__ float global_max;
  __shared__ float global_sum;
  for (int i = tid; i < N; i += nthreads) {
    max_val = fmaxf(max_val, A[row_offset + i]);
  }
  max_val = BlockReduceDynamic(max_val, ReduceOpType::MAX);
  if (tid == 0) {
    global_max = max_val;
  }
  __syncthreads();

  for (int i = tid; i < N; i += nthreads) {
    sum_val += __expf(A[row_offset + i] - global_max);
  }
  sum_val = BlockReduceDynamic(sum_val, ReduceOpType::SUM);
  if (tid == 0) {
    global_sum = sum_val;
  }
  __syncthreads();

  for (int i = tid; i < N; i += nthreads) {
    A[row_offset + i] = __expf(A[row_offset + i] - global_max) / global_sum;
  }
}

__global__ void slidw_window_attention() {}

extern "C" void solve(const float* Q, const float* K, const float* V, float* output, const int M,
                      const int d, const int window_size) {
  float* scores;
  CUDA_CHECK(cudaMalloc(&scores, sizeof(float) * M * M));
  const dim3 matmul_block(MATMUL_TILE, MATMUL_TILE);
  const dim3 qk_grid(CEIL(d, MATMUL_TILE), CEIL(M, MATMUL_TILE));
  // scores[N, N] = (Q_i[N, d] @ K_i[N, d]^T) / sqrt(d)
  const float scale = 1.0f / sqrtf(d);
  matmul_kernel<MATMUL_TILE><<<qk_grid, matmul_block>>>(Q, K, scores, M, d, M, scale);
  CUDA_CHECK(cudaGetLastError());

  // scores[M, M] = softmax(scores[M, M])
  row_softmax_kernel<<<M, 256>>>(scores, M, M);
  CUDA_CHECK(cudaGetLastError());

  // output[M, d] = scores[M, M] @ V[M, d]

  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaFree(scores));
}