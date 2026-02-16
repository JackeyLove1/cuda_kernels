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
Implement a program for multi-head self-attention. Given three input matrices Q (queries), K (keys),
and V (values) of size N×dmodel, compute: MultiHead(Q,K,V)=Concat(head₁,…,headₙ) where each head
compute: head; =softmax (dk=dmodel/h and Qᵢ,Kᵢ,Vᵢ being the i-th head's partition of the input
matrices. Implementation Requirements ·Use only native features (external libraries are not
permitted) ·The solve function signature must remain unchanged ·The final result must be stored in
the output array
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

// A @ B, A[M, N] B[N, K] C[M, K]
template <int TILE>
__global__ void matmul_nn_kernel(const float* __restrict__ A, const float* __restrict__ B,
                                 float* __restrict__ c, const int M, const int N, const int K,
                                 const float scale) {
  const int tx = threadIdx.x, ty = threadIdx.y;
  const int row = blockIdx.y * TILE + ty;
  const int col = blockIdx.x * TILE + tx;
  float sum{0.0f};
  const auto num_tiles = CEIL(N, TILE);
  __shared__ __align__(16) float sh_A[TILE][TILE + 1];
  __shared__ __align__(16) float sh_B[TILE][TILE + 1];

  for (int tile = 0; tile < num_tiles; ++tile) {
    const int k_a = tile * TILE + tx;
    const int k_b = tile * TILE + ty;
    sh_A[ty][tx] = (row < M && k_a < N) ? A[row * N + k_a] : 0.0f;
    sh_B[ty][tx] = (k_b < N && col < K) ? B[k_b * K + col] : 0.0f;
    __syncthreads();

#pragma unroll
    for (int kk = 0; kk < TILE; ++kk) {
      sum += sh_A[ty][kk] * sh_B[kk][tx];
    }
    __syncthreads();
  }

  if (row < M && col < K) {
    c[row * K + col] = sum / scale;
  }
}

// torch.softmax(A, dim=1) A.size() = (M, N)
// threadPerBlock(256) blockPerGrid(M)
// inplace operation
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

// scores[row, col] += alpha * (row - col)
__global__ void add_alibi_bias_kernel(float* __restrict__ scores, const int M, const int N,
                                      const float alpha) {
  const int row = blockIdx.y * blockDim.y + threadIdx.y;
  const int col = blockIdx.x * blockDim.x + threadIdx.x;
  if (row < M && col < N) {
    scores[row * N + col] += alpha * static_cast<float>(row - col);
  }
}

// src: [N, d_model], dst: [N, d]
__global__ void extract_head_kernel(const float* __restrict__ src, float* __restrict__ dst, int N,
                                    int d_model, int d, int head_idx) {
  const int row = blockIdx.y * blockDim.y + threadIdx.y;
  const int col = blockIdx.x * blockDim.x + threadIdx.x;
  if (row < N && col < d) {
    dst[row * d + col] = src[row * d_model + head_idx * d + col];
  }
}

// src: [N, d], dst: [N, d_model]
__global__ void scatter_head_kernel(const float* __restrict__ src, float* __restrict__ dst, int N,
                                    int d_model, int d, int head_idx) {
  const int row = blockIdx.y * blockDim.y + threadIdx.y;
  const int col = blockIdx.x * blockDim.x + threadIdx.x;
  if (row < N && col < d) {
    dst[row * d_model + head_idx * d + col] = src[row * d + col];
  }
}

// Q, K, V, output are device pointers
extern "C" void solve(const float* Q, const float* K, const float* V, float* output, int N,
                      int d_model, int h) {
  if (h <= 0 || d_model <= 0 || N <= 0 || d_model % h != 0) {
    fprintf(stderr, "Invalid MHA shape: N=%d d_model=%d h=%d\n", N, d_model, h);
    return;
  }

  const int d = d_model / h;
  const float scale = sqrtf(static_cast<float>(d));
  float* scores = nullptr;
  const size_t scores_bytes = static_cast<size_t>(N) * static_cast<size_t>(N) * sizeof(float);
  CUDA_CHECK(cudaMalloc(&scores, scores_bytes));
  const size_t qkv_size = static_cast<size_t>(N) * static_cast<size_t>(d) * sizeof(float);
  float *q_i, *k_i, *v_i, *o_i;
  CUDA_CHECK(cudaMalloc(&q_i, qkv_size));
  CUDA_CHECK(cudaMalloc(&k_i, qkv_size));
  CUDA_CHECK(cudaMalloc(&v_i, qkv_size));
  CUDA_CHECK(cudaMalloc(&o_i, qkv_size));

  const dim3 vec_block(16, 16);
  const dim3 vec_grid(CEIL(d, 16), CEIL(N, 16));
  for (int h_i = 0; h_i < h; ++h_i) {
    extract_head_kernel<<<vec_grid, vec_block>>>(Q, q_i, N, d_model, d, h_i);
    CUDA_CHECK(cudaGetLastError());
    extract_head_kernel<<<vec_grid, vec_block>>>(K, k_i, N, d_model, d, h_i);
    CUDA_CHECK(cudaGetLastError());
    extract_head_kernel<<<vec_grid, vec_block>>>(V, v_i, N, d_model, d, h_i);
    CUDA_CHECK(cudaGetLastError());

    const dim3 matmul_block(MATMUL_TILE, MATMUL_TILE);
    const dim3 qk_grid(CEIL(N, MATMUL_TILE), CEIL(N, MATMUL_TILE));
    // scores[N, N] = (Q_i[N, d] @ K_i[N, d]^T) / sqrt(d)
    matmul_kernel<MATMUL_TILE><<<qk_grid, matmul_block>>>(q_i, k_i, scores, N, d, N, scale);
    CUDA_CHECK(cudaGetLastError());

    row_softmax_kernel<<<N, 256>>>(scores, N, N);
    CUDA_CHECK(cudaGetLastError());

    const dim3 ov_grid(CEIL(d, MATMUL_TILE), CEIL(N, MATMUL_TILE));
    // head_out[N, d] = scores[N, N] @ V_i[N, d]
    matmul_nn_kernel<MATMUL_TILE><<<ov_grid, matmul_block>>>(scores, v_i, o_i, N, N, d, 1.0f);
    CUDA_CHECK(cudaGetLastError());

    // output[N, d_model] <- concat head_out on the last dimension.
    scatter_head_kernel<<<vec_grid, vec_block>>>(o_i, output, N, d_model, d, h_i);
    CUDA_CHECK(cudaGetLastError());
  }
  CUDA_CHECK(cudaFree(scores));
  CUDA_CHECK(cudaFree(q_i));
  CUDA_CHECK(cudaFree(k_i));
  CUDA_CHECK(cudaFree(v_i));
  CUDA_CHECK(cudaFree(o_i));
}