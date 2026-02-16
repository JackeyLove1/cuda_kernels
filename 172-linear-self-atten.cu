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
Implement Linear Attention for a given set of matrices, following the method described
in"Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention". Given the query
matrix Q of size Mxd, key matrix K of size Mxd, and value matrix V of size Mxd, your program should
compute the output matrix using the formula: LinearAttention(Q,K,V)=φ(Q)(φ(K)TV) φ(Q)(∑jφ(Kj)) where
φ(x) is a feature map applied element-wise, for example: φ(x)=ELU(x)+1=(x+1,x>0 ex,x≤0 All matrices
Q, K, V, and output are of type float32, and M and d are of type int32. Implementation Requirements
·Use only native features (external libraries are not permitted)
·The solve function signature must remain unchanged
·The final result must be stored in the output matrix output
 */

/**
```python
import torch
import torch.nn as nn

def phi(x: torch.Tensor) -> torch.Tensor:
   """
   Element-wise function: φ(x) = ELU(x) + 1
   φ(x) = { x + 1,   if x > 0
            e^x,     if x <= 0
   """
   # 使用 PyTorch 内置的 ELU 函数（alpha=1.0 是默认值）
   elu = nn.ELU(alpha=1.0)
   return elu(x) + 1.0

# Q, K, V, output are tensors on the GPU
def solve(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor,
         output: torch.Tensor, M: int, d: int):
   Q = Q.view(M, d).contiguous()
   K = K.view(M, d).contiguous()
   V = V.view(M, d).contiguous()
   Q_phi = phi(Q)                # (M, d)
   K_phi = phi(K)                # (M, d)
   V = V                         # (M, d)

   # Numerator: (M, d) @ (d, d) = (M, d)
   KV = torch.matmul(K_phi.T, V)     # (d, d)
   numerator = torch.matmul(Q_phi, KV)  # (M, d)

   # Denominator: (M, d) @ (d,) = (M,)
   K_sum = K_phi.sum(dim=0)          # (d,)
   denominator = torch.matmul(Q_phi, K_sum)  # (M,)

   # Avoid division by zero
   denominator = denominator.unsqueeze(-1) + 1e-6  # (M, 1)
   result = numerator / denominator                # (M, d)

   output.copy_(result)
```

Constraints
Matrix Q, K, and V are all of size M×d
1 ≤ M ≤ 10000
1 ≤ d ≤ 128
All elements in Q, K, and V are sampled from[-100.0, 100.0]
Data type for all matrices is float32
Performance is measured with M = 10,000
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

__forceinline__ __device__ float phi(float x) { return x > 0 ? (x + 1.0f) : __expf(x); }

// <<<CEIL(M, TILE), 32>>> 32 * 4 == 128
// TODO: use float4 accelerate
__global__ void phi_kernel_vec(const float* __restrict__ input, float* __restrict__ output,
                               const int M, const int d) {
  const int tid = threadIdx.x;
  const int row_base = blockIdx.x, row_stride = gridDim.x;
  const int row_threads = blockDim.x;
  const int vec_size = d >> 2;
  const int vec_tile = d & 3;
  for (int r = row_base; r < M; r += row_stride) {
    const auto row_offset = r * d;
    const float4* input_row_4 = LOAD4(input + row_offset);
    float4* output_row_4 = STORE4(output + row_offset);
    for (int i = tid; i < vec_size; i += row_threads) {
      const float4 input_val = input_row_4[i];
      const float4 output_val =
          make_float4(phi(input_val.x), phi(input_val.y), phi(input_val.z), phi(input_val.w));
      output_row_4[i] = output_val;
    }
    if (vec_tile && tid < vec_tile) {
      const int base_idx = (vec_size << 2) + tid;
      const float input_val = input[row_offset + base_idx];
      const float output_val = phi(input_val);
      output[row_offset + base_idx] = output_val;
    }
  }
}

__global__ void phi_kernel_common(const float* __restrict__ input, float* __restrict__ output,
                                  const int M, const int d) {
  const int tid = threadIdx.x;
  const int row_base = blockIdx.x, row_stride = gridDim.x;
  const int row_threads = blockDim.x;
  const int vec_size = d >> 2;
  const int vec_tile = d & 3;
  for (int r = row_base; r < M; r += row_stride) {
    const auto row_offset = r * d;
    const float* input_row = input + row_offset;
    float* output_row = output + row_offset;
    for (int i = tid; i < d; i += row_threads) {
      const float input_val = input_row[i];
      const float output_val = phi(input_val);
      output_row[i] = output_val;
    }
  }
}

__global__ void ksum_kernel(const float* __restrict__ K_phi, float* __restrict__ K_sum, const int M,
                            const int d) {
  const int col = blockIdx.x * blockDim.x + threadIdx.x;
  if (col >= d) return;
  float acc = 0.0f;
  for (int m = 0; m < M; ++m) {
    acc += K_phi[m * d + col];
  }
  K_sum[col] = acc;
}

// KV = K_phi^T @ V, where K_phi and V are [M, d], KV is [d, d].
__global__ void kv_kernel(const float* __restrict__ K_phi, const float* __restrict__ V,
                          float* __restrict__ KV, const int M, const int d) {
  const int row = blockIdx.y * blockDim.y + threadIdx.y;
  const int col = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= d || col >= d) return;

  float acc = 0.0f;
  for (int m = 0; m < M; ++m) {
    acc += K_phi[m * d + row] * V[m * d + col];
  }
  KV[row * d + col] = acc;
}

__global__ void normalize_kernel(const float* __restrict__ numerator,
                                 const float* __restrict__ denominator, float* __restrict__ output,
                                 const int M, const int d, const float eps) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int total = M * d;
  if (idx >= total) return;
  const int row = idx / d;
  output[idx] = numerator[idx] / (denominator[row] + eps);
}

extern "C" void solve(const float* Q, const float* K, const float* V, float* output, int M, int d) {
  float *Q_phi = nullptr, *K_phi = nullptr;
  float *KV = nullptr, *numerator = nullptr, *denominator = nullptr, *K_sum = nullptr;
  CUDA_CHECK(cudaMalloc(&Q_phi, M * d * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&K_phi, M * d * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&KV, d * d * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&numerator, M * d * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&denominator, M * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&K_sum, d * sizeof(float)));

  CUDA_CHECK(cudaMemset(Q_phi, 0, M * d * sizeof(float)));
  CUDA_CHECK(cudaMemset(K_phi, 0, M * d * sizeof(float)));
  CUDA_CHECK(cudaMemset(KV, 0, d * d * sizeof(float)));
  CUDA_CHECK(cudaMemset(numerator, 0, M * d * sizeof(float)));
  CUDA_CHECK(cudaMemset(denominator, 0, M * sizeof(float)));
  CUDA_CHECK(cudaMemset(K_sum, 0, d * sizeof(float)));

  const bool can_vec_q = ((d & 3) == 0) && ((reinterpret_cast<uintptr_t>(Q) & 0xF) == 0);
  const bool can_vec_k = ((d & 3) == 0) && ((reinterpret_cast<uintptr_t>(K) & 0xF) == 0);
  if (can_vec_q) {
    phi_kernel_vec<<<CEIL(M, 32), 32>>>(Q, Q_phi, M, d);
  } else {
    phi_kernel_common<<<CEIL(M, 32), 32>>>(Q, Q_phi, M, d);
  }
  if (can_vec_k) {
    phi_kernel_vec<<<CEIL(M, 32), 32>>>(K, K_phi, M, d);
  } else {
    phi_kernel_common<<<CEIL(M, 32), 32>>>(K, K_phi, M, d);
  }
  CUDA_CHECK(cudaGetLastError());

  dim3 kv_block(MATMUL_TILE, MATMUL_TILE);
  dim3 kv_grid(CEIL(d, MATMUL_TILE), CEIL(d, MATMUL_TILE));
  kv_kernel<<<kv_grid, kv_block>>>(K_phi, V, KV, M, d);
  CUDA_CHECK(cudaGetLastError());

  dim3 block(MATMUL_TILE, MATMUL_TILE);
  dim3 grid_num(CEIL(d, MATMUL_TILE), CEIL(M, MATMUL_TILE));
  matmul_nn_kernel<MATMUL_TILE><<<grid_num, block>>>(Q_phi, KV, numerator, M, d, d, 1.0f);
  CUDA_CHECK(cudaGetLastError());

  constexpr int REDUCE_THREADS = 128;
  ksum_kernel<<<CEIL(d, REDUCE_THREADS), REDUCE_THREADS>>>(K_phi, K_sum, M, d);
  CUDA_CHECK(cudaGetLastError());

  // denominator is [M, 1] = [M, d] @ [d, 1]
  dim3 grid_den(CEIL(1, MATMUL_TILE), CEIL(M, MATMUL_TILE));
  matmul_nn_kernel<MATMUL_TILE><<<grid_den, block>>>(Q_phi, K_sum, denominator, M, d, 1, 1.0f);
  CUDA_CHECK(cudaGetLastError());

  constexpr float kEps = 1e-6f;
  constexpr int NORM_THREADS = 256;
  normalize_kernel<<<CEIL(M * d, NORM_THREADS), NORM_THREADS>>>(numerator, denominator, output, M, d,
                                                                 kEps);
  CUDA_CHECK(cudaGetLastError());

  CUDA_CHECK(cudaFree(Q_phi));
  CUDA_CHECK(cudaFree(K_phi));
  CUDA_CHECK(cudaFree(KV));
  CUDA_CHECK(cudaFree(numerator));
  CUDA_CHECK(cudaFree(denominator));
  CUDA_CHECK(cudaFree(K_sum));
}