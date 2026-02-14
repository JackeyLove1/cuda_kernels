#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/sort.h>
#include <thrust/tabulate.h>

#include <algorithm>
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
[[maybe_unused]] static constexpr int THREADS_PER_BLOCK = 512;

static constexpr int TILE = 32;

/**
Write a program that multiplies two matrices of 32-bit floating point numbers on a GPU. Given matrix
A of dimensions M×N and matrix B of dimensions N×K, compute the product matrix C=A×B, which will
have dimensions M×K. All matrices are stored in row-major format.
 */
__global__ void matrix_multiplication_kernel(const float* __restrict__ A,
                                             const float* __restrict__ B, float* __restrict__ C,
                                             const int M, const int N, const int K) {
  static constexpr int kStage = 2;
  __shared__ __align__(16) float sh_A[kStage][TILE][TILE + 1];
  __shared__ __align__(16) float sh_B[kStage][TILE][TILE + 1];
  const int tx = threadIdx.x;
  const int ty = threadIdx.y;
  const int bx = blockIdx.x;
  const int by = blockIdx.y;
  const int col = bx * TILE + tx;
  const int row = by * TILE + ty;
  float sum{0.f};
  const auto num_tils = CEIL(N, TILE);

  cuda::pipeline<cuda::thread_scope_thread> pipe = cuda::make_pipeline();

  pipe.producer_acquire();
  if (row < M && tx < N) {
    cuda::memcpy_async(&sh_A[0][ty][tx], &A[row * N + tx], sizeof(float), pipe);
  } else {
    sh_A[0][ty][tx] = 0.f;
  }
  if (col < K && ty < N) {
    cuda::memcpy_async(&sh_B[0][ty][tx], &B[ty * K + col], sizeof(float), pipe);
  } else {
    sh_B[0][ty][tx] = 0.f;
  }
  pipe.producer_commit();

  for (int k_tile = 0; k_tile < num_tils; ++k_tile) {
    if (k_tile + 1 < num_tils) {
      pipe.producer_acquire();
      auto next_k = (k_tile + 1);
      auto write_stage = next_k % kStage;

      if (row < M && (next_k * TILE + tx) < N) {
        cuda::memcpy_async(&sh_A[write_stage][ty][tx], &A[row * N + (next_k * TILE + tx)],
                           sizeof(float), pipe);
      } else {
        sh_A[write_stage][ty][tx] = 0.f;
      }

      if (col < K && (next_k * TILE + ty) < N) {
        cuda::memcpy_async(&sh_B[write_stage][ty][tx], &B[(next_k * TILE + ty) * K + col],
                           sizeof(float), pipe);
      } else {
        sh_B[write_stage][ty][tx] = 0.f;
      }
      pipe.producer_commit();
    }

    pipe.consumer_wait();
    __syncthreads();

    auto read_stage = k_tile % kStage;

#pragma unroll 4
    for (int k = 0; k < TILE; ++k) {
      sum = fmaf(sh_A[read_stage][ty][k], sh_B[read_stage][k][tx], sum);
    }

    pipe.consumer_release();
    __syncthreads();
  }
  if (col < K && row < M) {
    C[row * K + col] = sum;
  }
}

/**
Implement a GPU program that raises a square matrix A of size N×N to an integer power P.
The solve function receives a扁tened input matrix input (row-major order), an empty output matrix
output of the same size, the dimension N, and the exponent P. You must compute output=A⁰ where
matrix multiplication is standard dense multiplication over 32-bit floating point numbers.
 */

// A, B, C are device pointers (i.e. pointers to memory on the GPU)

__global__ void diag_kernel(float* A, int n, int lda) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) A[i * lda + i] = 1.0f;
}

void cuda_eye_fast(float* dA, int n, int lda) {
  cudaMemset(dA, 0, sizeof(float) * lda * n);
  int threads = 256;
  int blocks = (n + threads - 1) / threads;
  diag_kernel<<<blocks, threads>>>(dA, n, lda);
}

extern "C" void solve(const float* d_matrix, float* d_result, int N, int P) {
  dim3 blockDim(TILE, TILE);
  dim3 gridDim(CEIL(N, TILE), CEIL(N, TILE));
  float *d_temp, *d_base;
  cudaMalloc(&d_temp, N * N * sizeof(float));
  cudaMalloc(&d_base, N * N * sizeof(float));
  cudaMemcpy(d_base, d_matrix, N * N * sizeof(float), cudaMemcpyDeviceToDevice);
  cuda_eye_fast(d_result, N, N);  // d_result = I (identity matrix)
  while (P > 0) {
    if (P & 1) {
      // result = result * base
      matrix_multiplication_kernel<<<gridDim, blockDim>>>(d_result, d_base, d_temp, N, N, N);
      cudaMemcpy(d_result, d_temp, N * N * sizeof(float), cudaMemcpyDeviceToDevice);
    }
    // base = base * base
    matrix_multiplication_kernel<<<gridDim, blockDim>>>(d_base, d_base, d_temp, N, N, N);
    cudaMemcpy(d_base, d_temp, N * N * sizeof(float), cudaMemcpyDeviceToDevice);
    P >>= 1;
  }
  cudaDeviceSynchronize();
}
