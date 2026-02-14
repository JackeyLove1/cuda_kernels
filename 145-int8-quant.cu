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

/**
Implement a quantized matrix multiplication program for 8-bit signed integer matrices. Given two
input matrices A of dimensions M×K and B of dimensions K×N, quantization scales scale_A, scale_B,
output scale scale_C, zero-points zero_point_A, zero_point_B, zero_point_C, compute:
C_{\text{quant}}(i,j)=\text{clamp}\left(\text{round}\left(\frac{\sum_{k=0}^{K-1}(A_{ik}-z_{A})(B_{kj}-z_{B})\cdot
s_{A}s_{B}}{s_{C}}\right)+z_{C},-128,127\right) where s_A = scale_A, z_A = zero_point_A, etc.
 */

static constexpr int TILE = 16;

__global__ void int8_quant_kernel(const int8_t* __restrict__ A, const int8_t* __restrict__ B,
                                  int8_t* __restrict__ C, const int M, const int N, const int K,
                                  const float scale_A, const float scale_B, const float scale_C,
                                  const int zero_point_A, const int zero_point_B,
                                  const int zero_point_C) {
  const int tx = threadIdx.x;
  const int ty = threadIdx.y;
  const int col = blockIdx.x * TILE + tx;
  const int row = blockIdx.y * TILE + ty;

  __shared__ int8_t smem_A[TILE][TILE + 1];
  __shared__ int8_t smem_B[TILE][TILE + 1];

  int acc = 0;
  const int num_tiles = CEIL(K, TILE);

  for (int tile = 0; tile < num_tiles; ++tile) {
    const int a_col = tile * TILE + tx;
    const int b_row = tile * TILE + ty;

    smem_A[ty][tx] = (row < M && a_col < K) ? A[row * K + a_col] : 0;
    smem_B[ty][tx] = (b_row < K && col < N) ? B[b_row * N + col] : 0;
    __syncthreads();

#pragma unroll
    for (int kk = 0; kk < TILE; ++kk) {
      const int a = static_cast<int>(smem_A[ty][kk]) - zero_point_A;
      const int b = static_cast<int>(smem_B[kk][tx]) - zero_point_B;
      acc += a * b;
    }
    __syncthreads();
  }

  if (row < M && col < N) {
    const float scaled = roundf(static_cast<float>(acc) * (scale_A * scale_B / scale_C)) +
                         static_cast<float>(zero_point_C);
    const float clipped = fmaxf(fminf(scaled, 127.0f), -128.0f);
    C[row * N + col] = static_cast<int8_t>(clipped);
  }
}

struct __align__(16) Int8x16 {
  int8_t v[16];
};

__global__ void int8_quant_kernel_128(const int8_t* __restrict__ A, const int8_t* __restrict__ B,
                                      int8_t* __restrict__ C, const int M, const int N, const int K,
                                      const float scale_A, const float scale_B, const float scale_C,
                                      const int zero_point_A, const int zero_point_B,
                                      const int zero_point_C) {}

extern "C" void solve(const int8_t* A, const int8_t* B, int8_t* C, int M, int N, int K,
                      float scale_A, float scale_B, float scale_C, int zero_point_A,
                      int zero_point_B, int zero_point_C) {
  dim3 nthrs(TILE, TILE);
  dim3 nblks(CEIL(N, TILE), CEIL(M, TILE));
  int8_quant_kernel<<<nblks, nthrs>>>(A, B, C, M, N, K, scale_A, scale_B, scale_C, zero_point_A,
                                      zero_point_B, zero_point_C);
  cudaDeviceSynchronize();
}