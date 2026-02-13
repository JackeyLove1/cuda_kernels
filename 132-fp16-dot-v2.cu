#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

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
static constexpr int THREADS_PER_BLOCK = 256;

/**
Implement a GPU program that compute s the dot product of two vectors containing 16-bit floating
point numbers (FP16/ half). The dot product is the sum of the products of the corresponding elements
of two vectors. Mathematically, the dot product of two vectors A and B of length n is defined as:
A·B=∑n-1i=0Ai·Bi=A₀·B₀+A₁·B₁+...+An-1·Bn-
All inputs are stored as 16-bit floating point numbers (FP16/ half). For best precision,
accumulation during multiplication should use FP32 before converting the final result to FP16.
 */
__device__ __forceinline__ float dot(const float4& vec_a, const float4& vec_b) {
  const half2* a_h2 = reinterpret_cast<const half2*>(&vec_a);
  const half2* b_h2 = reinterpret_cast<const half2*>(&vec_b);

  float sum = 0.0f;

#pragma unroll
  for (int i = 0; i < 4; ++i) {
    float2 fa = __half22float2(a_h2[i]);
    float2 fb = __half22float2(b_h2[i]);

    sum = __fmaf_rn(fa.x, fb.x, sum);
    sum = __fmaf_rn(fa.y, fb.y, sum);
  }
  return sum;
}

__global__ void half_dot_kernel(const half* __restrict__ A, const half* __restrict__ B,
                                float* __restrict__ result, const int N) {
  const auto tid = threadIdx.x;
  const auto gid = blockIdx.x * blockDim.x + tid;
  float sum = 0.0f;
  const auto stride = blockDim.x * gridDim.x;
  const float4* a4 = reinterpret_cast<const float4*>(A);
  const float4* b4 = reinterpret_cast<const float4*>(B);
  const auto vec_size = N >> 3;
  const auto vec_tail = N & 7;

  for (int i = gid; i < vec_size; i += stride) {
    const auto a = a4[i];
    const auto b = b4[i];
    sum += dot(a, b);
  }
  if (vec_tail && gid < vec_tail) {
    const auto idx = (vec_size << 3) + gid;
    sum += __half2float(A[idx]) * __half2float(B[idx]);
  }

  sum = BlockReduceDynamic<float>(sum, ReduceOpType::SUM);
  if (tid == 0) {
    atomicAdd(result, sum);
  }
}

// A, B, result are device pointers
extern "C" void solve(const half* A, const half* B, half* result, int N) {
  const int threads_per_block = THREADS_PER_BLOCK;
  const int blocks_per_grid = std::min(CEIL(N, threads_per_block * 8), MAX_BLOCKS);
  float* tmp_result = nullptr;
  cudaMalloc(&tmp_result, sizeof(float));
  cudaMemset(tmp_result, 0, sizeof(float));
  half_dot_kernel<<<blocks_per_grid, threads_per_block>>>(A, B, tmp_result, N);
  cudaDeviceSynchronize();
  float host_result = 0.0f;
  cudaMemcpy(&host_result, tmp_result, sizeof(float), cudaMemcpyDeviceToHost);
  const half host_half_result = __float2half(host_result);
  cudaMemcpy(result, &host_half_result, sizeof(half), cudaMemcpyHostToDevice);
  cudaFree(tmp_result);
}