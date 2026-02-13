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

[[maybe_unused]] static constexpr int CFACTOR = 1;
[[maybe_unused]] static constexpr int WARP_SIZE = 32;
[[maybe_unused]] static constexpr int MAX_BLOCKS = 4096;
[[maybe_unused]] static constexpr int THREADS_PER_BLOCK = 512;

/**
Write a program that performs a 2D convolution operation on the GPU. Given an input matrix and a
kernel (filter), compute the convolved output. The convolution should be performed with
a"valid"boundary condition, meaning the kernel is only applied where it fully overlaps with the
input. The input consists of: input : A 2D matrix of 32-bit floating-point numbers, represented as a
1D array in row-major order. kernel : A 2D kernel (filter) of 32-bit floating-point numbers, also
represented as a 1D array in row-major order.

Constraints
1 ≤ input_rows, input_cols ≤ 3072
1 ≤ kernel_rows, kernel_cols ≤ 31
kernel_rows ≤ input_rows
kernel_cols ≤ input_cols
Performance is measured with input_cols = 3,072, input_rows = 3,072, kernel_cols = 15, kernel_rows =
15
 */

__constant__ float kernel_constant[32 * 32];

template <int TILE_W, int TILE_H>
__global__ void conv2d_kernel(const float* __restrict__ input, float* __restrict__ output,
                              const int inH, const int inW, const int kH, const int kW,
                              const int outH, const int outW) {
  const int out_x0 = blockIdx.x * TILE_W;
  const int out_y0 = blockIdx.y * TILE_H;
  const int tx = threadIdx.x;
  const int ty = threadIdx.y;
  const int out_x = out_x0 + tx;
  const int out_y = out_y0 + ty;

  extern __shared__ float sh_in[];
  const int shW = TILE_W + kW - 1;
  const int shH = TILE_H + kH - 1;

  const auto threads = blockDim.x * blockDim.y;
  const auto tid = blockDim.x * ty + tx;
  const auto shared_mem_size = shW * shH;
  for (int i = tid; i < shared_mem_size; i += threads) {
    float value{0.f};
    auto sy = i / shW;
    auto sx = i % shW;
    auto in_x = out_x0 + sx;
    auto in_y = out_y0 + sy;
    if (in_x >= 0 && in_x < inW && in_y >= 0 && in_y < inH) {
      value = input[in_y * inW + in_x];
    }
    sh_in[i] = value;
  }

  __syncthreads();

  if (out_x < outW && out_y < outH) {
    float sum{0.f};
    auto base = shW * ty + tx;
    for (int ky = 0; ky < kH; ++ky) {
      auto sh_row = base + ky * shW;
      auto k_row = ky * kW;
#pragma unroll 1
      for (int kx = 0; kx < kW; ++kx) {
        auto k_idx = k_row + kx;
        auto sh_idx = sh_row + kx;
        sum = fmaf(sh_in[sh_idx], kernel_constant[k_idx], sum);
      }
    }
    output[out_y * outW + out_x] = sum;
  }
}

// input, kernel, output are device pointers
extern "C" void solve(const float* input, const float* kernel, float* output, int input_rows,
                      int input_cols, int kernel_rows, int kernel_cols) {
  cudaMemcpyToSymbol(kernel_constant, kernel, kernel_rows * kernel_cols * sizeof(float));
  const int inH = input_rows, inW = input_cols;
  const int kH = kernel_rows, kW = kernel_cols;
  // valid conv mode
  const int outH = inH - kH + 1, outW = inW - kW + 1;
  constexpr int TILE_H = 16;
  constexpr int TILE_W = 16;
  dim3 threadsPerBlock(TILE_W, TILE_H);
  dim3 blocksPerGrid(CEIL(outW, TILE_W), CEIL(outH, TILE_H));
  size_t shared_mem_size = (TILE_W + kW - 1) * (TILE_H + kH - 1) * sizeof(float);
  conv2d_kernel<TILE_W, TILE_H><<<blocksPerGrid, threadsPerBlock, shared_mem_size>>>(
      input, output, inH, inW, kH, kW, outH, outW);
}