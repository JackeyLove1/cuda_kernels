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
  const auto tid = threadIdx.z * blockDim.y * blockDim.x + threadIdx.y * blockDim.x + threadIdx.x;
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
    const int nums_warps = (blockDim.x * blockDim.y * blockDim.z + 31) >> 5;
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

static constexpr int WARP_SIZE = 32;
static constexpr int MAX_BLOCKS = 4096;
static constexpr int THREADS_PER_BLOCK = 256;
static constexpr int TILE_X = 16;
static constexpr int TILE_Y = 16;
static constexpr int TILE_SIZE = 8;
static constexpr int CFACTOR = 8;

#define CHECK_CUDA(call)                                                                           \
  do {                                                                                             \
    cudaError_t err__ = (call);                                                                    \
    if (err__ != cudaSuccess) {                                                                    \
      fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err__)); \
      exit(1);                                                                                     \
    }                                                                                              \
  } while (0)

/**
Implement a program that computes the sum of a 3D subarray of 32-bit integers. You are given an
input 3D array input of length N×M×K, and two depth indices S_DEP and E_DEP, and two row indices
S_ROW and E_ROW and two column indices S_COL and E_COL. S_DEP, E_DEP, S_ROW, E_ROW, S_COL and E_COL
are inclusive, 0-based start and end indices—compute the sum of
input[S_DEP..E_DEP][S_ROW..E_ROW][S_COL..E_COL].
 */

void __global__ subarray_sum_kernel(const int* __restrict__ input, int* __restrict__ output,
                                    const int N, const int M, const int K, const int S_DEP,
                                    const int E_DEP, const int S_ROW, const int E_ROW,
                                    const int S_COL, const int E_COL) {
  const int tx = threadIdx.x, ty = threadIdx.y, tz = threadIdx.z;
  const int dep_id = blockIdx.z * blockDim.z + tz;
  const int row_id = blockIdx.y * blockDim.y + ty;
  const int col_id = blockIdx.x * blockDim.x + tx;
  const int row_stride = gridDim.y * blockDim.y;
  const int col_stride = gridDim.x * blockDim.x;
  const int dep_stride = gridDim.z * blockDim.z;
  int thread_sum = 0;
  for (int dep = S_DEP + dep_id; dep <= E_DEP; dep += dep_stride) {
    for (int row = S_ROW + row_id; row <= E_ROW; row += row_stride) {
      for (int col = S_COL + col_id; col <= E_COL; col += col_stride) {
        thread_sum += input[dep * M * K + row * K + col];
      }
    }
  }
  thread_sum = BlockReduceDynamic<int>(thread_sum, ReduceOpType::SUM);
  if (tx == 0 && ty == 0 && tz == 0) {
    atomicAdd(output, thread_sum);
  }
}

// A, B, and C are device pointers
extern "C" void solve(const int* input, int* output, int N, int M, int K, int S_DEP, int E_DEP,
                      int S_ROW, int E_ROW, int S_COL, int E_COL) {
  CHECK_CUDA(cudaMemset(output, 0, sizeof(int)));
  dim3 threadsPerBlock(TILE_SIZE, TILE_SIZE, TILE_SIZE);
  dim3 blocksPerGrid(16, 16, 16);

  subarray_sum_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, output, N, M, K, S_DEP, E_DEP,
                                                          S_ROW, E_ROW, S_COL, E_COL);

  CHECK_CUDA(cudaDeviceSynchronize());
}
