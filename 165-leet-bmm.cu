#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/gather.h>
#include <thrust/host_vector.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>

#include <algorithm>
#include <cfloat>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cub/cub.cuh>
#include <cuda/pipeline>
#include <cuda/std/utility>
#include <limits>
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

template <typename T>
static inline T read_scalar(const T* ptr) {
  if (ptr == nullptr) return T{};

  cudaPointerAttributes attr{};
  const cudaError_t attr_status = cudaPointerGetAttributes(&attr, ptr);
  if (attr_status == cudaSuccess && attr.type == cudaMemoryTypeDevice) {
    T value{};
    cudaMemcpy(&value, ptr, sizeof(T), cudaMemcpyDeviceToHost);
    return value;
  }

  cudaGetLastError();
  return *ptr;
}

template <typename T>
static inline void write_scalar(T* ptr, T value) {
  if (ptr == nullptr) return;

  cudaPointerAttributes attr{};
  const cudaError_t attr_status = cudaPointerGetAttributes(&attr, ptr);
  if (attr_status == cudaSuccess && attr.type == cudaMemoryTypeDevice) {
    cudaMemcpy(ptr, &value, sizeof(T), cudaMemcpyHostToDevice);
    return;
  }

  cudaGetLastError();
  *ptr = value;
}

/**
Implement a batched matrix multiplication in FP32. Given a batch of matrices A of shape [B, M, K]
and a batch of matrices B of shape [B, K, N], compute the output batch C of shape [B, M, N] such
that for each batch index b: Cb=Ab×Bb All matrices are stored in row-major order and use 32-bit
floating point numbers (FP32).
 */

// A [M, K] B [K, N]
template <int TILE, int TM, int TN>
__global__ void kernel(const float* A, const float* B, float* C, int BATCH, const int M,
                       const int N, const int K) {
  const int tid = threadIdx.x;
  // Each thread computes a TM x TN output tile.
  // Threads are laid out as [TILE / TN] columns and [TILE / TM] rows.
  constexpr auto tilePerRow = TILE / TN;
  const auto tc = tid % tilePerRow;
  const auto tr = tid / tilePerRow;
  const auto col0 = blockIdx.x * TILE + tc * TN;
  const auto row0 = blockIdx.y * TILE + tr * TM;
  const auto batch = blockIdx.z;
  const auto idx = batch * M * N + row0 * N + col0;

  __shared__ float As[TILE][TILE + 1];
  __shared__ float Bs[TILE][TILE + 1];

  float acc[TM][TN];

  for (int i = 0; i < TM; ++i) {
#pragma unroll
    for (int j = 0; j < TN; ++j) {
      acc[i][j] = 0.0f;
    }
  }

  const int num_tiles = CEIL(K, TILE);
  for (int t = 0; t < num_tiles; ++t) {
    for (int r = 0; r < TM; ++r) {
      for (int c = 0; c < TN; ++c) {
        auto t_row = tr * TM + r;
        auto t_col = tc * TN + c;
        auto a_col = t * TILE + tc * TN;
        auto b_row = t * TILE + tr * TM;
        As[t_row][t_col] =
            (row0 + r < M && a_col + c < K) ? A[batch * M * K + (row0 + r) * K + a_col + c] : 0.0f;
        Bs[t_row][t_col] =
            (col0 + c < N && b_row + r < K) ? B[batch * K * N + (b_row + r) * N + col0 + c] : 0.0f;
      }
    }
    __syncthreads();

    for (int k = 0; k < TILE; ++k) {
      for (int r = 0; r < TM; ++r) {
        float a_val = As[tr * TM + r][k];
#pragma unroll
        for (int c = 0; c < TN; ++c) {
          acc[r][c] = fmaf(a_val, Bs[k][tc * TN + c], acc[r][c]);
        }
      }
    }
    __syncthreads();
  }

  // update output
  for (int r = 0; r < TM; r++) {
    for (int c = 0; c < TN; c++) {
      if (row0 + r < M && col0 + c < N) {
        C[(batch * M * N) + (row0 + r) * N + col0 + c] = acc[r][c];
      }
    }
  }
}

extern "C" void solve(const float* A, const float* B, float* C, int BATCH, int M, int N, int K) {
  constexpr int TILE = 32, TM = 8, TN = 4;
  const int nthrs = (TILE / TM) * (TILE / TN);
  dim3 nblks(CEIL(N, TILE), CEIL(M, TILE), BATCH);
  kernel<TILE, TM, TN><<<nblks, nthrs>>>(A, B, C, BATCH, M, N, K);
  CUDA_CHECK(cudaGetLastError());
}