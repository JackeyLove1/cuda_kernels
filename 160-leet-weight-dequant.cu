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
Implement a GPU program that"dequantizes"a weight matrix on the GPU. You are given an input matrix x
of shape [M, N] containing quantized values and a scale matrix s of shape [ceil(M/T), ceil(N/T)],
where T is the tile size. For each element X_{i,j}, the corresponding scale factor is S_{row,col}
where row=\lfloor i/T\rfloor and col=\lfloor j/T\rfloor. The output Y_{i,j} should be computed as:
Y_{i,j}=X_{i,j}×S_{row,col}

constrains:
1 ≤ M, N ≤ 8192
TILE_SIZE ∈ {16, 32, 64, 128}
Performance is measured with M = 8,192, N = 8,192
 */

template <int TILE_SIZE>
__global__ void weight_dequant_kernel(const float* __restrict__ X, const float* __restrict__ S,
                                      float* __restrict__ Y, const int M, const int N) {
  static_assert(TILE_SIZE == 16 || TILE_SIZE == 32 || TILE_SIZE == 64 || TILE_SIZE == 128,
                "TILE_SIZE must be 16, 32, 64, or 128");

  const int tx = threadIdx.x, ty = threadIdx.y;
  const int bx = blockIdx.x, by = blockIdx.y;

  extern __shared__ float scales[];
  const int idx = by * gridDim.x + bx;
  if (tx == 0 && ty == 0) {
    scales[idx] = S[idx];
  }
  __syncthreads();

  const int row_base = by * TILE_SIZE, row = row_base + ty;
  const auto num_elements = TILE_SIZE * TILE_SIZE / 256;
  const int col_base = bx * TILE_SIZE, col = col_base + tx * num_elements;

  const float scale = scales[idx];
  //   const auto vec_size = num_elements >> 2;
  //   const auto vec_tail = num_elements & 3;

  // #pragma unroll
  //   for (int i = 0; i < vec_size; ++i) {
  //     const int idx = row * N + col + i * 4;
  //     const float4* x4 = LOAD4<float>(X + idx);
  //     float4* y4 = STORE4<float>(Y + idx);
  //     y4->x = x4->x * scale;
  //     y4->y = x4->y * scale;
  //     y4->z = x4->z * scale;
  //     y4->w = x4->w * scale;
  //   }

  // #pragma unroll
  //   for (int i = 0; i < vec_tail; ++i) {
  //     const int idx = row * N + col + (vec_size << 2) + i;
  //     Y[idx] = X[idx] * scale;
  //   }
  for (int i = 0; i < num_elements; ++i) {
    if (row < M && (col + i) < N) {
      const int base_idx = row * N + col + i;
      Y[base_idx] = X[base_idx] * scale;
    }
  }
}

extern "C" void solve(const float* X, const float* S, float* Y, int M, int N, int TILE_SIZE) {
  dim3 nthrds((256 / TILE_SIZE), TILE_SIZE);  // one thread for TILE_SIZE * TILE_SIZE / 256 elements
  dim3 nblks(CEIL(N, TILE_SIZE), CEIL(M, TILE_SIZE));
  const int shared_size = CEIL(N, TILE_SIZE) * CEIL(M, TILE_SIZE) * sizeof(float);
  assert(TILE_SIZE == 16 || TILE_SIZE == 32 || TILE_SIZE == 64 || TILE_SIZE == 128);
  if (TILE_SIZE == 16) {
    weight_dequant_kernel<16><<<nblks, nthrds, shared_size>>>(X, S, Y, M, N);
  } else if (TILE_SIZE == 32) {
    weight_dequant_kernel<32><<<nblks, nthrds, shared_size>>>(X, S, Y, M, N);
  } else if (TILE_SIZE == 64) {
    weight_dequant_kernel<64><<<nblks, nthrds, shared_size>>>(X, S, Y, M, N);
  } else if (TILE_SIZE == 128) {
    weight_dequant_kernel<128><<<nblks, nthrds, shared_size>>>(X, S, Y, M, N);
  } else {
    throw std::invalid_argument("TILE_SIZE must be 16, 32, 64, or 128");
  }
  cudaDeviceSynchronize();
}
