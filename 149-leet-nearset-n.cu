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
Implement a GPU program that, for N three-dimensional points stored on the device, fills indices[i]
with the index j of the point closest to points[i]. Comparing squared Euclidean distance is
sufficient—you do not need to compute square-roots. Implementation Requirements The solve function
signature must remain unchanged External libraries are not permitted The final result must be stored
in the indices array

Input: points = [(0,0,0),(1,0,0),(5,5,5)]
indices = [-1,-1,-1]
N = 3
Output: indices = [1,0,1] # 0个1 are nearest, 2 is closest to 1
 */
static constexpr int TILE = 256;

// TODO: use cuda pipeline and float4 load/store
__global__ void nearest_n_kernel(const float* __restrict__ points, int* __restrict__ indices,
                                 const int N) {
  const auto tid = threadIdx.x;
  const auto gid = blockDim.x * blockIdx.x + tid;
  const bool invalid = (gid < N);

  __shared__ __align__(16) float smem[TILE * 3];

  float x1 = 0.0f;
  float x2 = 0.0f;
  float x3 = 0.0f;
  float best_distance = FLT_MAX;
  int best_index = -1;

  if (invalid) {
    x1 = points[gid * 3 + 0];
    x2 = points[gid * 3 + 1];
    x3 = points[gid * 3 + 2];
  }

  for (int base = 0; base < N; base += TILE) {
    int idx = base + tid;
    if (idx < N) {
      smem[tid * 3 + 0] = points[idx * 3 + 0];
      smem[tid * 3 + 1] = points[idx * 3 + 1];
      smem[tid * 3 + 2] = points[idx * 3 + 2];
    } else {
      smem[tid * 3 + 0] = 0.0f;
      smem[tid * 3 + 1] = 0.0f;
      smem[tid * 3 + 2] = 0.0f;
    }
    __syncthreads();

    if (invalid) {
      int limit = min(TILE, N - base);
#pragma unroll 4
      for (int i = 0; i < limit; ++i) {
        int curr_idx = base + i;
        if (curr_idx == gid) continue;
        float diff1 = x1 - smem[i * 3 + 0];
        float diff2 = x2 - smem[i * 3 + 1];
        float diff3 = x3 - smem[i * 3 + 2];
        float distance = diff1 * diff1 + diff2 * diff2 + diff3 * diff3;
        if (distance < best_distance) {
          best_distance = distance;
          best_index = curr_idx;
        }
      }
    }
    __syncthreads();
  }

  if (invalid) {
    indices[gid] = best_index;
  }
}

extern "C" void solve(const float* points, int* indices, int N) {
  dim3 nthrs(TILE);
  dim3 nblks(CEIL(N, TILE));
  nearest_n_kernel<<<nblks, nthrs>>>(points, indices, N);
  cudaDeviceSynchronize();
}