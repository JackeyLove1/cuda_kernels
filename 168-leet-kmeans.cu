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
Implement the k-means clustering algorithm for 2D points. Given arrays of x and y coordinates for
data points, initial centroids, and other parameters, assign each point to the nearest centroid and
update the centroids iteratively. The final centroids and labels should be stored in the output
arrays. Implementation Requirements ·External libraries are not permitted ·The solve function
signature must remain unchanged ·The final result must be stored in labels, final _ centroid _x, and
final _ centroid _y
 */

// calculate the distance between each data point and each centroid

__forceinline__ __device__ float compute_distance(const float2 a, const float2 b) {
  const float diff1 = a.x - b.x;
  const float diff2 = a.y - b.y;
  return diff1 * diff1 + diff2 * diff2;
}

namespace cg = cooperative_groups;

// Fused K-means kernel: label assignment + accumulation + centroid update
// Uses grid.sync() (cooperative launch) to synchronize all blocks between phases,
// eliminating the need for separate kernel launches.
// Shared memory layout: float2[k] centroids | float[k] sum_x | float[k] sum_y | int[k] count
__global__ void kmeans_fused_kernel(const float* data_x, const float* data_y,
                                    const float* centroid_x, const float* centroid_y,
                                    float* new_centroid_x, float* new_centroid_y, int* nums,
                                    int* labels, const int sample_size, const int k) {
  cg::grid_group grid = cg::this_grid();

  extern __shared__ char smem_raw[];
  float2* s_centroid = reinterpret_cast<float2*>(smem_raw);
  float* s_sum_x = reinterpret_cast<float*>(smem_raw + sizeof(float2) * k);
  float* s_sum_y = s_sum_x + k;
  int* s_count = reinterpret_cast<int*>(s_sum_y + k);

  const int tid = threadIdx.x;
  const int gid = blockDim.x * blockIdx.x + tid;
  const int stride = gridDim.x * blockDim.x;

  // Load current centroids into shared memory and zero accumulators
  for (int i = tid; i < k; i += blockDim.x) {
    s_centroid[i] = make_float2(centroid_x[i], centroid_y[i]);
    s_sum_x[i] = 0.0f;
    s_sum_y[i] = 0.0f;
    s_count[i] = 0;
  }
  __syncthreads();

  // Phase 1: Assign each point to nearest centroid + accumulate partial sums in shared memory
  for (int i = gid; i < sample_size; i += stride) {
    const float2 pt = make_float2(data_x[i], data_y[i]);
    float best_dist = INFINITY;
    int best_k = 0;
#pragma unroll
    for (int j = 0; j < k; j++) {
      const float d = compute_distance(pt, s_centroid[j]);
      if (d < best_dist) {
        best_dist = d;
        best_k = j;
      }
    }
    labels[i] = best_k;
    atomicAdd(s_sum_x + best_k, pt.x);
    atomicAdd(s_sum_y + best_k, pt.y);
    atomicAdd(s_count + best_k, 1);
  }
  __syncthreads();

  // Flush block-local partial sums to global accumulators
  for (int i = tid; i < k; i += blockDim.x) {
    if (s_count[i] > 0) {
      atomicAdd(new_centroid_x + i, s_sum_x[i]);
      atomicAdd(new_centroid_y + i, s_sum_y[i]);
      atomicAdd(nums + i, s_count[i]);
    }
  }

  // Grid-wide barrier: wait for ALL blocks to finish accumulation
  grid.sync();

  // Phase 2: Compute new centroids (divide accumulated sums by counts)
  for (int i = gid; i < k; i += stride) {
    if (nums[i] > 0) {
      new_centroid_x[i] /= nums[i];
      new_centroid_y[i] /= nums[i];
    } else {
      new_centroid_x[i] = centroid_x[i];
      new_centroid_y[i] = centroid_y[i];
    }
  }
}

// data_x, data_y, labels, initial_centroid_x, initial_centroid_y,
// final_centroid_x, final_centroid_y are device pointers
extern "C" void solve(const float* data_x, const float* data_y, int* labels,
                      float* initial_centroid_x, float* initial_centroid_y, float* final_centroid_x,
                      float* final_centroid_y, int sample_size, int k, int max_iterations) {
  static constexpr int nthrs = 256;
  float* final_centroid_x_out = final_centroid_x;
  float* final_centroid_y_out = final_centroid_y;
  int* nums;
  CUDA_CHECK(cudaMalloc(&nums, sizeof(int) * k));
  float* res_centroid_x;
  CUDA_CHECK(cudaMalloc(&res_centroid_x, sizeof(float) * k));
  float* res_centroid_y;
  CUDA_CHECK(cudaMalloc(&res_centroid_y, sizeof(float) * k));

  cudaMemcpy(final_centroid_x, initial_centroid_x, sizeof(float) * k, cudaMemcpyDeviceToDevice);
  cudaMemcpy(final_centroid_y, initial_centroid_y, sizeof(float) * k, cudaMemcpyDeviceToDevice);

  // Shared memory: float2[k] centroids + float[k] sum_x + float[k] sum_y + int[k] count
  const size_t smem_size = sizeof(float2) * k + sizeof(float) * k * 2 + sizeof(int) * k;

  // Query max concurrent blocks for cooperative launch (grid.sync requires all blocks co-resident)
  int max_blocks_per_sm = 0;
  CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&max_blocks_per_sm, kmeans_fused_kernel,
                                                           nthrs, smem_size));
  int device = 0;
  CUDA_CHECK(cudaGetDevice(&device));
  cudaDeviceProp prop;
  CUDA_CHECK(cudaGetDeviceProperties(&prop, device));

  const int max_blocks = max_blocks_per_sm * prop.multiProcessorCount;
  const int nblks = std::min(CEIL(sample_size, nthrs), max_blocks);

  for (int iter = 0; iter < max_iterations; ++iter) {
    CUDA_CHECK(cudaMemset(nums, 0, sizeof(int) * k));
    CUDA_CHECK(cudaMemset(res_centroid_x, 0, sizeof(float) * k));
    CUDA_CHECK(cudaMemset(res_centroid_y, 0, sizeof(float) * k));

    void* args[] = {(void*)&data_x,
                    (void*)&data_y,
                    (void*)&final_centroid_x,
                    (void*)&final_centroid_y,
                    (void*)&res_centroid_x,
                    (void*)&res_centroid_y,
                    (void*)&nums,
                    (void*)&labels,
                    (void*)&sample_size,
                    (void*)&k};

    CUDA_CHECK(cudaLaunchCooperativeKernel((void*)kmeans_fused_kernel, dim3(nblks), dim3(nthrs),
                                           args, smem_size));

    std::swap(final_centroid_x, res_centroid_x);
    std::swap(final_centroid_y, res_centroid_y);
  }

  if (final_centroid_x != final_centroid_x_out) {
    CUDA_CHECK(cudaMemcpy(final_centroid_x_out, final_centroid_x, sizeof(float) * k,
                          cudaMemcpyDeviceToDevice));
  }
  if (final_centroid_y != final_centroid_y_out) {
    CUDA_CHECK(cudaMemcpy(final_centroid_y_out, final_centroid_y, sizeof(float) * k,
                          cudaMemcpyDeviceToDevice));
  }

  cudaDeviceSynchronize();
  CUDA_CHECK(cudaFree(nums));
  CUDA_CHECK(cudaFree(res_centroid_x));
  CUDA_CHECK(cudaFree(res_centroid_y));
}