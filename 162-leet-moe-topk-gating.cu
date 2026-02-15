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
Implement a GPU program that performs Top-K Gating for Mixture of Experts (MoE) models. Given a
logit matrix of shape [M, E] where M is the number of tokens and E is the number of experts,
identify the k largest values in each row, extract their indices, and apply softmax to get mixing
weights. For each row i, the operation compute: indices; = argsort(logits;)[-k:] vals; = logits;
[indices; ] weights; = Softmax(vals; ) Implementation Requirements ·External libraries are not
permitted ·The solve function signature must remain unchanged ·The final result must be stored in
the topk _ weights and topk _ indices arrays

1 ≤ M ≤ 10,000 (number of tokens)
1 ≤ E ≤ 256 (number of experts)
1 ≤ k ≤ E (top-k selection, typically k=2)
All tensors are stored on GPU
Logits are 32-bit floats
Indices are 32-bit integers
Performance is measured with M = 1,024, k = 2
 */

__global__ void kernel_v1(const float* __restrict__ logits, float* __restrict__ topk_weights,
                          int* __restrict__ topk_indices, const int M, const int E, const int k) {
  const int row = blockIdx.x;
  const int tid = threadIdx.x;
  extern __shared__ float smem[];

  float* row_logits = smem;
  int* row_topk_indices = reinterpret_cast<int*>(smem + E);
  float* row_topk_weights = smem + E + k;

  // load shared memory for each row
  for (int i = 0; i < E; ++i) {
    row_logits[i] = logits[row * E + i];
  }

  // find top-k values and indices
  for (int i = 0; i < k; ++i) {  // top-i max value
    float k_max_value = -INFINITY;
    int k_max_index = -1;
    for (int j = 0; j < E; ++j) {  // each value in the row
      bool already_exist = false;
      if (row_logits[j] > k_max_value) {
        for (int l = 0; l < i; ++l) {
          if (row_topk_indices[l] == j) {
            already_exist = true;
            break;
          }
        }
        if (!already_exist) {
          k_max_value = row_logits[j];
          k_max_index = j;
        }
      }
    }
    row_topk_indices[i] = k_max_index;
    row_topk_weights[i] = k_max_value;
  }

  // softmax
  float global_max = row_topk_weights[0];
  float global_sum = 0.0f;
  for (int i = 0; i < k; ++i) {
    float value = __expf(row_topk_weights[i] - global_max);
    global_sum += value;
    row_topk_weights[i] = value;
  }
  for (int i = 0; i < k; ++i) {
    row_topk_weights[i] /= global_sum;
  }
  for (int i = 0; i < k; ++i) {
    topk_weights[row * k + i] = row_topk_weights[i];
    topk_indices[row * k + i] = row_topk_indices[i];
  }
}

// v2 version for specific k = 2 value optimize

__global__ void kernel_v2(const float* __restrict__ logits, float* __restrict__ topk_weights,
                          int* __restrict__ topk_indices, const int M, const int E, const int k) {
  const int row = blockIdx.x;
  const int tid = threadIdx.x;
  // extern __shared__ float row_logits[];
  float row_top1_weight = -INFINITY;
  int row_top1_index = -1;
  float row_top2_weight = -INFINITY;
  int row_top2_index = -1;

  // load shared memory for each row
  //   for (int i = 0; i < E; ++i) {
  //     row_logits[i] = logits[row * E + i];
  //   }

  // find top-2 values and indices
  for (int i = 0; i < E; ++i) {
    float value = logits[row * E + i];
    if (value > row_top1_weight) {
      row_top2_weight = row_top1_weight;
      row_top2_index = row_top1_index;
      row_top1_index = i;
      row_top1_weight = value;
    } else if (value > row_top2_weight) {
      row_top2_weight = value;
      row_top2_index = i;
    }
  }

  // softmax
  float global_max = row_top1_weight;
  row_top1_weight = __expf(row_top1_weight - global_max);
  row_top2_weight = __expf(row_top2_weight - global_max);
  float global_sum = row_top1_weight + row_top2_weight;
  topk_weights[row * k + 0] = row_top1_weight / global_sum;
  topk_weights[row * k + 1] = row_top2_weight / global_sum;
  topk_indices[row * k + 0] = row_top1_index;
  topk_indices[row * k + 1] = row_top2_index;
}

// v2 version for specific k = 2 value optimize + float4 load

__global__ void kernel_v3(const float* __restrict__ logits, float* __restrict__ topk_weights,
                          int* __restrict__ topk_indices, const int M, const int E, const int k) {
  const int row = blockIdx.x;
  const int tid = threadIdx.x;
  // extern __shared__ float row_logits[];
  float row_top1_weight = -INFINITY;
  int row_top1_index = -1;
  float row_top2_weight = -INFINITY;
  int row_top2_index = -1;

  // load shared memory for each row
  //   for (int i = 0; i < E; ++i) {
  //     row_logits[i] = logits[row * E + i];
  //   }

  const float* row_logits = logits + row * E;
  const auto combine_top2 = [&](float value, int index) {
    if (value > row_top1_weight) {
      row_top2_weight = row_top1_weight;
      row_top2_index = row_top1_index;
      row_top1_index = index;
      row_top1_weight = value;
    } else if (value > row_top2_weight) {
      row_top2_weight = value;
      row_top2_index = index;
    }
  };

  const int vec_size = E >> 2;
  const int vec_tail = E & 3;
  const uintptr_t row_addr = reinterpret_cast<uintptr_t>(row_logits);
  const bool aligned16 = ((row_addr & 0xF) == 0);

  // find top-2 values and indices
  if (aligned16) {
    const float4* row_logits4 = reinterpret_cast<const float4*>(row_logits);
#pragma unroll 2
    for (int i = 0; i < vec_size; ++i) {
      const float4 value = row_logits4[i];
      combine_top2(value.x, i * 4 + 0);
      combine_top2(value.y, i * 4 + 1);
      combine_top2(value.z, i * 4 + 2);
      combine_top2(value.w, i * 4 + 3);
    }
  } else {
#pragma unroll 2
    for (int i = 0; i < (vec_size << 2); ++i) {
      combine_top2(row_logits[i], i);
    }
  }

#pragma unroll
  for (int i = 0; i < vec_tail; ++i) {
    const int base_idx = (vec_size << 2) + i;
    combine_top2(row_logits[base_idx], base_idx);
  }

  // softmax
  float global_max = row_top1_weight;
  row_top1_weight = __expf(row_top1_weight - global_max);
  row_top2_weight = __expf(row_top2_weight - global_max);
  float global_sum = row_top1_weight + row_top2_weight;
  topk_weights[row * k + 0] = row_top1_weight / global_sum;
  topk_weights[row * k + 1] = row_top2_weight / global_sum;
  topk_indices[row * k + 0] = row_top1_index;
  topk_indices[row * k + 1] = row_top2_index;
}

extern "C" void solve(const float* logits, float* topk_weights, int* topk_indices, int M, int E,
                      int k) {
  const int nthrs = 1;
  const int nblks = M;

  if (k == 2) {
    // const int share_size = sizeof(float) * E;  // row-logits
    kernel_v3<<<nblks, nthrs>>>(logits, topk_weights, topk_indices, M, E, k);
  } else {
    const int share_size = sizeof(float) * E + sizeof(int) * k +
                           sizeof(float) * k;  // row-logits, row-topk_weights, row-topk_indices
    kernel_v1<<<nblks, nthrs, share_size>>>(logits, topk_weights, topk_indices, M, E, k);
  }
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());
}