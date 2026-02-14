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
__forceinline__ __device__ T WarpReduceOp(T value, unsigned int mask) {
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    value = Op::apply(value, __shfl_down_sync(mask, value, offset));
  }
  return value;
}

template <typename T, typename Op>
inline __device__ T BlockReduceOp(T value) {
  const auto tid = threadIdx.x;
  const auto lane_id = tid & 31;
  const auto warp_id = tid >> 5;
  const int num_warps = (blockDim.x + 31) >> 5;

  // warp reduce
  const unsigned int warp_mask = __activemask();
  value = WarpReduceOp<T, Op>(value, warp_mask);

  __shared__ T shared_data[32];
  if (lane_id == 0) {
    shared_data[warp_id] = value;
  }
  __syncthreads();

  // block reduce
  if (warp_id == 0) {
    const T lane_value = (lane_id < num_warps) ? shared_data[lane_id] : Op::identity();
    const unsigned int block_mask = __ballot_sync(FULL_MASK, lane_id < num_warps);
    value = WarpReduceOp<T, Op>(lane_value, block_mask);
  } else {
    value = Op::identity();
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
Implement a GPU program to calculate the categorical cross-entropy loss for a batch of predictions.
Given a matrix of predicted logots Z of size N×C and a vector of true class labels true _ labels of
size N, compute the average cross-entropy loss over the batch. The loss for a single sample j with
logots zj =[zj1,…,zjc] and true label yj is calculated using the numerically stable formula: Losssj
= log (zj = [zj1,…,zjc]) − zjyj The final output stored in the loss variable should be the average
loss over the N samples: L = 1/N∑j=1Losssj The input parameters are logists, true _ labels, N
(number of samples), and C (number of classes). The result should be stored in loss (a pointer to a
single float).

```python
import numpy as np

def categorical_cross_entropy(logits, true_labels):
    # Softmax
    exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
    probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

    # 计算loss
    N = logits.shape[0]
    losses = -np.log(probs[range(N), true_labels])

    return np.mean(losses)
```
 */

// TODO: use float4 + blockReduce to optimize
__global__ void categorical_cross_entropy_kernel(const float* __restrict__ logits,
                                                 const int* __restrict__ true_labels,
                                                 float* __restrict__ loss, const int N,
                                                 const int C) {
  const int tid = threadIdx.x;
  const int row = blockIdx.x;
  __shared__ float partial[256];

  const int row_offset = row * C;
  const int true_label = true_labels[row];

  // 1) row max for stable log-sum-exp
  float max_val = -INFINITY;
  for (int c = tid; c < C; c += blockDim.x) {
    max_val = fmaxf(max_val, logits[row_offset + c]);
  }
  partial[tid] = max_val;
  __syncthreads();
  if (tid == 0) {
    float row_max = -INFINITY;
    for (int t = 0; t < blockDim.x; ++t) {
      row_max = fmaxf(row_max, partial[t]);
    }
    partial[0] = row_max;
  }
  __syncthreads();
  max_val = partial[0];

  // 2) sum(exp(logit - max))
  float exp_sum = 0.0f;
  for (int c = tid; c < C; c += blockDim.x) {
    exp_sum += __expf(logits[row_offset + c] - max_val);
  }
  partial[tid] = exp_sum;
  __syncthreads();
  if (tid == 0) {
    float row_exp_sum = 0.0f;
    for (int t = 0; t < blockDim.x; ++t) {
      row_exp_sum += partial[t];
    }
    partial[0] = row_exp_sum;
  }
  __syncthreads();
  exp_sum = partial[0];

  // 3) loss_j = logsumexp(logits_j) - logits_j[y_j]
  if (tid == 0) {
    const float true_logit = logits[row_offset + true_label];
    const float sample_loss = __logf(exp_sum) + max_val - true_logit;
    atomicAdd(loss, sample_loss / static_cast<float>(N));
  }
}

extern "C" void solve(const float* logits, const int* true_labels, float* loss, int N, int C) {
  const int nths = std::min(256, C);
  dim3 threadsPerBlock(nths);
  dim3 blocksPerGrid(N);
  cudaMemset(loss, 0, sizeof(float));
  categorical_cross_entropy_kernel<<<blocksPerGrid, threadsPerBlock>>>(logits, true_labels, loss, N, C);
  cudaDeviceSynchronize();
}
