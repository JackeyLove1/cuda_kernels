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

static constexpr int PER_THREAD_WORK_ITEMS = 1;
static constexpr int WARP_SIZE = 32;
static constexpr int MAX_BLOCKS = 4096;
static constexpr int THREADS_PER_BLOCK = 1024;

/**
Implement RMS Normalization forward pass for 1D input vectors. Given an input tensor of shape [N]
where N is the number of elements, compute the normalized output using a scalar scale ( gamma ) and
shift ( beta ) parameter. RMS Normalization computes:
\begin{align}\text{rms}=\sqrt{\frac{1}{N}\sum_{i=1}^{N}x_{i}^{2}+\epsilon}
\\ \hat{x}_{i}=\frac{x_{i}}{\text{rms}} \\ y_{i}=\gamma\hat{x}_{i}+\beta\end{align}
 */

void __global__ rms_norm_kernel(const float* __restrict__ input, float gamma, float beta,
                                float* __restrict__ output, const int N, float eps) {
  const auto tid = threadIdx.x;
  const auto gid = blockIdx.x * blockDim.x + tid;
  const auto stride = blockDim.x * gridDim.x;
  const float4* input4 = LOAD4(input);
  float4* out4 = STORE4(output);
  const int vec_size = N >> 2;
  const int vec_tail = N & 3;
  float thread_sum{0.f};
  for (int i = gid; i < vec_size; i += stride) {
    float4 vec = input4[i];
    thread_sum += vec.x * vec.x + vec.y * vec.y + vec.z * vec.z + vec.w * vec.w;
  }
  if (vec_tail && gid < vec_tail) {
    const auto idx = (vec_size << 2) + gid;
    thread_sum += input[idx] * input[idx];
  }

  thread_sum = BlockReduceDynamic<float>(thread_sum, ReduceOpType::SUM);
  __shared__ float block_square_sum;
  if (tid == 0) {
    block_square_sum = thread_sum;
  }
  __syncthreads();

  float inv_rms = rsqrtf(block_square_sum / N + eps);
  auto op = [gamma, beta, inv_rms](float x) { return gamma * x * inv_rms + beta; };
  for (int i = gid; i < vec_size; i += stride) {
    float4 vec = input4[i];
    out4[i] = make_float4(op(vec.x), op(vec.y), op(vec.z), op(vec.w));
  }
  if (vec_tail && gid < vec_tail) {
    const auto idx = (vec_size << 2) + gid;
    output[idx] = op(input[idx]);
  }
}

// input, output are device pointers
extern "C" void solve(const float* input, float gamma, float beta, float* output, int N,
                      float eps) {
  rms_norm_kernel<<<1, 1024>>>(input, gamma, beta, output, N, eps);
  cudaDeviceSynchronize();
}
