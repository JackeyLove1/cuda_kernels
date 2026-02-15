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
Write a GPU program that implements top-p (nucleus) sampling for LLM inference.
Top-p sampling is a text generation technique where you sample from the smallest set of tokens whose
cumulative probability exceeds threshold p. This balances randomness and quality better than pure
top-k or greedy sampling. Given logits (unnormalized scores) from a language model:
1. Convert logits to probabilities using softmax
2. Sort tokens by probability (descending)
3. Find the smallest set where cumulative probability ≥ p (the"nucleus")
4. Renormalize the nucleus probabilities to sum to 1
5. Sample a token from the nucleus using the provided random seed
 */
#define MAX_IDENTITY (-INFINITY)
struct __align__(8) MaxSum {
  float max_val;
  float sum_val;
};

template <typename T, typename U>
__host__ __device__ __forceinline__ auto CDIV(T a, U b) {
  return (a + b - 1) / b;
}

// Online Softmax Update: Combine two (max, sum) pairs
__device__ __forceinline__ MaxSum combine_stats(MaxSum a, MaxSum b) {
  // Optimization & Safety: Handle identity cases to avoid NaN (-inf - -inf)
  // and skip expensive exp() calculations.
  if (a.max_val == MAX_IDENTITY) return b;
  if (b.max_val == MAX_IDENTITY) return a;

  MaxSum res;
  res.max_val = fmaxf(a.max_val, b.max_val);

  float scale_a = __expf(a.max_val - res.max_val);
  float scale_b = __expf(b.max_val - res.max_val);

  res.sum_val = a.sum_val * scale_a + b.sum_val * scale_b;
  return res;
}

// Warp Reduction
__device__ __forceinline__ MaxSum WarpReduceStats(MaxSum val) {
#pragma unroll
  for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
    float other_max = __shfl_down_sync(FULL_MASK, val.max_val, offset);
    float other_sum = __shfl_down_sync(FULL_MASK, val.sum_val, offset);
    val = combine_stats(val, {other_max, other_sum});
  }
  return val;
}

// --------------------------------------------------------------------------
// Optimized Kernel: Explicit ILP (Instruction Level Parallelism)
// --------------------------------------------------------------------------
__global__ void compute_stats_kernel_opt(const float* __restrict__ input,
                                         MaxSum* __restrict__ block_stats, const int N) {
  MaxSum local_val = {MAX_IDENTITY, 0.0f};

  const int tid = threadIdx.x;
  const int stride = gridDim.x * blockDim.x;
  int idx = blockIdx.x * blockDim.x + tid;

  // Process as float4 to improve memory throughput
  for (; idx * 4 < N; idx += stride) {
    const int base_id = idx * 4;
    const int remaining = N - base_id;

    MaxSum batch_val;

    if (remaining >= 4) {
      const float4 val = reinterpret_cast<const float4*>(input)[idx];

      // 1. Find Max within the batch (Instruction Parallelism!)
      float m0 = fmaxf(val.x, val.y);
      float m1 = fmaxf(val.z, val.w);
      float m_batch = fmaxf(m0, m1);

      // 2. Compute Sum within the batch (Independent expf calls!)
      // These 4 expf calls can be pipelined by the compiler/GPU
      float s_batch = __expf(val.x - m_batch) + __expf(val.y - m_batch) + __expf(val.z - m_batch) +
                      __expf(val.w - m_batch);

      batch_val = {m_batch, s_batch};
    } else {
      // Tail handling
      float m_batch = MAX_IDENTITY;
      float s_batch = 0.0f;
      for (int k = 0; k < remaining; ++k) {
        float v = input[base_id + k];
        float new_max = fmaxf(m_batch, v);
        float scale = __expf(m_batch - new_max);
        s_batch = s_batch * scale + __expf(v - new_max);
        m_batch = new_max;
      }
      batch_val = {m_batch, s_batch};
    }

    // 3. Combine with local accumulator (Only 1 combine per 4 elements)
    local_val = combine_stats(local_val, batch_val);
  }

  // Warp & Block Reduction
  local_val = WarpReduceStats(local_val);

  static __shared__ float smem_max[WARP_SIZE];
  static __shared__ float smem_sum[WARP_SIZE];

  const int warp_id = tid / WARP_SIZE;
  const int lane_id = tid % WARP_SIZE;

  if (lane_id == 0) {
    smem_max[warp_id] = local_val.max_val;
    smem_sum[warp_id] = local_val.sum_val;
  }
  __syncthreads();

  if (warp_id == 0) {
    MaxSum warp_val;
    const int num_warps = blockDim.x / WARP_SIZE;
    if (lane_id < num_warps) {
      warp_val.max_val = smem_max[lane_id];
      warp_val.sum_val = smem_sum[lane_id];
    } else {
      warp_val = {MAX_IDENTITY, 0.0f};
    }

    warp_val = WarpReduceStats(warp_val);

    if (lane_id == 0) {
      block_stats[blockIdx.x] = warp_val;
    }
  }
}

// Pass 2: Reduce Block Stats (Same as before)
__global__ void reduce_final_stats_kernel(const MaxSum* __restrict__ block_stats,
                                          float* __restrict__ global_max,
                                          float* __restrict__ global_sum, int num_blocks) {
  MaxSum local_val = {MAX_IDENTITY, 0.0f};
  for (int i = threadIdx.x; i < num_blocks; i += blockDim.x) {
    local_val = combine_stats(local_val, block_stats[i]);
  }
  local_val = WarpReduceStats(local_val);

  static __shared__ float smem_max[WARP_SIZE];
  static __shared__ float smem_sum[WARP_SIZE];

  const int tid = threadIdx.x;
  const int warp_id = tid / WARP_SIZE;
  const int lane_id = tid % WARP_SIZE;

  if (lane_id == 0) {
    smem_max[warp_id] = local_val.max_val;
    smem_sum[warp_id] = local_val.sum_val;
  }
  __syncthreads();

  if (warp_id == 0) {
    MaxSum warp_val;
    const int num_warps = blockDim.x / WARP_SIZE;
    if (lane_id < num_warps) {
      warp_val.max_val = smem_max[lane_id];
      warp_val.sum_val = smem_sum[lane_id];
    } else {
      warp_val = {MAX_IDENTITY, 0.0f};
    }
    warp_val = WarpReduceStats(warp_val);
    if (lane_id == 0) {
      *global_max = warp_val.max_val;
      *global_sum = warp_val.sum_val;
    }
  }
}

// Pass 3: Softmax (Optimized with loop unrolling hint)
__global__ void softmax_kernel_opt(const float* __restrict__ input, float* __restrict__ output,
                                   const float* __restrict__ global_max,
                                   const float* __restrict__ global_sum, const int N) {
  const auto stride = gridDim.x * blockDim.x;
  auto idx = threadIdx.x + blockIdx.x * blockDim.x;

  const float max_val = *global_max;
  const float inv_sum = 1.0f / (*global_sum);

  for (; idx * 4 < N; idx += stride) {
    const int base_id = idx * 4;
    const int remaining = N - base_id;

    if (remaining >= 4) {
      const float4 val = reinterpret_cast<const float4*>(input)[idx];
      float4 out;

      out.x = __expf(val.x - max_val) * inv_sum;
      out.y = __expf(val.y - max_val) * inv_sum;
      out.z = __expf(val.z - max_val) * inv_sum;
      out.w = __expf(val.w - max_val) * inv_sum;

      reinterpret_cast<float4*>(output)[idx] = out;
    } else {
      for (int j = base_id; j < N; ++j) {
        output[j] = __expf(input[j] - max_val) * inv_sum;
      }
    }
  }
}

extern "C" void solve(const float* logits, const float* p, const int* seed, int* sampled_token,
                      int vocab_size) {
  constexpr int threadsPerBlock = 256;
  constexpr auto max_blocks = 1024;  // Persistent style
  const auto blocksPerGrid = std::min(max_blocks, CDIV(vocab_size, threadsPerBlock * 4));
  // Note: blocksPerGrid logic slightly adjusted for throughput, *4 accounts for float4
  const int N = vocab_size;
  MaxSum* d_block_stats;
  float *d_final_max, *d_final_sum;
  float* output;

  cudaMalloc(&d_block_stats, blocksPerGrid * sizeof(MaxSum));
  cudaMalloc(&d_final_max, sizeof(float));
  cudaMalloc(&d_final_sum, sizeof(float));
  cudaMalloc(&output, vocab_size * sizeof(float));

  // Step 1: Compute Stats
  compute_stats_kernel_opt<<<blocksPerGrid, threadsPerBlock>>>(logits, d_block_stats, N);

  // Step 2: Reduce
  reduce_final_stats_kernel<<<1, 256>>>(d_block_stats, d_final_max, d_final_sum, blocksPerGrid);

  // Step 3: Normalize
  softmax_kernel_opt<<<blocksPerGrid, threadsPerBlock>>>(logits, output, d_final_max, d_final_sum,
                                                         N);

  // Step 4: Sort softmax output by probability (descending), keeping token indices.
  thrust::device_ptr<float> output_ptr(output);
  thrust::device_vector<float> sorted_probs(output_ptr, output_ptr + N);
  thrust::device_vector<int> sorted_indices(N);
  thrust::sequence(sorted_indices.begin(), sorted_indices.end());
  thrust::sort_by_key(sorted_probs.begin(), sorted_probs.end(), sorted_indices.begin(),
                      thrust::greater<float>());
  CUDA_CHECK(cudaDeviceSynchronize());

  thrust::host_vector<float> h_sorted_probs = sorted_probs;
  thrust::host_vector<int> h_sorted_indices = sorted_indices;

  const float top_p = fminf(fmaxf(read_scalar(p), 0.0f), 1.0f);
  const int seed_val = read_scalar(seed);

  float nucleus_sum = 0.0f;
  int nucleus_size = 0;
  for (; nucleus_size < N; ++nucleus_size) {
    nucleus_sum += h_sorted_probs[nucleus_size];
    if (nucleus_sum >= top_p) {
      ++nucleus_size;  // include current token
      break;
    }
  }
  if (nucleus_size <= 0) nucleus_size = 1;
  if (nucleus_size > N) nucleus_size = N;

  std::mt19937 rng(static_cast<uint32_t>(seed_val));
  std::vector<double> nucleus_probs(nucleus_size);
  if (nucleus_sum > 0.0f) {
    const double inv_nucleus_sum = 1.0 / static_cast<double>(nucleus_sum);
    for (int i = 0; i < nucleus_size; ++i) {
      nucleus_probs[i] = static_cast<double>(h_sorted_probs[i]) * inv_nucleus_sum;
    }
  } else {
    const double uniform_prob = 1.0 / static_cast<double>(nucleus_size);
    for (int i = 0; i < nucleus_size; ++i) {
      nucleus_probs[i] = uniform_prob;
    }
  }

  std::discrete_distribution<int> dist(nucleus_probs.begin(), nucleus_probs.end());
  const int selected_rank = dist(rng);
  const int selected_token = h_sorted_indices[selected_rank];
  write_scalar(sampled_token, selected_token);

  CUDA_CHECK(cudaFree(output));
  CUDA_CHECK(cudaFree(d_final_sum));
  CUDA_CHECK(cudaFree(d_final_max));
  CUDA_CHECK(cudaFree(d_block_stats));
}

int main() {
  constexpr int vocab_size = 3;
  const float h_logits[vocab_size] = {10.0f, 1.0f, 1.0f};
  const float h_p = 0.5f;
  const int h_seed = 123;
  int h_sampled_token = -1;

  float* d_logits = nullptr;
  float* d_p = nullptr;
  int* d_seed = nullptr;
  int* d_sampled_token = nullptr;

  cudaError_t status = cudaMalloc(&d_logits, vocab_size * sizeof(float));
  if (status != cudaSuccess) {
    fprintf(stderr, "cudaMalloc d_logits failed: %s\n", cudaGetErrorString(status));
    return 1;
  }
  status = cudaMalloc(&d_p, sizeof(float));
  if (status != cudaSuccess) {
    fprintf(stderr, "cudaMalloc d_p failed: %s\n", cudaGetErrorString(status));
    cudaFree(d_logits);
    return 1;
  }
  status = cudaMalloc(&d_seed, sizeof(int));
  if (status != cudaSuccess) {
    fprintf(stderr, "cudaMalloc d_seed failed: %s\n", cudaGetErrorString(status));
    cudaFree(d_p);
    cudaFree(d_logits);
    return 1;
  }
  status = cudaMalloc(&d_sampled_token, sizeof(int));
  if (status != cudaSuccess) {
    fprintf(stderr, "cudaMalloc d_sampled_token failed: %s\n", cudaGetErrorString(status));
    cudaFree(d_seed);
    cudaFree(d_p);
    cudaFree(d_logits);
    return 1;
  }

  status = cudaMemcpy(d_logits, h_logits, vocab_size * sizeof(float), cudaMemcpyHostToDevice);
  if (status != cudaSuccess) {
    fprintf(stderr, "cudaMemcpy logits failed: %s\n", cudaGetErrorString(status));
    cudaFree(d_sampled_token);
    cudaFree(d_seed);
    cudaFree(d_p);
    cudaFree(d_logits);
    return 1;
  }
  status = cudaMemcpy(d_p, &h_p, sizeof(float), cudaMemcpyHostToDevice);
  if (status != cudaSuccess) {
    fprintf(stderr, "cudaMemcpy p failed: %s\n", cudaGetErrorString(status));
    cudaFree(d_sampled_token);
    cudaFree(d_seed);
    cudaFree(d_p);
    cudaFree(d_logits);
    return 1;
  }
  status = cudaMemcpy(d_seed, &h_seed, sizeof(int), cudaMemcpyHostToDevice);
  if (status != cudaSuccess) {
    fprintf(stderr, "cudaMemcpy seed failed: %s\n", cudaGetErrorString(status));
    cudaFree(d_sampled_token);
    cudaFree(d_seed);
    cudaFree(d_p);
    cudaFree(d_logits);
    return 1;
  }

  solve(d_logits, d_p, d_seed, d_sampled_token, vocab_size);

  status = cudaMemcpy(&h_sampled_token, d_sampled_token, sizeof(int), cudaMemcpyDeviceToHost);
  if (status != cudaSuccess) {
    fprintf(stderr, "cudaMemcpy sampled_token failed: %s\n", cudaGetErrorString(status));
    cudaFree(d_sampled_token);
    cudaFree(d_seed);
    cudaFree(d_p);
    cudaFree(d_logits);
    return 1;
  }

  printf("Input:\n");
  printf("  logits = [10.0, 1.0, 1.0]\n");
  printf("  p = 0.5\n");
  printf("  seed = 123\n\n");
  printf("Output:\n");
  printf("  sampled_token = %d\n", h_sampled_token);
  printf("  (single token dominates the probability mass)\n");
  printf("  %s\n", (h_sampled_token == 0) ? "[PASS]" : "[FAIL]");

  cudaFree(d_sampled_token);
  cudaFree(d_seed);
  cudaFree(d_p);
  cudaFree(d_logits);
  return (h_sampled_token == 0) ? 0 : 2;
}