#include <cuda.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cuda/atomic>
#include <limits>
#include <numeric>

#define WARP_SIZE 32
#define FULL_MASK 0xFFFFFFFF
#define MAX_IDENTITY (-FLT_MAX)

template <typename T, typename U>
__host__ __device__ __forceinline__ auto CDIV(T a, U b) {
  return (a + b - 1) / b;
}

struct __align__(8) MaxSum {
  float max_val;
  float sum_val;
};

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

extern "C" void solve(const float* input, float* output, const int N) {
  constexpr int threadsPerBlock = 256;
  constexpr auto max_blocks = 1024;  // Persistent style
  const auto blocksPerGrid = std::min(max_blocks, CDIV(N, threadsPerBlock * 4));
  // Note: blocksPerGrid logic slightly adjusted for throughput, *4 accounts for float4

  MaxSum* d_block_stats;
  float *d_final_max, *d_final_sum;

  cudaMalloc(&d_block_stats, blocksPerGrid * sizeof(MaxSum));
  cudaMalloc(&d_final_max, sizeof(float));
  cudaMalloc(&d_final_sum, sizeof(float));

  // Step 1: Compute Stats
  compute_stats_kernel_opt<<<blocksPerGrid, threadsPerBlock>>>(input, d_block_stats, N);

  // Step 2: Reduce
  reduce_final_stats_kernel<<<1, 256>>>(d_block_stats, d_final_max, d_final_sum, blocksPerGrid);

  // Step 3: Normalize
  softmax_kernel_opt<<<blocksPerGrid, threadsPerBlock>>>(input, output, d_final_max, d_final_sum,
                                                         N);
}