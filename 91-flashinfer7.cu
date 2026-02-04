#include <cstdio>
#include <cassert>
#include <cuda_runtime.h>
#include <cuda_pipeline.h>
#include <cooperative_groups.h>
#include <include/utils.cuh>

namespace cg = cooperative_groups;

// 使用 Warp Shuffle 进行 Block 内归约
__device__ float block_reduce_sum(float val) {
  // 1. Warp 级归约
  for (int offset = 16; offset > 0; offset >>= 1) {
    val += __shfl_down_sync(0xffffffff, val, offset);
  }

  // 2. 将每个 Warp 的结果存入 Shared Memory
  static __shared__ float shared[32];
  const int lane_id = threadIdx.x & 31;
  const int warp_id = threadIdx.x >> 5;
  const int warp_nums = blockDim.x / warpSize;

  if (lane_id == 0) {
    shared[warp_id] = val;
  }
  __syncthreads();

  // 3. 第一个 Warp 读取 Shared Memory 并进行最后一次归约
  val = (lane_id < warp_nums) ? shared[lane_id] : 0;
  if (warp_id == 0) {
    for (int offset = 16; offset > 0; offset >>= 1) {
      val += __shfl_down_sync(0xffffffff, val, offset);
    }
  }
  return val;
}

__global__ void dot_product_kernel(const float* __restrict__ A,
                                   const float* __restrict__ B,
                                   float* __restrict__ result,
                                   int N) {
  assert(BLOCK_SIZE == blockDim.x);
  __shared__ float s_a[2][BLOCK_SIZE];
  __shared__ float s_b[2][BLOCK_SIZE];

  const int tid = threadIdx.x;
  const int stride = gridDim.x * blockDim.x;
  int idx = blockDim.x * blockIdx.x + threadIdx.x;

  float thread_sum{0.f};
  int compute_stage = 0;
  int fetch_stage = 1;

  // Prime the pipeline for the first iteration.
  if (idx < N) {
    __pipeline_memcpy_async(&s_a[compute_stage][tid], &A[idx], sizeof(float));
    __pipeline_memcpy_async(&s_b[compute_stage][tid], &B[idx], sizeof(float));
  } else {
    // Keep barriers uniform; make the consumed value defined.
    s_a[compute_stage][tid] = 0.0f;
    s_b[compute_stage][tid] = 0.0f;
  }
  __pipeline_commit();

  idx += stride;
  while (idx < N) {
    // Ensure the previously-committed async copies are visible before consuming.
    __pipeline_wait_prior(0);
    __syncthreads();

    thread_sum += s_a[compute_stage][tid] * s_b[compute_stage][tid];

    // Prefetch next element into the other stage.
    if (idx < N) {
      __pipeline_memcpy_async(&s_a[fetch_stage][tid], &A[idx], sizeof(float));
      __pipeline_memcpy_async(&s_b[fetch_stage][tid], &B[idx], sizeof(float));
    } else {
      s_a[fetch_stage][tid] = 0.0f;
      s_b[fetch_stage][tid] = 0.0f;
    }
    __pipeline_commit();

    compute_stage ^= 1;
    fetch_stage ^= 1;
    idx += stride;
  }

  // Consume the final stage.
  __pipeline_wait_prior(0);
  __syncthreads();
  thread_sum += s_a[compute_stage][tid] * s_b[compute_stage][tid];

  float block_sum = block_reduce_sum(thread_sum);
  if (tid == 0) {
    atomicAdd(result, block_sum);
  }
}

int main() {
  int N = 1024 * 1024 * 32; // 32M 元素
  size_t bytes = N * sizeof(float);

  float *h_A, *h_B, *h_res;
  float *d_A, *d_B, *d_res;

  // Host 分配
  cudaMallocHost(&h_A, bytes);
  cudaMallocHost(&h_B, bytes);
  cudaMallocHost(&h_res, sizeof(float));

  // 初始化数据
  for (int i = 0; i < N; i++) {
    h_A[i] = 1.0f;
    h_B[i] = 2.0f;
  }
  *h_res = 0.0f;

  // Device 分配
  CHECK_CUDA(cudaMalloc(&d_A, bytes));
  CHECK_CUDA(cudaMalloc(&d_B, bytes));
  CHECK_CUDA(cudaMalloc(&d_res, sizeof(float)));

  // 拷贝数据
  CHECK_CUDA(cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(d_res, h_res, sizeof(float), cudaMemcpyHostToDevice));

  // Kernel 配置
  int threads = BLOCK_SIZE;
  int blocks = (N + threads - 1) / threads;
  // 为了让 double buffer 生效，最好让每个 Block 跑多个循环。
  // 这里我们减少 blocks 数量，强制 grid-stride loop 发生。
  blocks = 1024; // 举例

  printf("Launching kernel with %d blocks, %d threads...\n", blocks, threads);

  dot_product_kernel<<<blocks, threads>>>(d_A, d_B, d_res, N);
  CHECK_CUDA(cudaDeviceSynchronize());

  // 拷贝结果回 Host
  CHECK_CUDA(cudaMemcpy(h_res, d_res, sizeof(float), cudaMemcpyDeviceToHost));

  printf("Dot Product Result: %.2f (Expected: %.2f)\n", *h_res, (float)N * 2.0f);

  // 清理
  cudaFreeHost(h_A); cudaFreeHost(h_B); cudaFreeHost(h_res);
  cudaFree(d_A); cudaFree(d_B); cudaFree(d_res);

  return 0;
}