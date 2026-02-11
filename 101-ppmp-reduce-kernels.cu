#include <cuda.h>
#include <cuda_runtime.h>

#include <cstdio>
#include <iostream>

#define MASK 0xFFFFFFFF
#define WARP_SIZE 32

template <typename T>
__device__ __forceinline__ T warpReduceSum(T val) {
#pragma unroll
  for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
    val += __shfl_down_sync(MASK, val, offset);
  }
  return val;
}

// dim3 nblks(512) dim3 nthrs(256)
__global__ void reduce_sum_basic(const float* __restrict__ in, float* __restrict__ out,
                                 const int N) {
  const auto tid = threadIdx.x;
  const auto gid = blockDim.x * blockIdx.x + tid;
  float sum{0.f};
  const auto stride = blockDim.x * gridDim.x;

  for (int i = gid; i < N; i += stride) {
    sum += in[i];
  }
  __syncthreads();
  const auto warp_id = tid >> 5;  // 0 ~ 31
  const auto lane_id = tid & 31;  // 0 ~ 31

  // warp reduce sum
  sum = warpReduceSum(sum);
  __shared__ float smem[WARP_SIZE];
  smem[warp_id] = sum;

  if (lane_id == 0) {
    smem[warp_id] = sum;
  }
  __syncthreads();

  // block reduce sum
  if (warp_id == 0) {
    const int nums_warp = (blockDim.x + WARP_SIZE - 1) / WARP_SIZE;
    smem[lane_id] = (lane_id < nums_warp) ? smem[lane_id] : 0.f;
    sum = warpReduceSum(smem[0]);
    if (lane_id == 0) {
      atomicAdd(out, sum);
    }
  }
}