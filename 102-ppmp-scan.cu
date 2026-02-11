#include <cuda.h>
#include <cuda_runtime.h>

#include <cstdio>
#include <iostream>

// Hillis-Steele algorithm
__global__ void inclusive_scan_block(const float* __restrict__ in, float* __restrict__ out, int N) {
  extern __shared__ float smem[];  // blockDim.x * sizeof(float)
  auto tid = threadIdx.x;
  auto gid = blockIdx.x * blockDim.x + tid;
  smem[tid] = (gid < N) ? in[gid] : 0.f;
  __syncthreads();

  for (int offset = 1; offset < blockDim.x; offset <<= 1) {
    float t = 0.f;
    if (tid > offset) t = smem[tid - offset];
    __syncthreads();

    smem[tid] += t;
    __syncthreads();
  }

  if (gid < N) out[gid] = smem[tid];
}