#include <cuda.h>
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <numeric>
#include <random>
#include <type_traits>
#include <vector>

#define CEIL(a, b) (((a) + (b) - 1) / (b))
#define LOAD4(ptr) (reinterpret_cast<const float4*>(ptr))
#define STORE4(ptr) (reinterpret_cast<float4*>(ptr))

#define PER_THREAD_WORK_ITEMS 1  // thread works per iteration

__forceinline__ __device__ __host__ float4 operator+(const float4 a, const float4 b) {
  return make_float4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w);
}

/**
Implement a program that copies an N×N matrix of 32-bit floating point numbers from input array A to
output array B on the GPU. The program should perform a direct element-wise copy so that B₁,j=A₁,j
for all valid indices.
 */
__global__ void copy_matrix_kernel(const float* __restrict__ A, float* __restrict__ B,
                                   const int N) {
  const int tid = threadIdx.x;
  const int gid = blockIdx.x * blockDim.x + tid;
  const int total = N * N;
  const int stride = blockDim.x * gridDim.x;
  const int vec_size = total >> 2;
  const int vec_tail = total & 3;
  const float4* a4 = LOAD4(A);
  float4* b4 = STORE4(B);
  for (int i = gid; i < vec_size; i += stride) {
    b4[i] = a4[i];
  }
  if (vec_tail && tid < 1) {
#pragma unroll
    for (int i = 0; i < vec_tail; ++i) {
      const int base = (vec_size << 2) + i;
      B[base] = A[base];
    }
  }
}

// A, B are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* A, float* B, int N) {
  const int total = N * N;
  const int threadsPerBlock = 256;
  int blocksPerGrid = std::min(CEIL(total, threadsPerBlock * 4), 2048);
  copy_matrix_kernel<<<blocksPerGrid, threadsPerBlock>>>(A, B, N);
  cudaDeviceSynchronize();
}
