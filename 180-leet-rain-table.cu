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

/**
Implement a program that performs the Rectified Linear Unit (ReLU) activation function on a vector
of 32-bit floating point numbers. The ReLU function sets all negative values to zero and leaves
positive values unchanged: ReLU(x)=max(0,x) Implementation Requirements ·External libraries are not
permitted ·The solve function signature must remain unchanged ·The final result must be stored in
output
 */
#define CEIL(a, b) (((a) + (b) - 1) / (b))
#define LOAD4(ptr) (reinterpret_cast<const int4*>(ptr))
#define STORE4(ptr) (reinterpret_cast<int4*>(ptr))

#define WORKS 1  // thread works per iteration

__forceinline__ __device__ __host__ float4 operator+(const float4 a, const float4 b) {
  return make_float4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w);
}

__forceinline__ __device__ unsigned int fnv1a_hash_u32(unsigned int v) {
  constexpr unsigned int FNV_PRIME = 16777619u;
  constexpr unsigned int OFFSET_BASIS = 2166136261u;

  unsigned int h = OFFSET_BASIS;
  h ^= (v & 0xFFu);
  h *= FNV_PRIME;

  h ^= ((v >> 8) & 0xFFu);
  h *= FNV_PRIME;

  h ^= ((v >> 16) & 0xFFu);
  h *= FNV_PRIME;

  h ^= ((v >> 24) & 0xFFu);
  h *= FNV_PRIME;

  return h;
}

__forceinline__ __device__ unsigned int fnv1a_hash_u32_opt(unsigned int v) {
  return fnv1a_hash_u32(v);
}

__forceinline__ __device__ unsigned int hash_R(unsigned int x, const int R) {
#pragma unroll
  for (int i = 0; i < R; ++i) x = fnv1a_hash_u32(x);
  return x;
}

template <int TPB>
__global__ __launch_bounds__(TPB, 2) void fnv1a_hash_kernel_unrolled_vec4(
    const int* __restrict__ input, unsigned int* __restrict__ output, int N, const int R) {
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  const int totalThreads = gridDim.x * blockDim.x;

  // 每线程处理 4 个
  int vecIdx = tid;  // vec index over int4
  const int vecStride = totalThreads;

  const int N4 = N >> 2;   // N / 4
  const int tail = N & 3;  // N % 4

  const int4* in4 = reinterpret_cast<const int4*>(input);
  uint4* out4 = reinterpret_cast<uint4*>(output);

  // Software pipeline: preload next vec4 while hashing current vec4.
  if (vecIdx < N4) {
    int currIdx = vecIdx;
    int4 curr = in4[currIdx];

    for (;;) {
      const int nextIdx = currIdx + vecStride;
      int4 next = make_int4(0, 0, 0, 0);
      if (nextIdx < N4) {
        next = in4[nextIdx];
      }

      unsigned int x0 = (unsigned int)curr.x;
      unsigned int x1 = (unsigned int)curr.y;
      unsigned int x2 = (unsigned int)curr.z;
      unsigned int x3 = (unsigned int)curr.w;

      // Interleave independent lanes to improve ILP.
      x0 = fnv1a_hash_u32_opt(x0);
      x1 = fnv1a_hash_u32_opt(x1);
      x2 = fnv1a_hash_u32_opt(x2);
      x3 = fnv1a_hash_u32_opt(x3);

#pragma unroll
      for (int r = 1; r < R; ++r) {
        x0 = fnv1a_hash_u32_opt(x0);
        x1 = fnv1a_hash_u32_opt(x1);
        x2 = fnv1a_hash_u32_opt(x2);
        x3 = fnv1a_hash_u32_opt(x3);
      }

      out4[currIdx] = make_uint4(x0, x1, x2, x3);

      if (nextIdx >= N4) break;
      currIdx = nextIdx;
      curr = next;
    }
  }

  if (tail && tid < tail) {
    int base = N4 << 2;
    unsigned int x = (unsigned int)input[base + tid];
    x = hash_R(x, R);
    output[base + tid] = x;
  }
}

// input, output are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const int* input, unsigned int* output, int N, int R) {
  constexpr int threadsPerBlock = 256;
  // 限制grid大小以提高SM占用率和L2缓存效率
  int maxBlocks = CEIL(N, threadsPerBlock * 4);
  int blocksPerGrid = std::min(maxBlocks, 2048);  // 限制最大block数

  fnv1a_hash_kernel_unrolled_vec4<threadsPerBlock>
      <<<blocksPerGrid, threadsPerBlock>>>(input, output, N, R);
  cudaDeviceSynchronize();
}