#include <cooperative_groups.h>
#include <cuda_runtime.h>

#include <cstdio>
#include <cub/cub.cuh>
#include <cuda/pipeline>

#define CUDA_CHECK(call)                                                                          \
  do {                                                                                            \
    cudaError_t _e = (call);                                                                      \
    if (_e != cudaSuccess) {                                                                      \
      std::fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(_e)); \
      return;                                                                                     \
    }                                                                                             \
  } while (0)

/**(lane & j) == 0：判断 lane 在第 j 这一位上是 0 还是 1

比如 j=1 时，看的是最低位：偶数 lane 为真，奇数 lane 为假

你举的 lane=3, j=1：

3 & 1 = 1 → (3 & 1)==0 是 False（因为 lane 3 是奇数） */
__global__ void warp_bitonic_sort(int& val) {
  static constexpr int WARP_SIZE = 32;
  static constexpr unsigned int FULL_MASK = 0xffffffff;
  const int lane_id = threadIdx.x & 31;

  for (int k = 2; k <= WARP_SIZE; k >>= 1) {
    for (int j = (k >> 1); j > 0; j >>= 1) {
      int other_val = __shfl_xor_sync(FULL_MASK, val, j);
      int low = max(val, other_val);
      int high = min(val, other_val);
      bool ascending = (lane_id & k) == 0;
      val = (((lane_id & j) == 0) == ascending) ? low : high;
    }
  }
}

__global__ void bitonic_sort_kernel(float* __restrict__ input, const int n_per_block) {
  extern __shared__ float s[];
  const int tid = threadIdx.x;
  const int gid = blockDim.x * blockIdx.x + threadIdx.x;
  if (tid < n_per_block) {
    s[tid] = input[gid];
  }
  __syncthreads();

  // bitonic sort
  for (int k = 2; k <= n_per_block; k <<= 1) {
    for (int j = (k >> 1); j > 0; j >>= 1) {
      int i = tid;
      int ixj = i ^ j;
      if (ixj > i) {
        bool ascending = ((i & k) == 0);
        float a = s[i];
        float b = s[ixj];
        if (ascending) {
          if (a > b) {
            s[i] = b;
            s[ixj] = a;
          }
        } else {
          if (a < b) {
            s[i] = b;
            s[ixj] = a;
          }
        }
      }
      __syncthreads();
    }
  }

  if (tid < n_per_block) {
    input[gid] = s[tid];
  }
}

int main() {
  const int n_per_block = 1024;  // must 2^k
  const int blocks = 1;          // one block sort
  const int N = blocks * n_per_block;

  float* h = (float*)malloc(N * sizeof(float));
  for (int i = 0; i < N; i++) h[i] = (float)(rand() % 10000);

  float* d;
  cudaMalloc(&d, N * sizeof(float));
  cudaMemcpy(d, h, N * sizeof(float), cudaMemcpyHostToDevice);

  dim3 grid(blocks), block(n_per_block);
  size_t shmem = n_per_block * sizeof(float);

  bitonic_sort_kernel<<<grid, block, shmem>>>(d, n_per_block);

  cudaMemcpy(h, d, N * sizeof(float), cudaMemcpyDeviceToHost);

  // 简单检查是否升序
  bool ok = true;
  for (int i = 1; i < N; i++)
    if (h[i - 1] > h[i]) {
      ok = false;
      break;
    }
  printf("sorted = %s\n", ok ? "true" : "false");

  cudaFree(d);
  free(h);
  return 0;
}