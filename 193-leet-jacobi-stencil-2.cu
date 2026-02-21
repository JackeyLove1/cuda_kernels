
#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
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

#define CEIL(a, b) (((a) + (b) - 1) / (b))
#define TILE 16
#define NUM_ITERS 100

/**
 * 2D Jacobi Stencil — 翻转指针 (Ping-Pong Buffer) 实现
 *
 * 迭代策略:
 *   ping -> pong (iter 0)
 *   pong -> ping (iter 1)
 *   ...
 * 每次迭代只交换 float* 指针，零额外拷贝。
 * 边界格子 (row==0, row==rows-1, col==0, col==cols-1) 原样复制。
 * 内部格子取上下左右四邻居均值。
 */
__global__ void jacobi2d_kernel(const float* __restrict__ src, float* __restrict__ dst,
                                const int rows, const int cols) {
  const int col = blockIdx.x * blockDim.x + threadIdx.x;
  const int row = blockIdx.y * blockDim.y + threadIdx.y;

  if (row >= rows || col >= cols) return;
}

extern "C" void solve(const float* input, float* output, int rows, int cols) {}
