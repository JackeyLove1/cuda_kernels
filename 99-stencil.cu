#include <cuda.h>
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include <iostream>

__constant__ float c[7] = {1, 2, 3, 4, 5, 6, 7};

__global__ void stencil_basic_kernel(const float* __restrict__ in, float* __restrict__ out,
                                     const int N) {
  const int i = blockIdx.z * blockDim.z + threadIdx.z;
  const int j = blockIdx.y * blockDim.y + threadIdx.y;
  const int k = blockIdx.x * blockDim.x + threadIdx.x;
  const auto stride_i = N * N;
  const auto stride_j = N;
  const auto stride_k = 1;
  if (i >= 1 && i < N - 1 && j >= 1 && j < N - 1 && k >= 1 && k < N - 1) {
    out[i * stride_i + j * stride_j + k * stride_k] =
        c[0] * in[i * stride_i + j * stride_j + k * stride_k] +
        c[1] * in[i * stride_i + j * stride_j + (k - 1) * stride_k] +
        c[2] * in[i * stride_i + j * stride_j + (k + 1) * stride_k] +
        c[3] * in[(i - 1) * stride_i + j * stride_j + k * stride_k] +
        c[4] * in[(i + 1) * stride_i + j * stride_j + k * stride_k] +
        c[5] * in[i * stride_i + (j - 1) * stride_j + k * stride_k] +
        c[6] * in[i * stride_i + (j + 1) * stride_j + k * stride_k];
  }
}

#define PAD 1
#define OUT_TILE 32
#define IN_TILE ((OUT_TILE) + 2 * (PAD))

__global__ void stencil_basic_kernel_tiled(const float* __restrict__ in, float* __restrict__ out,
                                           const int N) {
  const int i = blockIdx.z * OUT_TILE + threadIdx.z - PAD;
  const int j = blockIdx.y * OUT_TILE + threadIdx.y - PAD;
  const int k = blockIdx.x * OUT_TILE + threadIdx.x - PAD;
  __shared__ float tile[IN_TILE][IN_TILE][IN_TILE];
  const auto stride_i = N * N;
  const auto stride_j = N;
  const auto stride_k = 1;
  const auto tz = threadIdx.z, ty = threadIdx.y, tx = threadIdx.x;
  if (i >= 1 && i < N - 1 && j >= 1 && j < N - 1 && k >= 1 && k < N - 1) {
    tile[tz][ty][tx] = in[i * stride_i + j * stride_j + k * stride_k];
  } else {
    tile[tz][ty][tx] = 0.0f;
  }
  __syncthreads();
  if (i >= 1 && i < N - 1 && j >= 1 && j < N - 1 && k >= 1 && k < N - 1) {
    out[i * stride_i + j * stride_j + k * stride_k] =
        c[0] * tile[tz][ty][tx] + c[1] * tile[tz][ty][tx - 1] + c[2] * tile[tz][ty][tx + 1] +
        c[3] * tile[tz - 1][ty][tx] + c[4] * tile[tz + 1][ty][tx] + c[5] * tile[tz][ty - 1][tx] +
        c[6] * tile[tz][ty + 1][tx];
  }
}

// thread coarse
__global__ void stencil_kernel_tiled_coarse(const float* __restrict__ in, float* __restrict__ out,
                                            const int N) {
  const int iStart = blockIdx.z * OUT_TILE;
  const int j = blockIdx.y * OUT_TILE + threadIdx.y - PAD;
  const int k = blockIdx.x * OUT_TILE + threadIdx.x - PAD;
  __shared__ float inPrev[IN_TILE][IN_TILE];
  __shared__ float inCurr[IN_TILE][IN_TILE];
  __shared__ float inNext[IN_TILE][IN_TILE];

  const auto stride_i = N * N;
  const auto stride_j = N;
  const auto stride_k = 1;
  const auto tz = threadIdx.z, ty = threadIdx.y, tx = threadIdx.x;

  if (iStart - 1 >= 0 && iStart - 1 < N && j < N && j >= 0 && k < N && k >= 0) {
    inPrev[ty][tx] = in[(iStart - 1) * stride_i + j * stride_j + k * stride_k];
  }
  if (iStart >= 0 && iStart < N && j < N && j >= 0 && k < N && k >= 0) {
    inCurr[ty][tx] = in[iStart * stride_i + j * stride_j + k * stride_k];
  }
  for (int i = iStart; i < iStart + OUT_TILE; i++) {
    if (i < N && j < N && k < N && i >= 0 && j >= 0 && k >= 0) {
      inNext[ty][tx] = in[(i + 1) * stride_i + j * stride_j + k * stride_k];
    } else {
      inNext[ty][tx] = 0.0f;
    }
    __syncthreads();
    out[i * stride_i + j * stride_j + k * stride_k] =
        c[0] * inCurr[ty][tx] + c[1] * inPrev[ty][tx] + c[2] * inNext[ty][tx] +
        c[3] * inCurr[ty - 1][tx] + c[4] * inCurr[ty + 1][tx] + c[5] * inCurr[ty][tx - 1] +
        c[6] * inCurr[ty][tx + 1];
    __syncthreads();
    inPrev[ty][tx] = inCurr[ty][tx];
    inCurr[ty][tx] = inNext[ty][tx];
  }
}

// thread coarse register
__global__ void stencil_kernel_tiled_coarse_register(const float* __restrict__ in,
                                                     float* __restrict__ out, const int N) {
  const int iStart = blockIdx.z * OUT_TILE;
  const int j = blockIdx.y * OUT_TILE + threadIdx.y - PAD;
  const int k = blockIdx.x * OUT_TILE + threadIdx.x - PAD;

  float inPrev = 0.0f;
  float inNext = 0.0f;
  float inCurr = 0.0f;
  __shared__ float inCurr_s[IN_TILE][IN_TILE];

  const auto stride_i = N * N;
  const auto stride_j = N;
  const auto stride_k = 1;
  const auto ty = threadIdx.y, tx = threadIdx.x;

  const bool inJkRange = (j >= 0 && j < N && k >= 0 && k < N);
  if (inJkRange && iStart - 1 >= 0 && iStart - 1 < N) {
    inPrev = in[(iStart - 1) * stride_i + j * stride_j + k * stride_k];
  }
  if (inJkRange && iStart >= 0 && iStart < N) {
    inCurr = in[iStart * stride_i + j * stride_j + k * stride_k];
  }

  for (int i = iStart; i < iStart + OUT_TILE; ++i) {
    if (inJkRange && i + 1 >= 0 && i + 1 < N) {
      inNext = in[(i + 1) * stride_i + j * stride_j + k * stride_k];
    } else {
      inNext = 0.0f;
    }

    inCurr_s[ty][tx] = inCurr;
    __syncthreads();

    if (i >= 1 && i < N - 1 && j >= 1 && j < N - 1 && k >= 1 && k < N - 1 && ty >= 1 &&
        ty < IN_TILE - 1 && tx >= 1 && tx < IN_TILE - 1) {
      out[i * stride_i + j * stride_j + k * stride_k] =
          c[0] * inCurr_s[ty][tx] + c[1] * inPrev + c[2] * inNext + c[3] * inCurr_s[ty - 1][tx] +
          c[4] * inCurr_s[ty + 1][tx] + c[5] * inCurr_s[ty][tx - 1] + c[6] * inCurr_s[ty][tx + 1];
    }

    __syncthreads();
    inPrev = inCurr;
    inCurr = inNext;
  }
}

int main() {}