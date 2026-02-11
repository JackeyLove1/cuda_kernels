#include <cuda.h>
#include <cuda_runtime.h>

#include <cstdio>
#include <iostream>

#define NUM_BINS 1024

// dim3 nblks(1024) dim3 nthrs(256)
__global__ void histogram_basic(const char* __restrict__ data, const unsigned int N,
                                unsigned int* __restrict__ histogram) {
  __shared__ unsigned int s_histogram[NUM_BINS];
  const auto tid = threadIdx.x;
  const auto gid = blockIdx.x * blockDim.x + tid;
  for (int i = tid; i < NUM_BINS; i += blockDim.x) {
    s_histogram[i] = 0u;
  }
  __syncthreads();
  const auto stride = gridDim.x * blockDim.x;
  for (int i = gid; i < N; i += stride) {
    int alpha_pos = data[i] - 'a';
    if (alpha_pos >= 0 && alpha_pos < 26) {
      atomicAdd(&s_histogram[alpha_pos / 4], 1);
    }
  }
  __syncthreads();

  for (int i = tid; i < NUM_BINS; i += blockDim.x) {
    const auto bin_value = s_histogram[i];
    if (bin_value > 0) {
      atomicAdd(histogram, bin_value);
    }
  }
}

__global__ void histogram_registers(const char* __restrict__ data, const unsigned int N,
                                    unsigned int* __restrict__ histogram) {
  __shared__ unsigned int s_histogram[NUM_BINS];
  const auto tid = threadIdx.x;
  const auto gid = blockIdx.x * blockDim.x + tid;
  for (int i = tid; i < NUM_BINS; i += blockDim.x) {
    s_histogram[i] = 0u;
  }
  __syncthreads();
  int preAccumulated = 0;
  int preCharBin = 0;
  const auto stride = gridDim.x * blockDim.x;
  for (int i = gid; i < N; i += stride) {
    int alpha_pos = data[i] - 'a';
    auto alpha_bin = alpha_pos / 4;
    if (alpha_pos >= 0 && alpha_pos < 26) {
      if (alpha_bin != preCharBin) {
        atomicAdd(&s_histogram[preCharBin], preAccumulated);
        preAccumulated = 1;
        preCharBin = alpha_bin;
      } else {
        preAccumulated++;
      }
    }
  }
  // end of loop
  if (preAccumulated > 0) {
    atomicAdd(&s_histogram[preCharBin], preAccumulated);
  }
  __syncthreads();

  for (int i = tid; i < NUM_BINS; i += blockDim.x) {
    const auto bin_value = s_histogram[i];
    if (bin_value > 0) {
      atomicAdd(histogram, bin_value);
    }
  }
}

int main() {}