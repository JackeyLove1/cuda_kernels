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
#include <vector>

/**
Implement a program that performs a 1D convolution operation. Given an input array and a kernel
(filter), compute the convolved output. The convolution should be performed with a"valid"boundary
condition, meaning the kernel is only applied where it fully overlaps with the input. The input
consists of two arrays: input: A 1D array of 32-bit floating-point numbers. kernel: A 1D array of
32-bit floating-point numbers representing the convolution kernel. The output should be written to
the output array, which will have a size of input size-kernel size+1. The convolution operation is
defined mathematically as: output[i]=sumk=0kernel size-1input[i+j]·kernel[j] where i ranges from 0
to input size-kernel size.
 */
#define CEIL(a, b) (((a) + (b) - 1) / (b))
#define LOAD4(ptr) (reinterpret_cast<const float4*>(ptr))
#define STORE4(ptr) (reinterpret_cast<float4*>(ptr))

#define WORKS 1  // thread works per iteration

__forceinline__ __device__ float4 operator+(const float4 a, const float4 b) {
  return make_float4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w);
}

__global__ void convolution_1d_kernel(const float* input, const float* kernel, float* output,
                                      int input_size, int kernel_size) {}

// input, kernel, output are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* input, const float* kernel, float* output, int input_size,
                      int kernel_size) {
  int output_size = input_size - kernel_size + 1;
  int threadsPerBlock = 256;
  int blocksPerGrid = (output_size + threadsPerBlock - 1) / threadsPerBlock;

  convolution_1d_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, kernel, output, input_size,
                                                            kernel_size);
  cudaDeviceSynchronize();
}
