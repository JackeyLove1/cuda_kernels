#include <cuda_runtime.h>

// Implement a program that reverses an array of 32-bit floating point numbers in-place. The program
// should perform an in-place reversal of input.
__global__ void reverse_array(float* __restrict__ input, int N) {
  const long long half = static_cast<long long>(N) >> 1;
  const long long tid = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
  const long long stride = static_cast<long long>(blockDim.x) * gridDim.x;

  // Grid-stride loop keeps launch dimensions bounded for large arrays.
  for (long long i = tid; i < half; i += stride) {
    const long long j = static_cast<long long>(N) - 1 - i;
    const float left = input[i];
    const float right = input[j];
    input[i] = right;
    input[j] = left;
  }
}

// input is device pointer
extern "C" void solve(float* input, int N) {
  if (input == nullptr || N <= 1) {
    return;
  }

  const int threadsPerBlock = 256;
  const int workItems = N >> 1;
  int blocksPerGrid = (workItems + threadsPerBlock - 1) / threadsPerBlock;
  if (blocksPerGrid == 0) {
    return;
  }
  // Keep the launch shape valid on all devices; kernel handles the tail via stride.
  if (blocksPerGrid > 65535) {
    blocksPerGrid = 65535;
  }

  reverse_array<<<blocksPerGrid, threadsPerBlock>>>(input, N);
  cudaDeviceSynchronize();
}