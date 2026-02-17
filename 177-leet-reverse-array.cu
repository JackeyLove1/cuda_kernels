#include <cuda_runtime.h>
#include <cstdint>

// Implement a program that reverses an array of 32-bit floating point numbers in-place. The program
// should perform an in-place reversal of input.
__global__ void reverse_array_scalar(float* __restrict__ input, int N) {
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

__global__ void reverse_array_float4(float* __restrict__ input, int N) {
  const long long half = static_cast<long long>(N) >> 1;
  const long long tid = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
  const long long stride = static_cast<long long>(blockDim.x) * gridDim.x;
  const long long vecWidth = 4;
  const long long vecStride = stride * vecWidth;
  const long long vecEnd = half & ~(vecWidth - 1);
  float4* const input4 = reinterpret_cast<float4*>(input);

  // Fast path: true float4 vector load/store on aligned addresses.
  for (long long i = tid * vecWidth; i < vecEnd; i += vecStride) {
    const long long rightBase = static_cast<long long>(N) - 1 - i;
    const long long leftVec = i >> 2;
    const long long rightVec = (rightBase - 3) >> 2;
    const float4 left = input4[leftVec];
    const float4 right = input4[rightVec];

    input4[leftVec] = make_float4(right.w, right.z, right.y, right.x);
    input4[rightVec] = make_float4(left.w, left.z, left.y, left.x);
  }
}

// input is device pointer
extern "C" void solve(float* input, int N) {
  if (input == nullptr || N <= 1) {
    return;
  }

  const int threadsPerBlock = 256;
  const int half = N >> 1;
  const bool aligned16 = (reinterpret_cast<std::uintptr_t>(input) & 0xF) == 0;
  const bool useFloat4 = aligned16 && ((N & 3) == 0);

  if (useFloat4) {
    const int vecWorkItems = (half + 3) >> 2;
    int blocksPerGrid = (vecWorkItems + threadsPerBlock - 1) / threadsPerBlock;
    if (blocksPerGrid == 0) {
      return;
    }
    if (blocksPerGrid > 65535) {
      blocksPerGrid = 65535;
    }
    reverse_array_float4<<<blocksPerGrid, threadsPerBlock>>>(input, N);
  } else {
    int blocksPerGrid = (half + threadsPerBlock - 1) / threadsPerBlock;
    if (blocksPerGrid == 0) {
      return;
    }
    if (blocksPerGrid > 65535) {
      blocksPerGrid = 65535;
    }
    reverse_array_scalar<<<blocksPerGrid, threadsPerBlock>>>(input, N);
  }
  cudaDeviceSynchronize();
}