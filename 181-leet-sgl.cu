#include <cuda_runtime.h>

__device__ __forceinline__ float silu(float x) { return x * (1 / (1 + __expf(-x))); }
__device__ __forceinline__ float4 silu(float4 x) {
  return make_float4(silu(x.x), silu(x.y), silu(x.z), silu(x.w));
}

__global__ void swiglu_kernel(const float* input, float* output, int halfN) {
  const int gid = blockDim.x * blockIdx.x + threadIdx.x;
  const int stride = blockDim.x * gridDim.x;
  const int total4 = halfN >> 2;

  const float4* input1_4 = reinterpret_cast<const float4*>(input);
  const float4* input2_4 = reinterpret_cast<const float4*>(input + halfN);
  float4* output_4 = reinterpret_cast<float4*>(output);

  for (int i = gid; i < total4; i += stride) {
    float4 x1 = input1_4[i];
    float4 x2 = input2_4[i];
    float4 y = silu(x1);
    output_4[i] = make_float4(y.x * x2.x, y.y * x2.y, y.z * x2.z, y.w * x2.w);
  }

  const int tail_start = total4 * 4;
  for (int i = tail_start + gid; i < halfN; i += stride) {
    float x_1 = input[i];
    float x_2 = input[i + halfN];
    output[i] = silu(x_1) * x_2;
  }
}

// input, output are device pointers
extern "C" void solve(const float* input, float* output, int N) {
  int halfN = N / 2;
  int threadsPerBlock = 256;
  int blocksPerGrid = (halfN + threadsPerBlock * 4 - 1) / (threadsPerBlock * 4);
  if (blocksPerGrid == 0) blocksPerGrid = 1;

  swiglu_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, output, halfN);
  cudaDeviceSynchronize();
}