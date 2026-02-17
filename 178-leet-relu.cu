#include <cuda_runtime.h>

#define WARP_SIZE 32
#define FLOAT4(value) (reinterpret_cast<float4*>(&(value))[0])

__global__ __launch_bounds__(256) void relu_kernel(float* __restrict__ input,
                                                   float* __restrict__ output, const int N) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int vec_size = N / 4;  // Number of full float4 vectors
  const int vec_tail = N % 4;  // Remaining scalar elements

  // Vectorized main path.
  if (idx < vec_size) {
    const int stride_idx = idx * 4;
    const float4 x = FLOAT4(input[stride_idx]);
    float4 y;
    y.x = fmaxf(0.0f, x.x);
    y.y = fmaxf(0.0f, x.y);
    y.z = fmaxf(0.0f, x.z);
    y.w = fmaxf(0.0f, x.w);
    FLOAT4(output[stride_idx]) = y;
  }

  // Scalar tail path (at most 3 elements).
  if (idx < vec_tail) {
    const int tail_idx = vec_size * 4 + idx;
    output[tail_idx] = fmaxf(0.0f, input[tail_idx]);
  }
}

// input, output are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(float* input, float* output, int N) {
  if (N <= 0) return;

  int threadsPerBlock = 256;
  int vec_size = N / 4;
  int vec_tail = N % 4;
  int work_items = vec_size > vec_tail ? vec_size : vec_tail;
  int blocksPerGrid = (work_items + threadsPerBlock - 1) / threadsPerBlock;

  relu_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, output, N);
}