#include <cuda_runtime.h>

__global__ void clip_kernel(const float* __restrict__ input, float* __restrict__ output, float lo, float hi, int N) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;

  int n_vec = N / 4;
  const float4* input4 = reinterpret_cast<const float4*>(input);
  float4* output4 = reinterpret_cast<float4*>(output);

  // grid-stride: 每个线程处理多个float4向量，提高单线程工作量
  for (int id = tid; id < n_vec; id += stride) {
    float4 data = input4[id];
    data.x = fminf(fmaxf(data.x, lo), hi);
    data.y = fminf(fmaxf(data.y, lo), hi);
    data.z = fminf(fmaxf(data.z, lo), hi);
    data.w = fminf(fmaxf(data.w, lo), hi);
    output4[id] = data;
  }

  // 处理尾部不足4个的元素
  int tail_start = n_vec * 4;
  for (int i = tail_start + tid; i < N; i += stride) {
    output[i] = fminf(fmaxf(input[i], lo), hi);
  }
}

// input, output are device pointers
extern "C" void solve(const float* input, float* output, float lo, float hi, int N) {
  if (N <= 0) return;

  int threadsPerBlock = 256;
  int n_work = (N + 3) / 4;  // 按float4工作量估算grid大小
  int blocksPerGrid = (n_work + threadsPerBlock - 1) / threadsPerBlock;

  clip_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, output, lo, hi, N);
  cudaDeviceSynchronize();
}
