#include <cuda.h>
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <type_traits>

/**
Implement a program that performs the Rectified Linear Unit (ReLU) activation function on a vector
of 32-bit floating point numbers. The ReLU function sets all negative values to zero and leaves
positive values unchanged: ReLU(x)=max(0,x) Implementation Requirements ·External libraries are not
permitted ·The solve function signature must remain unchanged ·The final result must be stored in
output
 */
#define CEIL(a, b) (((a) + (b) - 1) / (b))
#define LOAD4(ptr) (reinterpret_cast<const float4*>(ptr))
#define STORE4(ptr) (reinterpret_cast<float4*>(ptr))

#define WORKS 1  // thread works per iteration

__forceinline__ __device__ float4 operator+(const float4 a, const float4 b) {
  return make_float4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w);
}

static constexpr float alpha = 0.01f;

template <typename T>
struct dependent_false : std::false_type {};

template <typename T>
__forceinline__ __device__ T _leaky_relu_op(const T x) {
  if constexpr (std::is_same_v<T, float>) {
    return x > 0.0f ? x : x * alpha;
  } else if constexpr (std::is_same_v<T, float4>) {
    return make_float4(_leaky_relu_op(x.x), _leaky_relu_op(x.y), _leaky_relu_op(x.z),
                       _leaky_relu_op(x.w));
  } else {
    static_assert(dependent_false<T>::value, "Unsupported type");
  }
}
__global__ __launch_bounds__(256) void leaky_relu_kernel(const float* __restrict__ input,
                                                         float* __restrict__ output, const int N) {
  const auto tid = threadIdx.x;
  const auto gid = blockIdx.x * blockDim.x + tid;
  const auto stride = blockDim.x * gridDim.x;
  const float4* in4 = LOAD4(input);
  float4* out4 = STORE4(output);
  const int total = N >> 2;

#pragma unroll
  for (int i = gid; i < total; i += stride) {
    float4 in = in4[i];
    float4 out = _leaky_relu_op(in);
    out4[i] = out;
  }

  const auto tail_start = total * 4;
  for (int i = tail_start + gid; i < N; i += stride) {
    output[i] = _leaky_relu_op(input[i]);
  }
}

// input, output are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* input, float* output, int N) {
  int threadsPerBlock = 256;
  int blocksPerGrid = CEIL(N, threadsPerBlock * 4);

  leaky_relu_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, output, N);
  cudaDeviceSynchronize();
}
