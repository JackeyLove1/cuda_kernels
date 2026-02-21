#include <cuda_runtime.h>
#include <math.h>
#include <type_traits>

#define CEIL(a, b) (((a) + (b) - 1) / (b))

template <typename>
struct dependent_false : std::false_type {};

template <typename T>
__forceinline__ __device__ T sigmoid(const T& x) {
  if constexpr (std::is_same_v<T, float>) {
    return 1.0f / (1.0f + __expf(-x));
  } else if constexpr (std::is_same_v<T, float4>) {
    float4 result;
    result.x = sigmoid(x.x);
    result.y = sigmoid(x.y);
    result.z = sigmoid(x.z);
    result.w = sigmoid(x.w);
    return result;
  } else {
    static_assert(dependent_false<T>::value, "Unsupported type");
    return T{};
  }
}

#define CFACTOR 4  // thread corase

__global__ void sigmoid_kernel(const float* __restrict__ X, float* __restrict__ Y, const int N) {
  const int gid = blockDim.x * blockIdx.x + threadIdx.x;
  const float4* X4 = reinterpret_cast<const float4*>(X);
  float4* Y4 = reinterpret_cast<float4*>(Y);
  const int total_threads = blockDim.x * gridDim.x;
  const int vec_size = N / 4;   // number of float4 elements
  const int vec_tail = N % 4;   // scalar tail after float4 vectorization
  const int stride = total_threads * CFACTOR;

  for (int i = gid * CFACTOR; i < vec_size; i += stride) {
#pragma unroll
    for (int j = 0; j < CFACTOR; ++j) {
      const int idx4 = i + j;
      if (idx4 < vec_size) {
        Y4[idx4] = sigmoid(X4[idx4]);
      }
    }
  }

  if (vec_tail && gid < vec_tail) {
    const int idx = vec_size * 4 + gid;
    Y[idx] = sigmoid(X[idx]);
  }
}

// X, Y are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* __restrict__ X, float* __restrict__ Y, const int N) {
  constexpr int threadsPerBlock = 256;
  const int blocksPerGrid = CEIL(N, threadsPerBlock * (4 * CFACTOR));
  sigmoid_kernel<<<blocksPerGrid, threadsPerBlock>>>(X, Y, N);
  // cudaDeviceSynchronize();
}
