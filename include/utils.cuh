#pragma once

#ifndef CUDA_EXAMPLES_UTILS_CUH
#define CUDA_EXAMPLES_UTILS_CUH

#define BLOCK_THREADS 256
#define ITEMS_PER_THREAD 4

constexpr static auto FULL_MASK = 0xFFFFFFFF;

template <typename T>
__forceinline__ __device__ T warpReduceSum(T val) {
#pragma unroll
  for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
    val += __shfl_down_sync(FULL_MASK, val, offset);
  }
}

template <typename T, typename U>
static __forceinline__ auto ceil_div(const T a, const U b) {
  return static_cast<T>((a + b - 1) / b);
}

template <typename T, typename U>
__forceinline__ __host__ __device__ auto MIN(T a, U b) {
  return (a < static_cast<T>(b)) ? a : b;
}

template <typename T, typename U>
__forceinline__ __host__ __device__ auto MAX(T a, U b) {
  return (a < static_cast<T>(b)) ? b : a;
}

#define CUDA_CHECK(expr_to_check)                                                        \
  do {                                                                                   \
    cudaError_t result = expr_to_check;                                                  \
    if (result != cudaSuccess) {                                                         \
      fprintf(stderr, "CUDA Runtime Error: %s:%i:%d = %s\n", __FILE__, __LINE__, result, \
              cudaGetErrorString(result));                                               \
    }                                                                                    \
  } while (0)

#define CUDA_CALL(func, ...)                                                                \
  {                                                                                         \
    cudaError_t e = (func);                                                                 \
    if (e != cudaSuccess) {                                                                 \
      std::cerr << "CUDA Error: " << cudaGetErrorString(e) << " (" << e << ") " << __FILE__ \
                << ": line " << __LINE__ << " at function " << STR(func) << std::endl;      \
      return e;                                                                             \
    }                                                                                       \
  }
#else

#endif  // CUDA_EXAMPLES_UTILS_CUH