#pragma once

#include <include/vec_dtypes.cuh>
#include <include/cp_async.cuh>
#include <include/math.cuh>
#include <include/state.cuh>

#ifndef CUDA_EXAMPLES_UTILS_CUH
#define CUDA_EXAMPLES_UTILS_CUH

#define BLOCK_SIZE 256
#define BLOCK_THREADS 256
#define ITEMS_PER_THREAD 4
#define WARP_SIZE 32

#ifndef DEVICE_INLINE
#define DEVICE_INLINE __device__ __forceinline__
#endif

// Define CHECK macro if not already defined
#ifndef CHECK
#define CHECK(call)                                                            \
  do {                                                                         \
    cudaError_t err = call;                                                    \
    if (err != cudaSuccess) {                                                  \
      fprintf(stderr, "CUDA error at %s:%d - %s\n", __FILE__, __LINE__,        \
              cudaGetErrorString(err));                                        \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  } while (0)
#endif

#ifndef CHECK_CUDA
#define CHECK_CUDA(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA Error: %s at line %d\n", cudaGetErrorString(err), __LINE__); \
            exit(1); \
        } \
    } while (0)
#endif

constexpr static auto FULL_MASK = 0xFFFFFFFF;

template <typename T>
__forceinline__ __device__ T WarpReduceSum(T val) {
#pragma unroll
  for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
    val += __shfl_down_sync(FULL_MASK, val, offset);
  }
  return val;
}

template <typename T, typename U>
static __forceinline__ __device__ __host__ auto CEIL_DIV(const T a, const U b) {
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
#endif  // CUDA_EXAMPLES_UTILS_CUH

#define STRINGFY(str) #str
#define TORCH_BINDING_COMMON_EXTENSION(func)                                   \
    m.def(STRINGFY(func), &func, STRINGFY(func));