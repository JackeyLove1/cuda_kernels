#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cub/cub.cuh>
#include <numeric>
#include <random>
#include <type_traits>
#include <vector>

#define CEIL(a, b) (((a) + (b) - 1) / (b))

template <typename>
struct dependent_false : std::false_type {};

template <typename T>
__forceinline__ __device__ auto LOAD4(const T* ptr) {
  if constexpr (std::is_same_v<T, int>) {
    return reinterpret_cast<const int4*>(ptr);
  } else if constexpr (std::is_same_v<T, float>) {
    return reinterpret_cast<const float4*>(ptr);
  } else if constexpr (std::is_same_v<T, double>) {
    return reinterpret_cast<const double4*>(ptr);
  } else if constexpr (std::is_same_v<T, short>) {
    return reinterpret_cast<const short4*>(ptr);
  } else {
    static_assert(dependent_false<T>::value, "Unsupported type for LOAD4");
    return nullptr;
  }
}

template <typename T>
__forceinline__ __device__ auto STORE4(T* ptr) {
  if constexpr (std::is_same_v<T, int>) {
    return reinterpret_cast<int4*>(ptr);
  } else if constexpr (std::is_same_v<T, float>) {
    return reinterpret_cast<float4*>(ptr);
  } else if constexpr (std::is_same_v<T, double>) {
    return reinterpret_cast<double4*>(ptr);
  } else if constexpr (std::is_same_v<T, short>) {
    return reinterpret_cast<short4*>(ptr);
  } else {
    static_assert(dependent_false<T>::value, "Unsupported type for STORE4");
    return nullptr;
  }
}

__forceinline__ __device__ __host__ float4 operator+(const float4 a, const float4 b) {
  return make_float4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w);
}

static inline __device__ float atomicMax(float* addr, float value) {
  float old = *addr, assumed;
  if (old >= value) return old;
  do {
    assumed = old;
    old = atomicCAS((unsigned int*)addr, __float_as_int(assumed), __float_as_int(value));

  } while (old != assumed);

  return old;
}

template <typename T>
struct ReduceOp {
  static constexpr char op_id = 0;
};

template <typename T>
struct SumOp : public ReduceOp<T> {
  static constexpr char op_id = 1;

  __forceinline__ __device__ static T apply(const T& a, const T& b) { return a + b; }

  __forceinline__ __device__ static constexpr auto identity() { return T{0}; }
};

template <typename T>
struct MaxOp : public ReduceOp<T> {
  static constexpr char op_id = 2;

  __forceinline__ __device__ static T apply(const T& a, const T& b) {
    if constexpr (std::is_same_v<T, float>) {
      return fmaxf(a, b);
    } else {
      return max(a, b);
    }
  }

  __forceinline__ __device__ static constexpr auto identity() {
    if constexpr (std::is_same_v<T, float>) {
      return -INFINITY;
    } else {
      return -HUGE_VAL;
    }
  }
};

static constexpr unsigned int FULL_MASK = 0xffffffff;

template <typename T, typename Op>
__forceinline__ __device__ T WarpReduceOp(T value) {
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    value = Op::apply(value, __shfl_down_sync(FULL_MASK, value, offset));
  }
  return value;
}

template <typename T, typename Op>
inline __device__ T BlockReduceOp(T value) {
  const auto tid = threadIdx.x;
  const auto lane_id = tid & 31;
  const auto warp_id = tid >> 5;

  // warp reduce
  value = WarpReduceOp<T, Op>(value);

  __shared__ T shared_data[32];
  if (lane_id == 0) {
    shared_data[warp_id] = value;
  }
  __syncthreads();

  // block reduce
  if (warp_id == 0) {
    const int nums_warps = blockDim.x >> 5;
    const T lane_value = (lane_id < nums_warps) ? shared_data[lane_id] : Op::identity();
    value = WarpReduceOp<T, Op>(lane_value);
  }
  return value;
}

enum class ReduceOpType {
  SUM = 1,
  MAX = 2,
};

template <typename T>
__forceinline__ __device__ T BlockReduceDynamic(T value, ReduceOpType op_type) {
  switch (op_type) {
    case ReduceOpType::SUM:
      return BlockReduceOp<T, SumOp<T>>(value);
    case ReduceOpType::MAX:
      return BlockReduceOp<T, MaxOp<T>>(value);
    default:
      // Runtime-dispatched op type: keep a runtime fallback instead of
      // compile-time static_assert in a potentially unreachable branch.
      return value;
  }
}

static constexpr int PER_THREAD_WORK_ITEMS = 1;
static constexpr int WARP_SIZE = 32;
static constexpr int MAX_BLOCKS = 4096;
static constexpr int THREADS_PER_BLOCK = 256;

#define CHECK_CUDA(call)                                                                           \
  do {                                                                                             \
    cudaError_t err__ = (call);                                                                    \
    if (err__ != cudaSuccess) {                                                                    \
      fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err__)); \
      exit(1);                                                                                     \
    }                                                                                              \
  } while (0)

/**
Implement a program that computes the sum of a subarray of 32-bit integers. You are given an input
array input of length N, and two indices S and E. S and E are inclusive, 0-based start and end
indices— compute the sum of input[S...E].
 */

void __global__ subarray_sum_kernel(const int* __restrict__ input, int* __restrict__ output,
                                    const int N, const int M, const int S_ROW, const int E_ROW,
                                    const int S_COL, const int E_COL) {
  const int tx = threadIdx.x, ty = threadIdx.y;
  const int row_id = blockIdx.y * blockDim.y + ty;
  const int col_id = blockIdx.x * blockDim.x + tx;
  // if (row_id >= N || col_id >= M) return;
  if (row_id >= S_ROW && row_id <= E_ROW && col_id >= S_COL && col_id <= E_COL) {
    atomicAdd(output, input[row_id * M + col_id]);
  }
}

// A, B, and C are device pointers
extern "C" void solve(const int* input, int* output, int N, int M, int S_ROW, int E_ROW, int S_COL,
                      int E_COL) {
  CHECK_CUDA(cudaMemset(output, 0, sizeof(int)));
  constexpr int BLOCK_X = 16;
  constexpr int BLOCK_Y = 16;
  dim3 threadsPerBlock(BLOCK_X, BLOCK_Y);
  dim3 blocksPerGrid(CEIL(M, BLOCK_X), CEIL(N, BLOCK_Y));

  subarray_sum_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, output, N, M, S_ROW, E_ROW, S_COL,
                                                          E_COL);
  CHECK_CUDA(cudaGetLastError());
  CHECK_CUDA(cudaDeviceSynchronize());
}

static int cpu_subarray_sum(const thrust::host_vector<int>& input, int N, int M, int S_ROW,
                            int E_ROW, int S_COL, int E_COL) {
  int ans = 0;
  for (int r = S_ROW; r <= E_ROW; ++r) {
    for (int c = S_COL; c <= E_COL; ++c) {
      ans += input[r * M + c];
    }
  }
  return ans;
}

int main(int argc, char** argv) {
  int N = 2;
  int M = 3;
  if (argc >= 3) {
    N = atoi(argv[1]);
    M = atoi(argv[2]);
  }

  printf("=== Subarray 2D Sum Unit Test & Benchmark ===\n");
  printf("Shape: N=%d, M=%d\n", N, M);

  thrust::host_vector<int> h_input(N * M);
  std::mt19937 rng(42);
  std::uniform_int_distribution<int> dist(-10, 10);
  for (int i = 0; i < N * M; ++i) h_input[i] = dist(rng);

  thrust::device_vector<int> d_input = h_input;
  thrust::device_vector<int> d_output(1);
  thrust::host_vector<int> h_output(1);

  bool pass = true;
  int mismatches = 0;
  constexpr int MAX_PRINT_MISMATCH = 10;

  for (int s_row = 0; s_row < N; ++s_row) {
    for (int e_row = s_row; e_row < N; ++e_row) {
      for (int s_col = 0; s_col < M; ++s_col) {
        for (int e_col = s_col; e_col < M; ++e_col) {
          solve(thrust::raw_pointer_cast(d_input.data()), thrust::raw_pointer_cast(d_output.data()),
                N, M, s_row, e_row, s_col, e_col);
          h_output = d_output;
          const int gpu_val = h_output[0];
          const int cpu_val = cpu_subarray_sum(h_input, N, M, s_row, e_row, s_col, e_col);
          if (gpu_val != cpu_val) {
            pass = false;
            if (mismatches < MAX_PRINT_MISMATCH) {
              printf("Mismatch #%d: [%d:%d, %d:%d] gpu=%d cpu=%d\n", mismatches + 1, s_row, e_row,
                     s_col, e_col, gpu_val, cpu_val);
            }
            ++mismatches;
          }
        }
      }
    }
  }

  printf("Correctness: %s (mismatches=%d)\n\n", pass ? "PASS" : "FAIL", mismatches);

  const int S_ROW = 0;
  const int E_ROW = N - 1;
  const int S_COL = 0;
  const int E_COL = M - 1;

  cudaEvent_t t0, t1;
  CHECK_CUDA(cudaEventCreate(&t0));
  CHECK_CUDA(cudaEventCreate(&t1));

  constexpr int WARMUP = 10;
  constexpr int NITER = 1000;
  for (int i = 0; i < WARMUP; ++i) {
    solve(thrust::raw_pointer_cast(d_input.data()), thrust::raw_pointer_cast(d_output.data()), N, M,
          S_ROW, E_ROW, S_COL, E_COL);
  }

  CHECK_CUDA(cudaEventRecord(t0));
  for (int i = 0; i < NITER; ++i) {
    solve(thrust::raw_pointer_cast(d_input.data()), thrust::raw_pointer_cast(d_output.data()), N, M,
          S_ROW, E_ROW, S_COL, E_COL);
  }
  CHECK_CUDA(cudaEventRecord(t1));
  CHECK_CUDA(cudaEventSynchronize(t1));

  float total_ms = 0.0f;
  CHECK_CUDA(cudaEventElapsedTime(&total_ms, t0, t1));
  const float avg_ms = total_ms / static_cast<float>(NITER);

  const double elements = static_cast<double>(N) * static_cast<double>(M);
  const double bytes_moved = elements * sizeof(int) + sizeof(int);
  const double gelem_per_sec = elements / (avg_ms * 1e-3) / 1e9;
  const double bandwidth_gbs = bytes_moved / (avg_ms * 1e-3) / 1e9;

  printf("=== Performance (%d runs) ===\n", NITER);
  printf("  Avg latency : %.6f ms\n", avg_ms);
  printf("  Throughput  : %.6f Gelem/s\n", gelem_per_sec);
  printf("  Bandwidth   : %.6f GB/s\n", bandwidth_gbs);

  CHECK_CUDA(cudaEventDestroy(t0));
  CHECK_CUDA(cudaEventDestroy(t1));
  return pass ? 0 : 1;
}
