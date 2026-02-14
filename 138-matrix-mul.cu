#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/sort.h>

#include <algorithm>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cub/cub.cuh>
#include <cuda/pipeline>
#include <limits>
#include <random>
#include <type_traits>

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
      return (a > b) ? a : b;
    }
  }

  __forceinline__ __device__ static constexpr auto identity() {
    if constexpr (std::is_same_v<T, float>) {
      return -INFINITY;
    } else if constexpr (std::is_same_v<T, half>) {
      return -65504;
    } else {
      return std::numeric_limits<T>::lowest();
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

[[maybe_unused]] static constexpr int CFACTOR = 1;
[[maybe_unused]] static constexpr int WARP_SIZE = 32;
[[maybe_unused]] static constexpr int MAX_BLOCKS = 4096;
[[maybe_unused]] static constexpr int THREADS_PER_BLOCK = 512;

static constexpr int TILE = 32;

/**
Write a program that multiplies two matrices of 32-bit floating point numbers on a GPU. Given matrix
A of dimensions M×N and matrix B of dimensions N×K, compute the product matrix C=A×B, which will
have dimensions M×K. All matrices are stored in row-major format.
 */
__global__ void matrix_multiplication_kernel(const float* __restrict__ A,
                                             const float* __restrict__ B, float* __restrict__ C,
                                             const int M, const int N, const int K) {
  static constexpr int kStage = 2;
  __shared__ __align__(16) float sh_A[kStage][TILE][TILE + 1];
  __shared__ __align__(16) float sh_B[kStage][TILE][TILE + 1];
  const int tx = threadIdx.x;
  const int ty = threadIdx.y;
  const int bx = blockIdx.x;
  const int by = blockIdx.y;
  const int col = bx * TILE + tx;
  const int row = by * TILE + ty;
  float sum{0.f};
  const auto num_tils = CEIL(N, TILE);

  cuda::pipeline<cuda::thread_scope_thread> pipe = cuda::make_pipeline();

  pipe.producer_acquire();
  if (row < M && tx < N) {
    cuda::memcpy_async(&sh_A[0][ty][tx], &A[row * N + tx], sizeof(float), pipe);
  } else {
    sh_A[0][ty][tx] = 0.f;
  }
  if (col < K && ty < N) {
    cuda::memcpy_async(&sh_B[0][ty][tx], &B[ty * K + col], sizeof(float), pipe);
  } else {
    sh_B[0][ty][tx] = 0.f;
  }
  pipe.producer_commit();

  for (int k_tile = 0; k_tile < num_tils; ++k_tile) {
    if (k_tile + 1 < num_tils) {
      pipe.producer_acquire();
      auto next_k = (k_tile + 1);
      auto write_stage = next_k % kStage;

      if (row < M && (next_k * TILE + tx) < N) {
        cuda::memcpy_async(&sh_A[write_stage][ty][tx], &A[row * N + (next_k * TILE + tx)],
                           sizeof(float), pipe);
      } else {
        sh_A[write_stage][ty][tx] = 0.f;
      }

      if (col < K && (next_k * TILE + ty) < N) {
        cuda::memcpy_async(&sh_B[write_stage][ty][tx], &B[(next_k * TILE + ty) * K + col],
                           sizeof(float), pipe);
      } else {
        sh_B[write_stage][ty][tx] = 0.f;
      }
      pipe.producer_commit();
    }

    pipe.consumer_wait();
    __syncthreads();

    auto read_stage = k_tile % kStage;

#pragma unroll 4
    for (int k = 0; k < TILE; ++k) {
      sum = fmaf(sh_A[read_stage][ty][k], sh_B[read_stage][k][tx], sum);
    }

    pipe.consumer_release();
    __syncthreads();
  }
  if (col < K && row < M) {
    C[row * K + col] = sum;
  }
}

// A, B, C are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* A, const float* B, float* C, int M, int N, int K) {
  dim3 threadsPerBlock(TILE, TILE);
  dim3 blocksPerGrid(CEIL(K, TILE), CEIL(M, TILE));

  matrix_multiplication_kernel<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, M, N, K);
  cudaDeviceSynchronize();
}

static float matmul_cpu_reference_at(const float* A, const float* B, int row, int col, int N,
                                     int K) {
  float sum = 0.0f;
  for (int i = 0; i < N; ++i) {
    sum += A[row * N + i] * B[i * K + col];
  }
  return sum;
}

static bool check_correctness_sampled(const thrust::host_vector<float>& gpu_result, const float* A,
                                      const float* B, int M, int N, int K, float rtol = 1e-4f,
                                      float atol = 1e-5f) {
  constexpr int SAMPLE_COUNT = 100;
  const int total_elems = M * K;
  const int check_count = std::min(SAMPLE_COUNT, total_elems);

  std::mt19937 rng(42);
  std::uniform_int_distribution<int> row_dist(0, M - 1);
  std::uniform_int_distribution<int> col_dist(0, K - 1);

  int errors = 0;
  constexpr int max_errors = 10;
  float max_diff = 0.0f;

  if (total_elems <= SAMPLE_COUNT) {
    for (int row = 0; row < M; ++row) {
      for (int col = 0; col < K; ++col) {
        const int idx = row * K + col;
        const float g = gpu_result[idx];
        const float c = matmul_cpu_reference_at(A, B, row, col, N, K);
        const float diff = fabsf(g - c);
        const float ref = fabsf(c);
        max_diff = fmaxf(max_diff, diff);
        if (diff > (atol + rtol * ref)) {
          if (++errors <= max_errors) {
            printf("  MISMATCH at [row=%d,col=%d]: gpu=%.6f cpu=%.6f diff=%.6e\n", row, col, g, c,
                   diff);
          }
        }
      }
    }
  } else {
    for (int k = 0; k < check_count; ++k) {
      const int row = row_dist(rng);
      const int col = col_dist(rng);
      const int idx = row * K + col;
      const float g = gpu_result[idx];
      const float c = matmul_cpu_reference_at(A, B, row, col, N, K);
      const float diff = fabsf(g - c);
      const float ref = fabsf(c);
      max_diff = fmaxf(max_diff, diff);
      if (diff > (atol + rtol * ref)) {
        if (++errors <= max_errors) {
          printf("  MISMATCH at [row=%d,col=%d]: gpu=%.6f cpu=%.6f diff=%.6e\n", row, col, g, c,
                 diff);
        }
      }
    }
  }

  printf("  Checked %d / %d elements (random sample), max_diff=%.6e, errors=%d\n", check_count,
         total_elems, max_diff, errors);
  return errors == 0;
}

int main(int argc, char** argv) {
  int M = 4096 * 2;
  int N = 2048;
  int K = 4096;
  if (argc == 4) {
    M = atoi(argv[1]);
    N = atoi(argv[2]);
    K = atoi(argv[3]);
  } else if (argc != 1) {
    printf("Usage: %s [M N K]\n", argv[0]);
    return 1;
  }

  const int size_a = M * N;
  const int size_b = N * K;
  const int size_c = M * K;
  printf("=== Matrix Multiplication Unit Test & Benchmark ===\n");
  printf("M=%d, N=%d, K=%d\n", M, N, K);
  printf("A: %d elems, B: %d elems, C: %d elems\n", size_a, size_b, size_c);

  thrust::host_vector<float> h_A(size_a);
  thrust::host_vector<float> h_B(size_b);
  thrust::device_vector<float> d_A(size_a);
  thrust::device_vector<float> d_B(size_b);
  thrust::device_vector<float> d_C(size_c);

  std::mt19937 rng(42);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  for (int i = 0; i < size_a; ++i) h_A[i] = dist(rng);
  for (int i = 0; i < size_b; ++i) h_B[i] = dist(rng);
  d_A = h_A;
  d_B = h_B;

  solve(thrust::raw_pointer_cast(d_A.data()), thrust::raw_pointer_cast(d_B.data()),
        thrust::raw_pointer_cast(d_C.data()), M, N, K);
  cudaDeviceSynchronize();

  thrust::host_vector<float> h_gpu_C = d_C;
  const bool pass = check_correctness_sampled(h_gpu_C, h_A.data(), h_B.data(), M, N, K);
  printf("Correctness: %s\n\n", pass ? "PASS" : "FAIL");

  cudaEvent_t t0, t1;
  cudaEventCreate(&t0);
  cudaEventCreate(&t1);

  constexpr int WARMUP = 10;
  for (int i = 0; i < WARMUP; ++i) {
    solve(thrust::raw_pointer_cast(d_A.data()), thrust::raw_pointer_cast(d_B.data()),
          thrust::raw_pointer_cast(d_C.data()), M, N, K);
  }
  cudaDeviceSynchronize();

  constexpr int NITER = 100;
  cudaEventRecord(t0);
  for (int i = 0; i < NITER; ++i) {
    solve(thrust::raw_pointer_cast(d_A.data()), thrust::raw_pointer_cast(d_B.data()),
          thrust::raw_pointer_cast(d_C.data()), M, N, K);
  }
  cudaEventRecord(t1);
  cudaEventSynchronize(t1);

  float total_ms = 0.0f;
  cudaEventElapsedTime(&total_ms, t0, t1);
  const float avg_ms = total_ms / NITER;

  const double bytes_moved = static_cast<double>(size_a + size_b + size_c) * sizeof(float);
  const double bandwidth_gb = bytes_moved / (avg_ms * 1e-3) / 1e9;
  const double outputs_per_sec = static_cast<double>(size_c) / (avg_ms * 1e-3);
  const double tflops = (2.0 * static_cast<double>(M) * N * K) / (avg_ms * 1e-3) / 1e12;

  printf("=== Performance (%d runs) ===\n", NITER);
  printf("  Avg latency : %.4f ms\n", avg_ms);
  printf("  Throughput  : %.2f Gelem/s\n", outputs_per_sec / 1e9);
  printf("  Bandwidth   : %.2f GB/s\n", bandwidth_gb);
  printf("  Compute     : %.2f TFLOPS\n", tflops);

  cudaEventDestroy(t0);
  cudaEventDestroy(t1);
  return pass ? 0 : 1;
}
