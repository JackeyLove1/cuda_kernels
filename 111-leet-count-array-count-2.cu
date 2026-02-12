#include <cuda.h>
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cub/cub.cuh>
#include <cuda/pipeline>
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

static constexpr unsigned int FULL_MASK = 0xffffffff;

template <typename T>
__forceinline__ __device__ T WarpReduceSum(T value) {
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    value += __shfl_down_sync(FULL_MASK, value, offset);
  }
  return value;
}

static constexpr int PER_THREAD_WORK_ITEMS = 2;
static constexpr int WARP_SIZE = 32;
static constexpr int MAX_BLOCKS = 1024;
static constexpr int THREADS_PER_BLOCK = 512;
/**
Write a GPU program that counts the number of elements with the integer value k in an array of
32-bit integers. The program should count the number of elements with k in an array. You are given
an input array input of length N and integer k.
 */
__global__ void count_equal_kernel(const int* __restrict__ input, int* __restrict__ output,
                                   const int N, const int K) {
  __shared__ int shared_count[THREADS_PER_BLOCK];
  const int tid = threadIdx.x;
  const int gid = blockIdx.x * THREADS_PER_BLOCK + tid;
  const int stride = gridDim.x * THREADS_PER_BLOCK * PER_THREAD_WORK_ITEMS;
  const int total = N;
  const int vec4_size = total / 4;
  const int vec_tail = total % 4;
  const int4* input4 = LOAD4(input);

  int thread_count = 0;
  for (int i = gid * PER_THREAD_WORK_ITEMS; i < vec4_size; i += stride) {
#pragma unroll
    for (int j = 0; j < PER_THREAD_WORK_ITEMS; ++j) {
      const int idx4 = i + j;
      if (idx4 < vec4_size) {
        const int4 value = input4[idx4];
        thread_count += (value.x == K) + (value.y == K) + (value.z == K) + (value.w == K);
      }
    }
  }
  if (vec_tail && gid < vec_tail) {
    const int base = (vec4_size * 4) + gid;
    thread_count += (input[base] == K);
  }
  shared_count[tid] = thread_count;
  __syncthreads();

  const int warp_id = tid >> 5;
  const int lane_id = tid & 31;
  const int warp_count = WarpReduceSum(thread_count);
  if (lane_id == 0) {
    shared_count[warp_id] = warp_count;
  }
  __syncthreads();

  if (warp_id == 0) {
    constexpr int WARP_COUNT = THREADS_PER_BLOCK / WARP_SIZE;
    const auto lane_value = (lane_id < WARP_COUNT) ? shared_count[lane_id] : 0;
    const int block_count = WarpReduceSum(lane_value);
    if (tid == 0) {
      atomicAdd(output, block_count);
    }
  }
}

// input, output are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const int* input, int* output, int N, int K) {
  int threadsPerBlock = THREADS_PER_BLOCK;
  int blocksPerGrid = std::min(CEIL(N, threadsPerBlock * 4 * PER_THREAD_WORK_ITEMS), MAX_BLOCKS);

  count_equal_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, output, N, K);
  cudaDeviceSynchronize();
}

int cpu_count_equal(const thrust::host_vector<int>& input, int K) {
  int count = 0;
  for (int i = 0; i < static_cast<int>(input.size()); ++i) {
    count += (input[i] == K);
  }
  return count;
}

int main(int argc, char** argv) {
  int K = 501010;
  int N = 100000000;
  if (argc > 1) {
    K = std::atoi(argv[1]);
  }
  if (argc > 2) {
    N = std::atoi(argv[2]);
  }

  if (N <= 0) {
    std::fprintf(stderr, "N must be > 0, got %d\n", N);
    return 1;
  }

  std::printf("=== Count Equal Kernel Test & Benchmark ===\n");
  std::printf("N = %d, K = %d\n", N, K);
  std::printf("Input size: %.2f MiB\n", static_cast<double>(N * sizeof(int)) / (1 << 20));

  thrust::host_vector<int> h_input(N);
  thrust::device_vector<int> d_input(N);
  thrust::device_vector<int> d_output(1, 0);

  std::mt19937 rng(42);
  std::uniform_int_distribution<int> dist(0, 1000000);
  for (int i = 0; i < N; ++i) {
    h_input[i] = dist(rng);
  }
  d_input = h_input;

  d_output[0] = 0;
  solve(thrust::raw_pointer_cast(d_input.data()), thrust::raw_pointer_cast(d_output.data()), N, K);
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    std::fprintf(stderr, "CUDA kernel launch failed: %s\n", cudaGetErrorString(err));
    return 1;
  }

  thrust::host_vector<int> h_output = d_output;
  const int gpu_count = h_output[0];
  const int cpu_count = cpu_count_equal(h_input, K);
  const bool pass = (gpu_count == cpu_count);
  std::printf("Correctness: %s (gpu=%d, cpu=%d)\n\n", pass ? "PASS" : "FAIL", gpu_count, cpu_count);

  cudaEvent_t t0, t1;
  cudaEventCreate(&t0);
  cudaEventCreate(&t1);

  constexpr int WARMUP = 10;
  constexpr int NITER = 100;
  for (int i = 0; i < WARMUP; ++i) {
    d_output[0] = 0;
    solve(thrust::raw_pointer_cast(d_input.data()), thrust::raw_pointer_cast(d_output.data()), N, K);
  }
  cudaDeviceSynchronize();

  cudaEventRecord(t0);
  for (int i = 0; i < NITER; ++i) {
    d_output[0] = 0;
    solve(thrust::raw_pointer_cast(d_input.data()), thrust::raw_pointer_cast(d_output.data()), N, K);
  }
  cudaEventRecord(t1);
  cudaEventSynchronize(t1);

  float total_ms = 0.f;
  cudaEventElapsedTime(&total_ms, t0, t1);
  const float avg_ms = total_ms / NITER;
  const double bytes_moved = static_cast<double>(N) * sizeof(int);
  const double bandwidth_gb = bytes_moved / (avg_ms * 1e-3) / 1e9;
  const double elem_per_sec = static_cast<double>(N) / (avg_ms * 1e-3);

  std::printf("=== Performance (%d runs) ===\n", NITER);
  std::printf("  Avg latency : %.4f ms\n", avg_ms);
  std::printf("  Throughput  : %.2f Gelem/s\n", elem_per_sec / 1e9);
  std::printf("  Bandwidth   : %.2f GB/s\n", bandwidth_gb);

  cudaEventDestroy(t0);
  cudaEventDestroy(t1);
  return pass ? 0 : 1;
}
