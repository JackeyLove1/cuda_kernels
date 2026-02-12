#include <cuda.h>
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <numeric>
#include <random>
#include <type_traits>
#include <vector>

/**
Implement a program that performs the Rectified Linear Unit (ReLU) activation function on a vector
of 32-bit floating point numbers. The ReLU function sets all negative values to zero and leaves
positive values unchanged: ReLU(x)=max(0,x) Implementation Requirements ·External libraries are not
permitted ·The solve function signature must remain unchanged ·The final result must be stored in
output
 */
#define CEIL(a, b) (((a) + (b) - 1) / (b))
#define LOAD4(ptr) (reinterpret_cast<const int4*>(ptr))
#define STORE4(ptr) (reinterpret_cast<int4*>(ptr))

#define WORKS 1  // thread works per iteration

__forceinline__ __device__ __host__ float4 operator+(const float4 a, const float4 b) {
  return make_float4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w);
}

// Fast FNV-1a for 32-bit word (treat input as raw 32-bit)
__forceinline__ __device__ __host__ unsigned int fnv1a_hash(unsigned int v) {
  const unsigned int FNV_PRIME = 16777619u;
  const unsigned int OFFSET_BASIS = 2166136261u;

  unsigned int h = OFFSET_BASIS;
  h = (h ^ (v & 0xFFu)) * FNV_PRIME;
  h = (h ^ ((v >> 8) & 0xFFu)) * FNV_PRIME;
  h = (h ^ ((v >> 16) & 0xFFu)) * FNV_PRIME;
  h = (h ^ ((v >> 24) & 0xFFu)) * FNV_PRIME;
  return h;
}

__device__ unsigned int _fnv1a_hash_rounds(unsigned int input, const int R) {
#pragma unroll
  for (int i = 0; i < R; i++) {
    input = fnv1a_hash(input);
  }
  return input;
}

__global__ __launch_bounds__(256, 2) void fnv1a_hash_kernel(const int* __restrict__ input,
                                                            unsigned int* __restrict__ output,
                                                            const int N, const int R) {
  const auto tid = threadIdx.x;
  const auto gid = blockIdx.x * blockDim.x + tid;
  const auto stride = blockDim.x * gridDim.x;
  const auto total = N >> 2;
  const int4* in4 = LOAD4(input);
  int4* out4 = STORE4(output);
  for (int i = gid; i < total; i += stride) {
    int4 in = in4[i];
    int4 out{};
    out.x = _fnv1a_hash_rounds(in.x, R);
    out.y = _fnv1a_hash_rounds(in.y, R);
    out.z = _fnv1a_hash_rounds(in.z, R);
    out.w = _fnv1a_hash_rounds(in.w, R);
    out4[i] = out;
  }
  int tail_start = total * 4;
  for (int i = tail_start + gid; i < N; i += stride) {
    output[i] = _fnv1a_hash_rounds(input[i], R);
  }
}

// input, output are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const int* input, unsigned int* output, int N, int R) {
  int threadsPerBlock = 256;
  // 限制grid大小以提高SM占用率和L2缓存效率
  int maxBlocks = CEIL(N, threadsPerBlock * 4);
  int blocksPerGrid = std::min(maxBlocks, 2048);  // 限制最大block数

  fnv1a_hash_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, output, N, R);
  cudaDeviceSynchronize();
}

inline unsigned int fnv1a_hash_host(unsigned int v) {
  const unsigned int FNV_PRIME = 16777619u;
  const unsigned int OFFSET_BASIS = 2166136261u;
  unsigned int h = OFFSET_BASIS;
  h = (h ^ (v & 0xFFu)) * FNV_PRIME;
  h = (h ^ ((v >> 8) & 0xFFu)) * FNV_PRIME;
  h = (h ^ ((v >> 16) & 0xFFu)) * FNV_PRIME;
  h = (h ^ ((v >> 24) & 0xFFu)) * FNV_PRIME;
  return h;
}

inline unsigned int fnv1a_hash_rounds_host(unsigned int input, int R) {
  for (int i = 0; i < R; ++i) {
    input = fnv1a_hash_host(input);
  }
  return input;
}

void cpu_reference(const int* input, unsigned int* output, int N, int R) {
  for (int i = 0; i < N; ++i) {
    output[i] = fnv1a_hash_rounds_host(static_cast<unsigned int>(input[i]), R);
  }
}

bool check_correctness(const thrust::host_vector<unsigned int>& gpu_result,
                       const thrust::host_vector<unsigned int>& cpu_result, int N) {
  constexpr int SAMPLE_THRESHOLD = 1 << 20;  // 1M
  constexpr int SAMPLE_COUNT = 8192;

  std::mt19937 rng(42);
  int check_count = N;
  std::vector<int> indices(N);
  std::iota(indices.begin(), indices.end(), 0);

  if (N > SAMPLE_THRESHOLD) {
    check_count = SAMPLE_COUNT;
    std::shuffle(indices.begin(), indices.end(), rng);
    indices.resize(SAMPLE_COUNT);
  }

  int max_errors = 10;
  int errors = 0;
  for (int k = 0; k < check_count; ++k) {
    int i = indices[k];
    if (gpu_result[i] != cpu_result[i]) {
      if (++errors <= max_errors) {
        printf("  MISMATCH at [%d]: gpu=%u cpu=%u\n", i, gpu_result[i], cpu_result[i]);
      }
    }
  }

  printf("  Checked %d / %d elements, errors=%d\n", check_count, N, errors);
  return errors == 0;
}

int main(int argc, char** argv) {
  int N = (argc > 1) ? std::atoi(argv[1]) : (1 << 24);
  int R = (argc > 2) ? std::atoi(argv[2]) : 16;

  printf("=== FNV1a Rainbow Table Kernel Unit Test & Benchmark ===\n");
  printf("N=%d, R=%d\n", N, R);
  printf("Input size: %.2f MB\n", static_cast<double>(N) * sizeof(int) / (1 << 20));

  thrust::host_vector<int> h_in(N);
  thrust::device_vector<int> d_in(N);
  thrust::device_vector<unsigned int> d_out(N);

  std::mt19937 rng(42);
  std::uniform_int_distribution<int> dist(-1000000, 1000000);
  for (int i = 0; i < N; ++i) h_in[i] = dist(rng);
  d_in = h_in;

  solve(thrust::raw_pointer_cast(d_in.data()), thrust::raw_pointer_cast(d_out.data()), N, R);
  cudaDeviceSynchronize();

  thrust::host_vector<unsigned int> h_gpu_out = d_out;
  thrust::host_vector<unsigned int> h_cpu_out(N);
  cpu_reference(h_in.data(), h_cpu_out.data(), N, R);

  bool pass = check_correctness(h_gpu_out, h_cpu_out, N);
  printf("Correctness: %s\n\n", pass ? "PASS" : "FAIL");

  cudaEvent_t t0, t1;
  cudaEventCreate(&t0);
  cudaEventCreate(&t1);

  constexpr int WARMUP = 10;
  for (int i = 0; i < WARMUP; ++i) {
    solve(thrust::raw_pointer_cast(d_in.data()), thrust::raw_pointer_cast(d_out.data()), N, R);
  }
  cudaDeviceSynchronize();

  constexpr int NITER = 100;
  cudaEventRecord(t0);
  for (int i = 0; i < NITER; ++i) {
    solve(thrust::raw_pointer_cast(d_in.data()), thrust::raw_pointer_cast(d_out.data()), N, R);
  }
  cudaEventRecord(t1);
  cudaEventSynchronize(t1);

  float total_ms = 0.0f;
  cudaEventElapsedTime(&total_ms, t0, t1);
  float avg_ms = total_ms / NITER;

  const double bytes_moved = static_cast<double>(N) * (sizeof(int) + sizeof(unsigned int));
  const double bandwidth_gb = bytes_moved / (avg_ms * 1e-3) / 1e9;
  const double elem_per_sec = static_cast<double>(N) / (avg_ms * 1e-3);

  printf("=== Performance (N=%d, R=%d, %d runs) ===\n", N, R, NITER);
  printf("  Avg latency : %.4f ms\n", avg_ms);
  printf("  Throughput  : %.2f Gelem/s\n", elem_per_sec / 1e9);
  printf("  Bandwidth   : %.2f GB/s\n", bandwidth_gb);

  cudaEventDestroy(t0);
  cudaEventDestroy(t1);
  return pass ? 0 : 1;
}
