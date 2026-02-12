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
#include <vector>

/**
Implement a program that performs element-wise addition of two N×N matrices containing 32-bit
floating point numbers on a GPU. The program should take two input matrices of equal dimensions and
produce a single output matrix containing their element-wise sum.
 */
#define CEIL(a, b) (((a) + (b) - 1) / (b))
#define LOAD4(ptr) (reinterpret_cast<const float4*>(ptr))
#define STORE4(ptr) (reinterpret_cast<float4*>(ptr))

#define WORKS 2  // thread works per iteration

__forceinline__ __device__ float4 operator+(const float4 a, const float4 b) {
  return make_float4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w);
}

__global__ void matrix_add(const float* __restrict__ A, const float* __restrict__ B,
                           float* __restrict__ C, int N) {
  const int gid = blockIdx.x * blockDim.x + threadIdx.x;
  const int total = N * N;
  const int stride = blockDim.x * gridDim.x;

  const int vec_total = total / 4;
  const float4* a4 = LOAD4(A);
  const float4* b4 = LOAD4(B);
  float4* c4 = STORE4(C);
  const int vec_stride = stride * WORKS;

  // Vectorized body: each thread handles WORKS float4 chunks per round.
  for (int base_vec_idx = gid * WORKS; base_vec_idx < vec_total; base_vec_idx += vec_stride) {
#pragma unroll
    for (int work = 0; work < WORKS; ++work) {
      const int vec_idx = base_vec_idx + work;
      if (vec_idx < vec_total) {
        c4[vec_idx] = a4[vec_idx] + b4[vec_idx];
      }
    }
  }

  // Scalar tail for sizes not divisible by 4.
  const int tail_start = vec_total * 4;
  for (int base_idx = tail_start + gid * WORKS; base_idx < total; base_idx += vec_stride) {
#pragma unroll
    for (int work = 0; work < WORKS; ++work) {
      const int idx = base_idx + work;
      if (idx < total) {
        C[idx] = A[idx] + B[idx];
      }
    }
  }
}

// A, B, C are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* A, const float* B, float* C, int N) {
  constexpr int threadsPerBlock = 256;
  const int total = N * N;
  const int work_per_thread = 4 * WORKS;
  const int blocksPerGrid = std::max(1, CEIL(total, threadsPerBlock * work_per_thread));

  matrix_add<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, N);
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("matrix_add launch failed: %s\n", cudaGetErrorString(err));
  }
  cudaDeviceSynchronize();
}

// N = 4096

void matrix_add_cpu(const float* A, const float* B, float* C, int N) {
  const int total = N * N;
  for (int i = 0; i < total; ++i) {
    C[i] = A[i] + B[i];
  }
}

bool check_correctness(const thrust::host_vector<float>& gpu_result,
                       const thrust::host_vector<float>& cpu_result, int total, float rtol = 1e-4f,
                       float atol = 1e-5f) {
  constexpr int SAMPLE_THRESHOLD = 1 << 20;  // 1M
  constexpr int SAMPLE_COUNT = 8192;
  std::mt19937 rng(42);

  std::vector<int> indices(total);
  std::iota(indices.begin(), indices.end(), 0);

  int check_count = total;
  if (total > SAMPLE_THRESHOLD) {
    check_count = std::min(SAMPLE_COUNT, total);
    std::shuffle(indices.begin(), indices.end(), rng);
    indices.resize(check_count);
  }

  constexpr int max_errors = 10;
  int errors = 0;
  float max_diff = 0.0f;
  for (int k = 0; k < check_count; ++k) {
    const int i = indices[k];
    const float diff = fabsf(gpu_result[i] - cpu_result[i]);
    const float ref = fabsf(cpu_result[i]);
    max_diff = fmaxf(max_diff, diff);
    if (diff > atol + rtol * ref) {
      if (++errors <= max_errors) {
        printf("  MISMATCH at [%d]: gpu=%.6f cpu=%.6f diff=%.6e\n", i, gpu_result[i], cpu_result[i],
               diff);
      }
    }
  }
  printf("  Checked %d / %d elements, max_diff=%.6e, errors=%d\n", check_count, total, max_diff,
         errors);
  return errors == 0;
}

int main(int argc, char** argv) {
  int N = 4096;
  if (argc > 1) {
    N = std::max(1, std::atoi(argv[1]));
  }
  const int total = N * N;

  printf("=== Matrix Add Kernel Unit Test & Benchmark ===\n");
  printf("N = %d, elements = %d, matrix bytes = %.2f MB\n", N, total,
         static_cast<double>(total * sizeof(float)) / (1 << 20));

  thrust::host_vector<float> h_A(total), h_B(total), h_cpu_C(total);
  thrust::device_vector<float> d_A(total), d_B(total), d_C(total);

  std::mt19937 rng(42);
  std::uniform_real_distribution<float> dist(-10.0f, 10.0f);
  for (int i = 0; i < total; ++i) {
    h_A[i] = dist(rng);
    h_B[i] = dist(rng);
  }

  d_A = h_A;
  d_B = h_B;

  solve(thrust::raw_pointer_cast(d_A.data()), thrust::raw_pointer_cast(d_B.data()),
        thrust::raw_pointer_cast(d_C.data()), N);
  cudaDeviceSynchronize();
  thrust::host_vector<float> h_gpu_C = d_C;

  matrix_add_cpu(h_A.data(), h_B.data(), h_cpu_C.data(), N);
  const bool pass = check_correctness(h_gpu_C, h_cpu_C, total);
  printf("Correctness: %s\n\n", pass ? "PASS" : "FAIL");

  cudaEvent_t t0, t1;
  cudaEventCreate(&t0);
  cudaEventCreate(&t1);

  constexpr int WARMUP = 10;
  for (int i = 0; i < WARMUP; ++i) {
    solve(thrust::raw_pointer_cast(d_A.data()), thrust::raw_pointer_cast(d_B.data()),
          thrust::raw_pointer_cast(d_C.data()), N);
  }
  cudaDeviceSynchronize();

  constexpr int NITER = 100;
  cudaEventRecord(t0);
  for (int i = 0; i < NITER; ++i) {
    solve(thrust::raw_pointer_cast(d_A.data()), thrust::raw_pointer_cast(d_B.data()),
          thrust::raw_pointer_cast(d_C.data()), N);
  }
  cudaEventRecord(t1);
  cudaEventSynchronize(t1);

  float total_ms = 0.0f;
  cudaEventElapsedTime(&total_ms, t0, t1);
  const float avg_ms = total_ms / NITER;
  const double bytes_moved = 3.0 * total * sizeof(float);  // A read + B read + C write
  const double bandwidth_gb = bytes_moved / (avg_ms * 1e-3) / 1e9;
  const double elem_per_sec = static_cast<double>(total) / (avg_ms * 1e-3);

  printf("=== Performance (%d runs) ===\n", NITER);
  printf("  Avg latency : %.4f ms\n", avg_ms);
  printf("  Throughput  : %.2f Gelem/s\n", elem_per_sec / 1e9);
  printf("  Bandwidth   : %.2f GB/s\n", bandwidth_gb);

  cudaEventDestroy(t0);
  cudaEventDestroy(t1);
  return pass ? 0 : 1;
}

/**
=== Matrix Add Kernel Unit Test & Benchmark ===
N = 4096, elements = 16777216, matrix bytes = 64.00 MB
  Checked 8192 / 16777216 elements, max_diff=0.000000e+00, errors=0
Correctness: PASS

=== Performance (100 runs) ===
  Avg latency : 0.8256 ms
  Throughput  : 20.32 Gelem/s
  Bandwidth   : 243.86 GB/s
 */