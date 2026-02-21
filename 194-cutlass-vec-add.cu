#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cute/tensor.hpp>
#include <numeric>
#include <random>
#include <vector>

using namespace cute;
// nvcc -arch=sm_120 -O3 -lineinfo -I 3rdparty/cutlass/include 194-cutlass-vec-add.cu -o 194

// Scalar: one float per thread
__global__ void vector_add(const float* A, const float* B, float* C, int N) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= N) return;
  auto a = make_tensor(make_gmem_ptr(A), make_shape(N), make_stride(Int<1>{}));
  auto b = make_tensor(make_gmem_ptr(B), make_shape(N), make_stride(Int<1>{}));
  auto c = make_tensor(make_gmem_ptr(C), make_shape(N), make_stride(Int<1>{}));
  c(i) = a(i) + b(i);
}

// float4: one float4 (4 floats) per thread — shape (N/4, 4), stride (4, 1)
__global__ void vector_add_float4(const float* A, const float* B, float* C, int N) {
  const int n4 = N / 4;
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n4) return;
  auto a = make_tensor(make_gmem_ptr(A), make_shape(n4, Int<4>{}), make_stride(Int<4>{}, Int<1>{}));
  auto b = make_tensor(make_gmem_ptr(B), make_shape(n4, Int<4>{}), make_stride(Int<4>{}, Int<1>{}));
  auto c = make_tensor(make_gmem_ptr(C), make_shape(n4, Int<4>{}), make_stride(Int<4>{}, Int<1>{}));
  // c(i) = a(i) + b(i);  // a(i), b(i), c(i) are length-4 slices
  // Write the 4 lanes explicitly.
  // ⚠️ THIS IS INCORRECT: c(i) = a(i) + b(i) is incorrect.
  // We need to write the 4 lanes explicitly.
  c(make_coord(i, Int<0>{})) = a(i, Int<0>{}) + b(i, Int<0>{});
  c(i, Int<1>{}) = a(i, Int<1>{}) + b(i, Int<1>{});
  c(i, Int<2>{}) = a(i, Int<2>{}) + b(i, Int<2>{});
  c(i, Int<3>{}) = a(i, Int<3>{}) + b(i, Int<3>{});
}

template <int BlockSize>
__global__ void vector_add_tiled(const float* A, const float* B, float* C, int N) {
  constexpr int Vec = 4;
  int n4 = N / Vec;
  auto a =
      make_tensor(make_gmem_ptr(A), make_shape(n4, Int<Vec>{}), make_stride(Int<Vec>{}, Int<1>{}));
  auto b =
      make_tensor(make_gmem_ptr(B), make_shape(n4, Int<Vec>{}), make_stride(Int<Vec>{}, Int<1>{}));
  auto c =
      make_tensor(make_gmem_ptr(C), make_shape(n4, Int<Vec>{}), make_stride(Int<Vec>{}, Int<1>{}));

  auto blk_A = local_tile(a, make_shape(Int<BlockSize>{}, Int<Vec>{}), make_coord(blockIdx.x, _));
  auto blk_B = local_tile(b, make_shape(Int<BlockSize>{}, Int<Vec>{}), make_coord(blockIdx.x, _));
  auto blk_C = local_tile(c, make_shape(Int<BlockSize>{}, Int<Vec>{}), make_coord(blockIdx.x, _));

  auto thr_layout = make_layout(make_shape(Int<BlockSize>{}));
  auto thr_A = local_partition(blk_A, thr_layout, threadIdx.x);
  auto thr_B = local_partition(blk_B, thr_layout, threadIdx.x);
  auto thr_C = local_partition(blk_C, thr_layout, threadIdx.x);

  int i4 = blockIdx.x * BlockSize + threadIdx.x;
  if (threadIdx.x < BlockSize && i4 < n4) {
#pragma unroll
    for (int v = 0; v < Vec; ++v) {
      thr_C(v) = thr_A(v) + thr_B(v);
    }
  }
}

extern "C" void solve(const float* A, const float* B, float* C, int N) {
  constexpr int threadsPerBlock = 256;
  int n4 = N / 4;
  int blocksPerGrid = (n4 + threadsPerBlock - 1) / threadsPerBlock;
  if (n4 > 0) {
    vector_add_tiled<threadsPerBlock><<<blocksPerGrid, threadsPerBlock>>>(A, B, C, N);
  }
  int tail = N - n4 * 4;
  if (tail > 0) {
    vector_add<<<1, tail>>>(A + n4 * 4, B + n4 * 4, C + n4 * 4, tail);
  }
  cudaDeviceSynchronize();
}

// ============================================================
// CPU reference (scalar)
// ============================================================
static void cpu_reference(const float* A, const float* B, float* C, int N) {
  for (int i = 0; i < N; ++i) C[i] = A[i] + B[i];
}

// ============================================================
// Correctness check — full check for N <= 1M, else random sample
// ============================================================
static bool check_correctness(const thrust::host_vector<float>& gpu_result,
                              const thrust::host_vector<float>& cpu_result, int N,
                              float rtol = 1e-4f, float atol = 1e-5f) {
  constexpr int SAMPLE_THRESHOLD = 1 << 20;
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

  int max_errors = 10, errors = 0;
  float max_diff = 0.f;
  for (int k = 0; k < check_count; ++k) {
    int i = indices[k];
    float diff = fabsf(gpu_result[i] - cpu_result[i]);
    float ref = fabsf(cpu_result[i]);
    max_diff = fmaxf(max_diff, diff);
    if (diff > atol + rtol * ref) {
      if (++errors <= max_errors)
        printf("  MISMATCH at [%d]: gpu=%.6f cpu=%.6f diff=%.6e\n", i, gpu_result[i], cpu_result[i],
               diff);
    }
  }
  printf("  Checked %d / %d elements, max_diff=%.6e, errors=%d\n", check_count, N, max_diff,
         errors);
  return errors == 0;
}

int main(int argc, char** argv) {
  constexpr int DEFAULT_N = 1 << 24;
  int N = DEFAULT_N;
  if (argc > 1) N = atoi(argv[1]);
  printf("=== Kernel Unit Test & Benchmark (vector add) ===\n");
  printf("N = %d (%.2f MB total input+output)\n", N, (double)N * 3 * sizeof(float) / (1 << 20));

  thrust::host_vector<float> h_A(N), h_B(N);
  thrust::device_vector<float> d_A(N), d_B(N), d_C(N);

  std::mt19937 rng(42);
  std::uniform_real_distribution<float> dist(-1.f, 1.f);
  for (int i = 0; i < N; ++i) {
    h_A[i] = dist(rng);
    h_B[i] = dist(rng);
  }
  d_A = h_A;
  d_B = h_B;

  solve(thrust::raw_pointer_cast(d_A.data()), thrust::raw_pointer_cast(d_B.data()),
        thrust::raw_pointer_cast(d_C.data()), N);

  thrust::host_vector<float> h_gpu = d_C;

  thrust::host_vector<float> h_cpu(N);
  cpu_reference(h_A.data(), h_B.data(), h_cpu.data(), N);

  bool pass = check_correctness(h_gpu, h_cpu, N);
  printf("Correctness: %s\n\n", pass ? "PASS" : "FAIL");

  cudaEvent_t t0, t1;
  cudaEventCreate(&t0);
  cudaEventCreate(&t1);

  constexpr int WARMUP = 10;
  for (int i = 0; i < WARMUP; ++i)
    solve(thrust::raw_pointer_cast(d_A.data()), thrust::raw_pointer_cast(d_B.data()),
          thrust::raw_pointer_cast(d_C.data()), N);
  cudaDeviceSynchronize();

  constexpr int NITER = 100;
  cudaEventRecord(t0);
  for (int i = 0; i < NITER; ++i)
    solve(thrust::raw_pointer_cast(d_A.data()), thrust::raw_pointer_cast(d_B.data()),
          thrust::raw_pointer_cast(d_C.data()), N);
  cudaEventRecord(t1);
  cudaEventSynchronize(t1);

  float total_ms = 0.f;
  cudaEventElapsedTime(&total_ms, t0, t1);
  float avg_ms = total_ms / NITER;

  double bytes_moved = (2.0 * N * sizeof(float)) + (N * sizeof(float));
  double bandwidth_gb = bytes_moved / (avg_ms * 1e-3) / 1e9;
  double elem_per_sec = (double)N / (avg_ms * 1e-3);

  printf("=== Performance (N=%d, %d runs) ===\n", N, NITER);
  printf("  Avg latency : %.4f ms\n", avg_ms);
  printf("  Throughput  : %.2f Gelem/s\n", elem_per_sec / 1e9);
  printf("  Bandwidth   : %.2f GB/s\n", bandwidth_gb);

  cudaEventDestroy(t0);
  cudaEventDestroy(t1);
  return pass ? 0 : 1;
}