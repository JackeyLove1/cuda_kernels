#include <cooperative_groups.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cuda/pipeline>
#include <numeric>
#include <random>
#include <vector>

#define BM 128
#define BN 128
#define BK 8
#define TM 8
#define TN 8

#define FLOAT4(x) (reinterpret_cast<float4*>(&(x))[0])
#define FLOAT4_CONST(x) (reinterpret_cast<const float4*>(&(x))[0])
#define CDIV(x, y) (((x) + (y) - 1) / (y))

__global__ void matrix_multiplication_kernel(const float* __restrict__ A,
                                             const float* __restrict__ B, float* __restrict__ C,
                                             const int M, const int K, const int N) {
  namespace cg = cooperative_groups;
  auto block = cg::this_thread_block();

  const int bx = blockIdx.x;
  const int by = blockIdx.y;
  const int tx = threadIdx.x;
  const int ty = threadIdx.y;
  const int tid = ty * blockDim.x + tx;

  constexpr int stages = 2;
  __shared__ float s_a[stages][BM][BK];
  __shared__ float s_b[stages][BK][BN];

  float r_c[TM][TN] = {0.0f};
  float r_a[TM];
  float r_b[TN];

  constexpr int threads_per_tile = (BM / TM) * (BN / TN);
  constexpr int threads_per_row_a = threads_per_tile / BM;
  constexpr int threads_per_row_b = threads_per_tile / BK;
  constexpr int threads_per_row_load_a = (BM * BK) / threads_per_tile;
  constexpr int threads_per_row_load_b = (BK * BN) / threads_per_tile;

  constexpr int reg_a_m_stride = BM / TM;
  constexpr int reg_b_n_stride = BN / TN;

  const int load_a_smem_m = tid / threads_per_row_a;
  const int load_a_smem_k = (tid % threads_per_row_a) * threads_per_row_load_a;

  const int load_b_smem_k = tid / threads_per_row_b;
  const int load_b_smem_n = (tid % threads_per_row_b) * threads_per_row_load_b;

  const int load_a_gemm_m = by * BM + load_a_smem_m;
  const int load_b_gemm_n = bx * BN + load_b_smem_n;

  const int k_tiles = CDIV(K, BK);
  auto pipe = cuda::make_pipeline();
  auto async_load_tile = [&](int stage, int tile_k) {
    const int load_a_gemm_k = tile_k * BK + load_a_smem_k;
    const int load_b_gemm_k = tile_k * BK + load_b_smem_k;
    const int load_a_gemm_addr = load_a_gemm_m * K + load_a_gemm_k;
    const int load_b_gemm_addr = load_b_gemm_k * N + load_b_gemm_n;

    pipe.producer_acquire();
    // load A and B from the global memory to the shared memory s_a and s_b
    for (int i = 0; i < threads_per_row_load_a; i++) {
      if ((load_a_gemm_m < M) && (load_a_gemm_k + i < K)) {
        cuda::memcpy_async(&s_a[stage][load_a_smem_m][load_a_smem_k + i], &A[load_a_gemm_addr + i],
                           sizeof(float), pipe);
      } else {
        s_a[stage][load_a_smem_m][load_a_smem_k + i] = 0.0f;
      }
    }
    for (int i = 0; i < threads_per_row_load_b; i++) {
      if ((load_b_gemm_k < K) && (load_b_gemm_n + i < N)) {
        cuda::memcpy_async(&s_b[stage][load_b_smem_k][load_b_smem_n + i], &B[load_b_gemm_addr + i],
                           sizeof(float), pipe);
      } else {
        s_b[stage][load_b_smem_k][load_b_smem_n + i] = 0.0f;
      }
    }
    pipe.producer_commit();
  };

  if (k_tiles > 0) {
    async_load_tile(0, 0);
    for (int bk = 0; bk < k_tiles; ++bk) {
      int read_stage = bk % stages;
      int next_tile = bk + 1;
      if (next_tile < k_tiles) {
        int write_stage = next_tile % stages;
        async_load_tile(write_stage, next_tile);
      }

      // calculation for this k-tile
      pipe.consumer_wait();
      __syncthreads();
#pragma unroll
      for (int k = 0; k < BK; ++k) {
#pragma unroll
        for (int m = 0; m < TM; ++m) {
          int reg_a_m = ty + m * reg_a_m_stride;
          r_a[m] = s_a[read_stage][reg_a_m][k];
        }
#pragma unroll
        for (int n = 0; n < TN; ++n) {
          int reg_b_n = tx + n * reg_b_n_stride;
          r_b[n] = s_b[read_stage][k][reg_b_n];
        }

#pragma unroll
        for (int m = 0; m < TM; ++m) {
          for (int n = 0; n < TN; ++n) {
            r_c[m][n] = fmaf(r_a[m], r_b[n], r_c[m][n]);
          }
        }
      }
      __syncthreads();
      pipe.consumer_release();
    }
  }

#pragma unroll
  for (int m = 0; m < TM; ++m) {
    const int store_c_gemm_m = by * BM + ty + m * reg_a_m_stride;
#pragma unroll
    for (int n = 0; n < TN; ++n) {
      const int store_c_gemm_n = bx * BN + tx + n * reg_b_n_stride;
      if ((store_c_gemm_n > N - 1) || (store_c_gemm_m > M - 1)) break;
      const int store_c_gemm_addr = store_c_gemm_m * N + store_c_gemm_n;
      C[store_c_gemm_addr] = r_c[m][n];
    }
  }
}

// A, B, C are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* A, const float* B, float* C, int M, int N, int K) {
  int M_new = M;
  int N_new = K;
  int K_new = N;

  dim3 threadsPerBlock(BN / TN, BM / TM);
  dim3 blocksPerGrid((N_new + BN - 1) / BN, (M_new + BM - 1) / BM);

  matrix_multiplication_kernel<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, M_new, K_new, N_new);
  cudaDeviceSynchronize();
}

void cpu_gemm_reference(const float* A, const float* B, float* C, int M, int N, int K) {
  for (int m = 0; m < M; ++m) {
    for (int k = 0; k < K; ++k) {
      float acc = 0.0f;
      for (int n = 0; n < N; ++n) {
        acc += A[m * N + n] * B[n * K + k];
      }
      C[m * K + k] = acc;
    }
  }
}

bool check_correctness(const thrust::host_vector<float>& gpu_result,
                       const thrust::host_vector<float>& cpu_result, int total_elems,
                       float rtol = 1e-4f, float atol = 1e-5f) {
  constexpr int SAMPLE_THRESHOLD = 1 << 20;
  constexpr int SAMPLE_COUNT = 8192;

  std::mt19937 rng(42);
  int check_count = total_elems;
  std::vector<int> indices(total_elems);
  std::iota(indices.begin(), indices.end(), 0);

  if (total_elems > SAMPLE_THRESHOLD) {
    check_count = SAMPLE_COUNT;
    std::shuffle(indices.begin(), indices.end(), rng);
    indices.resize(SAMPLE_COUNT);
  }

  int max_errors = 10;
  int errors = 0;
  float max_diff = 0.0f;
  for (int i = 0; i < check_count; ++i) {
    int idx = indices[i];
    float diff = fabsf(gpu_result[idx] - cpu_result[idx]);
    float ref = fabsf(cpu_result[idx]);
    max_diff = fmaxf(max_diff, diff);
    if (diff > atol + rtol * ref) {
      if (++errors <= max_errors) {
        printf("  MISMATCH at [%d]: gpu=%.6f cpu=%.6f diff=%.6e\n", idx, gpu_result[idx],
               cpu_result[idx], diff);
      }
    }
  }
  printf("  Checked %d / %d elements, max_diff=%.6e, errors=%d\n", check_count, total_elems,
         max_diff, errors);
  return errors == 0;
}

int main(int argc, char** argv) {
  int M = 512;
  int N = 512;
  int K = 512;
  if (argc >= 4) {
    M = atoi(argv[1]);
    N = atoi(argv[2]);
    K = atoi(argv[3]);
  }

  const int a_elems = M * N;
  const int b_elems = N * K;
  const int c_elems = M * K;
  printf("=== GEMM Unit Test & Benchmark ===\n");
  printf("M=%d N=%d K=%d\n", M, N, K);
  printf("A=%d elems, B=%d elems, C=%d elems\n", a_elems, b_elems, c_elems);

  thrust::host_vector<float> h_a(a_elems);
  thrust::host_vector<float> h_b(b_elems);
  thrust::device_vector<float> d_a(a_elems);
  thrust::device_vector<float> d_b(b_elems);
  thrust::device_vector<float> d_c(c_elems);

  std::mt19937 rng(42);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  for (int i = 0; i < a_elems; ++i) h_a[i] = dist(rng);
  for (int i = 0; i < b_elems; ++i) h_b[i] = dist(rng);

  d_a = h_a;
  d_b = h_b;

  solve(thrust::raw_pointer_cast(d_a.data()), thrust::raw_pointer_cast(d_b.data()),
        thrust::raw_pointer_cast(d_c.data()), M, N, K);
  thrust::host_vector<float> h_gpu_c = d_c;

  thrust::host_vector<float> h_cpu_c(c_elems);
  cpu_gemm_reference(h_a.data(), h_b.data(), h_cpu_c.data(), M, N, K);

  bool pass = check_correctness(h_gpu_c, h_cpu_c, c_elems);
  printf("Correctness: %s\n\n", pass ? "PASS" : "FAIL");

  cudaEvent_t t0, t1;
  cudaEventCreate(&t0);
  cudaEventCreate(&t1);

  constexpr int WARMUP = 3;
  for (int i = 0; i < WARMUP; ++i) {
    solve(thrust::raw_pointer_cast(d_a.data()), thrust::raw_pointer_cast(d_b.data()),
          thrust::raw_pointer_cast(d_c.data()), M, N, K);
  }
  cudaDeviceSynchronize();

  constexpr int NITER = 10;
  cudaEventRecord(t0);
  for (int i = 0; i < NITER; ++i) {
    solve(thrust::raw_pointer_cast(d_a.data()), thrust::raw_pointer_cast(d_b.data()),
          thrust::raw_pointer_cast(d_c.data()), M, N, K);
  }
  cudaEventRecord(t1);
  cudaEventSynchronize(t1);

  float total_ms = 0.0f;
  cudaEventElapsedTime(&total_ms, t0, t1);
  float avg_ms = total_ms / NITER;

  double bytes_moved = static_cast<double>(a_elems + b_elems + c_elems) * sizeof(float);
  double bandwidth_gb = bytes_moved / (avg_ms * 1e-3) / 1e9;
  double out_elem_per_sec = static_cast<double>(c_elems) / (avg_ms * 1e-3);
  double tflops = 2.0 * static_cast<double>(M) * N * K / (avg_ms * 1e-3) / 1e12;

  printf("=== Performance (%d runs) ===\n", NITER);
  printf("  Avg latency : %.4f ms\n", avg_ms);
  printf("  Throughput  : %.2f Gelem/s (output elems)\n", out_elem_per_sec / 1e9);
  printf("  Bandwidth   : %.2f GB/s\n", bandwidth_gb);
  printf("  Compute     : %.2f TFLOPS\n", tflops);

  cudaEventDestroy(t0);
  cudaEventDestroy(t1);
  return pass ? 0 : 1;
}