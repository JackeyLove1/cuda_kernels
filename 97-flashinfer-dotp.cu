#include <cooperative_groups.h>
#include <cooperative_groups/memcpy_async.h>
#include <cuda.h>
#include <cuda_pipeline.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <cuda/pipeline>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/inner_product.h>
#include <thrust/copy.h>
#include <random>

namespace cg = cooperative_groups;

#define PIPELINE_STAGES 4

#define TILE_SIZE 1024

__global__ void pipeline_dit_product_kernel(const float* __restrict__ A,
                                            const float* __restrict__ B, float* result,
                                            const size_t N) {
  extern __shared__ float smem[];
  // A_smem[stages][tile_size]
  // B_smem[stages][tile_size]
  float* A_smem = smem;
  float* B_smem = &smem[PIPELINE_STAGES * TILE_SIZE];

  const auto gid = blockIdx.x * TILE_SIZE;
  const auto tid = threadIdx.x;

  float local_sum{0.f};

  cuda::pipeline<cuda::thread_scope_thread> pipe = cuda::make_pipeline();

  auto fetch_idx = 0;

  for (; fetch_idx < PIPELINE_STAGES - 1; ++fetch_idx) {
    auto offset = (blockIdx.x * TILE_SIZE + fetch_idx * TILE_SIZE) + tid;
    pipe.producer_acquire();
    if (offset < N) {
      cuda::memcpy_async(&A_smem[(fetch_idx % PIPELINE_STAGES) * TILE_SIZE + tid],
        &A[offset], sizeof(float), pipe);
      cuda::memcpy_async(&B_smem[(fetch_idx % PIPELINE_STAGES) * TILE_SIZE + tid],
        &B[offset], sizeof(float), pipe);
    }
    pipe.producer_commit();
  }

  auto total_tile = (N + TILE_SIZE - 1) / TILE_SIZE;

  auto compute_idx = 0;
  for (; compute_idx < total_tile; ++compute_idx) {
    if (fetch_idx < total_tile) {
      auto offset = (blockIdx.x * TILE_SIZE + fetch_idx * TILE_SIZE) + tid;
      int stage_idx = fetch_idx % PIPELINE_STAGES;

      pipe.producer_acquire();
      if (offset < N) {
        cuda::memcpy_async(&A_smem[stage_idx * TILE_SIZE + tid],
          &A[offset], sizeof(float), pipe);
        cuda::memcpy_async(&B_smem[stage_idx * TILE_SIZE + tid],
          &B[offset], sizeof(float), pipe);
      }
      pipe.producer_commit();
      fetch_idx++;
    }

    pipe.consumer_wait();
    __syncthreads();

    int stage_idx = compute_idx % PIPELINE_STAGES;
    float a_val = A_smem[stage_idx * TILE_SIZE + tid];
    float b_val = B_smem[stage_idx * TILE_SIZE + tid];

    if ((compute_idx * TILE_SIZE + tid) < N) {
      local_sum += a_val * b_val;
    }

    pipe.consumer_release();
    __syncthreads();
  }

  for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
    local_sum += __shfl_down_sync(0xFFFFFFFF, local_sum, offset);
  }

  if (tid % 32 == 0) {
    atomicAdd(result, local_sum);
  }

}

int main() {
  const size_t N = 1 << 20;

  std::mt19937 rng(1234);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

  thrust::host_vector<float> hA(N);
  thrust::host_vector<float> hB(N);
  for (size_t i = 0; i < N; ++i) {
    hA[i] = dist(rng);
    hB[i] = dist(rng);
  }

  thrust::device_vector<float> dA = hA;
  thrust::device_vector<float> dB = hB;
  thrust::device_vector<float> dResult(1, 0.0f);

  const int threads = TILE_SIZE;
  const int blocks = 1;
  const size_t smem_size =
      static_cast<size_t>(2) * PIPELINE_STAGES * TILE_SIZE * sizeof(float);

  pipeline_dit_product_kernel<<<blocks, threads, smem_size>>>(
      thrust::raw_pointer_cast(dA.data()),
      thrust::raw_pointer_cast(dB.data()),
      thrust::raw_pointer_cast(dResult.data()), N);
  cudaDeviceSynchronize();

  float gpu_result = 0.0f;
  thrust::copy(dResult.begin(), dResult.end(), &gpu_result);

  double cpu_result = thrust::inner_product(hA.begin(), hA.end(), hB.begin(), 0.0);

  double abs_err = std::abs(cpu_result - static_cast<double>(gpu_result));
  double rel_err = abs_err / (std::abs(cpu_result) + 1e-9);

  printf("CPU result: %.6f\n", cpu_result);
  printf("GPU result: %.6f\n", gpu_result);
  printf("Abs error : %.6e\n", abs_err);
  printf("Rel error : %.6e\n", rel_err);

  const double tol = 1e-3;
  if (rel_err < tol || abs_err < tol) {
    printf("PASS\n");
    return 0;
  }
  printf("FAIL\n");
  return 1;
}

// ncu -f --set full --launch-count 10 -o report ./flashinfer12