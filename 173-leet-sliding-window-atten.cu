#include <cuda_runtime.h>
#include <stdio.h>

#define BLOCK_THREADS 128

__global__ void window_attn_kernel(const float* __restrict__ Q, const float* __restrict__ K,
                                   const float* __restrict__ V, float* __restrict__ output,
                                   const int M, const int d, const int window_size) {
  extern __shared__ float sh_scores[];
  __shared__ float sh_max;
  __shared__ float sh_sum_exp;

  const int row = blockIdx.x;
  const int tx = threadIdx.x;
  if (row >= M) return;

  const float scale = 1.0f / sqrtf((float)d);
  const int i = row;
  const int start = max(0, i - window_size);
  const int end = min(M - 1, i + window_size);
  const int local_len = end - start + 1;

  const float* q_ptr = &Q[i * d];

  for (int j = start + tx; j <= end; j += blockDim.x) {
    const float* k_ptr = &K[j * d];

    float dot = 0.0f;
    for (int k = 0; k < d; k++) {
      dot += q_ptr[k] * k_ptr[k];
    }

    sh_scores[j - start] = dot * scale;
  }

  __syncthreads();

  if (tx == 0) {
    float max_val = -INFINITY;
    for (int j = 0; j < local_len; j++) {
      max_val = fmaxf(max_val, sh_scores[j]);
    }
    sh_max = max_val;
  }
  __syncthreads();

  float max_val = sh_max;

  if (tx == 0) {
    float sum_exp = 0.0f;
    for (int j = 0; j < local_len; j++) {
      sum_exp += expf(sh_scores[j] - max_val);
    }
    sh_sum_exp = sum_exp;
  }
  __syncthreads();

  float sum_exp = sh_sum_exp;

  for (int k = tx; k < d; k += blockDim.x) {
    double final_val = 0.0;

    for (int j = 0; j < local_len; j++) {
      float weight = expf(sh_scores[j] - max_val) / sum_exp;
      const int v_idx = (start + j) * d + k;
      final_val += (double)weight * (double)V[v_idx];
    }

    output[i * d + k] = (float)final_val;
  }
}

extern "C" void solve(const float* Q, const float* K, const float* V, float* output, int M, int d,
                      int window_size) {
  const int max_local = 2 * window_size + 1;
  size_t shmem_size = max_local * sizeof(float);
  window_attn_kernel<<<M, BLOCK_THREADS, shmem_size>>>(Q, K, V, output, M, d, window_size);
}