#include <cuda_runtime.h>
#include <stdio.h>

#define BLOCK_THREADS 128

__global__ void causal_attn_kernel(const float* Q, const float* K, const float* V, float* output,
                                   int M, int d) {
  extern __shared__ float sh_scores[];

  __shared__ float sh_max;
  __shared__ float sh_sum_exp;

  int i = blockIdx.x;
  int tx = threadIdx.x;

  float scale = 1.0f / sqrtf((float)d);

  const float* q_ptr = &Q[i * d];

  for (int j = tx; j <= i; j += blockDim.x) {
    const float* k_ptr = &K[j * d];

    float dot = 0.0f;
    for (int k = 0; k < d; k++) {
      dot += q_ptr[k] * k_ptr[k];
    }

    sh_scores[j] = dot * scale;
  }

  __syncthreads();

  if (tx == 0) {
    float max_val = -INFINITY;
    for (int j = 0; j <= i; j++) {
      max_val = fmaxf(max_val, sh_scores[j]);
    }
    sh_max = max_val;
  }
  __syncthreads();

  float max_val = sh_max;

  if (tx == 0) {
    float sum_exp = 0.0f;
    for (int j = 0; j <= i; j++) {
      sum_exp += expf(sh_scores[j] - max_val);
    }
    sh_sum_exp = sum_exp;
  }
  __syncthreads();

  float sum_exp = sh_sum_exp;

  for (int k = tx; k < d; k += blockDim.x) {
    double final_val = 0.0;

    for (int j = 0; j <= i; j++) {
      float weight = expf(sh_scores[j] - max_val) / sum_exp;
      final_val += (double)weight * (double)V[j * d + k];
    }

    output[i * d + k] = (float)final_val;
  }
}

extern "C" void solve(const float* Q, const float* K, const float* V, float* output, int M, int d) {
  size_t shmem_size = M * sizeof(float);
  causal_attn_kernel<<<M, BLOCK_THREADS, shmem_size>>>(Q, K, V, output, M, d);
}