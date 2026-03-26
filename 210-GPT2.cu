#include <cuda_runtime.h>

#include <cstdio>
#include <cub/cub.cuh>
#include <cuda/pipeline>

/**
Weight Layout
All block parameters are packed into a single contiguous weights buffer (7,087,872 floats)in the
following order. Index into the buffer using the offsets below (e. g. Wąkv[i][j] is atweights[1536 +
i * 2304 + j] ). All 2D matrices are stored in row-major order. ParameterShapeSizeOffset 71(LN1
weight)(768,)7680 β₁(LN1 bias)(768,)768768 Wqkv(768,2304)1,769,4721,536 bqkv(2304,)2,3041,771,008
        Wattn(768,768)589,8241,773,312
        battn(768,)7682,363,136
        γ₂(LN2 weight)(768,)7682,363,904
        β₂(LN2 bias)(768,)7682,364,672
        Wfc(768,3072)2,359,2962,365,440
        bfc(3072,)3,0724,724,736
        Wproj(3072,768)2,359,2964,727,808
        bproj(768,)768 7,087,104
 */

#define FULL_MASK 0xFFFFFFFF
#define WARP_SIZE 32

static constexpr int DIM = 768;
static constexpr int N_HEADS = 12;
static constexpr int FFN_DIM = 2072;
static constexpr int SEQ_LEN = 1024;
static constexpr int WEIGHT_QKV = 768 * 2304;
static constexpr int BIAS_QKV = 2304;
static constexpr int WEIGHT_ATTN = 768 * 768;
static constexpr int BIAS_ATTN = 768;
static constexpr int WEIGHT_FC = 768 * 3072;
static constexpr int BIAS_FC = 3072;
static constexpr int WEIGHT_PROJ = 3072 * 768;
static constexpr int BIAS_PROJ = 768;
static constexpr float EPS = 1e-5;

__device__ __forceinline__ float WarpReduceSum(float value) {
#pragma unroll
  for (int offset = (WARP_SIZE >> 1); offset > 0; offset >>= 1) {
    value += __shfl_down_sync(FULL_MASK, value, offset);
  }
  return value;
}

__device__ inline void BlockReduceSum(float value, float* output) {
  const auto tid = threadIdx.x;
  const auto warp_id = tid >> 5;
  const auto lane_id = tid & 31;
  const auto warp_nums = blockDim.x >> 5;

  value = WarpReduceSum(value);
  __shared__ float shared_data[32];
  if (lane_id == 0) {
    shared_data[warp_id] = value;
  }
  __syncthreads();

  if (warp_id == 0) {
    value = (lane_id < warp_nums) ? shared_data[lane_id] : 0.f;
    value = WarpReduceSum(value);
    if (lane_id == 0) {
      atomicAdd(output, value);
    }
  }
}

// (seq_len, dim) -> (seq_len, dim)
// dim3 blocksPerGrid(seq_len), dim3 threadsPerBlock(128)
__global__ void layer_norm_kernel(float* input, const float* __restrict__ gamma,
                                  const float* __restrict__ beta, const int rows, const int cols,
                                  float* __restrict__ output) {
  __shared__ float smem[DIM];
  __shared__ float row_sum;
  __shared__ float row_mean;
  __shared__ float row_var_inv;

  const auto tid = threadIdx.x;
  const auto row = blockIdx.x;
  const auto row_offsets = row * cols;
  const auto stride = blockDim.x;

  if (tid == 0) {
    row_sum = 0;
    row_mean = 0;
    row_var_inv = 0;
  }
  __syncthreads();

  float local_sum{0.f};
#pragma unroll 4
  for (int i = tid; i < cols; i += stride) {
    float val = input[row_offsets + i];
    local_sum += val;
    smem[i] = val;
  }
  __syncthreads();

  BlockReduceSum(local_sum, &row_sum);
  if (tid == 0) {
    row_mean = row_sum / cols;
  }
  __syncthreads();

  float local_var{0.f};
  float mean = row_mean;
#pragma unroll 4
  for (int i = tid; i < cols; i += stride) {
    float val = smem[i] - mean;
    local_var = fmaf(val, val, local_var);
  }
  __syncthreads();

  BlockReduceSum(local_var, &row_var_inv);
  if (tid == 0) {
    row_var_inv = 1.f / sqrtf(row_var_inv / cols + EPS);
  }
  __syncthreads();

  float var_inv = row_var_inv;
#pragma unroll 4
  for (int i = tid; i < cols; i += stride) {
    float val = smem[i];
    val = (val - mean) * var_inv;
    val = fmaf(val, gamma[i], beta[i]);
    output[row_offsets + i] = val;
  }
}

__device__ __forceinline__ float gelu_tanh(float x) {
  // constants
  constexpr float kAlpha = 0.7978845608028654f;  // sqrt(2/pi)
  constexpr float kBeta = 0.044715f;

  float x2 = x * x;
  float x3 = x2 * x;
  float u = fmaf(kBeta, x3, x);
  float v = kAlpha * u;
  float t = tanhf(v);
  return 0.5f * x * (1.0f + t);
}

// (seq_len, dim) -> (seq_len, dim)
// dim3 blocksPerGrid(seq_len), dim3 threadsPerBlock(128)
__global__ void gelu_kernel(float* __restrict__ input, float* __restrict__ output, const int rows,
                            const int cols) {
  const auto tid = threadIdx.x;
  const auto row = blockIdx.x;
  const auto row_offsets = row * cols;
  const auto stride = blockDim.x;
  for (int i = tid; i < cols; i += stride) {
    output[row_offsets + i] = gelu_tanh(input[row_offsets + i]);
  }
}

// A: [M, K], B: [K, N], C: [M, N], D: [M, N]
// A @ B + C = D
#define BM 128
#define BN 128
#define BK 8
#define TM 8
#define TN 8
__global__ void matrix_multiplication_add_kernel(const float* __restrict__ A,
                                                 const float* __restrict__ B,
                                                 const float* __restrict__ C, float* __restrict__ D,
                                                 const int M, const int K, const int N) {
  const int bx = blockIdx.x;
  const int by = blockIdx.y;
  const int tx = threadIdx.x;
  const int ty = threadIdx.y;
  const int tid = ty * blockDim.x + tx;  // 转化一个block里面的线程的二维坐标为一维坐标（0～215）

  // Shared memory per block per round
  __shared__ float s_a[BM][BK];
  __shared__ float s_b[BK][BN];

  // register memory of C per thread
  float r_c[TM][TN] = {0.0f};
  float r_a[TM];
  float r_b[TN];

  int threads_per_row_a = BN / (TM * TN);            // 每行线程数 = [(BM/TM) × (BN/TN)] / BM
  int threads_per_row_b = BN * BM / (BK * TM * TN);  // 每行线程数 = [(BM/TM) × (BN/TN)] / BK

  // 当前线程负责加载的 s_a 共享内存的行号
  int load_a_smem_m = tid / threads_per_row_a;  // row of s_a
  // 当前线程负责加载的 s_a 共享内存的列号
  // 「A 分块总元素数」除以「总线程数」= 单线程加载元素数 = (BM × BK) / [(BM × BN) / (TM × TN)] = BK
  // * TM * TN / BN
  int load_a_smem_k = (tid % threads_per_row_a) * (BK * TM * TN / BN);  // col of s_a

  int load_b_smem_k = tid / threads_per_row_b;                        // row of s_b
  int load_b_smem_n = (tid % threads_per_row_b) * BK * TM * TN / BM;  // col of s_b

  int load_a_gmem_m = by * BM + load_a_smem_m;  // global row of a
  int load_b_gmem_n = bx * BN + load_b_smem_n;  // global col of b

  for (int bk = 0; bk < (K + BK - 1) / BK; bk++) {
    int load_a_gmem_k = bk * BK + load_a_smem_k;  // global col of a
    int load_b_gmem_k = bk * BK + load_b_smem_k;  // global row of b
    int load_a_gmem_addr = load_a_gmem_m * K + load_a_gmem_k;
    int load_b_gmem_addr = load_b_gmem_k * N + load_b_gmem_n;
    // load A and B from the global memory to the shared memory s_a and s_b
    // 单线程加载元素数 = (BM × BK) / [(BM × BN) / (TM × TN)] = BK * TM * TN / BN
    for (int i = 0; i < BK * TM * TN / BN; i++) {
      if ((load_a_gmem_m > M - 1) || (load_a_gmem_k + i > K - 1)) {
        s_a[load_a_smem_m][load_a_smem_k + i] = 0.0;
      } else {
        s_a[load_a_smem_m][load_a_smem_k + i] = A[load_a_gmem_addr + i];
      }
    }
    for (int i = 0; i < BK * TM * TN / BM; i++) {
      if ((load_b_gmem_k > K - 1) || (load_b_gmem_n + i > N - 1)) {
        s_b[load_b_smem_k][load_b_smem_n + i] = 0.0;
      } else {
        s_b[load_b_smem_k][load_b_smem_n + i] = B[load_b_gmem_addr + i];
      }
    }
    __syncthreads();

#pragma unroll
    for (int k = 0; k < BK; k++) {
#pragma unroll
      for (int m = 0; m < TM; m++) {
        // 要读取的 s_a 行号
        // ty=0   0 16 32 48 ....
        int comp_a_smem_m = ty + m * BM / TM;
        r_a[m] = s_a[comp_a_smem_m][k];
      }
#pragma unroll
      for (int n = 0; n < TN; n++) {
        int comp_b_smem_n = tx + n * BN / TN;
        r_b[n] = s_b[k][comp_b_smem_n];
      }
#pragma unroll
      for (int m = 0; m < TM; m++) {
#pragma unroll
        for (int n = 0; n < TN; n++) {
          r_c[m][n] += r_a[m] * r_b[n];
        }
      }
    }
    __syncthreads();
  }
#pragma unroll
  for (int i = 0; i < TM; i++) {
    int store_c_gmem_m = by * BM + ty + i * BM / TM;  // global row of c
#pragma unroll
    for (int j = 0; j < TN; j++) {
      int store_c_gmem_n = bx * BN + tx + j * BN / TN;  // global col of c
      if ((store_c_gmem_n > N - 1) || (store_c_gmem_m > M - 1)) break;
      int store_c_gmem_addr = store_c_gmem_m * N + store_c_gmem_n;
      D[store_c_gmem_addr] = C[store_c_gmem_addr] + r_c[i][j];
    }
  }
}

// softmax(Q @ K^T) @ V
// Q: [M, d], K: [N, d], V: [N, d], output: [M, d]
__global__ void softmax_attention_kernel(const float* __restrict__ Q, const float* __restrict__ K,
                                         const float* __restrict__ V, float* __restrict__ output,
                                         const int M, const int N, const int d) {}

// x, output, weights are device pointers
extern "C" void solve(const float* x, float* output, float* weights, int seq_len) {
  float* ln1_w = weights;
  float* ln1_b = ln1_w + DIM;
  float* qkv_w = ln1_b + DIM;
  float* qkv_b = qkv_w + WEIGHT_QKV;
  float* attn_w = qkv_b + BIAS_QKV;
  float* attn_b = attn_w + WEIGHT_ATTN;
  float* attn_ln2_w = attn_b + BIAS_ATTN;
  float* attn_ln2_b = attn_ln2_w + DIM;
  float* fc_w = attn_ln2_b + DIM;
  float* fc_b = fc_w + WEIGHT_FC;
  float* proj_w = fc_b + BIAS_FC;
  float* proj_b = proj_w + WEIGHT_PROJ;
}
