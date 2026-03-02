#include <cuda_runtime.h>

#define BM 128
#define BN 128
#define BK 8
#define TM 8
#define TN 8

#define FLOAT4(x) (reinterpret_cast<float4*>(&(x))[0])
#define FLOAT4_CONST(x) (reinterpret_cast<const float4*>(&(x))[0])

__global__ void matrix_multiplication_kernel(const float* A, const float* B, float* C, int M, int K,
                                             int N) {
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
      C[store_c_gmem_addr] = r_c[i][j];
    }
  }
}

__global__ void matrix_multiplication_float4_kernel(const float* A, const float* B, float* C, int M,
                                                    int K, int N) {
  const int tx = threadIdx.x;
  const int ty = threadIdx.y;
  const int bx = blockIdx.x;
  const int by = blockIdx.y;
  const int tid = ty * blockDim.x + tx;

  // 转置存入 sa
  __shared__ float sa[BK][BM];
  __shared__ float sb[BK][BN];

  float ra[TM];
  float rb[TN];
  float rc[TM][TN] = {0.f};

#pragma unroll
  for (int bk = 0; bk < ((K + BK - 1) / BK); bk++) {
    // load global to shared
    const int load_a_elements_per_thread = BM * BK / ((BM * BN) / (TM * TN));
    const int load_a_threads_per_row = BK / load_a_elements_per_thread;
    const int load_a_smem_m = tid / load_a_threads_per_row;
    const int load_a_smem_k = (tid % load_a_threads_per_row) * load_a_elements_per_thread;
    const int load_a_gmem_m = by * BM + load_a_smem_m;
    const int load_a_gmem_k = bk * BK + load_a_smem_k;
    const int load_a_gmem_ptr = load_a_gmem_m * K + load_a_gmem_k;
#pragma unroll
    for (int i = 0; i < load_a_elements_per_thread; i += 4) {
      if (load_a_gmem_m >= M || (load_a_gmem_k + i) >= K) {
        sa[load_a_smem_k + i + 0][load_a_smem_m] = 0.f;
        sa[load_a_smem_k + i + 1][load_a_smem_m] = 0.f;
        sa[load_a_smem_k + i + 2][load_a_smem_m] = 0.f;
        sa[load_a_smem_k + i + 3][load_a_smem_m] = 0.f;
      } else {
        float4 reg = FLOAT4_CONST(A[load_a_gmem_ptr + i]);
        sa[load_a_smem_k + i + 0][load_a_smem_m] = reg.x;
        sa[load_a_smem_k + i + 1][load_a_smem_m] = reg.y;
        sa[load_a_smem_k + i + 2][load_a_smem_m] = reg.z;
        sa[load_a_smem_k + i + 3][load_a_smem_m] = reg.w;
      }
    }
    const int load_b_elements_per_thread = BK * BN / ((BM * BN) / (TM * TN));
    const int load_b_threads_per_row = BN / load_b_elements_per_thread;
    const int load_b_smem_k = tid / load_b_threads_per_row;
    const int load_b_smem_n = (tid % load_b_threads_per_row) * load_b_elements_per_thread;
    const int load_b_gmem_k = bk * BK + load_b_smem_k;
    const int load_b_gmem_n = bx * BN + load_b_smem_n;
    const int load_b_gmem_ptr = load_b_gmem_k * N + load_b_gmem_n;
#pragma unroll
    for (int i = 0; i < load_b_elements_per_thread; i += 4) {
      if (load_b_gmem_k >= K || (load_b_gmem_n + i) >= N) {
        sb[load_b_smem_k][load_b_smem_n + i + 0] = 0.f;
        sb[load_b_smem_k][load_b_smem_n + i + 1] = 0.f;
        sb[load_b_smem_k][load_b_smem_n + i + 2] = 0.f;
        sb[load_b_smem_k][load_b_smem_n + i + 3] = 0.f;
      } else {
        FLOAT4(sb[load_b_smem_k][load_b_smem_n + i]) = FLOAT4_CONST(B[load_b_gmem_ptr + i]);
      }
    }
    __syncthreads();
#pragma unroll
    for (int k = 0; k < BK; k++) {
// load shared to register
#pragma unroll
      for (int m = 0; m < TM; m += 4) {
        const int comp_a_smem_m = ty * TM + m;
        FLOAT4(ra[m]) = FLOAT4(sa[k][comp_a_smem_m]);
      }
#pragma unroll
      for (int n = 0; n < TN; n += 4) {
        const int comp_b_smem_n = tx * TN + n;
        FLOAT4(rb[n]) = FLOAT4(sb[k][comp_b_smem_n]);
      }
// compute
#pragma unroll
      for (int m = 0; m < TM; m++) {
#pragma unroll
        for (int n = 0; n < TN; n++) {
          rc[m][n] += ra[m] * rb[n];
        }
      }
    }
    __syncthreads();
  }
// write back
#pragma unroll
  for (int m = 0; m < TM; m++) {
#pragma unroll
    for (int n = 0; n < TN; n += 4) {
      const int store_c_gmem_m = by * BM + ty * TM + m;
      const int store_c_gmem_n = bx * BN + tx * TN + n;
      if (store_c_gmem_m >= M || store_c_gmem_n >= N) break;
      const int store_c_gmem_ptr = store_c_gmem_m * N + store_c_gmem_n;
      FLOAT4(C[store_c_gmem_ptr]) = FLOAT4(rc[m][n]);
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

  if ((M_new % 4 == 0) && (N_new % 4 == 0) && (K_new % 4 == 0)) {
    matrix_multiplication_float4_kernel<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, M_new, K_new,
                                                                            N_new);
  } else {
    matrix_multiplication_kernel<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, M_new, K_new, N_new);
  }
  cudaDeviceSynchronize();
}
