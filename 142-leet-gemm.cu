#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <mma.h>
using namespace nvcuda;

#define CEIL_DIV(x, y) (((x) + (y) - 1) / (y))

#define WM 16
#define WN 16
#define WK 16

__global__ void gemm_tc(const __restrict__ half* A, const __restrict__ half* B,
                        __restrict__ half* C, int M, int N, int K, float alpha, float beta) {
  const int tile_m = blockIdx.y * WM;
  const int tile_n = blockIdx.x * WN;

  extern __shared__ unsigned char smem[];
  half* As = reinterpret_cast<half*>(smem);
  half* Bs = As + WM * WK;
  float* Cs = reinterpret_cast<float*>(Bs + WK * WN);

  wmma::fragment<wmma::accumulator, WM, WN, WK, float> c_frag;
  wmma::fill_fragment(c_frag, 0.0f);
  const int lane = threadIdx.x & 31;

  for (int k0 = 0; k0 < K; k0 += WK) {
    for (int i = lane; i < WM * WK; i += 32) {
      int row = i / WK;
      int col = i % WK;
      int global_row = tile_m + row;
      int global_col = k0 + col;
      half v = __float2half(0.0f);
      if (global_row < M && global_col < K) {
        v = A[global_row * K + global_col];
      }
      As[row * WK + col] = v;
    }
    for (int i = lane; i < WK * WN; i += 32) {
      int row = i / WN;
      int col = i % WN;
      int global_row = row + k0;
      int global_col = tile_n + col;
      half v = __float2half(0.0f);
      if (global_row < K && global_col < N) {
        v = B[global_row * N + global_col];
      }
      Bs[row * WN + col] = v;
    }
    __syncthreads();
    wmma::fragment<wmma::matrix_a, WM, WN, WK, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, WM, WN, WK, half, wmma::row_major> b_frag;

    wmma::load_matrix_sync(a_frag, As, WK);
    wmma::load_matrix_sync(b_frag, Bs, WN);
    wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);

    __syncthreads();
  }
  wmma::store_matrix_sync(Cs, c_frag, WN, wmma::mem_row_major);
  __syncthreads();
  for (int i = lane; i < WM * WN; i += 32) {
    int row = i / WM;
    int col = i % WN;
    int global_row = tile_m + row;
    int global_col = tile_n + col;
    if (global_row < M && global_col < N) {
      float ab = Cs[i];
      float old = __float2half(C[global_row * N + global_col]);
      float out = alpha * ab + beta * old;
      C[global_row * N + global_col] = out;
    }
  }
}

// A, B, and C are device pointers
extern "C" void solve(const half* A, const half* B, half* C, int M, int N, int K, float alpha,
                      float beta) {
  dim3 grid(CEIL_DIV(N, WN), CEIL_DIV(M, WN));
  dim3 block(32, 1, 1);
  size_t smem_size = WM * WK * sizeof(half) + WK * WN * sizeof(half) + WM * WN * sizeof(float);
  gemm_tc<<<grid, block, smem_size>>>(A, B, C, M, N, K, alpha, beta);
}
