#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdio.h>

#define CEIL_DIV(A, B) ((A + B - 1) / B)

template <int T, int Mr, int Mc>
__global__ void batched_mm(const half* A, const half* B, half* C, int BATCH, int M, int N, int K) {
  int tid = threadIdx.x;
  int tilesPerRow = T / Mc;
  int tc = tid % tilesPerRow;
  int tr = tid / tilesPerRow;
  int col0 = T * blockIdx.x + tc * Mc;
  int row0 = T * blockIdx.y + tr * Mr;
  int b = blockIdx.z;
  int idx = (b * M * N) + row0 * N + col0;

  // basic shared memory
  __shared__ float A_s[T][T];
  __shared__ float B_s[T][T];
  float acc[Mr][Mc];
  // move tile by tile through K-axis
  int numTiles = CEIL_DIV(K, T);

  for (int r = 0; r < Mr; r++) {
    for (int c = 0; c < Mc; c++) {
      acc[r][c] = 0.0f;
    }
  }

  for (int t = 0; t < numTiles; t++) {
    // laod tiles into shared memory
    for (int r = 0; r < Mr; r++) {
      for (int c = 0; c < Mc; c++) {
        int tRow = tr * Mr + r;
        int tCol = tc * Mc + c;
        int Acol = t * T + tc * Mc;
        int Brow = t * T + tr * Mr;
        A_s[tRow][tCol] = (row0 + r < M && Acol + c < K)
                              ? static_cast<float>(A[(b * M * K) + (row0 + r) * K + Acol + c])
                              : 0.0f;
        B_s[tRow][tCol] = (Brow + r < K && col0 + c < N)
                              ? static_cast<float>(B[(b * N * K) + (Brow + r) * N + col0 + c])
                              : 0.0f;
      }
    }

    __syncthreads();
    // dot product on tiles
    for (int k = 0; k < T; k++) {
      for (int r = 0; r < Mr; r++) {
        float A = A_s[tr * Mr + r][k];
#pragma unroll
        for (int c = 0; c < Mc; c++) {
          acc[r][c] += A * B_s[k][tc * Mc + c];
        }
      }
    }
    __syncthreads();
  }

  // update output
  for (int r = 0; r < Mr; r++) {
    for (int c = 0; c < Mc; c++) {
      if (row0 + r < M && col0 + c < N) {
        C[(b * M * N) + (row0 + r) * N + col0 + c] = __float2half(acc[r][c]);
      }
    }
  }
}

// A, B, C are device pointers
extern "C" void solve(const half* A, const half* B, half* C, int BATCH, int M, int N, int K) {
  int T = 64;
  int Mr = 8;
  int Mc = 4;
  int threadsPerBlock = (T / Mr) * (T / Mc);
  dim3 numBlocks(CEIL_DIV(N, T), CEIL_DIV(M, T), BATCH);
  batched_mm<64, 8, 4><<<numBlocks, threadsPerBlock>>>(A, B, C, BATCH, M, N, K);
}
