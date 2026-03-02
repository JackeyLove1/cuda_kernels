// Sparse (implicit zeros) A [M x N] times dense B [N x K] -> dense C [M x K]
// A, B, C are device pointers (float32, row-major).
//
// Key idea: treat A as "implicitly sparse" (stored dense but many zeros).
// Assign 1 warp to compute 32 output columns of ONE row at a time.
// For each k in [0..N-1], lane0 loads A[row,k] once and broadcasts to the warp.
// If A[row,k] != 0, all lanes multiply it with their B[k, col] and accumulate.
// This avoids 32x redundant loads of A for the same row,k.
//
// No external libs; pure CUDA native features.

#include <cuda_runtime.h>
#include <stdint.h>

#ifndef CHECK_CUDA
#define CHECK_CUDA(call)     \
  do {                       \
    cudaError_t _e = (call); \
    (void)_e;                \
  } while (0)
#endif

static __global__ void spmm_implicit_sparse_warp(const float* __restrict__ A,
                                                 const float* __restrict__ B, float* __restrict__ C,
                                                 int M, int N, int K) {
  // Block layout:
  // - blockIdx.y selects the row in A/C.
  // - blockIdx.x selects a "super-tile" of columns in C.
  // - Each block has WARPS_PER_BLOCK warps; each warp computes 32 columns.
  constexpr int WARPS_PER_BLOCK = 4;  // 4 warps => 128 threads
  constexpr int WARP_SIZE = 32;
  constexpr int COLS_PER_BLOCK = WARPS_PER_BLOCK * WARP_SIZE;

  int row = (int)blockIdx.y;
  if (row >= M) return;

  int tid = (int)threadIdx.x;
  int warp_id = tid / WARP_SIZE;     // [0..WARPS_PER_BLOCK-1]
  int lane = tid & (WARP_SIZE - 1);  // [0..31]

  // Global column this thread computes
  int col_base = ((int)blockIdx.x * COLS_PER_BLOCK) + (warp_id * WARP_SIZE);
  int col = col_base + lane;

  // Accumulator for C[row, col]
  float acc = 0.0f;

  // Pointers to the start of the row in A and C
  const float* Arow = A + (size_t)row * (size_t)N;

  // Iterate k across the row of A (and rows of B)
  // Lane 0 loads A value once per warp and broadcasts to other lanes.
  for (int k = 0; k < N; ++k) {
    float a = 0.0f;

    if (lane == 0) {
      a = Arow[k];
    }
    // Broadcast 'a' from lane0 to all lanes in this warp
    a = __shfl_sync(0xFFFFFFFFu, a, 0);

    // Skip zeros (sparsity exploitation)
    if (a != 0.0f) {
      if (col < K) {
        // B is row-major [N x K], so B[k, col] = B[k*K + col]
        float b = B[(size_t)k * (size_t)K + (size_t)col];
        acc = fmaf(a, b, acc);
      }
    }
  }

  // Store result
  if (col < K) {
    C[(size_t)row * (size_t)K + (size_t)col] = acc;
  }
}

// A, B, C are device pointers
extern "C" void solve(const float* A, const float* B, float* C, int M, int N, int K, int /*nnz*/) {
  // Choose a block with 4 warps (128 threads). Good baseline for many GPUs.
  constexpr int WARPS_PER_BLOCK = 4;
  constexpr int THREADS = WARPS_PER_BLOCK * 32;         // 128
  constexpr int COLS_PER_BLOCK = WARPS_PER_BLOCK * 32;  // 128 cols computed per block per row

  dim3 block(THREADS, 1, 1);
  dim3 grid((K + COLS_PER_BLOCK - 1) / COLS_PER_BLOCK, M, 1);

  spmm_implicit_sparse_warp<<<grid, block>>>(A, B, C, M, N, K);
  CHECK_CUDA(cudaGetLastError());
}