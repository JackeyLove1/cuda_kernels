#include <cuda_runtime.h>
#include <thrust/execution_policy.h>
#include <thrust/merge.h>

// A, B, C are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* A, const float* B, float* C, int M, int N) {
  const float* d_a = A;
  const float* d_b = B;
  float* d_c = C;

  // Merge two sorted sequences in O(M+N), no need to sort entire array.
  thrust::merge(thrust::device, d_a, d_a + M, d_b, d_b + N, d_c);
}
