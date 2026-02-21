
#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/sort.h>

#include <algorithm>
#include <cfloat>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cub/cub.cuh>
#include <cuda/pipeline>
#include <limits>
#include <random>
#include <type_traits>

#define CEIL(a, b) (((a) + (b) - 1) / (b))

template <typename>
struct dependent_false : std::false_type {};

__global__ void kernel(const float* __restrict__ input, float* __restrict__ output, const int rows,
                       const int cols) {
  const int gid = blockDim.x * blockIdx.x + threadIdx.x;
  const int row = gid / cols;
  const int col = gid % cols;
  const int row_prev = row - 1;
  const int row_next = row + 1;
  const int col_prev = col - 1;
  const int col_next = col + 1;
  const bool row_prev_valid = row_prev >= 0;
  const bool row_next_valid = row_next < rows;
  const bool col_prev_valid = col_prev >= 0;
  const bool col_next_valid = col_next < cols;
  if (row >= rows || col >= cols) return;

  const auto current = input[row * cols + col];
  if (row_prev_valid && col_prev_valid && row_next_valid && col_next_valid) {
    const auto left = input[row * cols + col_prev];
    const auto right = input[row * cols + col_next];
    const auto top = input[row_prev * cols + col];
    const auto bottom = input[row_next * cols + col];
    const auto new_value = (left + right + top + bottom) * 0.25f;
    output[row * cols + col] = new_value;
  } else {
    output[row * cols + col] = current;
  }
}

extern "C" void solve(const float* input, float* output, int rows, int cols) {
  constexpr int nthrs = 256;
  const int nblks = CEIL(rows * cols, nthrs);
  kernel<<<nblks, nthrs>>>(input, output, rows, cols);
  cudaDeviceSynchronize();
}
