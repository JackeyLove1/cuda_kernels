#include <cstdio>
#include <vector>
#include <random>
#include <cmath>
#include <cuda.h>
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

/*
P_{y,x}=\sum\limits_{j=-r_{y}}^{r_{y}}\sum\limits_{k=-r_{x}}^{r_{x}}f_{j+r_{y},k+r_{x}}N_{y+k,x+j}
 */

#define FILTER_RADIUS 1
#define OUT_SIZE(IN, PAD, KERNEL, STRIDE) ((IN + 2 * PAD - KERNEL) / STRIDE + 1)

#define IN_TILE_SIZE 32
#define OUT_TILE_SIZE ((IN_TILE_SIZE) - 2 * (FILTER_RADIUS))

__constant__ float F[2*FILTER_RADIUS+1][2*FILTER_RADIUS+1] = {
  {1, 2, 3},{4, 5, 6}, {7, 8, 9}
};
constexpr float H_F[2 * FILTER_RADIUS + 1][2 * FILTER_RADIUS + 1] = {
    {1, 2, 3}, {4, 5, 6}, {7, 8, 9}};

// 2d grid + 2d thread block
__global__ void convolution_2D_basic_kernel(
  const float * __restrict__ N, float *P,
  const int r, const int width, const int height) {

  const auto outRow = blockIdx.x * blockDim.x + threadIdx.x;
  const auto outCol = blockDim.y * blockIdx.y + threadIdx.y;
  const auto kernelSize = 2 * r + 1;
  float PValue{0.f};
  for (int fRow = 0; fRow < kernelSize; ++fRow) {
#pragma unroll 2
    for (int fCol = 0; fCol < kernelSize; ++fCol) {
      int inRow = outRow - r + fRow;
      int inCol = outCol - r + fCol;
      if (inRow >= 0 && inRow < height && inCol < width && inCol >= 0) {
        PValue += N[inRow * width + inCol] * F[fRow][fCol];
      }
    }
  }
  P[outRow * width + outCol] = PValue;
}

__global__ void convolution_2D_constant_tile_kernel(
  const float * __restrict__ N, float *P,
  const int r, const int width, const int height) {

  const auto tx = threadIdx.x , ty = threadIdx.y;
  const int col = blockIdx.x * OUT_TILE_SIZE + tx - FILTER_RADIUS;
  const int row = blockIdx.y * OUT_TILE_SIZE + ty - FILTER_RADIUS;

  __shared__ float tile[IN_TILE_SIZE][IN_TILE_SIZE];
  if (col >= 0 && col < width && row >= 0 && row < height) {
    tile[ty][tx] = N[row * width + col];
  } else {
    tile[ty][tx] = 0.f;
  }
  __syncthreads();

  int tileRow = ty - FILTER_RADIUS;
  int tileCol = tx - FILTER_RADIUS;
  if (col >= 0 && col < width && row >= 0 && row < height) {
    if (tileCol >= 0 && tileCol < OUT_TILE_SIZE && tileRow >= 0 && tileRow < OUT_TILE_SIZE) {
      float PValue {0.f};
      for (int fRow = 0; fRow < 2 * FILTER_RADIUS + 1 ; ++fRow) {
        for (int fCol = 0; fCol < 2 * FILTER_RADIUS + 1; ++fCol) {
          int inRow = tileRow + fRow;
          int inCol = tileCol + fCol;
          PValue += F[fRow][fCol] * tile[inRow][inCol];
        }
      }
      P[row * width + col] = PValue;
    }
  }
}


static void checkCuda(cudaError_t err, const char* what) {
  if (err != cudaSuccess) {
    std::fprintf(stderr, "CUDA error at %s: %s\n", what, cudaGetErrorString(err));
    std::exit(EXIT_FAILURE);
  }
}

static void conv2d_cpu_reference(
    const std::vector<float>& input,
    std::vector<float>& output,
    int width, int height, int r) {
  const int kernelSize = 2 * r + 1;
  for (int row = 0; row < height; ++row) {
    for (int col = 0; col < width; ++col) {
      float acc = 0.0f;
      for (int fRow = 0; fRow < kernelSize; ++fRow) {
        for (int fCol = 0; fCol < kernelSize; ++fCol) {
          const int inRow = row - r + fRow;
          const int inCol = col - r + fCol;
          if (inRow >= 0 && inRow < height && inCol >= 0 && inCol < width) {
            acc += input[inRow * width + inCol] * H_F[fRow][fCol];
          }
        }
      }
      output[row * width + col] = acc;
    }
  }
}

static bool compare_results(
    const std::vector<float>& ref,
    const std::vector<float>& got,
    int width, int height, float atol) {
  float maxAbsDiff = 0.0f;
  int badRow = -1;
  int badCol = -1;
  float badRef = 0.0f;
  float badGot = 0.0f;

  for (int row = 0; row < height; ++row) {
    for (int col = 0; col < width; ++col) {
      const int idx = row * width + col;
      const float diff = std::fabs(ref[idx] - got[idx]);
      if (diff > maxAbsDiff) {
        maxAbsDiff = diff;
      }
      if (diff > atol && badRow < 0) {
        badRow = row;
        badCol = col;
        badRef = ref[idx];
        badGot = got[idx];
      }
    }
  }

  std::printf("max |ref-gpu| = %.8f\n", maxAbsDiff);
  if (badRow >= 0) {
    std::printf(
        "FAILED: first mismatch at (row=%d, col=%d): ref=%.8f gpu=%.8f diff=%.8f (tol=%.8f)\n",
        badRow, badCol, badRef, badGot, std::fabs(badRef - badGot), atol);
    return false;
  }
  std::puts("PASSED: all elements within tolerance.");
  return true;
}

int main() {
  constexpr int width = 61;
  constexpr int height = 47;
  constexpr int r = FILTER_RADIUS;
  constexpr float atol = 1e-4f;
  const size_t numel = static_cast<size_t>(width) * static_cast<size_t>(height);
  const size_t bytes = numel * sizeof(float);

  std::vector<float> hInput(numel);
  std::vector<float> hRef(numel, 0.0f);
  std::vector<float> hOut(numel, 0.0f);

  std::mt19937 rng(20260210);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  for (size_t i = 0; i < numel; ++i) {
    hInput[i] = dist(rng);
  }

  conv2d_cpu_reference(hInput, hRef, width, height, r);

  float* dInput = nullptr;
  float* dOut = nullptr;
  checkCuda(cudaMalloc(&dInput, bytes), "cudaMalloc(dInput)");
  checkCuda(cudaMalloc(&dOut, bytes), "cudaMalloc(dOut)");
  checkCuda(cudaMemcpy(dInput, hInput.data(), bytes, cudaMemcpyHostToDevice), "H2D input");
  checkCuda(cudaMemset(dOut, 0, bytes), "memset dOut");

  dim3 block(IN_TILE_SIZE, IN_TILE_SIZE);
  dim3 grid(
      (width + OUT_TILE_SIZE - 1) / OUT_TILE_SIZE,
      (height + OUT_TILE_SIZE - 1) / OUT_TILE_SIZE);
  convolution_2D_constant_tile_kernel<<<grid, block>>>(dInput, dOut, r, width, height);
  checkCuda(cudaGetLastError(), "launch convolution_2D_constant_tile_kernel");
  checkCuda(cudaDeviceSynchronize(), "sync after convolution_2D_constant_tile_kernel");
  checkCuda(cudaMemcpy(hOut.data(), dOut, bytes, cudaMemcpyDeviceToHost), "D2H output");

  const bool ok = compare_results(hRef, hOut, width, height, atol);

  checkCuda(cudaFree(dInput), "cudaFree(dInput)");
  checkCuda(cudaFree(dOut), "cudaFree(dOut)");

  return ok ? EXIT_SUCCESS : EXIT_FAILURE;
}