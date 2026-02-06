#include <cuda.h>
#include <cuda_runtime.h>
#ifdef TEST
#include <torch/extension.h>
#include <torch/types.h>
#endif

#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <iostream>
#include <string>
#include <type_traits>
#include <vector>

#include "include/vec_dtypes.cuh"

// button for test or pybind
#define TEST 1

// ncu -f --set full --launch-count 10 -o report ./flashinfer10

enum class PositionEncodingMode { KNone = 0, KRope = 1, KAiBli = 2 };

static constexpr float theta_scale = 10000.0f;

// [seq, rope_dim]  rope_dim = head_dim // 2
// chunk rope / unintervaled
__global__ void rope_kernel_v1(const float* __restrict__ x, float* __restrict__ y,
                               const int seq_len, const int rope_dim) {
  const auto seq_idx = blockIdx.x;
  const auto tid = threadIdx.x;
  if (seq_idx >= seq_len || tid >= rope_dim) return;
  const auto head_dim = rope_dim * 2;
  const float x1 = x[seq_idx * head_dim + tid];
  const float x2 = x[seq_idx * head_dim + tid + rope_dim];
  const float inv_freq = 1.0f / __powf(theta_scale, 2 * tid / static_cast<float>(head_dim));
  // const float sin = __sinf(seq_idx * inv_freq);
  // const float cos = __cosf(seq_idx * inv_freq);
  float sin, cos;
  __sincosf(seq_idx * inv_freq, &sin, &cos);
  const float y1 = x1 * cos - x2 * sin;
  const float y2 = x1 * sin + x2 * cos;
  y[seq_idx * head_dim + tid] = y1;
  y[seq_idx * head_dim + tid + rope_dim] = y2;
}

// use float4
#define LOAD_FLOAT4(ptr) (reinterpret_cast<const float4*>(ptr))
#define STORE_FLOAT4(ptr) (reinterpret_cast<float4*>(ptr))

__global__ void rope_kernel_v2(const float* __restrict__ x, float* __restrict__ y,
                               const int seq_len, const int rope_dim) {
  const auto seq_idx = blockIdx.x;
  const auto tid = threadIdx.x;
  if (seq_idx >= seq_len || tid * 4 >= rope_dim) return;
  const auto head_dim = rope_dim * 2;
  const int row4 = (seq_idx * head_dim) / 4;  // float4 index for row start
  const int rope4 = rope_dim / 4;             // float4 offset for second half
  const float4 x1 = LOAD_FLOAT4(x)[row4 + tid];
  const float4 x2 = LOAD_FLOAT4(x)[row4 + rope4 + tid];
  auto rope_compute = [&](const float x1, const float x2, const int idx) {
    const float inv_freq =
        1.0f / __powf(theta_scale, (2 * (tid * 4 + idx)) / static_cast<float>(head_dim));
    float sin, cos;
    __sincosf(seq_idx * inv_freq, &sin, &cos);
    return float2{x1 * cos - x2 * sin, x1 * sin + x2 * cos};
  };
  float2 y1 = rope_compute(x1.x, x2.x, 0);
  float2 y2 = rope_compute(x1.y, x2.y, 1);
  float2 y3 = rope_compute(x1.z, x2.z, 2);
  float2 y4 = rope_compute(x1.w, x2.w, 3);
  STORE_FLOAT4(y)[row4 + tid] = make_float4(y1.x, y2.x, y3.x, y4.x);
  STORE_FLOAT4(y)[row4 + rope4 + tid] = make_float4(y1.y, y2.y, y3.y, y4.y);
}

#ifndef TEST
void apply_rope_v1(torch::Tensor x, torch::Tensor y) {
  assert(x.dim() == 2);
  assert(y.dim() == 2);
  assert(x.size(0) == y.size(0));
  assert(x.size(1) == y.size(1));
  assert(x.size(1) % 2 == 0);
  assert(y.size(1) % 2 == 0);
  assert(x.size(1) == y.size(1));
  assert(x.size(1) == y.size(1));
  assert(x.device().is_cuda());
  assert(y.device().is_cuda());
  assert(x.dtype() == torch::kFloat32);
  assert(y.dtype() == torch::kFloat32);

  const auto seq_len = x.size(0);
  const auto head_dim = x.size(1);
  const auto rope_dim = head_dim / 2;
  dim3 nblks(seq_len);
  dim3 nthrs(rope_dim);
  rope_kernel_v1<<<nblks, nthrs>>>(x.data_ptr<float>(), y.data_ptr<float>(), seq_len, rope_dim);
  return;
}

void apply_rope_v1(torch::Tensor x, torch::Tensor y) {
  assert(x.dim() == 2);
  assert(y.dim() == 2);
  assert(x.size(0) == y.size(0));
  assert(x.size(1) == y.size(1));
  assert(x.size(1) % 8 == 0);
  assert(y.size(1) % 8 == 0);
  assert(x.size(1) == y.size(1));
  assert(x.size(1) == y.size(1));
  assert(x.device().is_cuda());
  assert(y.device().is_cuda());
  assert(x.dtype() == torch::kFloat32);
  assert(y.dtype() == torch::kFloat32);

  const auto seq_len = x.size(0);
  const auto head_dim = x.size(1);
  const auto rope_dim = head_dim / 2;
  dim3 nblks(seq_len);
  dim3 nthrs(rope_dim / 4);
  rope_kernel_v2<<<nblks, nthrs>>>(x.data_ptr<float>(), y.data_ptr<float>(), seq_len, rope_dim);
  return;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("apply_rope_v1", &apply_rope_v1, "Apply rope v1");
}
#else
static inline void cuda_check(cudaError_t err, const char* what) {
  if (err != cudaSuccess) {
    std::fprintf(stderr, "CUDA error: %s: %s\n", what, cudaGetErrorString(err));
    std::exit(1);
  }
}

static void rope_ref_uninterleaved(const float* x, float* y, int seq_len, int rope_dim) {
  const int head_dim = rope_dim * 2;
  for (int s = 0; s < seq_len; ++s) {
    for (int tid = 0; tid < rope_dim; ++tid) {
      const float x1 = x[s * head_dim + tid];
      const float x2 = x[s * head_dim + tid + rope_dim];
      const float inv_freq = 1.0f / std::pow(theta_scale, (2.0f * static_cast<float>(tid)) /
                                                              static_cast<float>(head_dim));
      const float angle = static_cast<float>(s) * inv_freq;
      const float si = std::sin(angle);
      const float co = std::cos(angle);
      y[s * head_dim + tid] = x1 * co - x2 * si;
      y[s * head_dim + tid + rope_dim] = x1 * si + x2 * co;
    }
  }
}

int main() {
  // Build with: nvcc -O3 -DROPE_TEST_MAIN 95-flashinfer-rope.cu -o rope_test
  // This validates the "uninterleaved" layout: first half x1, second half x2.
  const int seq_len = 128;
  const int rope_dim = 1024;  // head_dim = 2048
  const int head_dim = rope_dim * 2;
  const size_t n = static_cast<size_t>(seq_len) * static_cast<size_t>(head_dim);
  const size_t bytes = n * sizeof(float);

  std::vector<float> h_x(n), h_y_v2(n, 0.0f), h_y_ref(n, 0.0f);
  // Deterministic pseudo-random input.
  uint32_t state = 123456789u;
  auto next_f = [&]() -> float {
    state = 1664525u * state + 1013904223u;
    // Map to [-1, 1]
    const float u = static_cast<float>(state) / static_cast<float>(0xFFFFFFFFu);
    return 2.0f * u - 1.0f;
  };
  for (size_t i = 0; i < n; ++i) h_x[i] = next_f();

  rope_ref_uninterleaved(h_x.data(), h_y_ref.data(), seq_len, rope_dim);

  float* d_x = nullptr;
  float* d_y = nullptr;
  cuda_check(cudaMalloc(&d_x, bytes), "cudaMalloc d_x");
  cuda_check(cudaMalloc(&d_y, bytes), "cudaMalloc d_y");
  cuda_check(cudaMemcpy(d_x, h_x.data(), bytes, cudaMemcpyHostToDevice), "H2D x");

  dim3 nblks(seq_len);
  // v2 uses float4: one thread handles 4 rope elements.
  // NOTE: current v2 assumes rope_dim % 4 == 0 (true for rope_dim=1024 here).
  dim3 nthrs(rope_dim / 4);
  rope_kernel_v2<<<nblks, nthrs>>>(d_x, d_y, seq_len, rope_dim);
  cuda_check(cudaGetLastError(), "kernel launch v2");
  cuda_check(cudaDeviceSynchronize(), "cudaDeviceSynchronize v2");

  cuda_check(cudaMemcpy(h_y_v2.data(), d_y, bytes, cudaMemcpyDeviceToHost), "D2H y_v2");
  cuda_check(cudaFree(d_x), "cudaFree d_x");
  cuda_check(cudaFree(d_y), "cudaFree d_y");

  float max_abs_diff = 0.0f;
  size_t max_i = 0;
  for (size_t i = 0; i < n; ++i) {
    const float diff = std::fabs(h_y_v2[i] - h_y_ref[i]);
    if (diff > max_abs_diff) {
      max_abs_diff = diff;
      max_i = i;
    }
  }

  std::printf("RoPE v2 (float4) uninterleaved test: seq_len=%d rope_dim=%d head_dim=%d\n", seq_len,
              rope_dim, head_dim);
  std::printf("Max abs diff = %.9g at i=%zu (gpu=%.9g ref=%.9g)\n", max_abs_diff, max_i,
              h_y_v2[max_i], h_y_ref[max_i]);

  // __powf is fast-math; allow a small tolerance.
  const float tol = 2e-4f;
  if (max_abs_diff > tol) {
    std::fprintf(stderr, "FAIL: max_abs_diff %.9g > tol %.9g\n", max_abs_diff, tol);
    return 1;
  }
  std::printf("PASS\n");
  return 0;
}
#endif