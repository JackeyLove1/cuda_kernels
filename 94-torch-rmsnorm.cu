#include <cstdio>
#include <cassert>
#include <vector>
#include <random>
#include <cmath>
#include <numeric>
#include <cuda_runtime.h>
#include <cuda_pipeline.h>
#include <cooperative_groups.h>
#include <cub/cub.cuh>
#include <torch/extension.h>
#include <torch/types.h>
#include <ATen/cuda/CUDAContext.h>
#include <pybind11/pybind11.h>

#include <include/utils.cuh>

namespace cg = cooperative_groups;
namespace py = pybind11;

template <uint32_t vec_size, typename T>
__global__ void RMSNormKernel(const T* __restrict__ input, const T* __restrict__ weight,
                             T* __restrict__ output, uint32_t d, uint32_t stride_input,
                             uint32_t stride_output, float eps) {
  const uint32_t tx = threadIdx.x;
  const uint32_t ty = threadIdx.y;
  constexpr uint32_t warp_size = 32;
  const uint32_t warp_id = ty;
  const uint32_t lane_id = tx;
  const uint32_t num_warps = blockDim.y;
  const uint32_t num_threads = warp_size * num_warps;
  const uint32_t tid = tx + ty * warp_size;
  const uint32_t rounds = CEIL_DIV(d, vec_size * num_threads);
  extern __shared__ float smem[];  // size: num_warps * sizeof(float)

  using vec_t = flashinfer::vec_t<T, vec_size>;

  // 1) sum(x^2) over d
  float sum_sq = 0.f;
  for (uint32_t i = 0; i < rounds; ++i) {
    vec_t input_vec;
    input_vec.fill(T(0));
    const uint32_t d_offset = (i * num_threads + tid) * vec_size;
    if (d_offset < d) {
      const uint32_t batch_offset = uint32_t(blockIdx.x) * stride_input;
      input_vec.load(input + batch_offset + d_offset);
    }
#pragma unroll
    for (uint32_t j = 0; j < vec_size; ++j) {
      float x = (float)input_vec[j];
      sum_sq += x * x;
    }
  }

  // 2) reduce within each warp
#pragma unroll
  for (uint32_t offset = warp_size / 2; offset > 0; offset >>= 1) {
    sum_sq += __shfl_down_sync(FULL_MASK, sum_sq, offset, warp_size);
  }
  if (lane_id == 0) smem[warp_id] = sum_sq;
  __syncthreads();

  // 3) warp0 reduces partial sums from all warps
  float block_sum_sq = 0.f;
  if (warp_id == 0) {
    block_sum_sq = (lane_id < num_warps) ? smem[lane_id] : 0.f;
#pragma unroll
    for (uint32_t offset = warp_size / 2; offset > 0; offset >>= 1) {
      block_sum_sq += __shfl_down_sync(FULL_MASK, block_sum_sq, offset, warp_size);
    }
    if (lane_id == 0) smem[0] = block_sum_sq;
  }
  __syncthreads();

  const float rms_rcp = flashinfer::math::rsqrt(smem[0] / (float)d + eps);

  // 4) y = x * rms_rcp * weight
  for (uint32_t i = 0; i < rounds; ++i) {
    vec_t input_vec;
    vec_t weight_vec;
    vec_t output_vec;
    input_vec.fill(T(0));
    weight_vec.fill(T(0));

    const uint32_t d_offset = (i * num_threads + tid) * vec_size;
    if (d_offset < d) {
      const uint32_t batch_offset_in = uint32_t(blockIdx.x) * stride_input;
      input_vec.load(input + batch_offset_in + d_offset);
      weight_vec.load(weight + d_offset);
    }

#pragma unroll
    for (uint32_t j = 0; j < vec_size; ++j) {
      float x = (float)input_vec[j];
      float w = (float)weight_vec[j];
      output_vec[j] = (T)(x * rms_rcp * w);
    }

    if (d_offset < d) {
      const uint32_t batch_offset_out = uint32_t(blockIdx.x) * stride_output;
      output_vec.store(output + batch_offset_out + d_offset);
    }
  }
}

// input  shape: [batch_size, d]
// output shape: [batch_size, d]
// weight shape: [d]
template <typename T>
void RMSNorm(const T* input, const T* weight, T* output, uint32_t batch_size, uint32_t d,
             uint32_t stride_input, uint32_t stride_output, float eps = 1e-5f,
             cudaStream_t stream = 0) {
  assert(BLOCK_SIZE % 32 == 0);
  const uint32_t vec_size = (uint32_t)std::gcd((uint32_t)(16 / sizeof(T)), d);
  assert(d % vec_size == 0);
  const uint32_t num_warps = CEIL_DIV((uint32_t)BLOCK_SIZE, (uint32_t)32);
  const dim3 threads(32, num_warps);
  const dim3 blocks(batch_size);
  const size_t smem_bytes = num_warps * sizeof(float);

  switch (vec_size) {
    case 1:
      RMSNormKernel<1, T><<<blocks, threads, smem_bytes, stream>>>(input, weight, output, d,
                                                                   stride_input, stride_output,
                                                                   eps);
      break;
    case 2:
      RMSNormKernel<2, T><<<blocks, threads, smem_bytes, stream>>>(input, weight, output, d,
                                                                   stride_input, stride_output,
                                                                   eps);
      break;
    case 4:
      RMSNormKernel<4, T><<<blocks, threads, smem_bytes, stream>>>(input, weight, output, d,
                                                                   stride_input, stride_output,
                                                                   eps);
      break;
    case 8:
      RMSNormKernel<8, T><<<blocks, threads, smem_bytes, stream>>>(input, weight, output, d,
                                                                   stride_input, stride_output,
                                                                   eps);
      break;
    default:
      // fallback: smallest supported vec_size
      RMSNormKernel<1, T><<<blocks, threads, smem_bytes, stream>>>(input, weight, output, d,
                                                                   stride_input, stride_output,
                                                                   eps);
      break;
  }
  // Don't synchronize here. Benchmarking (and callers) should control synchronization.
}

torch::Tensor cuda_rms_norm(torch::Tensor input, torch::Tensor weight, float eps = 1e-5) {
  TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
  TORCH_CHECK(weight.is_cuda(), "weight must be a CUDA tensor");
  TORCH_CHECK(input.device() == weight.device(), "input and weight must be on the same device");
  TORCH_CHECK(input.dim() == 2, "input must have shape [batch, d]");
  TORCH_CHECK(weight.dim() == 1, "weight must have shape [d]");
  TORCH_CHECK(input.size(1) == weight.numel(), "weight.numel() must equal input.size(1)");
  TORCH_CHECK(input.scalar_type() == weight.scalar_type(), "input and weight must have same dtype");
  // Test-only: keep the binding simple and only support fp32.
  TORCH_CHECK(input.scalar_type() == at::kFloat, "only float32 (fp32) is supported in this test build");
  TORCH_CHECK(input.stride(1) == 1, "input must be contiguous in the last dimension (stride(1)==1)");
  TORCH_CHECK(weight.is_contiguous(), "weight must be contiguous");

  auto output = torch::empty_like(input);

  const uint32_t batch_size = static_cast<uint32_t>(input.size(0));
  const uint32_t d = static_cast<uint32_t>(input.size(1));
  const uint32_t stride_input = static_cast<uint32_t>(input.stride(0));
  const uint32_t stride_output = static_cast<uint32_t>(output.stride(0));
  const float eps_f = static_cast<float>(eps);
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  using T = float;
  RMSNorm<T>(input.data_ptr<T>(), weight.data_ptr<T>(), output.data_ptr<T>(), batch_size, d,
             stride_input, stride_output, eps_f, stream);

  // Lightweight kernel launch check (no device-wide synchronization).
  auto err = cudaGetLastError();
  TORCH_CHECK(err == cudaSuccess, "RMSNorm kernel launch failed: ", cudaGetErrorString(err));
  return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("rms_norm", &cuda_rms_norm, py::arg("input"), py::arg("weight"),
        py::arg("eps") = 1e-5, "RMSNorm (CUDA)");
  // Backward-compatible alias (older scripts used this name).
  m.def("rms_norm_lib", &cuda_rms_norm, py::arg("input"), py::arg("weight"),
        py::arg("eps") = 1e-5, "RMSNorm (CUDA) [alias]");
}

