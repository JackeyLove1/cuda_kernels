#include <cuda_runtime.h>
#include <cstdint>

#define CEIL(a, b) (((a) + (b) - 1) / (b))
__constant__ float kernel_constants[2048];

constexpr int TILE_SIZE = 1024;
constexpr int REG_SIZE = 4;
constexpr int THREADS_PER_BLOCK = 256;
constexpr int TM = (TILE_SIZE + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;  // 4

__device__ __forceinline__ float get_kernel_value(const int idx, const int kernel_size) {
  return (idx >= 0 && idx < kernel_size) ? kernel_constants[idx] : 0.0f;
}

__global__ void convolution_1d_kernel(const float* __restrict__ input, float* __restrict__ output,
                                      const int input_size, const int kernel_size) {
  extern __shared__ float tile_cache[];
  const int tx = threadIdx.x;
  const int tile_base = blockIdx.x * TILE_SIZE;
  const int tile_span = TILE_SIZE + kernel_size - 1;

  // Cooperative load into shared memory with float4 vector path and explicit zero padding.
  int load_base = tile_base + tx * REG_SIZE;
  for (int offset = 0; offset + tx * REG_SIZE < tile_span; offset += blockDim.x * REG_SIZE) {
    const int smem_idx = offset + tx * REG_SIZE;
    const int gmem_idx = load_base + offset;
    if (smem_idx + 3 < tile_span && gmem_idx + 3 < input_size &&
        ((reinterpret_cast<uintptr_t>(input + gmem_idx) & static_cast<uintptr_t>(0xF)) == 0U)) {
      reinterpret_cast<float4*>(&tile_cache[smem_idx])[0] =
          reinterpret_cast<const float4*>(&input[gmem_idx])[0];
    } else {
#pragma unroll
      for (int j = 0; j < REG_SIZE; ++j) {
        const int cur_s = smem_idx + j;
        const int cur_g = gmem_idx + j;
        if (cur_s < tile_span) {
          tile_cache[cur_s] = (cur_g < input_size) ? input[cur_g] : 0.0f;
        }
      }
    }
  }
  __syncthreads();

  const int start = tx * TM;
  float reg[REG_SIZE] = {0.0f, 0.0f, 0.0f, 0.0f};
  float out[TM] = {0.0f, 0.0f, 0.0f, 0.0f};

  // Keep 8 kernel values in two float4 vectors:
  // low:[k-4..k-1], high:[k..k+3].
  float4 k_vec_low = make_float4(get_kernel_value(-4, kernel_size), get_kernel_value(-3, kernel_size),
                                 get_kernel_value(-2, kernel_size), get_kernel_value(-1, kernel_size));
  float4 k_vec_high;
  if (kernel_size >= 4) {
    k_vec_high = reinterpret_cast<const float4*>(&kernel_constants[0])[0];
  } else {
    k_vec_high = make_float4(get_kernel_value(0, kernel_size), get_kernel_value(1, kernel_size),
                             get_kernel_value(2, kernel_size), get_kernel_value(3, kernel_size));
  }

  for (int k_off = 0; k_off < TM + kernel_size - 1; k_off += REG_SIZE) {
    if (start + k_off + 3 < tile_span) {
      reinterpret_cast<float4*>(reg)[0] = reinterpret_cast<float4*>(&tile_cache[start + k_off])[0];
    } else {
#pragma unroll
      for (int i = 0; i < REG_SIZE; ++i) {
        const int idx = start + k_off + i;
        reg[i] = (idx < tile_span) ? tile_cache[idx] : 0.0f;
      }
    }

    out[0] = fmaf(reg[0], k_vec_high.x, out[0]);
    out[0] = fmaf(reg[1], k_vec_high.y, out[0]);
    out[0] = fmaf(reg[2], k_vec_high.z, out[0]);
    out[0] = fmaf(reg[3], k_vec_high.w, out[0]);

    out[1] = fmaf(reg[0], k_vec_low.w, out[1]);
    out[1] = fmaf(reg[1], k_vec_high.x, out[1]);
    out[1] = fmaf(reg[2], k_vec_high.y, out[1]);
    out[1] = fmaf(reg[3], k_vec_high.z, out[1]);

    out[2] = fmaf(reg[0], k_vec_low.z, out[2]);
    out[2] = fmaf(reg[1], k_vec_low.w, out[2]);
    out[2] = fmaf(reg[2], k_vec_high.x, out[2]);
    out[2] = fmaf(reg[3], k_vec_high.y, out[2]);

    out[3] = fmaf(reg[0], k_vec_low.y, out[3]);
    out[3] = fmaf(reg[1], k_vec_low.z, out[3]);
    out[3] = fmaf(reg[2], k_vec_low.w, out[3]);
    out[3] = fmaf(reg[3], k_vec_high.x, out[3]);

    k_vec_low = k_vec_high;
    const int next_k = k_off + 4;
    if (next_k + 3 < kernel_size) {
      k_vec_high = reinterpret_cast<const float4*>(&kernel_constants[next_k])[0];
    } else {
      k_vec_high.x = get_kernel_value(next_k, kernel_size);
      k_vec_high.y = get_kernel_value(next_k + 1, kernel_size);
      k_vec_high.z = get_kernel_value(next_k + 2, kernel_size);
      k_vec_high.w = get_kernel_value(next_k + 3, kernel_size);
    }
  }

  const int output_size = input_size - kernel_size + 1;
  const int out_base = tile_base + start;
  for (int i = 0; i < TM; i += 4) {
    if (out_base + i + 3 < output_size &&
        ((reinterpret_cast<uintptr_t>(output + out_base + i) & static_cast<uintptr_t>(0xF)) == 0U)) {
      reinterpret_cast<float4*>(&output[out_base + i])[0] = reinterpret_cast<float4*>(&out[i])[0];
    } else {
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        const int out_idx = out_base + i + j;
        if (out_idx < output_size) {
          output[out_idx] = out[i + j];
        }
      }
    }
  }
}

// input, kernel, output are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* input, const float* kernel, float* output, int input_size,
                      int kernel_size) {
  const int output_size = input_size - kernel_size + 1;
  if (output_size <= 0 || kernel_size <= 0 || kernel_size > 2048) {
    return;
  }
  const int threadsPerBlock = THREADS_PER_BLOCK;
  const int blocksPerGrid = CEIL(output_size, TILE_SIZE);
  cudaMemcpyToSymbol(kernel_constants, kernel, kernel_size * sizeof(float));
  const size_t shared_mem_bytes = static_cast<size_t>(TILE_SIZE + kernel_size - 1) * sizeof(float);
  convolution_1d_kernel<<<blocksPerGrid, threadsPerBlock, shared_mem_bytes>>>(input, output, input_size,
                                                                               kernel_size);
  cudaDeviceSynchronize();
}