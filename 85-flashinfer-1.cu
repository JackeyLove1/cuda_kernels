#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_fp4.h>
#include <cuda_bf16.h>
#include <bit>
#include <cstdio>
#include <cstdint>
#include <iostream>
#include <iomanip>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#define FLASHINFER_INLINE inline __attribute__((always_inline)) __device__

template <typename dst_t, typename src_t>
struct vec_cast {
  template <size_t vec_size>
  FLASHINFER_INLINE static void cast(dst_t* dst, const src_t* src) {
    #pragma unroll
    for (size_t i = 0; i < vec_size; ++i) {
      dst[i] = (dst_t)src[i];
    }
  }
};

template <>
struct vec_cast<__nv_fp8_e4m3, float> {
  template <size_t vec_size>
  FLASHINFER_INLINE static void cast(__nv_fp8_e4m3* dst, const float* src) {
    if constexpr (vec_size == 1) {
      dst[0] = __nv_fp8_e4m3(src[0]);
    } else {
      #pragma unroll
      for (size_t i = 0; i < vec_size / 2; ++i) {
        ((__nv_fp8x2_storage_t*)dst)[i] =
          __nv_cvt_float2_to_fp8x2(((float2*)src)[i], __NV_SATFINITE, __NV_E4M3);
      }
    }
  }
};


template <>
struct vec_cast<float, half> {
  template <size_t vec_size>
  FLASHINFER_INLINE static void cast(float* dst, const half* src) {
    if constexpr (vec_size == 1) {
      dst[0] = (float)src[0];
    } else {
      #pragma unroll
      for (size_t i = 0; i < vec_size / 2; ++i) {
        ((float2*)dst)[i] = __half22float2(((half2*)src)[i]);
      }
    }
  }
};

__device__ __forceinline__ void st_global_release(int4 const & val, int4* addr) {
  asm volatile("st.release.global.sys.v4.b32 [%4], {%0, %1, %2, %3};" ::"r"(val.x), "r"(val.y),
               "r"(val.z), "r"(val.w), "l"(addr));
}

template <typename T>
constexpr FLASHINFER_INLINE int get_exponent_bits() {
  if constexpr (std::is_same_v<T, __nv_fp8_e4m3>) {
    return 4;
  } else if constexpr (std::is_same_v<T, __nv_fp8_e5m2>) {
    return 5;
  } else if constexpr (std::is_same_v<T, half>) {
    return 5;
  } else if constexpr (std::is_same_v<T, nv_bfloat16>) {
    return 8;
  }
}

template <typename T>
constexpr FLASHINFER_INLINE int get_mantissa_bits() {
  if constexpr (std::is_same_v<T, __nv_fp8_e4m3>) {
    return 3;
  } else if constexpr (std::is_same_v<T, __nv_fp8_e5m2>) {
    return 2;
  } else if constexpr (std::is_same_v<T, half>) {
    return 11;
  } else if constexpr (std::is_same_v<T, nv_bfloat16>) {
    return 7;
  }
}

template <typename fp8_dtype, typename fp16_dtype>
__device__ void fast_dequant_f8f16x4(uint32_t* input, uint2* output) {
  uint32_t q = *input;
  if constexpr (std::is_same_v<fp8_dtype, __nv_fp8_e5m2> && std::is_same_v<fp16_dtype, half>) {
    output->x = __byte_perm(0U, q, 0x5140);
    output->y = __byte_perm(0U, q, 0x7362);
  } else {
    constexpr int FP8_EXPONENT = get_exponent_bits<fp8_dtype>();
    constexpr int FP8_MANTISSA = get_mantissa_bits<fp8_dtype>();
    constexpr int FP16_EXPONENT = get_exponent_bits<fp16_dtype>();

    constexpr int RIGHT_SHIFT = FP16_EXPONENT - FP8_EXPONENT;
    // Calculate MASK for extracting mantissa and exponent
    constexpr int MASK1 = 0x80000000;
    constexpr int MASK2 = MASK1 >> (FP8_EXPONENT + FP8_MANTISSA);
    constexpr int MASK3 = MASK2 & 0x7fffffff;
    constexpr int MASK = MASK3 | (MASK3 >> 16);
    q = __byte_perm(q, q, 0x1302);

    // Extract and shift FP8 values to FP16 format
    uint32_t Out1 = (q & 0x80008000) | ((q & MASK) >> RIGHT_SHIFT);
    uint32_t Out2 = ((q << 8) & 0x80008000) | (((q << 8) & MASK) >> RIGHT_SHIFT);

    constexpr int BIAS_OFFSET = (1 << (FP16_EXPONENT - 1)) - (1 << (FP8_EXPONENT - 1));
    // Construct and apply exponent bias
    if constexpr (std::is_same_v<fp16_dtype, half>) {
      const half2 bias_reg = __float2half2_rn(float(1 << BIAS_OFFSET));

      // Convert to half2 and apply bias
      *(half2*)&(output->x) = __hmul2(*reinterpret_cast<const half2*>(&Out1), bias_reg);
      *(half2*)&(output->y) = __hmul2(*reinterpret_cast<const half2*>(&Out2), bias_reg);
    } else {
      constexpr uint32_t BIAS = (BIAS_OFFSET + 127) << 23;
      const nv_bfloat162 bias_reg = __float2bfloat162_rn(*reinterpret_cast<const float*>(&BIAS));
      // Convert to bfloat162 and apply bias
      *(nv_bfloat162*)&(output->x) =
          __hmul2(*reinterpret_cast<const nv_bfloat162*>(&Out1), bias_reg);
      *(nv_bfloat162*)&(output->y) =
          __hmul2(*reinterpret_cast<const nv_bfloat162*>(&Out2), bias_reg);
    }
  }
}


__global__ void shuflPermKernel(uint32_t* data_in_out) {
  unsigned int land_id = threadIdx.x % 32;

  uint32_t x = data_in_out[threadIdx.x];

  uint32_t tmp_shfl = __shfl_xor_sync(0xffffffff, x, 1);

  x = __byte_perm(x, tmp_shfl, 0x3254);

  data_in_out[threadIdx.x] = x;

}

int main() {
  const auto NUM_THREADS = 4;
  const auto BLOCK_SIZE = NUM_THREADS;
  const auto GRID = 1;

  // Host 端数据
  thrust::host_vector<uint32_t> h_data(NUM_THREADS);

  // 初始化数据
  h_data[0] = 0x00001100; // 线程 0 持有 '11'
  h_data[1] = 0x00002200; // 线程 1 持有 '22'
  h_data[2] = 0x00003300; // 线程 2 持有 '33'
  h_data[3] = 0x00004400; // 线程 3 持有 '44'

  std::cout << "--- Initial Host Data ---" << std::endl;
  for (int i = 0; i < NUM_THREADS; ++i) {
    std::cout << "h_data[" << i << "]: 0x"
              << std::hex << std::setfill('0') << std::setw(8) << h_data[i] << std::endl;
  }
  std::cout << std::dec << std::endl; // 恢复十进制输出

  thrust::device_vector<uint32_t> d_data = h_data;

  shuflPermKernel<<<GRID, BLOCK_SIZE>>>(thrust::raw_pointer_cast(d_data.data()));


  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    std::cerr << "CUDA Error: " << cudaGetErrorString(err) << std::endl;
    return 1;
  }

}