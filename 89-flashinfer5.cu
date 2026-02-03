#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <cstdio>
#include <cuda_fp4.hpp>
#include <cute/config.hpp>


// lane id and warp id PTX
__forceinline__ __device__ int get_lane_id() {
  int land_id;
  asm("mov.u32 %0, %%laneid;" : "=r"(land_id));
  return land_id;
}

__forceinline__ __device__ int get_warp_id() {
  int warp_id;
  asm("mov.u32 %0, %%wrapid;" : "=r"(warp_id));
  return warp_id;
}

// 2. Kernel 函数
__global__ void warp_leader_demo_kernel() {
  int global_tid = blockIdx.x * blockDim.x + threadIdx.x;

  int lane_id = get_lane_id();

  int val = 1;

  for (int offset = 16; offset > 0; offset /= 2) {
    val += __shfl_down_sync(0xFFFFFFFF, val, offset);
  }

  if (lane_id == 0) {
    printf("Block: %d, Warp ID (in block): %d, Result: %d\n", blockIdx.x, threadIdx.x / 32, val);
  }
}

// store and load PTX
__forceinline__ __device__ int4 ld_na_global_v4(const int4* addr) {
  int4 val;
  asm volatile("ld.global.cs.v4.b32 {%0, %1, %2, %3}, [%4];"
               : "=r"(val.x), "=r"(val.y), "=r"(val.w), "=r"(val.z)
               : "l"(addr));
  return val;
}

__forceinline__ __device__ void st_na_global_v4(int4* addr, int4 val) {
  // 注意：这里需要 4 个输入寄存器 {%1, %2, %3, %4}
  asm volatile("st.global.cs.v4.b32 [%0], {%1, %2, %3, %4};" ::"l"(addr), "r"(val.x), "r"(val.y),
               "r"(val.z), "r"(val.w));
}

__device__ __forceinline__ void st_global_release(int4 const& val, int4* addr) {
  asm volatile("st.release.global.sys.v4.b32 [%4], {%0, %1, %2, %3};"
               :
               : "r"(val.x), "r"(val.y), "r"(val.z), "r"(val.w), "l"(addr)
               : "memory");
}

__device__ __forceinline__ int4 ld_global_acquire(int4* addr) {
  int4 val;
  asm volatile("ld.acquire.global.sys.v4.b32 {%0, %1, %2, %3}, [%4];"
               : "=r"(val.x), "=r"(val.y), "=r"(val.z), "=r"(val.w)
               : "l"(addr)
               : "memory");
  return val;
}

__device__ __forceinline__ void st_global_volatile(int4 const& val, int4* addr) {
  asm volatile("st.volatile.global.v4.b32 [%4], {%0, %1, %2, %3};" ::"r"(val.x), "r"(val.y),
               "r"(val.z), "r"(val.w), "l"(addr));
}

__device__ __forceinline__ int4 ld_global_volatile(int4* addr) {
  int4 val;
  asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];"
               : "=r"(val.x), "=r"(val.y), "=r"(val.z), "=r"(val.w)
               : "l"(addr));
  return val;
}

#define INLINE __forceinline__ __device__

template <typename dst, typename src>
struct vec_cast {
  template <size_t vec_size>
  INLINE static void cast(dst* d, const src* s) {
#pragma unroll
    for (size_t i = 0; i < vec_size; ++i) {
      d[i] = static_cast<dst>(s[i]);
    }
  }
};

template <>
struct vec_cast<__nv_fp8_e4m3, float> {
  template <size_t vec_size>
  INLINE static void cast(__nv_fp8_e4m3* dst, const float* src) {
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


template <typename T>
constexpr INLINE int get_exponent_bits() {
  if constexpr (std::is_same_v<T, __nv_fp4_e2m1>) {
    return 4;
  } else if constexpr (std::is_same_v<T, __nv_fp8_e4m3>) {
    return 4;
  } else if constexpr (std::is_same_v<T, __nv_fp8_e5m2>) {
    return 5;
  } else if constexpr (std::is_same_v<T, half>) {
    return 5;
  } else if constexpr (std::is_same_v<T, __nv_bfloat16>) {
    return 8;
  }
  CUTE_GCC_UNREACHABLE;
}

constexpr float log2e = 1.44269504088896340736f;

constexpr float loge2 = 1.0f / log2e;

constexpr float inf = 5e4;

__forceinline__ __device__ half2 uint32_as_half2(uint32_t x) { return *(half2*)&x; }

__forceinline__ __device__ uint32_t half2_as_uint32(half2 x) { return *(uint32_t*)&x; }

template<typename T>
INLINE T ptx_exp2(T x) {
  if constexpr (std::is_same_v<T, float>) {
    float y;
    asm volatile ("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
  } else if constexpr (std::is_same_v<T, half2>) {
    uint32_t y_u32;
    uint32_t x_u32 = half2_as_uint32(x);
    asm volatile("ex2.approx.f16x2 %0, %1;" : "=r"(y_u32) : "r"(x_u32));
    return uint32_as_half2(y_u32);
  } else if constexpr (std::is_same_v<T, half>) {
    ushort y_u16;
    asm volatile("ex2.approx.f16 %0, %1;" : "=h"(y_u16) : "h"(__half_as_ushort(x)));
    return __ushort_as_half(y_u16);
  }
  CUTE_GCC_UNREACHABLE;
}

template <typename T>
INLINE T ptx_log2(T x) {
  if constexpr (std::is_same_v<T, float>) {
    float y;
    asm volatile("lg2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
  } else if constexpr (std::is_same_v<T, half2>) {
    uint32_t y_u32;
    uint32_t x_u32 = half2_as_uint32(x);
    asm volatile("lg2.approx.f16x2 %0, %1;" : "=r"(y_u32) : "r"(x_u32));
    return uint32_as_half2(y_u32);
  } else if constexpr (std::is_same_v<T, half>) {
    ushort y_u16;
    asm volatile("lg2.approx.f16 %0, %1;" : "=h"(y_u16) : "h"(__half_as_ushort(x)));
    return __ushort_as_half(y_u16);
  }
  CUTE_GCC_UNREACHABLE;
}

int main() {
  warp_leader_demo_kernel<<<2, 64>>>();

  // 同步设备以确保 printf 输出显示
  cudaDeviceSynchronize();


  return 0;
}