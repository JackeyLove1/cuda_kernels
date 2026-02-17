#include <cuda_runtime.h>

#define CEIL(a, b) (((a) + (b) - 1) / (b))
template <typename>
struct dependent_false : std::false_type {};

template <typename T>
__forceinline__ __device__ auto LOAD4(const T* ptr) {
  if constexpr (std::is_same_v<T, int>) {
    return reinterpret_cast<const int4*>(ptr);
  } else if constexpr (std::is_same_v<T, unsigned int>) {
    return reinterpret_cast<const uint4*>(ptr);
  } else if constexpr (std::is_same_v<T, float>) {
    return reinterpret_cast<const float4*>(ptr);
  } else if constexpr (std::is_same_v<T, double>) {
    return reinterpret_cast<const double4*>(ptr);
  } else if constexpr (std::is_same_v<T, short>) {
    return reinterpret_cast<const short4*>(ptr);
  } else {
    static_assert(dependent_false<T>::value, "Unsupported type for LOAD4");
    return nullptr;
  }
}

template <typename T>
__forceinline__ __device__ auto STORE4(T* ptr) {
  if constexpr (std::is_same_v<T, int>) {
    return reinterpret_cast<int4*>(ptr);
  } else if constexpr (std::is_same_v<T, unsigned int>) {
    return reinterpret_cast<uint4*>(ptr);
  } else if constexpr (std::is_same_v<T, float>) {
    return reinterpret_cast<float4*>(ptr);
  } else if constexpr (std::is_same_v<T, double>) {
    return reinterpret_cast<double4*>(ptr);
  } else if constexpr (std::is_same_v<T, short>) {
    return reinterpret_cast<short4*>(ptr);
  } else {
    static_assert(dependent_false<T>::value, "Unsupported type for STORE4");
    return nullptr;
  }
}

// Invert RGB, keep alpha for one pixel (RGBA as uint32: bytes 0,1,2,3 = R,G,B,A)
__forceinline__ __device__ unsigned int invert_rgb32(unsigned int v) {
  const unsigned int r = (v >> 0) & 0xffu;
  const unsigned int g = (v >> 8) & 0xffu;
  const unsigned int b = (v >> 16) & 0xffu;
  const unsigned int a = (v >> 24) & 0xffu;
  return (a << 24) | ((255u - b) << 16) | ((255u - g) << 8) | (255u - r);
}

__global__ void invert_kernel(unsigned char* __restrict__ image, const int width,
                              const int height) {
  const int idx = blockDim.x * blockIdx.x + threadIdx.x;
  const int numPixels = width * height;
  const int stride = blockDim.x * gridDim.x;
  const int vec_size = numPixels >> 2;  // number of 128-bit (uint4) elements
  const int vec_tail = numPixels & 3;

  unsigned int* __restrict__ img = reinterpret_cast<unsigned int*>(image);

// 128-bit coalesced loads/stores (lg128) via __ldg
#pragma unroll
  for (int i = idx; i < vec_size; i += stride) {
    const uint4 p = __ldg(reinterpret_cast<const uint4*>(img + (i << 2)));
    const uint4 out =
        make_uint4(invert_rgb32(p.x), invert_rgb32(p.y), invert_rgb32(p.z), invert_rgb32(p.w));
    *reinterpret_cast<uint4*>(img + (i << 2)) = out;
  }

  // Tail: 1–3 pixels not covered by full uint4 (one thread per pixel)
  if (vec_tail > 0 && idx < vec_tail) {
    const int pix = (vec_size << 2) + idx;
    img[pix] = invert_rgb32(__ldg(img + pix));
  }
}

// image_input, image_output are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(unsigned char* image, int width, int height) {
  int numPixels = width * height;
  int threadsPerBlock = 256;
  int blocksPerGrid = CEIL(numPixels, threadsPerBlock * 4);

  invert_kernel<<<blocksPerGrid, threadsPerBlock>>>(image, width, height);
  cudaDeviceSynchronize();
}