#include <cuda_runtime.h>
#include <cuda.h>
#include <cuda_fp16.h>
#include <cstdio>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <cub/cub.cuh>

#include "utils.cuh"

__global__ void permuteKernel(float f_val) {
  const auto u = __float_as_uint(f_val);
  uint32_t b0 = __byte_perm(u, 0, 0x0);
  uint32_t b1 = __byte_perm(u, 1, 0x1);
  uint32_t b2 = __byte_perm(u, 2, 0x2);
  uint32_t b3 = __byte_perm(u, 3, 0x3);

  printf("Float Value: %f\n", f_val);
  printf("Full Hex   : 0x%08x\n", u);
  printf("Byte 0 (Addr+0): 0x%02x  <-- 低位在前\n", b0);
  printf("Byte 1 (Addr+1): 0x%02x\n", b1);
  printf("Byte 2 (Addr+2): 0x%02x\n", b2);
  printf("Byte 3 (Addr+3): 0x%02x  <-- 高位在后\n", b3);
}

__device__ __forceinline__ int wrap_reduce_sum(int val) {
  for (int offset = 16; offset > 0; offset >>= 1) {
    val += __shfl_down_sync(0xffffffff, val, offset);
  }
  return val;
}

__global__ void wrapReduceSumKernel() {}

template <typename T>
__global__ void cubReduceSum(const T* __restrict__ input, T* output,
                             const int N) {
  using BlockReduce = cub::BlockReduce<float, BLOCK_THREADS>;
  __shared__ typename BlockReduce::TempStorage temp_storage;

  const auto idx = blockDim.x * blockIdx.x + threadIdx.x;
  auto data = (idx < N) ? input[idx] : T{};
  auto blockSum = BlockReduce(temp_storage).Sum(data);
  if (threadIdx.x == 0) {
    output[blockIdx.x] = blockSum;
  }

}

int main() {
  // permuteKernel<<<1, 1>>>(2.0f);

  const size_t N = 1e8;
  const size_t threadPerBlock = BLOCK_THREADS;
  const auto blockPerGrid = ceil_div(N, threadPerBlock);
  thrust::host_vector<int> hv(N);
  thrust::fill(hv.begin(), hv.end(), 1);

  thrust::device_vector<int> input = hv;
  thrust::device_vector<int> output(blockPerGrid);


  cubReduceSum<<<blockPerGrid, threadPerBlock>>>(
      thrust::raw_pointer_cast(input.data()),
      thrust::raw_pointer_cast(output.data()),
      N);

  cudaDeviceSynchronize();
  thrust::host_vector<int> output_host = output;
  for (const auto& num : output_host) {
    std::cout << num << " ";
  }


  auto err = cudaGetLastError();
  if (err != cudaSuccess) {
    std::cerr << "Error: " << cudaGetErrorString(err) << std::endl;
  }
  return 0;
}