#include <cuda_runtime.h>
#include <cuda.h>
#include <cuda_fp16.h>
#include <cstdio>
#include <algorithm>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/fill.h>
#include <thrust/reduce.h>
#include <thrust/functional.h>
#include <cub/cub.cuh>
#include <cutlass/fast_math.h>
#include "utils.cuh"

// #include <torch/extension.h>
// #include <torch/types.h>


constexpr auto VEC_PER_THREAD = 4;

__global__ void CunKernel(const int* __restrict__ input,
  int* output, const int N) {
  const auto BlockLoadStragy = cub::BlockLoadAlgorithm::BLOCK_LOAD_STRIPED;

  const int4* input_vec = reinterpret_cast<const int4*>(input);
  using BlockLoadT = cub::BlockLoad<int4, BLOCK_THREADS, VEC_PER_THREAD, BlockLoadStragy> ;
  using BlockReduceT = cub::BlockReduce<int, BLOCK_THREADS>;

  __shared__ union {
    typename BlockLoadT::TempStorage load;
    typename BlockReduceT::TempStorage reduce;
  } temp_storage;

  int4 thread_data[VEC_PER_THREAD];
  // IMPORTANT: input_vec is int4*, so offsets/counts for BlockLoad are in int4 units (not int units).
  // This kernel assumes N is divisible by 4 (so N_vec is an integer) and input is sufficiently aligned.
  const int N_vec = N / 4;
  const int block_offset = blockIdx.x * BLOCK_THREADS * VEC_PER_THREAD; // int4 index
  const int remaining = MAX(N_vec - block_offset, 0);
  const int valid_items = MIN(BLOCK_THREADS * VEC_PER_THREAD, remaining);

  // load
  BlockLoadT(temp_storage.load).Load(input_vec + block_offset, thread_data, valid_items, make_int4(0, 0, 0, 0));
  // __threadfence_block();

  // reduce sum
  int thread_sum = 0;
  #pragma unroll
  for (int i = 0; i < VEC_PER_THREAD; ++i) {
    thread_sum += thread_data[i].x + thread_data[i].y + thread_data[i].z + thread_data[i].w;
  }
  __syncthreads();

  int block_sum = BlockReduceT(temp_storage.reduce).Sum(thread_sum);
  if (threadIdx.x == 0) {
    output[blockIdx.x] = block_sum;
  }
}

static void LaunchCunKernelThrust(int N) {
  thrust::device_vector<int> d_input(static_cast<size_t>(N));
  thrust::fill(d_input.begin(), d_input.end(), 1);

  // Vectorized int4 path: the kernel processes N_vec = N/4 int4 elements.
  // For correctness, N should be divisible by 4.
  const int N_vec = N / 4;
  const int items_per_block = BLOCK_THREADS * VEC_PER_THREAD; // int4 items per block
  const int num_blocks = ceil_div(N_vec, items_per_block);
  thrust::device_vector<int> d_output(static_cast<size_t>(num_blocks));

  cudaEvent_t start{}, stop{};
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));

  // warmup
  CunKernel<<<num_blocks, BLOCK_THREADS>>>(
    thrust::raw_pointer_cast(d_input.data()),
    thrust::raw_pointer_cast(d_output.data()),
    N);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  constexpr int kIters = 10;
  CUDA_CHECK(cudaEventRecord(start));
  for (int i = 0; i < kIters; ++i) {
    CunKernel<<<num_blocks, BLOCK_THREADS>>>(
      thrust::raw_pointer_cast(d_input.data()),
      thrust::raw_pointer_cast(d_output.data()),
      N);
  }
  CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaEventSynchronize(stop));
  CUDA_CHECK(cudaGetLastError());

  float ms = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));

  const long long got =
    thrust::reduce(d_output.begin(), d_output.end(), 0LL, thrust::plus<long long>());
  const long long expected = static_cast<long long>(N);

  std::printf("N=%d, blocks=%d, got=%lld, expected=%lld, time=%.3f ms (avg=%.3f ms)\n",
              N, num_blocks, got, expected, ms, ms / kIters);

  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));
}

int main() {
  constexpr int N = 1e8; // 1e8
  LaunchCunKernelThrust(N);
  return 0;
}

// ncu --set full --export flashinfer3_report -f --kernel-name-base function --kernel-name "CunKernel" --launch-count 1 ./flashinfer3
// accelerate speed --launch-skip 1 --launch-count 1

// version 1: N=100000000, blocks=97657, got=100000000, expected=100000000, time=369.211 ms (avg=36.921 ms)

// version 2: coalesced memory access N=100000000, blocks=97657, got=100000000, expected=100000000, time=369.783 ms (avg=36.978 ms)

// version 3: coalesced memory access + vector int4