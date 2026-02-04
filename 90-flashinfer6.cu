#include <cstdio>
#include <cuda_runtime.h>
#include <cuda/pipeline>
#include <cub/cub.cuh>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/sequence.h>
#include <thrust/transform.h>

#include <include/utils.cuh>


DEVICE_INLINE void cp_async_16(void* smem_ptr, void* gmem_ptr, int src_cap) {
  unsigned smem_int = __cvta_generic_to_shared(smem_ptr);
  asm volatile ("cp.async.cg.shared.global.L2::128B [%0], [%1], %2, %3;\n"
    :: "r"(smem_int), "l"(gmem_ptr), "n"(16), "r"(src_cap));
}

DEVICE_INLINE void cp_async_wait_all() {
  asm volatile("cp.async.wait_all;\n"::);
}

DEVICE_INLINE void cp_async_commit() {
  asm volatile("cp.async.commit_group;\n"::);
}

__global__ void async_copy_kernel(const int* __restrict__ g_in, int* g_out, int N) {
  extern __shared__ int s_data[];
  const auto tid = threadIdx.x;
  const auto gid = blockDim.x * blockIdx.x + threadIdx.x;
  if (gid < N) {
    cp_async_16((void*)&s_data[tid * 4], (void*)&g_in[gid * 4], (N - gid * 4) * sizeof(int));
    cp_async_commit();
    cp_async_wait_all();
    __syncthreads();
    int sum{0};
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      sum += s_data[tid * 4 + i];
    }
    g_out[gid] = sum;
  }
}

int main() {
  const int N = 1024;  // Number of threads (each processes 4 integers)
  const int data_size = N * 4;  // Total number of integers

  // Use Thrust host_vector for automatic memory management
  thrust::host_vector<int> h_in(data_size);
  thrust::host_vector<int> h_ref(N);

  // Initialize input data using Thrust
  for (int i = 0; i < data_size; ++i) {
    h_in[i] = i % 10;  // Values 0-9 repeating
  }

  // Compute reference results on CPU
  for (int i = 0; i < N; ++i) {
    int sum = 0;
    for (int j = 0; j < 4; ++j) {
      sum += h_in[i * 4 + j];
    }
    h_ref[i] = sum;
  }

  // Use Thrust device_vector for automatic GPU memory management
  // Copy from host to device automatically
  thrust::device_vector<int> d_in = h_in;
  thrust::device_vector<int> d_out(N);

  // Get raw pointers for kernel launch
  int* d_in_ptr = thrust::raw_pointer_cast(d_in.data());
  int* d_out_ptr = thrust::raw_pointer_cast(d_out.data());

  // Launch kernel
  const int block_size = 256;
  const int grid_size = (N + block_size - 1) / block_size;
  const int smem_size = block_size * 4 * sizeof(int);

  printf("Launching kernel with grid_size=%d, block_size=%d, smem_size=%d bytes\n",
         grid_size, block_size, smem_size);

  async_copy_kernel<<<grid_size, block_size, smem_size>>>(d_in_ptr, d_out_ptr, N);
  CHECK(cudaGetLastError());
  CHECK(cudaDeviceSynchronize());

  // Copy results back to host automatically using Thrust
  thrust::host_vector<int> h_out = d_out;

  // Verify results
  bool correct = true;
  int max_errors = 10;
  int error_count = 0;
  for (int i = 0; i < N; ++i) {
    if (h_out[i] != h_ref[i]) {
      if (error_count < max_errors) {
        printf("Mismatch at index %d: GPU=%d, CPU=%d\n", i, h_out[i], h_ref[i]);
      }
      error_count++;
      correct = false;
    }
  }

  if (correct) {
    printf("✓ Test PASSED! All %d results match.\n", N);
  } else {
    printf("✗ Test FAILED! %d/%d mismatches found.\n", error_count, N);
  }

  return correct ? 0 : 1;
}

