// nvcc >= 13.1, e.g. /usr/local/cuda-13.1/bin/nvcc -arch=sm_120 -O0 -lineinfo 184-um.cu -o 184
// run: LD_LIBRARY_PATH=/usr/local/cuda-13.1/lib64 ./184
#include <cuda.h>
#include <cuda_runtime.h>

#include <cstdio>
#include <cstring>

// Unified Memory: same address valid on host and device (sm_120 / Blackwell)
__managed__ char type_global[] = "global";
__managed__ char global_string[32] = "Hello World";

__global__ void kernel(const char* type, const char* data) {
  static const int n_char = 8;
  printf("%s - first %d characters: '", type, n_char);
  for (int i = 0; i < n_char; ++i) printf("%c", data[i]);
  printf("'\n");
}

int main() {
  // Kernel printf goes to a device FIFO; must set non-zero size or nothing is printed
  cudaError_t e = cudaDeviceSetLimit(cudaLimitPrintfFifoSize, 1024 * 1024);
  if (e != cudaSuccess) fprintf(stderr, "cudaDeviceSetLimit: %s\n", cudaGetErrorString(e));
  kernel<<<1, 1>>>(type_global, global_string);
  e = cudaDeviceSynchronize();
  if (e != cudaSuccess) fprintf(stderr, "cudaDeviceSynchronize: %s\n", cudaGetErrorString(e));
  fflush(stdout);
  return 0;
}