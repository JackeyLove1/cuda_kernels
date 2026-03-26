#include <cuda_runtime.h>
#include <stdio.h>

__global__ void print_from_gpu() {
  printf("print_from_gpu from thread [%d, %d] \n", threadIdx.x, blockIdx.x);
}

int main() {
  print_from_gpu<<<1, 1>>>();
  cudaDeviceSynchronize();
  return 0;
}