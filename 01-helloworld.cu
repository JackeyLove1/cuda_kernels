#include <cstdio>
#include <cstdlib>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void print_from_gpu(){
    const auto tid = threadIdx.x;
    const auto bid = blockIdx.x;
    printf("print_from_gpu from thread [%d, %d] \n", tid, bid);
}


int main() {
    print_from_gpu<<<2, 4>>>();
    cudaDeviceSynchronize();
}