#include <cuda_runtime.h>
#include <cstdlib>
#include <cstdio>

__global__ void printDim()
{
    printf("grid dim: %d, block dim:%d,  tid: %d\n", gridDim.x, blockDim.x, threadIdx.x);
}

int main()
{
    printDim<<<2,4>>>();
    cudaDeviceSynchronize();
}