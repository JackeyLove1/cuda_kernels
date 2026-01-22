#include <cuda.h>
#include <cuda_runtime.h>

#include <cassert>
#include "cuda_array.h"

// 1d grid block
__global__ void vector_add(const float * a, const float * b, float * out, size_t n) {
    const auto idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = a[idx] + b[idx];
    }
}

void launch_add(CudaArray& a, CudaArray& b, CudaArray& out) {
    const auto n = a.size();
    assert(a.size() == b.size());
    assert(a.size() == out.size());

    const int threadPerBlock = 256;
    const int blockPerGrid = (n + threadPerBlock - 1) / threadPerBlock;
    vector_add<<<blockPerGrid, threadPerBlock>>>(
        a.get_raw_ptr(),
        b.get_raw_ptr(),
        out.get_raw_ptr(),
        n);
    cudaDeviceSynchronize();
}