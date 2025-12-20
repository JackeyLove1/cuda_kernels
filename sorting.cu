#include <cuda_runtime.h>
#include <cub/cub.cuh>
#include <cstdio>

#define CUDA_CHECK(call)                                                     \
do {                                                                       \
cudaError_t _e = (call);                                                 \
if (_e != cudaSuccess) {                                                 \
std::fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,     \
cudaGetErrorString(_e));                                  \
return;                                                                \
}                                                                        \
} while (0)

// a, b are device pointers to int32 arrays
extern "C" void solution(const int* a, int* b, size_t n) {
    if (!a || !b || n == 0) return;

    // 1) Query temp storage size
    void* d_temp = nullptr;
    size_t temp_bytes = 0;

    // Sort full 32 bits of int
    CUDA_CHECK(cub::DeviceRadixSort::SortKeys(
        d_temp, temp_bytes,
        a, b,
        static_cast<int>(n),
        /*begin_bit=*/0,
        /*end_bit=*/sizeof(int) * 8,
        /*stream=*/0));

    // 2) Allocate temp storage
    CUDA_CHECK(cudaMalloc(&d_temp, temp_bytes));

    // 3) Run sort
    CUDA_CHECK(cub::DeviceRadixSort::SortKeys(
        d_temp, temp_bytes,
        a, b,
        static_cast<int>(n),
        0, sizeof(int) * 8,
        0));

    // 4) Cleanup
    CUDA_CHECK(cudaFree(d_temp));

    // Optional: ensure completion if caller expects b ready immediately
    CUDA_CHECK(cudaDeviceSynchronize());
}
