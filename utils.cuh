

#ifndef CUDA_EXAMPLES_UTILS_CUH
#define CUDA_EXAMPLES_UTILS_CUH

constexpr static auto FULL_MASK = 0xFFFFFFFF;

template <typename T>
__forceinline__ __device__ warpReduceSum(T val)
{
    #pragma unroll
    for (int offset = warpSize / 2; offset > 0; offset >>= 1)
    {
        val += __shfl_down_sync(FULL_MASK, val, offset);
    }
}

#define CUDA_CHECK(expr_to_check) do {            \
cudaError_t result  = expr_to_check;          \
if(result != cudaSuccess)                     \
{                                             \
fprintf(stderr,                           \
"CUDA Runtime Error: %s:%i:%d = %s\n", \
__FILE__,                         \
__LINE__,                         \
result,\
cudaGetErrorString(result));      \
}                                             \
} while(0)

#endif //CUDA_EXAMPLES_UTILS_CUH