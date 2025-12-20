#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>


#define FULL_MASK 0xFFFFFFFF
#define WRAP_SIZE 32

template <typename T, typename K> __host__ __device__ __forceinline__ auto CDIV(T a, K b) { return (a + b - 1) / b; }

template <typename T> __device__ __forceinline__ T WarpReduceSum(T value)
{
#pragma unroll
    for (auto offset = WRAP_SIZE / 2; offset > 0; offset >>= 1) {
        value += __shfl_down_sync(FULL_MASK, value, offset);
    }
    return value;
}

__global__ void sumKernel(const float *__restrict__ input, float *__restrict__ output, const int N)
{
    __shared__ float smem[WRAP_SIZE];

    const auto tid    = threadIdx.x;
    const auto stride = blockDim.x * gridDim.x;
    const auto landId = tid & (WRAP_SIZE - 1);
    const auto warpId = tid / WRAP_SIZE;
    auto       id     = threadIdx.x + blockIdx.x * blockDim.x;

    float val = 0.0f;
    for (; id * 4 < N; id += stride) {
        const auto baseIdx   = id * 4;
        const auto remaining = N - baseIdx;
        if (remaining > 3) {
            const auto data = reinterpret_cast<const float4 *>(input)[id];
            val += data.x + data.y + data.z + data.w;
        }
        else {
#pragma unroll
            for (auto i = baseIdx; i < N; i++) {
                val += input[i];
            }
        }
    }

    val = WarpReduceSum(val);
    if (landId == 0) {
        smem[warpId] = val;
    }
    __syncthreads();

    if (warpId == 0) {
        const auto warp_nums  = CDIV(blockDim.x, warpSize);
        float      lane_value = (landId < warp_nums) ? smem[landId] : 0.0f;
        const auto block_sum  = WarpReduceSum(lane_value);
        if (landId == 0) {
            atomicAdd(output, block_sum);
        }
    }
}


// input, output are device pointers
extern "C" void solve(const float *input, float *output, int N)
{
    constexpr int threadsPerBlock = 1024;
    constexpr int maxBlocks       = 64;
    int           blocksPerGrid   = std::min(maxBlocks, CDIV(N, threadsPerBlock * 4));
    sumKernel<<<blocksPerGrid, threadsPerBlock>>>(input, output, N);
    // cudaDeviceSynchronize();
}


int main()
{
    const int N               = 100'000'000;
    const int threadsPerBlock = 512;
    // Calculate padded size to handle lack of boundary checks in kernel
    const int blocksPerGrid = CDIV(N, threadsPerBlock);
    const int padded_N      = blocksPerGrid * threadsPerBlock;

    size_t size_input  = padded_N * sizeof(float);
    size_t size_output = sizeof(float);

    float *h_input  = (float *)malloc(size_input);
    float *h_output = (float *)malloc(size_output);

    // Initialize input with random data
    for (int i = 0; i < N; ++i) {
        h_input[i] = (float)(rand() % 100) / 10.0f;
    }
    // Pad the rest with 0
    for (int i = N; i < padded_N; ++i) {
        h_input[i] = 0.0f;
    }

    float *d_input, *d_output;
    cudaMalloc((void **)&d_input, size_input);
    cudaMalloc((void **)&d_output, size_output);

    cudaMemcpy(d_input, h_input, size_input, cudaMemcpyHostToDevice);
    cudaMemset(d_output, 0, size_output);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Warmup
    solve(d_input, d_output, N);

    cudaMemset(d_output, 0, size_output);
    cudaEventRecord(start);
    solve(d_input, d_output, N);
    cudaEventRecord(stop);

    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    printf("Time: %f ms\n", milliseconds);
    printf("Bandwidth: %f GB/s\n", (size_input / 1e9) / (milliseconds / 1000));

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    cudaMemcpy(h_output, d_output, size_output, cudaMemcpyDeviceToHost);

    // Verify
    double expected = 0.0;
    for (int i = 0; i < N; ++i) {
        expected += (double)h_input[i];
    }

    printf("Input size: %d\n", N);
    printf("GPU Result: %f\n", *h_output);
    printf("CPU Result: %f\n", expected);

    // Check for errors (relative error or absolute error depending on magnitude)
    double diff = fabs((double)(*h_output) - expected);
    // Relative error check might be better for large sums
    double relative_error = expected != 0.0 ? diff / fabs(expected) : diff;

    if (relative_error < 1e-4) {
        printf("PASSED\n");
    }
    else {
        printf("FAILED (diff: %f, rel: %e)\n", diff, relative_error);
    }

    cudaFree(d_input);
    cudaFree(d_output);
    free(h_input);
    free(h_output);

    return 0;
}

/**
Time: 0.559104 ms
Bandwidth: 715.432203 GB/s
Input size: 100000000
GPU Result: 495011424.000000
CPU Result: 495011476.122117
 */