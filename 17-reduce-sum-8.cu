#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>


#define FULL_MASK 0xFFFFFFFF
#define WARP_SIZE 32

template <typename T, typename K> __host__ __device__ __forceinline__ auto CDIV(T a, K b) { return (a + b - 1) / b; }

template <typename T> __device__ __forceinline__ T WarpReduceSum(T value)
{
#pragma unroll
    for (auto offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        value += __shfl_down_sync(FULL_MASK, value, offset);
    }
    return value;
}

// 使用 __launch_bounds__ 提示编译器优化寄存器使用，提高 Occupancy
__global__ void  __launch_bounds__(512) ReduceSumKernelPhase1(const float *__restrict__ input, float *__restrict__ partial_sum, const int N)
{
    __shared__ float smem[WARP_SIZE];

    const auto tid    = threadIdx.x;
    const auto stride = blockDim.x * gridDim.x;
    const auto laneId = tid & (WARP_SIZE - 1);
    const auto warpId = tid / WARP_SIZE;
    auto       idx     = threadIdx.x + blockIdx.x * blockDim.x;

    // ILP optimize
    float sum1 = 0.0f, sum2 = 0.0f, sum3 = 0.0f, sum4 = 0.0f;

    int i = idx;
    for (; i * 16 < N; i += stride) {
        // 一次加载16个float (4个float4)
        float4 d1 = reinterpret_cast<const float4 *>(input)[i * 4];
        float4 d2 = reinterpret_cast<const float4 *>(input)[i * 4 + 1];
        float4 d3 = reinterpret_cast<const float4 *>(input)[i * 4 + 2];
        float4 d4 = reinterpret_cast<const float4 *>(input)[i * 4 + 3];

        sum1 += d1.x + d1.y + d1.z + d1.w;
        sum2 += d2.x + d2.y + d2.z + d2.w;
        sum3 += d3.x + d3.y + d3.z + d3.w;
        sum4 += d4.x + d4.y + d4.z + d4.w;
    }

    // 处理剩余的4元素块
    for (i = i * 4; i * 4 < N; i += stride) {
        int base = i * 4;
        int remaining = N - base;
        if (remaining >= 4) {
            float4 data = reinterpret_cast<const float4 *>(input)[i];
            sum1 += data.x + data.y + data.z + data.w;
        } else {
            for (int j = 0; j < remaining; j++) {
                sum1 += input[base + j];
            }
        }
    }

    float val = WarpReduceSum(sum1 + sum2 + sum3 + sum4);
    if (laneId == 0) {
        smem[warpId] = val;
    }
    __syncthreads();

    if (warpId == 0) {
        int numWarps = CDIV(blockDim.x, WARP_SIZE);
        float warpSum = (laneId < numWarps) ? smem[laneId] : 0.0f;
        warpSum = WarpReduceSum(warpSum);
        if (laneId == 0) {
            partial_sum[blockIdx.x] = warpSum;
        }
    }
}

__global__ void ReduceSumKernelPhase2(const float *__restrict__ input, float *__restrict__ output, const int N)
{
    __shared__ float smem[WARP_SIZE];

    const auto tid    = threadIdx.x;
    const auto stride = blockDim.x * gridDim.x;
    const auto laneId = tid & (WARP_SIZE - 1);
    const auto warpId = tid / WARP_SIZE;
    auto       idx     = threadIdx.x + blockIdx.x * blockDim.x;

    int i = idx;
    float sum = 0.f;
    for (; i * 4 < N; i += stride) {
        int base = i * 4;
        int remaining = N - base;
        if (remaining >= 4) {
            float4 data = reinterpret_cast<const float4 *>(input)[i];
            sum += data.x + data.y + data.z + data.w;
        } else {
#pragma unroll
            for (int j = 0; j < remaining; j++) {
                sum += input[base + j];
            }
        }
    }

    sum = WarpReduceSum(sum);
    if (laneId == 0) {
        smem[warpId] = sum;
    }
    __syncthreads();

    if (warpId == 0) {
        const int num_warps = CDIV(blockDim.x, WARP_SIZE);
        float warpSum = (laneId < num_warps) ? smem[laneId] : 0.0f;
        warpSum = WarpReduceSum(warpSum);
        if (laneId == 0) {
            *output = warpSum;
        }
    }
}



// input, output are device pointers
extern "C" void solve(const float *input, float *output, int N, float *partial_sum)
{
    constexpr int threadsPerBlock = 512;
    constexpr int maxBlocks       = 64;
    int           blocksPerGrid   = std::min(maxBlocks, CDIV(N, threadsPerBlock * 4));
    
    ReduceSumKernelPhase1<<<blocksPerGrid, threadsPerBlock>>>(input, partial_sum, N);
    ReduceSumKernelPhase2<<<1, threadsPerBlock>>>(partial_sum, output, blocksPerGrid);
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
    // Allocate maxBlocks for partial sum
    size_t size_partial = 64 * sizeof(float); 

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

    float *d_input, *d_output, *d_partial_sum;
    cudaMalloc((void **)&d_input, size_input);
    cudaMalloc((void **)&d_output, size_output);
    cudaMalloc((void **)&d_partial_sum, size_partial);

    cudaMemcpy(d_input, h_input, size_input, cudaMemcpyHostToDevice);
    cudaMemset(d_output, 0, size_output);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Warmup
    solve(d_input, d_output, N, d_partial_sum);

    cudaMemset(d_output, 0, size_output);
    cudaEventRecord(start);
    solve(d_input, d_output, N, d_partial_sum);
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
    cudaFree(d_partial_sum);
    free(h_input);
    free(h_output);

    return 0;
}

/**
ime: 0.559104 ms
Bandwidth: 715.432203 GB/s
Input size: 100000000
GPU Result: 495011456.000000
CPU Result: 495011476.122117
 */