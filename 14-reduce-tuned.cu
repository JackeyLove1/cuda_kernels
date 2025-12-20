#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>


#define FULL_MASK 0xFFFFFFFF

template <typename T, typename K> __host__ __device__ __forceinline__ auto CDIV(T a, K b)
{
    return (a + b - 1) / b;
}

__global__ void sumKernel(const float *__restrict__ input, float *__restrict__ output, const int N)
{
    __shared__ float smem[32];

    // Grid-Stride Loop logic
    // 每次迭代处理 gridDim.x * blockDim.x 个 float4 (即 * 4 个 float)
    const int tid       = threadIdx.x;
    const int warpId    = tid >> 5;
    const int laneId    = tid & (32 - 1);

    float val = 0.0f;

    // Grid Stride Loop: 让当前线程处理它负责的所有数据块
    // 注意：这里的步长 stride 是以 float4 为单位的
    int idx = blockDim.x * blockIdx.x + tid;
    const int stride = blockDim.x * gridDim.x;

    for (; idx * 4 < N; idx += stride) {
        const int baseIdx = idx * 4;
        const int remaining = N - baseIdx;

        if (remaining >= 4) {
            // 向量化加载
            const float4 v4 = reinterpret_cast<const float4 *>(input)[idx];
            val += v4.x + v4.y + v4.z + v4.w;
        } else {
            // 处理数组末尾不足4个元素的情况
            for (int i = 0; i < remaining; i++) {
                val += input[baseIdx + i];
            }
        }
    }

    // warp reduce sum
    const unsigned mask = __activemask();
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(mask, val, offset);
    }

    if (laneId == 0)
        smem[warpId] = val;

    __syncthreads();

    // block reduce sum
    // 这里的 warpNum 取决于 blockDim.x，如果是 512 线程，则是 16
    const int warpNum = CDIV(blockDim.x, 32);

    if (warpId == 0) {
        // 只有第一个 warp 需要工作
        // 读取刚才每个 warp 存入 smem 的总和
        val = (laneId < warpNum) ? smem[laneId] : 0.0f;

#pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            val += __shfl_down_sync(FULL_MASK, val, offset);
        }

        if (laneId == 0)
            atomicAdd(output, val);
    }
}


// input, output are device pointers
extern "C" void solve(const float *input, float *output, int N)
{
    constexpr int threadsPerBlock = 512;
    // Lower maxBlocks to increase work per thread
    constexpr int maxBlocks = 64; 

    // 依然可以根据 N 动态调整，但要有上限
    int blocksPerGrid = std::min(maxBlocks, CDIV(N, threadsPerBlock * 4));

    sumKernel<<<blocksPerGrid, threadsPerBlock>>>(input, output, N);
    // cudaDeviceSynchronize(); // Removed for pure kernel timing (event handles it)
}


int main()
{
    const int N               = 1000000;
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
Time: 0.008192 ms
Bandwidth: 488.500014 GB/s
Input size: 1000000
GPU Result: 4949858.500000
CPU Result: 4949858.300299
PASSED
 */