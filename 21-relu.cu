#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <type_traits>

#define WARP_SIZE     32
#define FLOAT4(value) (reinterpret_cast<float4 *>(&(value))[0])


__global__ __launch_bounds__(512) void relu_kernel(float *__restrict__ input, float *__restrict__ output, const int N)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;

    // Grid-Stride Loop over float4 vectors
    // i represents the index of the float4 vector
    for (int i = idx; i * 4 < N; i += stride) {
        // Check if the full float4 is within bounds
        if ((i + 1) * 4 <= N) {
            float4 v = reinterpret_cast<float4*>(input)[i];
            v.x = fmaxf(0.0f, v.x);
            v.y = fmaxf(0.0f, v.y);
            v.z = fmaxf(0.0f, v.z);
            v.w = fmaxf(0.0f, v.w);
            reinterpret_cast<float4*>(output)[i] = v;
        } else {
            // Handle remaining elements (1, 2, or 3 floats)
            // These are contiguous at the end of the array, so we increment by 1
            int offset = i * 4;
            for (int j = offset; j < N; ++j) {
                output[j] = fmaxf(0.0f, input[j]);
            }
        }
    }
}

extern "C" void solve(float *input, float *output, int N)
{
    int threadsPerBlock = 256;
    // We process 4 elements per thread logic roughly, but grid stride handles any size
    int num_blocks = (N / 4 + threadsPerBlock - 1) / threadsPerBlock;
    // Cap grid size to avoid overhead for very large N if needed, but here simple calculation is fine
    if (num_blocks > 65535) num_blocks = 65535; // Optional protection
    if (num_blocks == 0) num_blocks = 1;

    relu_kernel<<<num_blocks, threadsPerBlock>>>(input, output, N);
    cudaDeviceSynchronize();
}

bool check(float *host_out, float *host_in, int N)
{
    for (int i = 0; i < N; i++) {
        float expected = fmaxf(0.0f, host_in[i]);
        if (fabs(host_out[i] - expected) > 1e-5) {
            printf("Error at index %d: GPU %f, CPU %f\n", i, host_out[i], expected);
            return false;
        }
    }
    return true;
}

int main()
{
    int    N     = 1 << 25; // 32M elements
    size_t bytes = N * sizeof(float);

    float *h_in  = (float *)malloc(bytes);
    float *h_out = (float *)malloc(bytes);

    // Initialize input
    for (int i = 0; i < N; i++) {
        h_in[i] = (float)(rand() % 100 - 50); // range -50 to 49
    }

    float *d_in, *d_out;
    cudaMalloc(&d_in, bytes);
    cudaMalloc(&d_out, bytes);

    cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice);

    // Warmup
    solve(d_in, d_out, N);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    int repeat = 100;
    for (int i = 0; i < repeat; i++) {
        solve(d_in, d_out, N);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    float avg_ms = milliseconds / repeat;

    printf("Time: %.3f ms\n", avg_ms);
    // Bandwidth: Read N floats, Write N floats. 2 * N * 4 bytes.
    double bandwidth = 2.0 * N * sizeof(float) / (avg_ms / 1000.0) / 1e9;
    printf("Bandwidth: %.2f GB/s\n", bandwidth);

    cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost);

    if (check(h_out, h_in, N)) {
        printf("Verification PASSED\n");
    }
    else {
        printf("Verification FAILED\n");
    }

    cudaFree(d_in);
    cudaFree(d_out);
    free(h_in);
    free(h_out);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
