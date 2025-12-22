#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>
#include <cub/cub.cuh>
#include <limits>
#include <cfloat>
#include <algorithm>

// Functor for the transform iterator: exp(x - max)
struct ExpMinusMax {
    const float* d_max;
    
    __device__ __forceinline__
    float operator()(const float& x) const {
        return expf(x - *d_max);
    }
};

// Vectorized Normalize Kernel
__global__ void normalize_kernel(const float *__restrict__ input, 
                                 float *__restrict__ output, 
                                 const float *__restrict__ d_max, 
                                 const float *__restrict__ d_sum, 
                                 int N) 
{
    const float max_val = *d_max;
    const float inv_sum = 1.0f / (*d_sum);

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    // Vectorized part using float4
    for (; idx * 4 < N; idx += stride) {
        int base = idx * 4;
        int remaining = N - base;
        
        if (remaining >= 4) {
            float4 val = reinterpret_cast<const float4*>(input)[idx];
            float4 out;
            
            out.x = expf(val.x - max_val) * inv_sum;
            out.y = expf(val.y - max_val) * inv_sum;
            out.z = expf(val.z - max_val) * inv_sum;
            out.w = expf(val.w - max_val) * inv_sum;
            
            reinterpret_cast<float4*>(output)[idx] = out;
        } else {
            for (int k = 0; k < remaining; k++) {
                int i = base + k;
                output[i] = expf(input[i] - max_val) * inv_sum;
            }
        }
    }
}

extern "C" void solve(const float *input, float *output, int N) {
    float *d_max = nullptr;
    float *d_sum = nullptr;
    cudaMalloc(&d_max, sizeof(float));
    cudaMalloc(&d_sum, sizeof(float));

    void *d_temp_storage = nullptr;
    size_t temp_storage_bytes = 0;
    size_t temp_storage_bytes_max = 0;
    size_t temp_storage_bytes_sum = 0;

    // 1. Get size for Max
    cub::DeviceReduce::Max(nullptr, temp_storage_bytes_max, input, d_max, N);
    
    // 2. Get size for Sum
    ExpMinusMax transform_op = {d_max};
    cub::TransformInputIterator<float, ExpMinusMax, const float*> itr(input, transform_op);
    cub::DeviceReduce::Sum(nullptr, temp_storage_bytes_sum, itr, d_sum, N);

    // Allocate max required storage
    temp_storage_bytes = std::max(temp_storage_bytes_max, temp_storage_bytes_sum);
    cudaMalloc(&d_temp_storage, temp_storage_bytes);

    // Perform Max
    cub::DeviceReduce::Max(d_temp_storage, temp_storage_bytes, input, d_max, N);
    
    // Perform Sum (stats depend on d_max from previous step, so it is safe)
    cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, itr, d_sum, N);

    // Perform Normalize
    int threads = 256;
    int blocks = std::min(1024, (N + threads * 4 - 1) / (threads * 4));
    normalize_kernel<<<blocks, threads>>>(input, output, d_max, d_sum, N);

    cudaFree(d_max);
    cudaFree(d_sum);
    cudaFree(d_temp_storage);
}

void softmax_cpu(const float *input, float *output, int N) {
    float max_val = -FLT_MAX;
    for (int i = 0; i < N; i++) {
        if (input[i] > max_val) max_val = input[i];
    }
    float sum = 0.0f;
    for (int i = 0; i < N; i++) {
        sum += expf(input[i] - max_val);
    }
    for (int i = 0; i < N; i++) {
        output[i] = expf(input[i] - max_val) / sum;
    }
}

bool verify_results(const float *gpu_out, const float *cpu_in, int N, float tolerance = 1e-5f) {
    float *cpu_out = (float *)malloc(N * sizeof(float));
    softmax_cpu(cpu_in, cpu_out, N);
    bool all_correct = true;
    for (int i = 0; i < N; i++) {
        if (fabsf(gpu_out[i] - cpu_out[i]) > tolerance) {
            printf("Error at %d: GPU=%f CPU=%f\n", i, gpu_out[i], cpu_out[i]);
            all_correct = false;
            break;
        }
    }
    free(cpu_out);
    return all_correct;
}

int main(int argc, char *argv[]) {
    int N = 1 << 25; 
    size_t bytes = N * sizeof(float);
    
    float *h_in = (float *)malloc(bytes);
    float *h_out = (float *)malloc(bytes);
    
    srand(42);
    for (int i = 0; i < N; i++) h_in[i] = (float)(rand() % 1000) / 100.0f;

    float *d_in, *d_out;
    cudaMalloc(&d_in, bytes);
    cudaMalloc(&d_out, bytes);
    cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice);

    // Warmup
    for(int i=0; i<10; i++) solve(d_in, d_out, N);
    cudaDeviceSynchronize();

    cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);
    
    cudaEventRecord(start);
    for(int i=0; i<100; i++) solve(d_in, d_out, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);
    float avg_ms = ms / 100.0f;
    
    // Bandwidth: 3 Reads + 1 Write = 4 * N * 4 bytes
    double gb_s = (4.0 * (double)N * 4.0) / (avg_ms * 1000000.0);

    printf("CUB Safe Softmax\n");
    printf("N = %d\n", N);
    printf("Time: %.4f ms\n", avg_ms);
    printf("Bandwidth: %.2f GB/s\n", gb_s);

    cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost);
    if(verify_results(h_out, h_in, 1000)) printf("Verification Passed\n");
    else printf("Verification Failed\n");

    cudaFree(d_in); cudaFree(d_out);
    free(h_in); free(h_out);
    return 0;
}

/**
Time: 0.9522 ms
Bandwidth: 563.80 GB/s
 **/