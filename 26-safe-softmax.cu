#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda.h>
#include <cuda/atomic>
#include <cuda_runtime.h>
#include <numeric>
#include <limits>
#include <algorithm>
#include <cfloat>

#define WARP_SIZE     32
#define FULL_MASK     0xFFFFFFFF
#define LOG2E_F       1.44269504089f
#define CEIL(a, b)    (((a) + (b) - 1) / (b))

// Use -Infinity for max identity, not FLT_MIN (which is +1e-38)
#define MAX_IDENTITY  (-FLT_MAX) 

template <typename T, typename U> __host__ __device__ __forceinline__ auto CDIV(T a, U b) { return (a + b - 1) / b; }

struct __align__(8) MaxSum {
    float max_val;
    float sum_val;
};

// Online Softmax Update: Combine two (max, sum) pairs
// (m1, s1) and (m2, s2) -> (m_new, s_new)
// m_new = max(m1, m2)
// s_new = s1 * exp(m1 - m_new) + s2 * exp(m2 - m_new)
__device__ __forceinline__ MaxSum combine_stats(MaxSum a, MaxSum b) {
    MaxSum res;
    res.max_val = fmaxf(a.max_val, b.max_val);
    
    // Optimization: avoid exp calculation if diff is very large (underflow)
    // But for simplicity and correctness within float range:
    float scale_a = __expf(a.max_val - res.max_val);
    float scale_b = __expf(b.max_val - res.max_val);
    
    res.sum_val = a.sum_val * scale_a + b.sum_val * scale_b;
    return res;
}

// Warp Reduction for (Max, Sum) pair
__device__ __forceinline__ MaxSum WarpReduceStats(MaxSum val) {
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        float other_max = __shfl_down_sync(FULL_MASK, val.max_val, offset);
        float other_sum = __shfl_down_sync(FULL_MASK, val.sum_val, offset);
        val = combine_stats(val, {other_max, other_sum});
    }
    return val;
}

// Pass 1: Compute stats (Max, Sum) per block
// Grid-Stride Loop
__global__ void compute_stats_kernel(const float *__restrict__ input, 
                                     MaxSum *__restrict__ block_stats, 
                                     const int N) 
{
    // Initialize local stats
    MaxSum local_val = {MAX_IDENTITY, 0.0f};

    const int tid = threadIdx.x;
    const int stride = gridDim.x * blockDim.x;
    int idx = blockIdx.x * blockDim.x + tid;

    // Vectorized load loop (float4)
    for (; idx * 4 < N; idx += stride) {
        // Handle vector part
        const int base_id = idx * 4;
        const int remaining = N - base_id;
        
        if (remaining >= 4) {
            const float4 val = reinterpret_cast<const float4*>(input)[idx];
            
            // Process 4 elements
            float v[4] = {val.x, val.y, val.z, val.w};
            for (int k = 0; k < 4; ++k) {
                // Online update per-element
                float new_max = fmaxf(local_val.max_val, v[k]);
                float scale = __expf(local_val.max_val - new_max);
                local_val.sum_val = local_val.sum_val * scale + __expf(v[k] - new_max);
                local_val.max_val = new_max;
            }
        } else {
            // Handle scalar tail
            for (int k = 0; k < remaining; ++k) {
                float v = input[base_id + k];
                float new_max = fmaxf(local_val.max_val, v);
                float scale = __expf(local_val.max_val - new_max);
                local_val.sum_val = local_val.sum_val * scale + __expf(v - new_max);
                local_val.max_val = new_max;
            }
        }
    }

    // Warp Reduction
    local_val = WarpReduceStats(local_val);

    // Block Reduction (Shared Memory)
    static __shared__ float smem_max[WARP_SIZE];
    static __shared__ float smem_sum[WARP_SIZE];

    const int warp_id = tid / WARP_SIZE;
    const int lane_id = tid % WARP_SIZE;

    if (lane_id == 0) {
        smem_max[warp_id] = local_val.max_val;
        smem_sum[warp_id] = local_val.sum_val;
    }
    __syncthreads();

    // Final reduction by first warp
    if (warp_id == 0) {
        MaxSum warp_val;
        // Load from shared memory
        // BUG FIX: use lane_id to index smem, not warp_id
        const int num_warps = blockDim.x / WARP_SIZE;
        if (lane_id < num_warps) {
            warp_val.max_val = smem_max[lane_id];
            warp_val.sum_val = smem_sum[lane_id];
        } else {
            warp_val = {MAX_IDENTITY, 0.0f};
        }
        
        warp_val = WarpReduceStats(warp_val);

        if (lane_id == 0) {
            block_stats[blockIdx.x] = warp_val;
        }
    }
}

// Pass 2: Reduce Block Stats to Global Stats
// This kernel runs with a single block (or few) to reduce the per-block results
__global__ void reduce_final_stats_kernel(const MaxSum *__restrict__ block_stats, 
                                          float *__restrict__ global_max,
                                          float *__restrict__ global_sum,
                                          int num_blocks) 
{
    MaxSum local_val = {MAX_IDENTITY, 0.0f};
    
    // Simple loop over all blocks
    for (int i = threadIdx.x; i < num_blocks; i += blockDim.x) {
        local_val = combine_stats(local_val, block_stats[i]);
    }
    
    // Reduce within block
    local_val = WarpReduceStats(local_val);
    
    static __shared__ float smem_max[WARP_SIZE];
    static __shared__ float smem_sum[WARP_SIZE];
    
    const int tid = threadIdx.x;
    const int warp_id = tid / WARP_SIZE;
    const int lane_id = tid % WARP_SIZE;
    
    if (lane_id == 0) {
        smem_max[warp_id] = local_val.max_val;
        smem_sum[warp_id] = local_val.sum_val;
    }
    __syncthreads();
    
    if (warp_id == 0) {
        MaxSum warp_val;
        const int num_warps = blockDim.x / WARP_SIZE;
        if (lane_id < num_warps) {
            warp_val.max_val = smem_max[lane_id];
            warp_val.sum_val = smem_sum[lane_id];
        } else {
            warp_val = {MAX_IDENTITY, 0.0f};
        }
        
        warp_val = WarpReduceStats(warp_val);
        
        if (lane_id == 0) {
            *global_max = warp_val.max_val;
            *global_sum = warp_val.sum_val;
        }
    }
}

// Pass 3: Softmax (Normalize)
__global__ void softmax_kernel(const float *__restrict__ input,
                               float *__restrict__ output,
                               const float* __restrict__ global_max,
                               const float* __restrict__ global_sum,
                               const int N)
{
    const auto stride = gridDim.x * blockDim.x;
    auto idx = threadIdx.x + blockIdx.x * blockDim.x;
    
    // Load scalars to registers
    const float max_val = *global_max;
    const float sum_val = *global_sum;
    const float inv_sum = 1.0f / sum_val; // Precompute division

    for (; idx * 4 < N; idx += stride) {
        const int base_id = idx * 4;
        const int remaining = N - base_id;
        
        if (remaining >= 4) {
            const float4 val = reinterpret_cast<const float4 *>(input)[idx];
            float4 out;
            
            out.x = __expf(val.x - max_val) * inv_sum;
            out.y = __expf(val.y - max_val) * inv_sum;
            out.z = __expf(val.z - max_val) * inv_sum;
            out.w = __expf(val.w - max_val) * inv_sum;
            
            reinterpret_cast<float4 *>(output)[idx] = out;
        }
        else {
            for (int j = base_id; j < N; ++j) {
                output[j] = __expf(input[j] - max_val) * inv_sum;
            }
        }
    }
}

extern "C" void solve(const float *input, float *output, const int N)
{
    constexpr int threadsPerBlock = 256;
    constexpr auto max_blocks = 1024;
    const auto blocksPerGrid = std::min(max_blocks, CEIL(N, threadsPerBlock / 4));
    
    // Memory Management
    // 1. Block Stats Buffer
    MaxSum *d_block_stats;
    float *d_final_max, *d_final_sum;
    
    cudaMalloc(&d_block_stats, blocksPerGrid * sizeof(MaxSum));
    cudaMalloc(&d_final_max, sizeof(float));
    cudaMalloc(&d_final_sum, sizeof(float));
    
    // Step 1: Compute Stats (Max, Sum) per block
    compute_stats_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, d_block_stats, N);
    
    // Step 2: Reduce Block Stats to Global Stats
    // Launch 1 block with enough threads to cover num_blocks (or loop)
    reduce_final_stats_kernel<<<1, 256>>>(d_block_stats, d_final_max, d_final_sum, blocksPerGrid);
    
    // Step 3: Normalize
    softmax_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, output, d_final_max, d_final_sum, N);
    
    cudaFree(d_block_stats);
    cudaFree(d_final_max);
    cudaFree(d_final_sum);
}

// CPU reference implementation
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

// Verification logic
bool verify_results(const float *gpu_out, const float *cpu_in, int N, float tolerance = 1e-5f) {
    float *cpu_out = (float *)malloc(N * sizeof(float));
    if (!cpu_out) return false;
    
    softmax_cpu(cpu_in, cpu_out, N);
    
    bool all_correct = true;
    int error_count = 0;
    for (int i = 0; i < N; i++) {
        float diff = fabsf(gpu_out[i] - cpu_out[i]);
        if (diff > tolerance) {
            if (error_count < 10) {
                printf("Mismatch at index %d: GPU=%.6f, CPU=%.6f, diff=%.6f\n",
                       i, gpu_out[i], cpu_out[i], diff);
            }
            error_count++;
            all_correct = false;
        }
    }
    free(cpu_out);
    return all_correct;
}

int main(int argc, char *argv[]) {
    int N = 1 << 25;
    if (argc > 1) N = atoi(argv[1]);

    size_t bytes = N * sizeof(float);
    printf("Testing Optimized Online Softmax Kernel\n");
    printf("=======================================\n");
    printf("Array size: %d elements (%.2f MB)\n", N, bytes / (1024.0f * 1024.0f));

    float *h_in = (float *)malloc(bytes);
    float *h_out = (float *)malloc(bytes);
    
    // Random init
    srand(42);
    for (int i = 0; i < N; i++) {
        h_in[i] = (float)(rand() % 20000 - 10000) / 1000.0f;
    }

    float *d_in, *d_out;
    cudaMalloc(&d_in, bytes);
    cudaMalloc(&d_out, bytes);
    
    cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice);

    // Warmup
    for (int i = 0; i < 10; i++) solve(d_in, d_out, N);
    cudaDeviceSynchronize();

    // Benchmark
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    for (int i = 0; i < 100; i++) solve(d_in, d_out, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float total_ms = 0.0f;
    cudaEventElapsedTime(&total_ms, start, stop);
    float avg_ms = total_ms / 100.0f;

    // Bandwidth:
    // Pass 1: Read Input (N) + Write Stats (negligible)
    // Pass 2: Read Stats + Write Global Stats (negligible)
    // Pass 3: Read Input (N) + Write Output (N)
    // Total Reads: 2N. Total Writes: 1N.
    // Total bytes = 3 * N * 4.
    double total_bytes = 3.0 * N * sizeof(float);
    double bandwidth_gb_s = (total_bytes / (avg_ms / 1000.0)) / 1e9;

    printf("\nPerformance Results:\n");
    printf("-------------------\n");
    printf("Average time: %.4f ms\n", avg_ms);
    printf("Bandwidth: %.2f GB/s\n", bandwidth_gb_s);
    
    // Verify
    cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost);
    
    printf("\nVerification:\n");
    int verify_count = (N < 1000) ? N : 1000;
    // Check first 1000
    if (verify_results(h_out, h_in, verify_count)) {
        printf("PASSED first %d\n", verify_count);
    } else {
        printf("FAILED\n");
    }

    // Check negatives
    printf("\nChecking All-Negative Input...\n");
    float h_neg[] = {-10.0f, -20.0f, -30.0f, -40.0f};
    cudaMemcpy(d_in, h_neg, sizeof(h_neg), cudaMemcpyHostToDevice);
    solve(d_in, d_out, 4);
    cudaMemcpy(h_out, d_out, sizeof(h_neg), cudaMemcpyDeviceToHost);
    if (verify_results(h_out, h_neg, 4)) {
        printf("PASSED Negative Inputs\n");
    } else {
        printf("FAILED Negative Inputs\n");
    }

    cudaFree(d_in);
    cudaFree(d_out);
    free(h_in);
    free(h_out);
    return 0;
}

/**
Average time: 0.8755 ms
Bandwidth: 459.91 GB/s
 */