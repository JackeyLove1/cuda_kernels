---
name: add-cuda-test
description: Adds CUDA kernel unit tests and performance benchmarks to .cu files. Uses thrust for memory management, CPU random sampling for large-data correctness checks, and CUDA events for latency/throughput measurement. Use when the user asks to add tests, benchmarks, unit tests, or performance measurement to a CUDA kernel.
---

# Add CUDA Kernel Unit Test & Performance Benchmark

When the user asks to add tests to a CUDA kernel file, generate a complete `main()` function that includes **correctness verification** and **performance benchmarking**.

## Workflow

1. **Read the target `.cu` file** — identify all `__global__` kernels and the `solve()` entry point (if present)
2. **Determine kernel signature** — inputs, outputs, data types, problem size `N`
3. **Write a CPU reference** — a simple, obviously-correct scalar implementation
4. **Generate `main()`** with the sections described below

## Code Structure Template

```cpp
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <random>
#include <algorithm>

// ============================================================
// CPU reference (scalar, easy to verify)
// ============================================================
void cpu_reference(/* same logical params */) {
    // Straightforward loop implementation
}

// ============================================================
// Correctness check — supports random sampling for large N
// ============================================================
bool check_correctness(const thrust::host_vector<float>& gpu_result,
                       const thrust::host_vector<float>& cpu_result,
                       int N, float rtol = 1e-4f, float atol = 1e-5f) {
    // If N <= SAMPLE_THRESHOLD, check all elements
    // Otherwise, randomly sample SAMPLE_COUNT indices
    constexpr int SAMPLE_THRESHOLD = 1 << 20;  // 1M
    constexpr int SAMPLE_COUNT     = 8192;

    std::mt19937 rng(42);
    int check_count = N;
    std::vector<int> indices(N);
    std::iota(indices.begin(), indices.end(), 0);

    if (N > SAMPLE_THRESHOLD) {
        check_count = SAMPLE_COUNT;
        std::shuffle(indices.begin(), indices.end(), rng);
        indices.resize(SAMPLE_COUNT);
    }

    int    max_errors = 10;
    int    errors     = 0;
    float  max_diff   = 0.f;
    for (int k = 0; k < check_count; ++k) {
        int i = indices[k];
        float diff = fabsf(gpu_result[i] - cpu_result[i]);
        float ref  = fabsf(cpu_result[i]);
        max_diff   = fmaxf(max_diff, diff);
        if (diff > atol + rtol * ref) {
            if (++errors <= max_errors)
                printf("  MISMATCH at [%d]: gpu=%.6f cpu=%.6f diff=%.6e\n",
                       i, gpu_result[i], cpu_result[i], diff);
        }
    }
    printf("  Checked %d / %d elements, max_diff=%.6e, errors=%d\n",
           check_count, N, max_diff, errors);
    return errors == 0;
}

int main(int argc, char** argv) {
    // ----------------------------------------------------------
    // 1. Parse problem size from argv or use default
    // ----------------------------------------------------------
    int N = DEFAULT_N;
    if (argc > 1) N = atoi(argv[1]);
    printf("=== Kernel Unit Test & Benchmark ===\n");
    printf("N = %d (%.2f MB)\n", N,
           (double)N * sizeof(float) / (1 << 20));

    // ----------------------------------------------------------
    // 2. Allocate with thrust (RAII, no manual free)
    // ----------------------------------------------------------
    thrust::host_vector<float>   h_in(N);
    thrust::device_vector<float> d_in(N);
    thrust::device_vector<float> d_out(N);
    // ... (more vectors as needed)

    // ----------------------------------------------------------
    // 3. Initialize input data
    // ----------------------------------------------------------
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.f, 1.f);
    for (int i = 0; i < N; ++i) h_in[i] = dist(rng);
    d_in = h_in;  // host → device copy

    // ----------------------------------------------------------
    // 4. Run kernel once for correctness
    // ----------------------------------------------------------
    solve(thrust::raw_pointer_cast(d_in.data()),
          thrust::raw_pointer_cast(d_out.data()), N);
    cudaDeviceSynchronize();

    thrust::host_vector<float> h_gpu_out = d_out;  // device → host

    // ----------------------------------------------------------
    // 5. CPU reference & compare
    // ----------------------------------------------------------
    thrust::host_vector<float> h_cpu_out(N);
    cpu_reference(h_in.data(), h_cpu_out.data(), N);

    bool pass = check_correctness(h_gpu_out, h_cpu_out, N);
    printf("Correctness: %s\n\n", pass ? "PASS" : "FAIL");

    // ----------------------------------------------------------
    // 6. Performance benchmark
    // ----------------------------------------------------------
    cudaEvent_t t0, t1;
    cudaEventCreate(&t0);
    cudaEventCreate(&t1);

    // Warmup
    constexpr int WARMUP = 10;
    for (int i = 0; i < WARMUP; ++i)
        solve(/* ... */);
    cudaDeviceSynchronize();

    // Timed runs
    constexpr int NITER = 100;
    cudaEventRecord(t0);
    for (int i = 0; i < NITER; ++i)
        solve(/* ... */);
    cudaEventRecord(t1);
    cudaEventSynchronize(t1);

    float total_ms = 0.f;
    cudaEventElapsedTime(&total_ms, t0, t1);
    float avg_ms = total_ms / NITER;

    // Throughput: total memory moved per kernel call
    double bytes_moved  = /* READ + WRITE bytes */;
    double bandwidth_gb = bytes_moved / (avg_ms * 1e-3) / 1e9;
    double elem_per_sec = (double)N / (avg_ms * 1e-3);

    printf("=== Performance (N=%d, %d runs) ===\n", N, NITER);
    printf("  Avg latency : %.4f ms\n", avg_ms);
    printf("  Throughput  : %.2f Gelem/s\n", elem_per_sec / 1e9);
    printf("  Bandwidth   : %.2f GB/s\n", bandwidth_gb);

    cudaEventDestroy(t0);
    cudaEventDestroy(t1);
    return pass ? 0 : 1;
}
```

## Key Rules

### Memory management — thrust only
- Use `thrust::device_vector<T>` / `thrust::host_vector<T>` for all allocations.
- Pass raw pointers to kernels via `thrust::raw_pointer_cast(vec.data())`.
- Never use `cudaMalloc` / `cudaFree` / `malloc` / `free` in generated test code.

### Correctness check strategy
| N range | Strategy |
|---------|----------|
| N <= 1M | Check **every** element |
| N > 1M  | **Random sample** 8192 indices, compare with tolerance |

- Default tolerances: `rtol=1e-4`, `atol=1e-5` (float32). Adjust for fp16/bf16 (`rtol=1e-2`, `atol=1e-2`).
- Print first 10 mismatches with index + values, then summary.

### Performance measurement
- Use `cudaEvent_t` for GPU timing (NOT `std::chrono`).
- **Warmup**: 10 runs before measurement.
- **Timed**: 100 runs, report average.
- Report all three metrics:
  - **Latency** (ms) — average kernel time
  - **Throughput** (Gelem/s) — `N / avg_time`
  - **Bandwidth** (GB/s) — `total_bytes_moved / avg_time`
- Compute `bytes_moved` by analyzing kernel memory access pattern (reads + writes).

### Data initialization
- Use `std::mt19937` with **fixed seed 42** for reproducibility.
- Choose value range appropriate to the kernel (e.g. `[-10, 10]` for sigmoid, `[0, 1]` for softmax input).

### Return code
- Return `0` on pass, `1` on fail — enables CI integration.

## Adapting to Kernel Types

### Element-wise (sigmoid, relu, silu, add)
- `bytes_moved = 2 * N * sizeof(T)` (1 read + 1 write per element)
- CPU ref: simple loop applying the same math

### Reduction (sum, max, min)
- `bytes_moved = N * sizeof(T)` (read only, output is scalar)
- CPU ref: `std::accumulate` or loop

### GEMM (matrix multiply)
- Inputs: A(M×K), B(K×N), Output: C(M×N)
- `bytes_moved = (M*K + K*N + M*N) * sizeof(T)`
- CPU ref: triple nested loop (only for small M,N,K — sample check for large)
- Also report TFLOPS: `2.0 * M * N * K / (avg_ms * 1e-3) / 1e12`

### Scan (prefix sum)
- `bytes_moved = 2 * N * sizeof(T)`
- CPU ref: `std::partial_sum`

## Example — Adding Test to a Sigmoid Kernel

Given a file with `sigmoid_kernel` and `solve(const float* input, float* output, int N)`:

```cpp
// CPU reference
void sigmoid_cpu(const float* in, float* out, int N) {
    for (int i = 0; i < N; ++i)
        out[i] = 1.0f / (1.0f + expf(-in[i]));
}

int main(int argc, char** argv) {
    int N = (argc > 1) ? atoi(argv[1]) : (1 << 25);
    printf("=== Sigmoid Kernel Test & Benchmark (N=%d) ===\n", N);

    // Allocate
    thrust::host_vector<float>   h_in(N);
    thrust::device_vector<float> d_in(N), d_out(N);

    // Init
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-10.f, 10.f);
    for (int i = 0; i < N; ++i) h_in[i] = dist(rng);
    d_in = h_in;

    // Run GPU
    solve(thrust::raw_pointer_cast(d_in.data()),
          thrust::raw_pointer_cast(d_out.data()), N);
    cudaDeviceSynchronize();
    thrust::host_vector<float> h_gpu = d_out;

    // CPU ref
    thrust::host_vector<float> h_cpu(N);
    sigmoid_cpu(h_in.data(), h_cpu.data(), N);

    // Check
    bool pass = check_correctness(h_gpu, h_cpu, N);
    printf("Result: %s\n\n", pass ? "PASS" : "FAIL");

    // Benchmark
    cudaEvent_t t0, t1;
    cudaEventCreate(&t0); cudaEventCreate(&t1);
    for (int i = 0; i < 10; ++i)
        solve(thrust::raw_pointer_cast(d_in.data()),
              thrust::raw_pointer_cast(d_out.data()), N);
    cudaDeviceSynchronize();

    constexpr int NITER = 100;
    cudaEventRecord(t0);
    for (int i = 0; i < NITER; ++i)
        solve(thrust::raw_pointer_cast(d_in.data()),
              thrust::raw_pointer_cast(d_out.data()), N);
    cudaEventRecord(t1);
    cudaEventSynchronize(t1);

    float ms = 0.f;
    cudaEventElapsedTime(&ms, t0, t1);
    ms /= NITER;

    double bw = 2.0 * N * sizeof(float) / (ms * 1e-3) / 1e9;
    printf("  Latency  : %.4f ms\n", ms);
    printf("  Throughput: %.2f Gelem/s\n", (double)N / (ms*1e-3) / 1e9);
    printf("  Bandwidth : %.2f GB/s\n", bw);

    cudaEventDestroy(t0); cudaEventDestroy(t1);
    return pass ? 0 : 1;
}
```
