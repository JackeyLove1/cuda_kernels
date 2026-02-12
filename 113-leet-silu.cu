#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cub/cub.cuh>
#include <cuda/pipeline>
#include <numeric>
#include <random>
#include <type_traits>
#include <vector>

#define CEIL(a, b) (((a) + (b) - 1) / (b))

template <typename>
struct dependent_false : std::false_type {};

template <typename T>
__forceinline__ __device__ auto LOAD4(const T* ptr) {
  if constexpr (std::is_same_v<T, int>) {
    return reinterpret_cast<const int4*>(ptr);
  } else if constexpr (std::is_same_v<T, float>) {
    return reinterpret_cast<const float4*>(ptr);
  } else if constexpr (std::is_same_v<T, double>) {
    return reinterpret_cast<const double4*>(ptr);
  } else if constexpr (std::is_same_v<T, short>) {
    return reinterpret_cast<const short4*>(ptr);
  } else {
    static_assert(dependent_false<T>::value, "Unsupported type for LOAD4");
    return nullptr;
  }
}

template <typename T>
__forceinline__ __device__ auto STORE4(T* ptr) {
  if constexpr (std::is_same_v<T, int>) {
    return reinterpret_cast<int4*>(ptr);
  } else if constexpr (std::is_same_v<T, float>) {
    return reinterpret_cast<float4*>(ptr);
  } else if constexpr (std::is_same_v<T, double>) {
    return reinterpret_cast<double4*>(ptr);
  } else if constexpr (std::is_same_v<T, short>) {
    return reinterpret_cast<short4*>(ptr);
  } else {
    static_assert(dependent_false<T>::value, "Unsupported type for STORE4");
    return nullptr;
  }
}

__forceinline__ __device__ __host__ float4 operator+(const float4 a, const float4 b) {
  return make_float4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w);
}

static constexpr unsigned int FULL_MASK = 0xffffffff;

template <typename T>
__forceinline__ __device__ T WarpReduceSum(T value) {
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    value += __shfl_down_sync(FULL_MASK, value, offset);
  }
  return value;
}

static constexpr int PER_THREAD_WORK_ITEMS = 1;
static constexpr int WARP_SIZE = 32;
static constexpr int MAX_BLOCKS = 2048;
static constexpr int THREADS_PER_BLOCK = 256;

__device__ __forceinline__ float sigmoid(float x) { return 1.0f / (1.0f + expf(-x)); }

__device__ __forceinline__ float sigmoid_fast(float x) {
  return __fdividef(1.0f, 1.0f + __expf(-x));  // 使用快速除法
}

__device__ __forceinline__ float sigmoid_tanh(float x) { return 0.5f * (1.0f + tanhf(0.5f * x)); }

__device__ __forceinline__ float sigmoid_approx(float x) {
  // 这种方法没有 exp，只有绝对值和除法，非常快
  // 但导数性质与标准 sigmoid 不同，训练时慎用
  float abs_x = fabsf(x);
  return 0.5f * (__fdividef(x, 1.0f + abs_x) + 1.0f);
}

__device__ __forceinline__ float sigmoid_fast_2(float x) {
  // 可选：对常见训练/推理范围，提前饱和能省不少 exp
  // 这两个阈值很“工程化”：expf(±10) 已经足够让 sigmoid ~ 0/1
  if (x >= 10.0f) return 1.0f;
  if (x <= -10.0f) return 0.0f;

  float e = __expf(-x);  // fast exp
  // 1 / (1 + e) 的快速倒数版本：
  return __frcp_rn(1.0f + e);  // 或者 __fdividef(1.0f, 1.0f + e)
}

__device__ __forceinline__ float silu(float x) { return x * sigmoid_fast(x); }

__global__ void silu_kernel(const float* __restrict__ input, float* __restrict__ output,
                            const int N) {
  const int tid = threadIdx.x;
  const int gid = blockIdx.x * blockDim.x + tid;
  const int stride = gridDim.x * blockDim.x;
  const int total = N;
  const int vec_size = total >> 2;
  const int vec_tail = total & 3;
  const float4* input4 = LOAD4(input);
  float4* output4 = STORE4(output);
  for (int i = gid; i < vec_size; i += stride) {
    const float4 value = input4[i];
    const float4 result = make_float4(silu(value.x), silu(value.y), silu(value.z), silu(value.w));
    output4[i] = result;
  }
  if (vec_tail && gid < vec_tail) {
    const int base = (vec_size << 2) + gid;
    const float value = input[base];
    output[base] = silu(value);
  }
}

// input, output are device pointers
extern "C" void solve(const float* input, float* output, int N) {
  constexpr int threadsPerBlock = 256;
  int blocksPerGrid = std::min(CEIL(N, threadsPerBlock * 4), MAX_BLOCKS);

  silu_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, output, N);
  cudaDeviceSynchronize();
}

void silu_cpu_reference(const float* input, float* output, int n) {
  for (int i = 0; i < n; ++i) {
    const float x = input[i];
    output[i] = x / (1.0f + std::exp(-x));
  }
}

bool check_correctness(const thrust::host_vector<float>& gpu_result,
                       const thrust::host_vector<float>& cpu_result, int n, float rtol = 1e-4f,
                       float atol = 1e-5f) {
  constexpr int SAMPLE_THRESHOLD = 1 << 20;
  constexpr int SAMPLE_COUNT = 8192;

  std::mt19937 rng(42);
  std::vector<int> indices(n);
  std::iota(indices.begin(), indices.end(), 0);

  int check_count = n;
  if (n > SAMPLE_THRESHOLD) {
    std::shuffle(indices.begin(), indices.end(), rng);
    check_count = std::min(SAMPLE_COUNT, n);
    indices.resize(check_count);
  }

  int errors = 0;
  int max_errors = 10;
  float max_diff = 0.0f;
  for (int k = 0; k < check_count; ++k) {
    const int i = indices[k];
    const float diff = std::fabs(gpu_result[i] - cpu_result[i]);
    const float ref = std::fabs(cpu_result[i]);
    max_diff = std::max(max_diff, diff);
    if (diff > atol + rtol * ref) {
      if (++errors <= max_errors) {
        std::printf("  MISMATCH at [%d]: gpu=%.6f cpu=%.6f diff=%.6e\n", i, gpu_result[i],
                    cpu_result[i], diff);
      }
    }
  }

  std::printf("  Checked %d / %d elements, max_diff=%.6e, errors=%d\n", check_count, n, max_diff,
              errors);
  return errors == 0;
}

int main(int argc, char** argv) {
  int n = 50000;
  if (argc > 1) {
    n = std::atoi(argv[1]);
  }

  std::printf("=== SiLU Kernel Unit Test & Benchmark ===\n");
  std::printf("N = %d (%.3f MB)\n", n, static_cast<double>(n) * sizeof(float) / (1 << 20));

  thrust::host_vector<float> h_input(n);
  thrust::host_vector<float> h_cpu_output(n);
  thrust::device_vector<float> d_input(n);
  thrust::device_vector<float> d_output(n);

  std::mt19937 rng(42);
  std::uniform_real_distribution<float> dist(-10.0f, 10.0f);
  for (int i = 0; i < n; ++i) {
    h_input[i] = dist(rng);
  }
  d_input = h_input;

  solve(thrust::raw_pointer_cast(d_input.data()), thrust::raw_pointer_cast(d_output.data()), n);

  thrust::host_vector<float> h_gpu_output = d_output;
  silu_cpu_reference(h_input.data(), h_cpu_output.data(), n);
  bool pass = check_correctness(h_gpu_output, h_cpu_output, n);
  std::printf("Correctness: %s\n\n", pass ? "PASS" : "FAIL");

  cudaEvent_t t0, t1;
  cudaEventCreate(&t0);
  cudaEventCreate(&t1);

  constexpr int WARMUP = 10;
  for (int i = 0; i < WARMUP; ++i) {
    solve(thrust::raw_pointer_cast(d_input.data()), thrust::raw_pointer_cast(d_output.data()), n);
  }

  constexpr int NITER = 100;
  cudaEventRecord(t0);
  for (int i = 0; i < NITER; ++i) {
    solve(thrust::raw_pointer_cast(d_input.data()), thrust::raw_pointer_cast(d_output.data()), n);
  }
  cudaEventRecord(t1);
  cudaEventSynchronize(t1);

  float total_ms = 0.0f;
  cudaEventElapsedTime(&total_ms, t0, t1);
  const float avg_ms = total_ms / NITER;
  const double sec = avg_ms * 1e-3;
  const double throughput = static_cast<double>(n) / sec;
  const double bandwidth = (2.0 * n * sizeof(float)) / sec / 1e9;

  std::printf("=== Performance (N=%d, %d runs) ===\n", n, NITER);
  std::printf("  Avg latency : %.6f ms\n", avg_ms);
  std::printf("  Throughput  : %.3f Gelem/s\n", throughput / 1e9);
  std::printf("  Bandwidth   : %.3f GB/s\n", bandwidth);

  cudaEventDestroy(t0);
  cudaEventDestroy(t1);
  return pass ? 0 : 1;
}

// nvcc -arch=sm_120 -lineinfo -ptx 113-leet-silu.cu -o 113-leet-silu.ptx