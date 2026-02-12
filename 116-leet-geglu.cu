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
static constexpr int MAX_BLOCKS = 4096;
static constexpr int THREADS_PER_BLOCK = 256;

__device__ __forceinline__ float gelu_erf_fast(float x) {
  // 0.5*x*(1+erf(x/sqrt(2)))
  // rsqrt(2) = 0.7071067811865475
  return 0.5f * x * (1.0f + erff(x * 0.7071067811865475f));
}

__device__ __forceinline__ float tanh_fast(float u) {
  // tanh(u) = (1 - e^{-2u}) / (1 + e^{-2u})
  // 为避免溢出/浪费，可对 u 做工程截断
  if (u > 10.0f) return 1.0f;
  if (u < -10.0f) return -1.0f;

  float e = __expf(-2.0f * u);
  // (1 - e) / (1 + e)
  return (1.0f - e) * __frcp_rn(1.0f + e);
}

__device__ __forceinline__ float gelu_tanh_fast(float x) {
  const float a = 0.7978845608028654f;
  const float b = 0.044714998453855515f;

  float x2 = x * x;
  float x3 = x * x2;
  float u = a * (x + b * x3);

  float t = tanh_fast(u);
  return 0.5f * x * (1.0f + t);
}

__device__ __forceinline__ float op(float x1, float x2) { return x1 * gelu_erf_fast(x2); }

__global__ void geglu_kernel(const float* __restrict__ input, float* __restrict__ output,
                             const int halfN) {
  const int tid = threadIdx.x;
  const int gid = blockIdx.x * blockDim.x + tid;
  const int stride = gridDim.x * blockDim.x;
  const int total = halfN;
  const int interval4 = halfN >> 2;
  const int vec_size = total >> 2;
  const int vec_tail = total & 3;
  const float4* in4 = LOAD4(input);
  float4* output4 = STORE4(output);
  for (int i = gid; i < vec_size; i += stride) {
    const float4 x1 = in4[i];
    const float4 x2 = in4[i + interval4];
    const float4 result =
        make_float4(op(x1.x, x2.x), op(x1.y, x2.y), op(x1.z, x2.z), op(x1.w, x2.w));
    output4[i] = result;
  }
  if (vec_tail && gid < vec_tail) {
    const int base = (vec_size << 2) + gid;
    const float x1 = input[base];
    const float x2 = input[base + halfN];
    output[base] = op(x1, x2);
  }
}

// input, output are device pointers
extern "C" void solve(const float* input, float* output, int N) {
  const int halfN = N / 2;
  const int threadsPerBlock = THREADS_PER_BLOCK;
  const int blocksPerGrid = std::min(CEIL(halfN, threadsPerBlock * 4), MAX_BLOCKS);

  geglu_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, output, halfN);
  cudaDeviceSynchronize();
}

void geglu_cpu_reference(const float* input, float* output, int N) {
  const int halfN = N / 2;
  for (int i = 0; i < halfN; ++i) {
    output[i] = input[i] * (0.5f * input[i + halfN] *
                            (1.0f + std::erf(input[i + halfN] * 0.7071067811865475f)));
  }
}

bool check_correctness(const thrust::host_vector<float>& gpu_result,
                       const thrust::host_vector<float>& cpu_result, int N, float rtol = 1e-4f,
                       float atol = 1e-4f) {
  constexpr int SAMPLE_THRESHOLD = 1 << 20;
  constexpr int SAMPLE_COUNT = 8192;
  constexpr int MAX_ERRORS_TO_PRINT = 10;

  const int halfN = N / 2;
  std::vector<int> indices(halfN);
  std::iota(indices.begin(), indices.end(), 0);

  std::mt19937 rng(42);
  int check_count = halfN;
  if (halfN > SAMPLE_THRESHOLD) {
    std::shuffle(indices.begin(), indices.end(), rng);
    indices.resize(SAMPLE_COUNT);
    check_count = SAMPLE_COUNT;
  }

  int errors = 0;
  float max_diff = 0.0f;
  for (int k = 0; k < check_count; ++k) {
    const int i = indices[k];
    const float diff = std::fabs(gpu_result[i] - cpu_result[i]);
    const float ref = std::fabs(cpu_result[i]);
    max_diff = std::max(max_diff, diff);
    if (diff > atol + rtol * ref) {
      if (errors < MAX_ERRORS_TO_PRINT) {
        std::printf("  MISMATCH at [%d]: gpu=%.8f cpu=%.8f diff=%.8e\n", i, gpu_result[i],
                    cpu_result[i], diff);
      }
      ++errors;
    }
  }

  std::printf("  Checked %d / %d output elements, max_diff=%.8e, errors=%d\n", check_count, halfN,
              max_diff, errors);
  return errors == 0;
}

int main(int argc, char** argv) {
  constexpr int N_MIN = 1;
  constexpr int N_MAX = 1'000'000;
  constexpr int PERF_N = 1'000'000;
  constexpr float INPUT_MIN = -100.0f;
  constexpr float INPUT_MAX = 100.0f;
  constexpr int WARMUP = 10;
  constexpr int NITER = 100;

  int testN = PERF_N;
  if (argc > 1) {
    testN = std::atoi(argv[1]);
  }

  if (testN < N_MIN || testN > N_MAX) {
    std::fprintf(stderr, "Error: N must satisfy 1 <= N <= 1,000,000, got %d\n", testN);
    return 1;
  }
  if ((testN & 1) != 0) {
    std::fprintf(stderr, "Error: N must be an even number, got %d\n", testN);
    return 1;
  }

  std::printf("=== GeGLU Kernel Unit Test & Benchmark ===\n");
  std::printf("Correctness test N = %d, benchmark N = %d\n", testN, PERF_N);
  std::printf("Input range = [%.1f, %.1f]\n\n", INPUT_MIN, INPUT_MAX);

  // Correctness section
  thrust::host_vector<float> h_in(testN);
  thrust::host_vector<float> h_cpu_out(testN / 2);
  thrust::device_vector<float> d_in(testN);
  thrust::device_vector<float> d_out(testN / 2);

  std::mt19937 rng(42);
  std::uniform_real_distribution<float> dist(INPUT_MIN, INPUT_MAX);
  for (int i = 0; i < testN; ++i) {
    h_in[i] = dist(rng);
  }

  d_in = h_in;
  solve(thrust::raw_pointer_cast(d_in.data()), thrust::raw_pointer_cast(d_out.data()), testN);
  cudaDeviceSynchronize();

  thrust::host_vector<float> h_gpu_out = d_out;
  geglu_cpu_reference(h_in.data(), h_cpu_out.data(), testN);
  const bool pass = check_correctness(h_gpu_out, h_cpu_out, testN);
  std::printf("Correctness: %s\n\n", pass ? "PASS" : "FAIL");

  // Performance section (fixed N = 1,000,000)
  thrust::host_vector<float> h_perf_in(PERF_N);
  thrust::device_vector<float> d_perf_in(PERF_N);
  thrust::device_vector<float> d_perf_out(PERF_N / 2);
  for (int i = 0; i < PERF_N; ++i) {
    h_perf_in[i] = dist(rng);
  }
  d_perf_in = h_perf_in;

  for (int i = 0; i < WARMUP; ++i) {
    solve(thrust::raw_pointer_cast(d_perf_in.data()), thrust::raw_pointer_cast(d_perf_out.data()),
          PERF_N);
  }
  cudaDeviceSynchronize();

  cudaEvent_t t0, t1;
  cudaEventCreate(&t0);
  cudaEventCreate(&t1);

  cudaEventRecord(t0);
  for (int i = 0; i < NITER; ++i) {
    solve(thrust::raw_pointer_cast(d_perf_in.data()), thrust::raw_pointer_cast(d_perf_out.data()),
          PERF_N);
  }
  cudaEventRecord(t1);
  cudaEventSynchronize(t1);

  float total_ms = 0.0f;
  cudaEventElapsedTime(&total_ms, t0, t1);
  const float avg_ms = total_ms / static_cast<float>(NITER);

  const double bytes_moved = 1.5 * static_cast<double>(PERF_N) * sizeof(float);
  const double throughput_gelem = (static_cast<double>(PERF_N) / 2.0) / (avg_ms * 1e-3) / 1e9;
  const double bandwidth_gbs = bytes_moved / (avg_ms * 1e-3) / 1e9;

  std::printf("=== Performance (N=%d, runs=%d) ===\n", PERF_N, NITER);
  std::printf("  Avg latency : %.4f ms\n", avg_ms);
  std::printf("  Throughput  : %.4f Gelem/s (output elements)\n", throughput_gelem);
  std::printf("  Bandwidth   : %.2f GB/s\n", bandwidth_gbs);

  cudaEventDestroy(t0);
  cudaEventDestroy(t1);

  return pass ? 0 : 1;
}
