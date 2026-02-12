#include <cuda.h>
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <type_traits>

/**
Implement a program that performs the Rectified Linear Unit (ReLU) activation function on a vector
of 32-bit floating point numbers. The ReLU function sets all negative values to zero and leaves
positive values unchanged: ReLU(x)=max(0,x) Implementation Requirements ·External libraries are not
permitted ·The solve function signature must remain unchanged ·The final result must be stored in
output
 */
#define CEIL(a, b) (((a) + (b) - 1) / (b))
#define LOAD4(ptr) (reinterpret_cast<const float4*>(ptr))
#define STORE4(ptr) (reinterpret_cast<float4*>(ptr))

#define WORKS 1  // thread works per iteration

__forceinline__ __device__ float4 operator+(const float4 a, const float4 b) {
  return make_float4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w);
}

static constexpr float alpha = 0.01f;

template <typename T>
struct dependent_false : std::false_type {};

template <typename T>
__forceinline__ __device__ T _leaky_relu_op(const T x) {
  if constexpr (std::is_same_v<T, float>) {
    return x > 0.0f ? x : x * alpha;
  } else if constexpr (std::is_same_v<T, float4>) {
    return make_float4(_leaky_relu_op(x.x), _leaky_relu_op(x.y), _leaky_relu_op(x.z),
                       _leaky_relu_op(x.w));
  } else {
    static_assert(dependent_false<T>::value, "Unsupported type");
  }
}
__global__ __launch_bounds__(256) void leaky_relu_kernel(const float* __restrict__ input,
                                                         float* __restrict__ output, const int N) {
  const auto tid = threadIdx.x;
  const auto gid = blockIdx.x * blockDim.x + tid;
  const auto stride = blockDim.x * gridDim.x;
  const float4* in4 = LOAD4(input);
  float4* out4 = STORE4(output);
  const int total = N >> 2;

  for (int i = gid; i < total; i += stride) {
    float4 in = in4[i];
    float4 out = _leaky_relu_op(in);
    out4[i] = out;
  }

  const auto tail_start = total * 4;
  for (int i = tail_start + gid; i < N; i += stride) {
    output[i] = _leaky_relu_op(input[i]);
  }
}

// input, output are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* input, float* output, int N) {
  int threadsPerBlock = 256;
  int blocksPerGrid = 512;

  leaky_relu_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, output, N);
  cudaDeviceSynchronize();
}

void leaky_relu_cpu(const float* input, float* output, int N) {
  for (int i = 0; i < N; ++i) {
    output[i] = input[i] > 0.0f ? input[i] : input[i] * alpha;
  }
}

bool check_correctness(const thrust::host_vector<float>& gpu_result,
                       const thrust::host_vector<float>& cpu_result, int N, float rtol = 1e-4f,
                       float atol = 1e-5f) {
  constexpr int SAMPLE_THRESHOLD = 1 << 20;  // 1M
  constexpr int SAMPLE_COUNT = 8192;
  constexpr int MAX_ERRORS = 10;

  std::mt19937 rng(42);
  std::uniform_int_distribution<int> sample_dist(0, N - 1);

  int check_count = (N > SAMPLE_THRESHOLD) ? SAMPLE_COUNT : N;
  int errors = 0;
  float max_diff = 0.0f;
  for (int k = 0; k < check_count; ++k) {
    int i = (N > SAMPLE_THRESHOLD) ? sample_dist(rng) : k;
    float diff = fabsf(gpu_result[i] - cpu_result[i]);
    float ref = fabsf(cpu_result[i]);
    max_diff = fmaxf(max_diff, diff);
    if (diff > atol + rtol * ref) {
      if (errors < MAX_ERRORS) {
        printf("  MISMATCH at [%d]: gpu=%.6f cpu=%.6f diff=%.6e\n", i, gpu_result[i], cpu_result[i],
               diff);
      }
      ++errors;
    }
  }

  printf("  Checked %d / %d elements, max_diff=%.6e, errors=%d\n", check_count, N, max_diff, errors);
  return errors == 0;
}

int main(int argc, char** argv) {
  int N = (1 << 24);
  if (argc > 1) {
    N = atoi(argv[1]);
  }
  if (N <= 0) {
    printf("N must be > 0\n");
    return 1;
  }

  printf("=== Leaky ReLU Kernel Unit Test & Benchmark ===\n");
  printf("N = %d (%.2f MB per vector)\n", N, static_cast<double>(N) * sizeof(float) / (1 << 20));

  thrust::host_vector<float> h_in(N);
  thrust::host_vector<float> h_cpu_out(N);
  thrust::device_vector<float> d_in(N);
  thrust::device_vector<float> d_out(N);

  std::mt19937 rng(42);
  std::uniform_real_distribution<float> dist(-10.0f, 10.0f);
  for (int i = 0; i < N; ++i) {
    h_in[i] = dist(rng);
  }
  d_in = h_in;

  solve(thrust::raw_pointer_cast(d_in.data()), thrust::raw_pointer_cast(d_out.data()), N);
  cudaDeviceSynchronize();

  thrust::host_vector<float> h_gpu_out = d_out;
  leaky_relu_cpu(h_in.data(), h_cpu_out.data(), N);

  bool pass = check_correctness(h_gpu_out, h_cpu_out, N);
  printf("Correctness: %s\n\n", pass ? "PASS" : "FAIL");

  cudaEvent_t t0, t1;
  cudaEventCreate(&t0);
  cudaEventCreate(&t1);

  constexpr int WARMUP = 10;
  for (int i = 0; i < WARMUP; ++i) {
    solve(thrust::raw_pointer_cast(d_in.data()), thrust::raw_pointer_cast(d_out.data()), N);
  }
  cudaDeviceSynchronize();

  constexpr int NITER = 100;
  cudaEventRecord(t0);
  for (int i = 0; i < NITER; ++i) {
    solve(thrust::raw_pointer_cast(d_in.data()), thrust::raw_pointer_cast(d_out.data()), N);
  }
  cudaEventRecord(t1);
  cudaEventSynchronize(t1);

  float total_ms = 0.0f;
  cudaEventElapsedTime(&total_ms, t0, t1);
  float avg_ms = total_ms / NITER;

  double bytes_moved = 2.0 * static_cast<double>(N) * sizeof(float);  // one read + one write
  double bandwidth_gb = bytes_moved / (avg_ms * 1e-3) / 1e9;
  double elem_per_sec = static_cast<double>(N) / (avg_ms * 1e-3);

  printf("=== Performance (N=%d, %d runs) ===\n", N, NITER);
  printf("  Avg latency : %.4f ms\n", avg_ms);
  printf("  Throughput  : %.2f Gelem/s\n", elem_per_sec / 1e9);
  printf("  Bandwidth   : %.2f GB/s\n", bandwidth_gb);

  cudaEventDestroy(t0);
  cudaEventDestroy(t1);
  return pass ? 0 : 1;
}