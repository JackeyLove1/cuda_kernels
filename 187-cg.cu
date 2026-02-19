#include <cooperative_groups.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cooperative_groups/scan.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda/ptx>
#include <cstdio>
#include <functional>
#include <vector>

namespace cg = cooperative_groups;

#define CEIL(a, b) (((a) + (b) - 1) / (b))

// ── Version A: classic multi-block reduce via atomicAdd ──────────────────────
__global__ void cg_reduce_kernel(const float* __restrict__ input, float* __restrict__ output,
                                 const int N) {
  cg::thread_block cta = cg::this_thread_block();
  auto warp = cg::tiled_partition<32>(cta);
  const int gid = blockDim.x * blockIdx.x + threadIdx.x;
  const int stride = blockDim.x * gridDim.x;
  float val = 0.0f;
  for (int i = gid; i < N; i += stride) val += input[i];
  float warp_sum = cg::reduce(warp, val, cg::plus<float>());

  __shared__ float block_sum[32];
  int warp_id = cta.thread_rank() >> 5;
  int lane_id = cta.thread_rank() & 31;
  if (lane_id == 0) block_sum[warp_id] = warp_sum;
  cta.sync();

  int num_warps = blockDim.x >> 5;
  if (warp_id == 0) {
    float block_val = (lane_id < num_warps) ? block_sum[lane_id] : 0.0f;
    float block_total_sum = cg::reduce(warp, block_val, cg::plus<float>());
    if (lane_id == 0) atomicAdd(output, block_total_sum);
  }
}

// ── Version B: grid.sync() — NO atomicAdd ────────────────────────────────────
// 必须通过 cudaLaunchCooperativeKernel 启动，nblks ≤ 最大同时驻留 block 数
__global__ void cg_grid_reduce_kernel(const float* __restrict__ input, float* __restrict__ output,
                                      float* __restrict__ partial, const int N) {
  cg::grid_group grid = cg::this_grid();
  cg::thread_block cta = cg::this_thread_block();
  auto warp = cg::tiled_partition<32>(cta);

  const int gid = blockDim.x * blockIdx.x + threadIdx.x;
  const int stride = blockDim.x * gridDim.x;

  // Phase 1: 每个线程累加自己负责的元素
  float val = 0.0f;
  for (int i = gid; i < N; i += stride) val += input[i];

  // Phase 2: warp 内归约
  float warp_sum = cg::reduce(warp, val, cg::plus<float>());

  // Phase 3: block 内归约，结果写入 partial[blockIdx.x]（无 atomicAdd）
  __shared__ float smem[32];
  int warp_id = cta.thread_rank() >> 5;
  int lane_id = cta.thread_rank() & 31;
  if (lane_id == 0) smem[warp_id] = warp_sum;
  cta.sync();

  int num_warps = blockDim.x >> 5;
  if (warp_id == 0) {
    float block_val = (lane_id < num_warps) ? smem[lane_id] : 0.0f;
    float block_total = cg::reduce(warp, block_val, cg::plus<float>());
    // 直接按 blockIdx.x 写入，不需要 atomicAdd
    if (lane_id == 0) partial[blockIdx.x] = block_total;
  }

  // Phase 4: 等待所有 block 都写完各自的 partial sum
  // grid.sync() 是全局屏障：所有 block 到达后才继续
  grid.sync();

  // Phase 5: block 0 对 partial[] 做最终归约，直接写 output
  if (blockIdx.x == 0) {
    // 每个线程跨步累加多个 partial（当 gridDim.x > blockDim.x 时）
    float v = 0.0f;
    for (int i = threadIdx.x; i < static_cast<int>(gridDim.x); i += blockDim.x) v += partial[i];

    float ws = cg::reduce(warp, v, cg::plus<float>());
    if (lane_id == 0) smem[warp_id] = ws;
    cta.sync();

    if (warp_id == 0) {
      float bv = (lane_id < num_warps) ? smem[lane_id] : 0.0f;
      float total = cg::reduce(warp, bv, cg::plus<float>());
      if (lane_id == 0) *output = total;  // 直接写，无 atomicAdd
    }
  }
}

extern "C" void solve(const float* input, float* output, int N) {
  constexpr int nthrs = 256;
  const int nblks = std::min(CEIL(N, nthrs), 2048);
  cg_reduce_kernel<<<nblks, nthrs>>>(input, output, N);
  cudaDeviceSynchronize();
}

#define CUDA_CHECK(call)                                                                       \
  do {                                                                                         \
    cudaError_t _e = (call);                                                                   \
    if (_e != cudaSuccess) {                                                                   \
      printf("[CUDA ERROR] %s:%d  %s\n", __FILE__, __LINE__, cudaGetErrorString(_e));          \
    }                                                                                          \
  } while (0)

// grid.sync 版本：使用 cudaLaunchCooperativeKernel
// nblks 由 occupancy API 计算，保证 ≤ GPU 最大同时驻留 block 数（grid.sync 安全约束）
extern "C" void solve_grid(const float* input, float* output, int N) {
  constexpr int nthrs = 256;

  static int nblks = 0;
  static float* partial = nullptr;
  if (nblks == 0) {
    int device;
    CUDA_CHECK(cudaGetDevice(&device));
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
    int max_blocks_per_sm;
    CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&max_blocks_per_sm,
                                                             cg_grid_reduce_kernel, nthrs, 0));
    nblks = std::min(max_blocks_per_sm * prop.multiProcessorCount, CEIL(N, nthrs));
    CUDA_CHECK(cudaMalloc(&partial, nblks * sizeof(float)));
  }

  void* args[] = {(void*)&input, (void*)&output, (void*)&partial, (void*)&N};
  CUDA_CHECK(
      cudaLaunchCooperativeKernel((void*)cg_grid_reduce_kernel, nblks, nthrs, args, 0, nullptr));
  CUDA_CHECK(cudaDeviceSynchronize());
}

__global__ void print_cg() {
  const auto grid = cg::this_grid();
  const auto thread_block = cg::this_thread_block();
  const auto thread_nums = thread_block.num_threads();
  const auto thread_rank = thread_block.thread_rank();
  printf("thread nums: %d, thread rank:%d\n", thread_nums, thread_rank);
}

static float time_kernel(std::function<void()> fn, int warmup = 5, int iters = 100) {
  for (int i = 0; i < warmup; ++i) fn();
  cudaDeviceSynchronize();
  cudaEvent_t t0, t1;
  cudaEventCreate(&t0);
  cudaEventCreate(&t1);
  cudaEventRecord(t0);
  for (int i = 0; i < iters; ++i) fn();
  cudaEventRecord(t1);
  cudaEventSynchronize(t1);
  float ms = 0;
  cudaEventElapsedTime(&ms, t0, t1);
  cudaEventDestroy(t0);
  cudaEventDestroy(t1);
  return ms / iters;
}

int main() {
  constexpr int N = 1 << 24;  // 16M elements
  std::vector<float> h_in(N, 1.0f);

  float *d_in, *d_out_a, *d_out_b;
  CUDA_CHECK(cudaMalloc(&d_in, N * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_out_a, sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_out_b, sizeof(float)));
  CUDA_CHECK(cudaMemcpy(d_in, h_in.data(), N * sizeof(float), cudaMemcpyHostToDevice));

  // ── 正确性验证 ──────────────────────────────────────────────────────────────
  CUDA_CHECK(cudaMemset(d_out_a, 0, sizeof(float)));
  solve(d_in, d_out_a, N);
  float ra = 0;
  cudaMemcpy(&ra, d_out_a, sizeof(float), cudaMemcpyDeviceToHost);

  CUDA_CHECK(cudaMemset(d_out_b, 0, sizeof(float)));
  solve_grid(d_in, d_out_b, N);
  float rb = 0;
  cudaMemcpy(&rb, d_out_b, sizeof(float), cudaMemcpyDeviceToHost);

  printf("Version A (atomicAdd) : %.0f\n", ra);
  printf("Version B (grid.sync) : %.0f\n", rb);
  printf("Expected              : %d\n\n", N);

  // ── 性能测试 ────────────────────────────────────────────────────────────────
  auto bench_a = [&]() {
    cudaMemset(d_out_a, 0, sizeof(float));
    solve(d_in, d_out_a, N);
  };
  auto bench_b = [&]() {
    cudaMemset(d_out_b, 0, sizeof(float));
    solve_grid(d_in, d_out_b, N);
  };

  float ms_a = time_kernel(bench_a);
  float ms_b = time_kernel(bench_b);

  double bw_a = (double)N * sizeof(float) / (ms_a * 1e-3) / 1e9;
  double bw_b = (double)N * sizeof(float) / (ms_b * 1e-3) / 1e9;

  printf("%-30s %6.3f ms  %6.1f GB/s\n", "Version A (atomicAdd):", ms_a, bw_a);
  printf("%-30s %6.3f ms  %6.1f GB/s\n", "Version B (grid.sync):", ms_b, bw_b);
  printf("Speedup B/A: %.2fx\n", ms_a / ms_b);

  cudaFree(d_in);
  cudaFree(d_out_a);
  cudaFree(d_out_b);
}