// cg_reduce_example.cu
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

#include <cstdio>
#include <vector>

namespace cg = cooperative_groups;

__global__ void block_reduce_sum(const int* __restrict__ data, int* __restrict__ result, int N) {
  cg::thread_block cta = cg::this_thread_block();
  // cg::reduce 只支持 tile group（如 tiled_partition<32>），不支持整个 thread_block
  auto warp = cg::tiled_partition<32>(cta);

  int gid = blockIdx.x * blockDim.x + threadIdx.x;
  int val  = (gid < N) ? data[gid] : 0;

  // 第一级：warp 内归约
  int warp_sum = cg::reduce(warp, val, cg::plus<int>());

  // 第二级：各 warp 的部分和写入 shared memory，再由第一个 warp 汇总
  __shared__ int smem[32];  // 最多 1024/32 = 32 个 warp
  int warp_id = threadIdx.x >> 5;   // threadIdx.x / 32
  int lane_id = warp.thread_rank(); // threadIdx.x % 32

  if (lane_id == 0) smem[warp_id] = warp_sum;
  cta.sync();

  // 第一个 warp 负责最终汇总
  int num_warps = blockDim.x >> 5;
  if (warp_id == 0) {
    int block_val  = (lane_id < num_warps) ? smem[lane_id] : 0;
    int block_sum  = cg::reduce(warp, block_val, cg::plus<int>());
    if (lane_id == 0) result[blockIdx.x] = block_sum;
  }
}

static void checkCuda(cudaError_t e, const char* msg) {
  if (e != cudaSuccess) {
    std::fprintf(stderr, "CUDA error (%s): %s\n", msg, cudaGetErrorString(e));
    std::exit(1);
  }
}

int main() {
  // 准备一些数据：1..N
  int N = 1 << 20;  // 1048576
  std::vector<int> h_data(N);
  for (int i = 0; i < N; ++i) h_data[i] = 1;  // 简单点：全 1，方便验证

  int threads = 256;
  int blocks = (N + threads - 1) / threads;

  // device 内存
  int *d_data = nullptr, *d_out = nullptr;
  checkCuda(cudaMalloc(&d_data, N * sizeof(int)), "malloc d_data");
  checkCuda(cudaMalloc(&d_out, blocks * sizeof(int)), "malloc d_out");

  checkCuda(cudaMemcpy(d_data, h_data.data(), N * sizeof(int), cudaMemcpyHostToDevice), "copy H2D");

  // kernel
  block_reduce_sum<<<blocks, threads>>>(d_data, d_out, N);
  checkCuda(cudaGetLastError(), "launch kernel");
  checkCuda(cudaDeviceSynchronize(), "sync");

  // 拿回每个 block 的部分和
  std::vector<int> h_out(blocks);
  checkCuda(cudaMemcpy(h_out.data(), d_out, blocks * sizeof(int), cudaMemcpyDeviceToHost),
            "copy D2H");

  // CPU 汇总验证
  long long total = 0;
  for (int b = 0; b < blocks; ++b) total += h_out[b];

  std::printf("blocks=%d, threads=%d\n", blocks, threads);
  std::printf("GPU partial sums reduced on CPU: %lld\n", total);
  std::printf("Expected: %d\n", N);

  cudaFree(d_data);
  cudaFree(d_out);
  return 0;
}
