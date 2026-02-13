#include <cuda.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <climits>
#include <cstdio>

//  warp 级规约：寻找 warp 内的最大值
__device__ __forceinline__ int warpReduceMax(int val) {
  for (int offset = 16; offset > 0; offset /= 2) {
    val = max(val, __shfl_down_sync(0xffffffff, val, offset));
  }
  return val;
}

// Block 级规约：寻找 Block 内的最大值
__device__ __forceinline__ int blockReduceMax(int val) {
  // 1. 先进行 Warp 内规约
  val = warpReduceMax(val);

  // 2. 每个 Warp 的第一个线程将结果写入共享内存
  static __shared__ int shared[32];  // 假设最大 Block Size 为 1024，即 32 个 Warps
  int lane = threadIdx.x % 32;
  int wid = threadIdx.x / 32;

  if (lane == 0) {
    shared[wid] = val;
  }
  __syncthreads();

  // 3. 第一个 Warp 读取共享内存并再次规约
  val = (threadIdx.x < blockDim.x / 32) ? shared[lane] : INT_MIN;

  if (wid == 0) {
    val = warpReduceMax(val);
  }
  return val;
}

__global__ void optimized_sliding_window_kernel(const int* __restrict__ input,
                                                int* __restrict__ output, const int N, const int K,
                                                const int total_windows) {
  // 计算当前线程需要处理的窗口范围
  // 我们将所有窗口均匀分块给线程，而不是跨步，这样有利于滑动计算
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int num_threads = gridDim.x * blockDim.x;

  // 向上取整，计算每个线程处理的窗口数量
  int windows_per_thread = (total_windows + num_threads - 1) / num_threads;

  int start_idx = tid * windows_per_thread;
  int end_idx = min(start_idx + windows_per_thread, total_windows);

  int local_max = INT_MIN;

  if (start_idx < total_windows) {
    // 1. Setup: 计算该线程负责的第一个窗口的和
    // 时间复杂度 O(K)，但在 N=50000 且数据在缓存中时非常快
    int current_sum = 0;
// 使用 pragma unroll 提示编译器展开小循环
#pragma unroll 4
    for (int i = 0; i < K; ++i) {
      current_sum += input[start_idx + i];
    }
    local_max = current_sum;

    // 2. Sliding: 滑动窗口计算后续的和
    // 时间复杂度 O(M)，M 为该线程分配到的窗口数
    for (int i = start_idx + 1; i < end_idx; ++i) {
      // 新窗口和 = 旧窗口和 - 移出的元素 + 移入的元素
      // input[i - 1] 是移出的元素
      // input[i + K - 1] 是移入的元素
      // 使用 __ldg 显式通过只读缓存加载 (在新架构上通常由 const __restrict__ 自动处理)
      int leaving = input[i - 1];
      int entering = input[i + K - 1];
      current_sum = current_sum - leaving + entering;
      local_max = max(local_max, current_sum);
    }
  }

  // 3. Block 规约
  local_max = blockReduceMax(local_max);

  // 4. 原子更新全局最大值
  if (threadIdx.x == 0) {
    atomicMax(output, local_max);
  }
}

extern "C" void solve(const int* input, int* output, int N, int window_size) {
  // 边界条件处理
  if (N <= 0 || window_size <= 0 || window_size > N) {
    // 无法直接赋值，需要一次 Memcpy，或者让 Kernel 处理空跑
    // 这里为了简单，假设调用者已分配好 output 内存
    int init_val = INT_MIN;
    cudaMemcpy(output, &init_val, sizeof(int), cudaMemcpyHostToDevice);
    return;
  }

  // 初始化输出为 INT_MIN
  int init_val = INT_MIN;
  cudaMemcpy(output, &init_val, sizeof(int), cudaMemcpyHostToDevice);

  const int total_windows = N - window_size + 1;

  // 针对 N=50,000，256 线程 x 若干 Block 足够填满 GPU
  // 同时也足够让每个线程处理适量的窗口以掩盖指令延迟
  const int threads_per_block = 256;
  // 限制最大 Block 数，避免开启过多 Block 导致尾部效应，计算足够覆盖所有窗口即可
  // 实际上对于 N=50000，一个 Block 也能跑完，但多几个 Block 并行度更好
  const int num_blocks = std::min((total_windows + threads_per_block - 1) / threads_per_block, 128);

  optimized_sliding_window_kernel<<<num_blocks, threads_per_block>>>(input, output, N, window_size,
                                                                     total_windows);

  // 只有在需要 debug 或计时的上下文中才强制同步
  cudaDeviceSynchronize();
}