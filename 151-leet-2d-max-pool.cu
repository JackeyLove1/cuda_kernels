#include <cuda_runtime.h>

#include <algorithm>
#include <cfloat>
#include <cmath>

// CUDA Kernel: 2D Max Pooling
// 使用 __restrict__ 帮助编译器优化缓存加载
__global__ void max_pool_2d_kernel(const int nthreads, const float* __restrict__ input,
                                   float* __restrict__ output, const int N, const int C,
                                   const int H, const int W, const int H_out, const int W_out,
                                   const int kernel_size, const int stride, const int padding) {
  // 1. 扁平化的一维索引，确保连续线程处理连续内存
  int index = blockIdx.x * blockDim.x + threadIdx.x;

  if (index < nthreads) {
    // 2. 将一维索引逆映射回 (n, c, h_out, w_out)
    // 索引顺序假设为 NCHW (即 w 变化最快)
    int w_out = index % W_out;
    int h_out = (index / W_out) % H_out;
    int c = (index / (W_out * H_out)) % C;
    int n = index / (W_out * H_out * C);

    // 3. 计算输入窗口在 Input Tensor 中的对应位置
    // 对应公式：h = h_out * stride - padding
    int h_in_start = h_out * stride - padding;
    int w_in_start = w_out * stride - padding;

    // 4. 优化：边界预计算 (Loop Bound Optimization)
    // 直接计算出循环的有效范围，避免在内层循环中使用 if 判断
    int h_start = max(0, h_in_start);
    int h_end = min(H, h_in_start + kernel_size);
    int w_start = max(0, w_in_start);
    int w_end = min(W, w_in_start + kernel_size);

    // 初始化最大值为最小的浮点数
    float max_val = -FLT_MAX;

    // 5. 计算 Input 的基础偏移量 (针对 n 和 c)
    // input 的 layout 是 [N, C, H, W]
    int input_offset_base = (n * C + c) * H * W;

    // 6. 执行池化操作
    // 此时循环内没有任何 if 分支，流水线效率最高
    for (int h = h_start; h < h_end; ++h) {
      for (int w = w_start; w < w_end; ++w) {
        int input_idx = input_offset_base + h * W + w;
        float val = input[input_idx];
        max_val = fmaxf(max_val, val);
      }
    }

    // 写入结果
    output[index] = max_val;
  }
}

extern "C" void solve(const float* input, float* output, int N, int C, int H, int W,
                      int kernel_size, int stride, int padding) {
  // 1. 计算输出尺寸
  // 公式: floor((input + 2*pad - kernel) / stride) + 1
  int H_out = (H + 2 * padding - kernel_size) / stride + 1;
  int W_out = (W + 2 * padding - kernel_size) / stride + 1;

  // 2. 计算总元素数量
  int total_output_elements = N * C * H_out * W_out;

  // 3. 配置 CUDA Launch 参数
  // 使用 256 或 512 作为 block size 是常见选择
  int threads_per_block = 256;
  // 计算 grid size，确保覆盖所有元素
  int blocks_per_grid = (total_output_elements + threads_per_block - 1) / threads_per_block;

  // 4. 启动 Kernel
  max_pool_2d_kernel<<<blocks_per_grid, threads_per_block>>>(
      total_output_elements, input, output, N, C, H, W, H_out, W_out, kernel_size, stride, padding);

  // 注意：在实际工程或 benchmakr 中通常需要 cudaDeviceSynchronize() 或检查错误，
  // 但题目仅要求 solve 函数实现。
}