#include <cuda_runtime.h>
#include <math.h>

// 定义 PI 常量
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// 使用 float2 作为复数容器: .x 为实部, .y 为虚部
typedef float2 Complex;

// ---------------------------------------------------------
// Device Helper Functions (GPU端复数运算辅助函数)
// ---------------------------------------------------------

__device__ inline Complex c_add(Complex a, Complex b) { return make_float2(a.x + b.x, a.y + b.y); }

__device__ inline Complex c_sub(Complex a, Complex b) { return make_float2(a.x - b.x, a.y - b.y); }

// 复数乘法: (a+bi)(c+di) = (ac-bd) + (ad+bc)i
__device__ inline Complex c_mul(Complex a, Complex b) {
  return make_float2(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}

// ---------------------------------------------------------
// Kernels
// ---------------------------------------------------------

/**
 * Kernel 1: 位反转拷贝 (Bit Reversal Permutation)
 * 将输入信号按照位反转顺序拷贝到输出数组，作为 FFT 的预处理。
 */
__global__ void bit_reverse_kernel(const float* __restrict__ input, float* __restrict__ output,
                                   int N, int logN) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;

  if (tid < N) {
    unsigned int rev = 0;
    unsigned int val = tid;

    // 执行位反转算法
    for (int i = 0; i < logN; i++) {
      rev = (rev << 1) | (val & 1);
      val >>= 1;
    }

    // 按照 interleaved layout (real, imag) 读取和写入
    // input[2*tid] 是实部, input[2*tid+1] 是虚部
    output[2 * rev] = input[2 * tid];
    output[2 * rev + 1] = input[2 * tid + 1];
  }
}

/**
 * Kernel 2: 蝶形运算 (FFT Stage)
 * 执行 Cooley-Tukey 算法的一个阶段。
 * * spectrum: 正在处理的频谱数组 (in-place 更新)
 * N: 信号总长度
 * m: 当前阶段的 DFT 长度 (2, 4, 8, ..., N)
 * m2: m 的一半，即蝶形运算的跨度 (stride)
 */
__global__ void fft_stage_kernel(float* spectrum, int N, int m, int m2) {
  // 总共有 N/2 个蝶形运算需要并行处理
  int tid = blockIdx.x * blockDim.x + threadIdx.x;

  if (tid < N / 2) {
    // 映射 tid 到具体的蝶形运算索引
    // 蝶形运算分组进行。每组大小为 m，组内包含 m2 个蝶形运算。

    // k: 在当前 DFT 组内的偏移量 (0 到 m2-1)，用于计算旋转因子
    int k = tid % m2;

    // group_start: 当前蝶形运算所在组的起始索引
    int group_start = (tid / m2) * m;

    // 参与蝶形运算的两个元素的索引
    int idx1 = group_start + k;
    int idx2 = idx1 + m2;

    // 读取数据 (Interleaved layout)
    Complex u = make_float2(spectrum[2 * idx1], spectrum[2 * idx1 + 1]);
    Complex t_val = make_float2(spectrum[2 * idx2], spectrum[2 * idx2 + 1]);

    // 计算旋转因子 W_m^k = exp(-j * 2 * pi * k / m)
    double angle = -2.0 * M_PI * (double)k / (double)m;
    double sin_a, cos_a;
    sincos(angle, &sin_a, &cos_a);
    // 蝶形运算核心公式（中间过程用 double 降低累计误差）
    double ur = (double)u.x;
    double ui = (double)u.y;
    double vr = (double)t_val.x;
    double vi = (double)t_val.y;
    double tr = cos_a * vr - sin_a * vi;
    double ti = cos_a * vi + sin_a * vr;

    // 写回结果
    spectrum[2 * idx1] = (float)(ur + tr);
    spectrum[2 * idx1 + 1] = (float)(ui + ti);
    spectrum[2 * idx2] = (float)(ur - tr);
    spectrum[2 * idx2 + 1] = (float)(ui - ti);
  }
}

/**
 * Kernel 3: 通用 DFT (支持任意 N)
 * 每个线程计算一个输出频点 k:
 *   X[k] = sum_{n=0}^{N-1} x[n] * exp(-j * 2pi * k * n / N)
 */
__global__ void dft_kernel(const float* __restrict__ input, float* __restrict__ output, int N) {
  int k = blockIdx.x * blockDim.x + threadIdx.x;
  if (k >= N) return;

  float sum_real = 0.0f;
  float sum_imag = 0.0f;

  for (int n = 0; n < N; ++n) {
    float xr = input[2 * n];
    float xi = input[2 * n + 1];

    double angle = -2.0 * M_PI * (double)k * (double)n / (double)N;
    float sin_a, cos_a;
    __sincosf(angle, &sin_a, &cos_a);

    // (xr + j*xi) * (cos + j*sin)
    sum_real += xr * cos_a - xi * sin_a;
    sum_imag += xr * sin_a + xi * cos_a;
  }

  output[2 * k] = sum_real;
  output[2 * k + 1] = sum_imag;
}

// ---------------------------------------------------------
// Host Function: solve
// ---------------------------------------------------------

extern "C" void solve(const float* signal, float* spectrum, int N) {
  if (N <= 0) return;

  // 1. 分配 GPU 内存
  float *d_input, *d_spectrum;
  size_t size_bytes = N * 2 * sizeof(float);  // N 个复数 = 2N 个浮点数

  cudaMalloc((void**)&d_input, size_bytes);
  cudaMalloc((void**)&d_spectrum, size_bytes);

  // 2. 将输入数据从 Host 拷贝到 Device
  cudaMemcpy(d_input, signal, size_bytes, cudaMemcpyHostToDevice);

  // 3. 配置 Kernel 启动参数
  int blockSize = 256;
  int numBlocks = (N + blockSize - 1) / blockSize;

  // 4. 判断是否为 2 的幂；是则走 radix-2 FFT，否则走通用 DFT
  bool is_pow2 = (N & (N - 1)) == 0;

  if (is_pow2) {
    int logN = 0;
    int tempN = N;
    while (tempN > 1) {
      tempN >>= 1;
      logN++;
    }

    int numBlocksButterflies = (N / 2 + blockSize - 1) / blockSize;
    bit_reverse_kernel<<<numBlocks, blockSize>>>(d_input, d_spectrum, N, logN);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
      cudaFree(d_input);
      cudaFree(d_spectrum);
      return;
    }

    for (int m = 2; m <= N; m *= 2) {
      int m2 = m / 2;
      fft_stage_kernel<<<numBlocksButterflies, blockSize>>>(d_spectrum, N, m, m2);
    }
  } else {
    dft_kernel<<<numBlocks, blockSize>>>(d_input, d_spectrum, N);
  }

  // 5. 将结果从 Device 拷贝回 Host
  cudaMemcpy(spectrum, d_spectrum, size_bytes, cudaMemcpyDeviceToHost);

  // 6. 释放 GPU 内存
  cudaFree(d_input);
  cudaFree(d_spectrum);
}