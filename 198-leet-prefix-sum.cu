#include <cuda_runtime.h>

#define BLOCK_THREADS 1024
#define ELEMENTS_PER_BLOCK (BLOCK_THREADS * 2)

// ── 块内 Blelloch Exclusive Scan（内部使用，用于块和的递归扫描）──────────────
__global__ void k_block_scan_exclusive(const float* __restrict__ in, float* __restrict__ out,
                                       float* __restrict__ block_sums, int n) {
  extern __shared__ float s[];
  const int tid = threadIdx.x;
  const int base = blockIdx.x * ELEMENTS_PER_BLOCK;

  s[tid] = (base + tid < n) ? in[base + tid] : 0.f;
  s[tid + BLOCK_THREADS] = (base + tid + BLOCK_THREADS < n) ? in[base + tid + BLOCK_THREADS] : 0.f;
  __syncthreads();

  // Up-Sweep
  for (int stride = 1; stride < ELEMENTS_PER_BLOCK; stride <<= 1) {
    int idx = (tid + 1) * (stride << 1) - 1;
    if (idx < ELEMENTS_PER_BLOCK) s[idx] += s[idx - stride];
    __syncthreads();
  }

  if (tid == 0) {
    if (block_sums) block_sums[blockIdx.x] = s[ELEMENTS_PER_BLOCK - 1];
    s[ELEMENTS_PER_BLOCK - 1] = 0.f;
  }
  __syncthreads();

  // Down-Sweep
  for (int stride = ELEMENTS_PER_BLOCK >> 1; stride >= 1; stride >>= 1) {
    int idx = (tid + 1) * (stride << 1) - 1;
    if (idx < ELEMENTS_PER_BLOCK) {
      float left = s[idx - stride];
      float right = s[idx];
      s[idx - stride] = right;
      s[idx] = left + right;
    }
    __syncthreads();
  }

  // 写回 Exclusive 结果（不加原始值）
  int i0 = base + tid, i1 = i0 + BLOCK_THREADS;
  if (i0 < n) out[i0] = s[tid];
  if (i1 < n) out[i1] = s[tid + BLOCK_THREADS];
}

// ── 块内 Blelloch，最终输出 Inclusive Scan ────────────────────────────────────
__global__ void k_block_scan_inclusive(const float* __restrict__ in, float* __restrict__ out,
                                       float* __restrict__ block_sums, int n) {
  extern __shared__ float s[];
  const int tid = threadIdx.x;
  const int base = blockIdx.x * ELEMENTS_PER_BLOCK;

  // 保存原始值供最后转换
  float a0 = (base + tid < n) ? in[base + tid] : 0.f;
  float a1 = (base + tid + BLOCK_THREADS < n) ? in[base + tid + BLOCK_THREADS] : 0.f;
  s[tid] = a0;
  s[tid + BLOCK_THREADS] = a1;
  __syncthreads();

  // Up-Sweep
  for (int stride = 1; stride < ELEMENTS_PER_BLOCK; stride <<= 1) {
    int idx = (tid + 1) * (stride << 1) - 1;
    if (idx < ELEMENTS_PER_BLOCK) s[idx] += s[idx - stride];
    __syncthreads();
  }

  if (tid == 0) {
    if (block_sums) block_sums[blockIdx.x] = s[ELEMENTS_PER_BLOCK - 1];
    s[ELEMENTS_PER_BLOCK - 1] = 0.f;
  }
  __syncthreads();

  // Down-Sweep
  for (int stride = ELEMENTS_PER_BLOCK >> 1; stride >= 1; stride >>= 1) {
    int idx = (tid + 1) * (stride << 1) - 1;
    if (idx < ELEMENTS_PER_BLOCK) {
      float left = s[idx - stride];
      float right = s[idx];
      s[idx - stride] = right;
      s[idx] = left + right;
    }
    __syncthreads();
  }

  // Exclusive + 原始值 = Inclusive
  int i0 = base + tid, i1 = i0 + BLOCK_THREADS;
  if (i0 < n) out[i0] = s[tid] + a0;
  if (i1 < n) out[i1] = s[tid + BLOCK_THREADS] + a1;
}

// ── 加块偏移（块间 Exclusive 偏移加到块内 Inclusive 结果上）─────────────────
__global__ void k_add_offsets(float* __restrict__ data, const float* __restrict__ offsets, int n) {
  const int base = blockIdx.x * ELEMENTS_PER_BLOCK;
  const float off = offsets[blockIdx.x];
  int i0 = base + threadIdx.x, i1 = i0 + BLOCK_THREADS;
  if (i0 < n) data[i0] += off;
  if (i1 < n) data[i1] += off;
}

// ── 块和的递归 Exclusive Scan（内部专用）─────────────────────────────────────
static void recursive_exclusive_scan(const float* d_in, float* d_out, int n) {
  const int smem = ELEMENTS_PER_BLOCK * sizeof(float);
  const int nblks = (n + ELEMENTS_PER_BLOCK - 1) / ELEMENTS_PER_BLOCK;

  if (nblks == 1) {
    k_block_scan_exclusive<<<1, BLOCK_THREADS, smem>>>(d_in, d_out, nullptr, n);
    return;
  }

  float *d_bsums, *d_bscanned;
  cudaMalloc(&d_bsums, nblks * sizeof(float));
  cudaMalloc(&d_bscanned, nblks * sizeof(float));

  k_block_scan_exclusive<<<nblks, BLOCK_THREADS, smem>>>(d_in, d_out, d_bsums, n);
  recursive_exclusive_scan(d_bsums, d_bscanned, nblks);
  k_add_offsets<<<nblks, BLOCK_THREADS>>>(d_out, d_bscanned, n);

  cudaFree(d_bsums);
  cudaFree(d_bscanned);
}

// ── 对外 Inclusive Scan ───────────────────────────────────────────────────────
static void recursive_inclusive_scan(const float* d_in, float* d_out, int n) {
  const int smem = ELEMENTS_PER_BLOCK * sizeof(float);
  const int nblks = (n + ELEMENTS_PER_BLOCK - 1) / ELEMENTS_PER_BLOCK;

  if (nblks == 1) {
    k_block_scan_inclusive<<<1, BLOCK_THREADS, smem>>>(d_in, d_out, nullptr, n);
    return;
  }

  float *d_bsums, *d_bscanned;
  cudaMalloc(&d_bsums, nblks * sizeof(float));
  cudaMalloc(&d_bscanned, nblks * sizeof(float));

  // Phase 1：块内 Inclusive Scan，同时收集块总和
  k_block_scan_inclusive<<<nblks, BLOCK_THREADS, smem>>>(d_in, d_out, d_bsums, n);

  // Phase 2：对块总和做 Exclusive Scan，得到块间偏移
  recursive_exclusive_scan(d_bsums, d_bscanned, nblks);

  // Phase 3：将块间 Exclusive 偏移叠加到各块 Inclusive 结果
  k_add_offsets<<<nblks, BLOCK_THREADS>>>(d_out, d_bscanned, n);

  cudaFree(d_bsums);
  cudaFree(d_bscanned);
}

extern "C" void solve(const float* input, float* output, int N) {
  recursive_inclusive_scan(input, output, N);
}