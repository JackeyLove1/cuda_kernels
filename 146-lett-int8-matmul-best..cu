#include <cuda_runtime.h>

#define cdiv(_a, _b) (((_a) + (_b) - 1) / (_b))

static __device__ __forceinline__ void cp_async_16(unsigned int dst, const void* src) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"(dst), "l"(src));
}

static __device__ __forceinline__ void cp_async_wait() { asm volatile("cp.async.wait_all;"); }

static __device__ __forceinline__ int warp_reduce_sum(int x) {
#pragma unroll
  for (int i = 16; i >= 1; i /= 2) x += __shfl_xor_sync(0xffffffff, x, i);
  return x;
}

static __device__ __forceinline__ int8_t rescale(int x, float sa, float sb, float sc, int zc) {
  float fx = x * sa * sb / sc;
  int ix = (int)roundf(fx);
  ix += zc;
  ix = min(ix, 127);
  ix = max(ix, -128);
  return (int8_t)ix;
}

__global__ void matmulq8_simple(const int8_t* A, const int8_t* B, int8_t* C, int M, int N, int K,
                                float scale_A, float scale_B, float scale_C, int zero_point_A,
                                int zero_point_B, int zero_point_C) {
  const int mi = blockIdx.x;
  for (int ni = threadIdx.x; ni < N; ni += blockDim.x) {
    int s = 0;
    for (int ki = 0; ki < K; ki++) {
      int a = (int)A[mi * K + ki];
      int b = (int)B[ki * N + ni];
      s += (a - zero_point_A) * (b - zero_point_B);
    }
    int8_t c = rescale(s, scale_A, scale_B, scale_C, zero_point_C);
    C[mi * N + ni] = c;
  }
}

template <int M, int N, int BSIZE>
static __device__ __forceinline__ void ldmat_sync(int8_t* dst, const int8_t* src, int stride,
                                                  int size_m, int size_n, int8_t dft) {
#pragma unroll
  for (int i = 0; i < M * N; i += BSIZE * 16) {
    const int index = i + threadIdx.x * 16;
    const int ni = index % N;
    const int mi = index / N;
    int4* d = (int4*)(dst + index);
    const int4* s = (const int4*)(src + mi * stride + ni);
    if (ni < size_n && mi < size_m)
      *d = *s;
    else
      *d = make_int4(dft, dft, dft, dft);
  }
}

template <int STRIDE>
static __device__ __forceinline__ void ldmat4x4reg(char4* dst, const int8_t* src) {
#pragma unroll
  for (int i = 0; i < 4; i++) dst[i] = *(const char4*)(src + STRIDE * i);
}

template <int STRIDE>
static __device__ __forceinline__ void stmat4x4reg(int8_t* dst, const char4* src) {
#pragma unroll
  for (int i = 0; i < 4; i++) *(char4*)(dst + STRIDE * i) = src[i];
}

static __device__ __forceinline__ void transmat4x4reg(char4* dst, const char4* src) {
  dst[0] = make_char4(src[0].x, src[1].x, src[2].x, src[3].x);
  dst[1] = make_char4(src[0].y, src[1].y, src[2].y, src[3].y);
  dst[2] = make_char4(src[0].z, src[1].z, src[2].z, src[3].z);
  dst[3] = make_char4(src[0].w, src[1].w, src[2].w, src[3].w);
}

template <int TILE, int BSIZE>
__global__ void matmulq8_aligned(const int8_t* A, const int8_t* B, int8_t* C, int M, int N, int K,
                                 float scale_A, float scale_B, float scale_C, int zero_point_A,
                                 int zero_point_B, int zero_point_C) {
  __shared__ int8_t cached_a[TILE][TILE];
  __shared__ int8_t cached_b[TILE][TILE];
  __shared__ int suma[TILE];
  __shared__ int sumb[TILE];
  char4 reg_a[2][4];
  char4 reg_b[2][4], reg_tb[4];
  int reg_c[4 * 4] = {0};

  const int start_n = blockIdx.x * TILE;
  const int start_m = blockIdx.y * TILE;
  const int size_n = min(TILE, N - start_n);
  const int size_m = min(TILE, M - start_m);
  const int ssn = (threadIdx.x % (TILE / 4)) * 4;
  const int ssm = (threadIdx.x / (TILE / 4)) * 4;
  const int ssk = ssm;
  const int warp_id = threadIdx.x / 32;
  const int id_in_warp = threadIdx.x % 32;
  const int warps = blockDim.x / 32;
  const char4 one4 = make_char4(1, 1, 1, 1);

  for (int i = threadIdx.x; i < TILE; i += blockDim.x) suma[i] = sumb[i] = 0;

  for (int start_k = 0; start_k < K; start_k += TILE) {
    const int size_k = min(TILE, K - start_k);
    ldmat_sync<TILE, TILE, BSIZE>(&cached_a[0][0], A + start_m * K + start_k, K, size_m, size_k,
                                  (int8_t)zero_point_A);
    ldmat_sync<TILE, TILE, BSIZE>(&cached_b[0][0], B + start_k * N + start_n, N, size_k, size_n,
                                  (int8_t)zero_point_B);
    __syncthreads();

    // transpose B
    ldmat4x4reg<TILE>(reg_tb, &cached_b[ssk][ssn]);
    transmat4x4reg(reg_b[0], reg_tb);
    stmat4x4reg<TILE>(&cached_b[ssk][ssn], reg_b[0]);
    __syncthreads();

    // calc a sum
    for (int mi = warp_id; mi < TILE; mi += warps) {
      int sum = 0;
      for (int ki = id_in_warp * 4; ki < TILE; ki += 32 * 4) {
        char4* src = (char4*)(&cached_a[mi][ki]);
        sum = __dp4a(*src, one4, sum);
      }
      sum = warp_reduce_sum(sum);
      if (id_in_warp == 0) suma[mi] += sum;
    }
    // calc b sum
    if (threadIdx.x < TILE) {
      const int offk = threadIdx.x / (TILE / 4);
      const int offn = (threadIdx.x % (TILE / 4)) * 4;
      const int real_ni = offn + offk;
      int sum = 0;
      for (int ki = 0; ki < TILE; ki += 4) {
        char4* src = (char4*)(&cached_b[ki + offk][offn]);
        sum = __dp4a(*src, one4, sum);
      }
      sumb[real_ni] += sum;
    }

    // do int8 mma
    int f = 0;
    ldmat4x4reg<TILE>(reg_a[0], &cached_a[ssm][0]);  // broadcast
    ldmat4x4reg<TILE>(reg_b[0], &cached_b[0][ssn]);
#pragma unroll
    for (int small_start_k = 0; small_start_k < TILE; small_start_k += 4) {
      if (small_start_k + 4 < TILE) {
        ldmat4x4reg<TILE>(reg_a[f ^ 1], &cached_a[ssm][small_start_k + 4]);  // broadcast
        ldmat4x4reg<TILE>(reg_b[f ^ 1], &cached_b[small_start_k + 4][ssn]);
      }
#pragma unroll
      for (int mi = 0; mi < 4; mi++) {
#pragma unroll
        for (int ni = 0; ni < 4; ni++) {
          reg_c[mi * 4 + ni] = __dp4a(reg_a[f][mi], reg_b[f][ni], reg_c[mi * 4 + ni]);
        }
      }
      f ^= 1;
    }
    __syncthreads();
  }

#pragma unroll
  for (int mi = 0; mi < 4; mi++) {
    char4* pc = (char4*)(C + (start_m + ssm + mi) * N + start_n + ssn);
    int8_t rc[4];
#pragma unroll
    for (int ni = 0; ni < 4; ni++) {
      int x = reg_c[mi * 4 + ni] - zero_point_A * sumb[ssn + ni] - zero_point_B * suma[ssm + mi] +
              TILE * zero_point_A * zero_point_B;
      rc[ni] = rescale(x, scale_A, scale_B, scale_C, zero_point_C);
    }
    if (start_m + ssm + mi < M && start_n + ssn < N) *pc = make_char4(rc[0], rc[1], rc[2], rc[3]);
  }
}

// A, B, C are device pointers
extern "C" void solve(const int8_t* A, const int8_t* B, int8_t* C, int M, int N, int K,
                      float scale_A, float scale_B, float scale_C, int zero_point_A,
                      int zero_point_B, int zero_point_C) {
  if (M % 16 == 0 && N % 16 == 0 && K % 16 == 0) {
    constexpr int TILE = 128;
    constexpr int THREADS = (TILE / 4) * (TILE / 4);
    dim3 blocks(cdiv(N, TILE), cdiv(M, TILE));
    matmulq8_aligned<TILE, THREADS><<<blocks, THREADS>>>(
        A, B, C, M, N, K, scale_A, scale_B, scale_C, zero_point_A, zero_point_B, zero_point_C);
  } else {
    int threads = 256;
    int blocks = M;
    matmulq8_simple<<<blocks, threads>>>(A, B, C, M, N, K, scale_A, scale_B, scale_C, zero_point_A,
                                         zero_point_B, zero_point_C);
  }
  cudaDeviceSynchronize();
}