#include <cute/tensor.hpp>
#include <cstdlib>
#include <cuda_runtime.h>
#include <cute/tensor.hpp>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <random>

// 14.41ms
template <class ProblemShape, class CtaTiler,
          class TA, class AStride, class ASmemLayout, class TiledCopyA,
          class TB, class BStride, class BSmemLayout, class TiledCopyB,
          class TC, class CStride, class CSmemLayout, class TiledMma>
__global__ static
__launch_bounds__(decltype(size(TiledMma{}))::value)
void
gemm_kernel(ProblemShape shape_MNK, CtaTiler cta_tiler,
            TA const* A, AStride dA, ASmemLayout sA_layout, TiledCopyA copy_a,
            TB const* B, BStride dB, BSmemLayout sB_layout, TiledCopyB copy_b,
            TC      * C, CStride dC, CSmemLayout          , TiledMma mma){
  using namespace cute;

  auto mA =
      make_tensor(make_gmem_ptr(A), select<0, 2>(shape_MNK), dA);  // (M, K)
  auto mB =
      make_tensor(make_gmem_ptr(B), select<1, 2>(shape_MNK), dB);  // (N, K)
  auto mC =
      make_tensor(make_gmem_ptr(C), select<0, 1>(shape_MNK), dC);  // (M, N)

  auto block_coord = make_coord(blockIdx.x, blockIdx.y, _);
  auto gA =
      local_tile(mA, cta_tiler, block_coord, Step<_1, X, _1>{});  // (BM, BK, k)
  auto gB =
      local_tile(mB, cta_tiler, block_coord, Step<X, _1, _1>{});  // (BN, BK, k)
  auto gC =
      local_tile(mC, cta_tiler, block_coord, Step<_1, _1, X>{});  // (BM, BN)

  __shared__ TA smemA[cosize_v<ASmemLayout>];
  __shared__ TB smemB[cosize_v<BSmemLayout>];
  auto sA = make_tensor(make_smem_ptr(smemA), sA_layout);  // (BM, BK)
  auto sB = make_tensor(make_smem_ptr(smemB), sB_layout);  // (BN, BK)

  const auto tid = threadIdx.x;

  ThrCopy thr_copy_a = copy_a.get_slice(tid);
  auto tAgA = thr_copy_a.partition_S(gA); // (CPY, TM, TK, K)
  auto tAsA = thr_copy_a.partition_D(sA); // (CPY, TM, TK)
  auto tArA = make_fragment_like(tAsA);   // (CPY, TM, TK)

  ThrCopy thr_copy_b = copy_b.get_slice(tid);
  auto tBgB = thr_copy_b.partition_S(gB); // (CPY, TN, TK, K)
  auto tBsB = thr_copy_b.partition_D(sB); // (CPY, TN, TK)
  auto tBrB = make_fragment_like(tBsB);   // (CPY, TN, TK)

  ThrMMA thr_mma = mma.get_slice(tid);
  auto tCsA = thr_mma.partition_A(sA); // (MMA, MM, MK)
  auto tCsB = thr_mma.partition_B(sB); // (MMA, MN, MK)
  auto tCgC = thr_mma.partition_C(gC); // (MMA, MM, MN)
  auto tCrC = make_fragment_like(tCgC); // (MMA, MM, MN)
  clear(tCrC);


  copy(copy_a, tAgA(_, _, _, 0), tArA);
  copy(copy_b, tBgB(_, _, _, 0), tBrB);

  const auto K_MAX_TILE = size<3>(tAgA);
  for (auto k_tile = 0; k_tile < K_MAX_TILE; ++k_tile) {
    __syncthreads();
    copy(tArA, tAsA);
    copy(tBrB, tBsB);
    __syncthreads();

    int k_tile_next = (k_tile + 1 < K_MAX_TILE) ? (k_tile + 1) : k_tile;
    copy(tAgA(_, _, _, k_tile_next), tArA);
    copy(tBgB(_, _, _, k_tile_next), tBrB);

    gemm(mma, tCsA, tCsB, tCrC);
  }

  __syncthreads();
  copy(tCrC, tCgC);

}

int main() {
  srand(42);

  using T = float;
  using TA = float;
  using TB = float;
  using TC = float;

  auto gen_num = []() { return static_cast<T>((rand() % 1000) / 1000.0f); };

  using namespace cute;

  const int m = 5120, n = 1280, k = 4096;
  thrust::host_vector<TA> ha(m * k);
  thrust::host_vector<TB> hb(n * k);
  thrust::host_vector<TC> hc(m * n);
  for (int i = 0; i < m * k; ++i) ha[i] = gen_num();
  for (int i = 0; i < n * k; ++i) hb[i] = gen_num();
  for (int i = 0; i < m * n; ++i) hc[i] = gen_num();

  thrust::device_vector<TA> da = ha;
  thrust::device_vector<TB> db = hb;
  thrust::device_vector<TC> dc = hc;

  TA* pa = thrust::raw_pointer_cast(da.data());
  TB* pb = thrust::raw_pointer_cast(db.data());
  TC* pc = thrust::raw_pointer_cast(dc.data());

  using namespace cute;
  int M = int(m), N = int(n), K = int(k);
  auto problem_shape = make_shape(M, N, K);

  int ldA = int(m);  // A is row-major
  int ldB = int(n);  // B is col-major
  int ldC = int(k);  // C is row-major

  auto dA = make_stride(Int<1>{}, ldA);
  auto dB = make_stride(Int<1>{}, ldB);
  auto dC = make_stride(Int<1>{}, ldC);

  auto bM = Int<128>{};
  auto bN = Int<128>{};
  auto bK = Int<8>{};
  auto cta_shape = make_shape(bM, bN, bK);  // (BM, BN, BK)

  // smem layout
  auto sA = make_layout(make_shape(bM, bK));  // (BM, BK)
  auto sB = make_layout(make_shape(bN, bK));  // (BN, BK)
  auto sC = make_layout(make_shape(bM, bN));  // (BM, BN)

  using copy_op = UniversalCopy<uint128_t>;
  using copy_atom = Copy_Atom<copy_op, T>;

  auto cpy_thr_layout = make_layout(make_shape(Int<32>{}, Int<8>{}));
  auto cpy_val_layout = make_layout(make_shape(Int<4>{}, Int<1>{}));

  auto copyA = make_tiled_copy(copy_atom{},
    cpy_thr_layout, cpy_val_layout);
  auto copyB = make_tiled_copy(copy_atom{},
    cpy_thr_layout, cpy_val_layout);

  // float * float = float
  using mma_op = UniversalFMA<T>;
  using mma_atom = MMA_Atom<mma_op>;
  auto mma_thr_layout = make_layout(make_shape(Int<16>{}, Int<16>{}, Int<1>{}));
  auto mmaC = make_tiled_mma(mma_atom{},
    mma_thr_layout);

  dim3 threadPerBlock(size(mmaC));  // 256 threads
  dim3 blockPerGrid(size(ceil_div(M, bM)), size(ceil_div(N, bN)));

  gemm_kernel<<<blockPerGrid, threadPerBlock>>>(
      problem_shape, cta_shape,
      pa, dA, sA, copyA,
      pb, dB, sB, copyB,
      pc, dC, sC, mmaC);

  cudaDeviceSynchronize();
  auto cudaResult = cudaGetLastError();
  if (cudaResult != cudaSuccess) {
    std::cerr << "Error: " << cudaResult << std::endl;
    return -1;
  }
  return 0;
}