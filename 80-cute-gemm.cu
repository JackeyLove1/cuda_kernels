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

// 14.1 ms
template <class ProblemShape, class CtaTiler, class TA, class AStride,
          class ASmemLayout, class AThreadLayout, class TB, class BStride,
          class BSmemLayout, class BThreadLayout, class TC, class CStride,
          class CSmemLayout, class CThreadLayout>
__global__ static __launch_bounds__(decltype(size(
    CThreadLayout{}))::value) void gemm_kernel(ProblemShape shape_MNK,
                                               CtaTiler cta_tiler, TA const* A,
                                               AStride dA,
                                               ASmemLayout sA_layout,
                                               AThreadLayout tA_layout,
                                               TB const* B, BStride dB,
                                               BSmemLayout sB_layout,
                                               BThreadLayout tB_layout, TC* C,
                                               CStride dC,
                                               CSmemLayout sC_layout,
                                               CThreadLayout tC_layout) {
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

  auto tAgA = local_partition(gA, tA_layout, tid);  // (TM, TK, k)
  auto tAsA = local_partition(sA, tA_layout, tid);  // (TM, TK)

  auto tBgB = local_partition(gB, tB_layout, tid);  // (TN, TK, k)
  auto tBsB = local_partition(sB, tB_layout, tid);  // (TN, TK)

  auto tCsA = local_partition(sA, tC_layout, tid, Step<_1, X>{});  // (TM, BK)
  auto tCsB = local_partition(sB, tC_layout, tid, Step<X, _1>{});  // (TN, BK)
  auto tCgC = local_partition(gC, tC_layout, tid);                 // (TM, TN)
  auto tCrC = make_fragment_like(tCgC);
  clear(tCrC);

  const auto K_TILE = size<2>(tAgA);
  for (int k = 0; k < K_TILE; ++k) {
    copy(tAgA(_, _, k), tAsA);
    copy(tBgB(_, _, k), tBsB);

    cp_async_fence();
    cp_async_wait<0>();
    __syncthreads();

    gemm(tCsA, tCsB, tCrC);
    __syncthreads();
  }
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

  // thread layout
  auto tA = make_layout(make_shape(Int<32>{}, Int<8>{}));
  auto tB = make_layout(make_shape(Int<32>{}, Int<8>{}));
  auto tC = make_layout(make_shape(Int<16>{}, Int<16>{}));

  dim3 threadPerBlock(size(tC));  // 256 threads
  dim3 blockPerGrid(size(ceil_div(M, bM)), size(ceil_div(N, bN)));

  gemm_kernel<<<blockPerGrid, threadPerBlock>>>(
      problem_shape, cta_shape, pa, dA, sA, tA, pb, dB, sB, tB, pc, dC, sC, tC);

  cudaDeviceSynchronize();
  auto cudaResult = cudaGetLastError();
  if (cudaResult != cudaSuccess) {
    std::cerr << "Error: " << cudaResult << std::endl;
    return -1;
  }
  return 0;
}