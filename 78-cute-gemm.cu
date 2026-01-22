#include "cutlass/util/helper_cuda.hpp"

#include <cstdlib>
#include <cstdio>
#include <cassert>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include <cute/tensor.hpp>

template <typename T, unsigned KTileM, unsigned KTileN>
__global__ void kernel(const T* A) {
  using namespace cute;
  const auto tid = threadIdx.x;
  __shared__ T smem[KTileM * KTileN];
  auto gA = make_tensor(make_gmem_ptr(A),
    make_shape(Int<KTileM>{}, Int<KTileN>{}),
    make_stride(Int<1>{}, Int<KTileM>{})); // n-major (M, N)
  auto sA = make_tensor(make_smem_ptr(static_cast<T*>(smem)),
    make_shape(Int<KTileM>{}, Int<KTileN>{}),
    make_stride(Int<1>{}, Int<KTileM>{})); // n-major (M, N)

  using CopyOp = UniversalCopy<uint128_t>;
  using CopyTrait = Copy_Traits<CopyOp>;
  using CopyAtom = Copy_Atom<CopyTrait, T>;

  auto thr_layout = make_layout(make_shape(Int<32>{}, Int<1>{}));
  auto val_layout = make_layout(make_shape(Int<1>{}, Int<8>{}));
  auto tiled_g2s = make_tiled_copy(CopyAtom{}, thr_layout, val_layout);

  auto thr_g2s = tiled_g2s.get_thread_slice(tid);
  auto thr_s = thr_g2s.partition_S(gA);
  auto thr_d = thr_g2s.partition_D(sA);

  print("gA: ");print(gA);print("\n");
  print("sA: ");print(sA);print("\n");
  print("thr_s: ");print(thr_s);print("\n");
  print("thr_d: ");print(thr_d);print("\n");

  copy(thr_s, thr_d);

  __syncthreads();
}

int main() {
  const int m = 512, n = 512, k = 32;
  using T = float;
  using TA = T;
  using TB = T;
  using TC = T;

  using namespace cute;
  cute::device_init(0);

  thrust::host_vector<TA> h_A(m * k);
  thrust::host_vector<TB> h_B(n * k);
  thrust::host_vector<TC> h_C(m * n);

  for (int j = 0; j < m * k; ++j)
    h_A[j] = static_cast<TA>(2 * (rand() / double(RAND_MAX)) - 1);
  for (int j = 0; j < n * k; ++j)
    h_B[j] = static_cast<TB>(2 * (rand() / double(RAND_MAX)) - 1);
  for (int j = 0; j < m * n; ++j) h_C[j] = static_cast<TC>(-1);

  thrust::device_vector<TA> d_A = h_A;
  thrust::device_vector<TB> d_B = h_B;
  thrust::device_vector<TC> d_C = h_C;

  kernel<float, 128, 32><<<1, 1>>>(thrust::raw_pointer_cast(d_A.data()));
  cudaDeviceSynchronize();
  auto result = cudaGetLastError();
  if (result != cudaSuccess) {
    std::cerr << "Error: " << result << std::endl;
    return -1;
  }
  return 0;
}