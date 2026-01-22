#include "cute/layout.hpp"
#include "cute/pointer.hpp"
#include "cute/swizzle_layout.hpp"
#include "cute/tensor_impl.hpp"
#include <cstdlib>
#include <cuda_runtime.h>
#include <cute/tensor.hpp>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <iostream>

template <class TensorS, class TensorD, class ThreadLayout>
__global__ void copy_kernel(TensorS S, TensorD D, ThreadLayout) {
  using namespace cute;

  CUTE_STATIC_ASSERT_V(congruent(S) == congruent(D));

  Tensor tile_S = S(make_coord(_, _), blockIdx.x, blockIdx.y);  // (BM, BN)
  Tensor tile_D = D(make_coord(_, _), blockIdx.x, blockIdx.y);  // (BM, BN)

  Tensor thr_tile_S =
      local_partition(tile_S, ThreadLayout{}, threadIdx.x);  // (TM, TN)
  Tensor thr_tile_D =
      local_partition(tile_D, ThreadLayout{}, threadIdx.x);  // (TM, TN)

  Tensor fragment = make_fragment_like(thr_tile_S);

  copy(thr_tile_S, fragment);
  copy(fragment, thr_tile_D);
}

template <class TensorS, class TensorD, class Tiled_Copy>
__global__ void copy_kernel_vectorize(TensorS S, TensorD D,
                                      Tiled_Copy tiled_copy) {
  using namespace cute;

  Tensor tile_S = S(make_coord(_, _), blockIdx.x, blockIdx.y);  // (BM, BN)
  Tensor tile_D = D(make_coord(_, _), blockIdx.x, blockIdx.y);  // (BM, BN)

  auto thr_copy = tiled_copy.get_thread_slice(threadIdx.x);
  auto thr_tile_S = thr_copy.partition_S(tile_S);
  auto thr_tile_D = thr_copy.partition_D(tile_D);

  Tensor fragment = make_fragment_like(thr_tile_S);

  copy(thr_tile_S, fragment);
  copy(fragment, thr_tile_D);
}

int main() {
  srand(42);

  using T = float;

  auto gen_num = []() { return static_cast<T>((rand() % 1000) / 1000.0f); };

  using namespace cute;

  const int m = 5120, n = 5120, k = 4096;
  auto tensor_shape = make_shape(m, n);
  thrust::host_vector<T> hs(size(tensor_shape));
  thrust::host_vector<T> hd(size(tensor_shape));
  thrust::host_vector<T> rd(size(tensor_shape));
  thrust::fill(hd.begin(), hd.end(), 0);
  for (int i = 0; i < m * n; ++i) {
    hs[i] = gen_num();
  }
  thrust::copy(hs.begin(), hs.end(), rd.begin());

  thrust::device_vector<T> ds = hs;
  thrust::device_vector<T> dd = hd;

  Tensor ts = make_tensor(make_gmem_ptr(thrust::raw_pointer_cast(ds.data())),
                          make_layout(tensor_shape));  // (M, N)
  Tensor td = make_tensor(make_gmem_ptr(thrust::raw_pointer_cast(dd.data())),
                          make_layout(tensor_shape));  // (M, N)

  auto block_shape = make_shape(Int<128>{}, Int<64>{});

  if (size<0>(tensor_shape) % size<0>(block_shape) ||
      size<1>(tensor_shape) % size<1>(block_shape)) {
    std::cerr << "Error: tensor shape is not divisible by block shape"
              << std::endl;
    return -1;
  }

  Tensor tile_tensor_S = tiled_divide(ts, block_shape);  // ((M, N), m, n)
  Tensor tile_tensor_D = tiled_divide(td, block_shape);  // ((M, N), m, n)

  Layout thr_layout = make_layout(make_shape(Int<32>{}, Int<8>{}));  // (32, 8)
  Layout val_layout = make_layout(make_shape(Int<4>{}, Int<1>{}));   // (4, 1)

  using CopyOp = UniversalCopy<uint_byte_t<sizeof(T) * size<0>(val_layout)>>;
  using CopyAtom = Copy_Atom<CopyOp, T>;

  auto tiled_copy = make_tiled_copy(CopyAtom{}, thr_layout, val_layout);

  dim3 gridDim(size<1>(tile_tensor_S), size<2>(tile_tensor_D));  // (m, n)
  dim3 blockDim(size(thr_layout));

  copy_kernel_vectorize<<<gridDim, blockDim>>>(tile_tensor_S, tile_tensor_D,
                                               tiled_copy);

  cudaDeviceSynchronize();

  auto result = cudaGetLastError();
  if (result != cudaSuccess) {
    std::cerr << "CUDA Runtime error: " << cudaGetErrorString(result)
              << std::endl;
    return -1;
  }

  hd = dd;

  for (int i = 0; i < 100; i++) {
    if (rd[i] != hd[i]) {
      std::cerr << "index: " << i << " rd: " << rd[i] << " hd: " << hd[i]
                << std::endl;
      return -1;
    }
  }

  std::cout << "Pass !\n";

  return 0;
}