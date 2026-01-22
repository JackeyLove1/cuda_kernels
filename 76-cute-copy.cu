#include <cute/tensor.hpp>
#include <cstdlib>
#include <cuda_runtime.h>
#include <cute/tensor.hpp>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <iostream>

template <class TensorS, class TensorD, class BlockShape, class Tiled_Copy>
__global__ __launch_bounds__(256) void copy_kernel_vectorize(
    TensorS S, TensorD D, BlockShape block_shape, Tiled_Copy tiled_copy) {
  using namespace cute;


  auto coord = make_coord(blockIdx.x, blockIdx.y);
  auto tile_S = local_tile(S, block_shape, coord);
  auto tile_D = local_tile(D, block_shape, coord);

  ThrCopy thr_copy = tiled_copy.get_thread_slice(threadIdx.x);
  auto thr_S = thr_copy.partition_S(tile_S);
  auto thr_D = thr_copy.partition_D(tile_D);

  // Avoid register-owned "fragments" here: for local_tile() the per-thread
  // layout can be dynamic, which CUTE doesn't support for owning tensors.
  // Instead, copy directly using the CopyAtom inside tiled_copy on the
  // already-partitioned per-thread tensors.
  copy(tiled_copy, thr_S, thr_D);
}

int main() {
  srand(42);

  using T = float;

  auto gen_num = []() { return static_cast<T>((rand() % 1000) / 1000.0f); };

  using namespace cute;

  const int m = 512, n = 128;
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

  Layout thr_layout = make_layout(make_shape(Int<32>{}, Int<8>{}));  // (32, 8)
  Layout val_layout = make_layout(make_shape(Int<4>{}, Int<1>{}));   // (4, 1)

  using CopyOp = UniversalCopy<uint_byte_t<sizeof(T) * size<0>(val_layout)>>;
  using CopyAtom = Copy_Atom<CopyOp, T>;

  auto tiled_copy = make_tiled_copy(CopyAtom{}, thr_layout, val_layout);

  dim3 gridDim(size<0>(tensor_shape) / size<0>(block_shape),
               size<1>(tensor_shape) / size<1>(block_shape));
  dim3 blockDim(size(thr_layout));

  copy_kernel_vectorize<<<gridDim, blockDim>>>(ts, td, block_shape, tiled_copy);

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