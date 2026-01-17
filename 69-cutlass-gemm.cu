#include <iostream>
#include <cstdio>
#include <cuda_runtime.h>
#include <cuda.h>
#include <cute/tensor.hpp>
#include <thrust/device_vector.h>
#include <thrust/device_ptr.h>
#include <thrust/host_vector.h>

template<typename T, int kTileM, int kTileN, int kTileK, typename TiledMMA>
__global__ void gemm(T* Cptr, const T* Aptr, const T* Bptr, const int m, const int n, const int k)
{
    using namespace cute;

    Tensor A = make_tensor(make_gmem_ptr(Aptr), make_shape(m, k), make_stride(k, Int<1>{})); // (m, k) m-major
    Tensor B = make_tensor(make_gmem_ptr(Bptr), make_shape(n, k), make_stride(k, Int<1>{})); // (n, k) n-major
    Tensor C = make_tensor(make_gmem_ptr(Cptr), make_shape(m, n), make_stride(n, Int<1>{})); // (m, n) m-major

    const int bx = blockIdx.x;
    const int by = blockIdx.y;

    Tensor gA = local_tile(A, make_tile(Int<kTileM>{}, Int<kTileK>{}), make_coord(by, _)); // (BM, BK, num_tile_k)
    Tensor gB = local_tile(B, make_tile(Int<kTileN>{}, Int<kTileK>{}), make_coord(bx, _)); // (BN, BK, num_tile_k)
    Tensor gC = local_tile(C, make_tile(Int<kTileM>{}, Int<kTileN>{}), make_coord(by, bx)); // (BM, BN);

    TiledMMA tiled_mma;
    ThrMMA thr_mma = tiled_mma.get_slice(threadIdx.x);
    auto tAgA = thr_mma.partition_A(gA); // (MMA, MMA_M, MMA_K, num_tile_k)
    auto tBgB = thr_mma.partition_B(gB); // (MMA, MMA_N, MMA_K, num_tile_k)
    auto tCgC = thr_mma.partition_C(gC); // (MMA, MMA_M, MMA_N)

    auto tArA = thr_mma.partition_fragment_A(gA(_, _, 0)); // (MMA, MMA_M, MMA_K)
    auto tBrB = thr_mma.partition_fragment_B(gB(_, _, 0)); // (MMA, MMA_N, MMA_K)
    auto tCrC = thr_mma.partition_fragment_C(gC(_, _));          // (MMA, MMA_M, MMA_N)

    clear(tCrC);

    int num_tile_k = size<2>(gA);
    CUTE_UNROLL
    for (int k_tile = 0; k_tile < num_tile_k; ++k_tile) {
        cute::copy(tAgA(_, _,_, k_tile), tArA);
        cute::copy(tBgB(_, _,_, k_tile), tBrB);

        cute::gemm(tiled_mma, tCrC, tArA, tBrB, tCrC);
    }

    cute::copy(tCrC, tCgC);
}

int main()
{
    const int m = 5120, n = 5120, k = 4096;
    using T = half;

    thrust::host_vector<T> ha(m * k);
    thrust::host_vector<T> hb(n * k);
    thrust::host_vector<T> hc(m * n);

    thrust::device_vector<T> da = ha;
    thrust::device_vector<T> db = hb;
    thrust::device_vector<T> dc = hc;

    using namespace cute;
    using mma_op = SM80_16x8x8_F16F16F16F16_TN;
    using mma_traits = MMA_Traits<mma_op>;
    using mma_atom = MMA_Atom<mma_traits>;

    using MMA = decltype(make_tiled_mma(mma_atom{},
                        make_layout(Shape<_2, _2, _1>{}),
                        make_layout(Shape<_1, _1, _1>{})));

    print_latex(MMA{});

    constexpr int kTileM = 128;
    constexpr int kTileN = 128;
    constexpr int kTileK = 32;

    dim3 block(size(MMA{}));
    dim3 grid(ceil_div(n, kTileN), ceil_div(m, kTileM));
    for (int i = 0; i < 100; ++i) {
        // gemm<T, kTileM, kTileN, kTileK, MMA><<<grid, block>>>(
        //     thrust::raw_pointer_cast(dc.data()),
        //     thrust::raw_pointer_cast(da.data()),
        //     thrust::raw_pointer_cast(db.data()),
        //     m, n, k);
    }


}