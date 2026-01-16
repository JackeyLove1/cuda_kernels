#include <cstdlib>
#include <cuda_runtime.h>
#include <cute/tensor.hpp>
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include "cutlass/util/GPU_Clock.hpp"


template <class ProblemShape, class CtaTiler,
          class TA, class AStride, class ASmemLayout, class TiledCopyA,
          class TB, class BStride, class BSmemLayout, class TiledCopyB,
          class TC, class CStride, class CSmemLayout, class TiledMma,
          class Alpha, class Beta>
__global__ static
__launch_bounds__(decltype(size(TiledMma{}))::value)
void
gemm_device(ProblemShape shape_MNK, CtaTiler cta_tiler,
            TA const* A, AStride dA, ASmemLayout sA_layout, TiledCopyA copy_a,
            TB const* B, BStride dB, BSmemLayout sB_layout, TiledCopyB copy_b,
            TC      * C, CStride dC, CSmemLayout          , TiledMma mma,
            Alpha alpha, Beta beta)
{

    using namespace cute;
    CUTE_STATIC_ASSERT_V(rank(shape_MNK) == Int<3>{});
    CUTE_STATIC_ASSERT_V(rank(cta_tiler) == Int<3>{});

    CUTE_STATIC_ASSERT_V(size(copy_a) == size(mma));
    CUTE_STATIC_ASSERT_V(size(copy_b) == size(mma));

    CUTE_STATIC_ASSERT(is_static_v<ASmemLayout>);
    CUTE_STATIC_ASSERT(is_static_v<BSmemLayout>);
    CUTE_STATIC_ASSERT(is_static_v<CSmemLayout>);

    CUTE_STATIC_ASSERT_V(size<0>(ASmemLayout{}) == size<0>(cta_tiler));  // BLK_M
    CUTE_STATIC_ASSERT_V(size<0>(CSmemLayout{}) == size<0>(cta_tiler));  // BLK_M
    CUTE_STATIC_ASSERT_V(size<0>(BSmemLayout{}) == size<1>(cta_tiler));  // BLK_N
    CUTE_STATIC_ASSERT_V(size<1>(CSmemLayout{}) == size<1>(cta_tiler));  // BLK_N
    CUTE_STATIC_ASSERT_V(size<1>(ASmemLayout{}) == size<2>(cta_tiler));  // BLK_K
    CUTE_STATIC_ASSERT_V(size<1>(BSmemLayout{}) == size<2>(cta_tiler));  // BLK_K

    CUTE_STATIC_ASSERT_V(congruent(select<0,2>(shape_MNK), dA));         // dA strides for shape MK
    CUTE_STATIC_ASSERT_V(congruent(select<1,2>(shape_MNK), dB));         // dB strides for shape NK
    CUTE_STATIC_ASSERT_V(congruent(select<0,1>(shape_MNK), dC));         // dC strides for shape MN

    Tensor mA = make_tensor(make_gmem_ptr(A), select<0,2>(shape_MNK), dA);
    Tensor mB = make_tensor(make_gmem_ptr(B), select<1,2>(shape_MNK), dB);
    Tensor mC = make_tensor(make_gmem_ptr(C), select<0,1>(shape_MNK), dC);

    auto cta_coord = make_coord(blockIdx.x, blockIdx.y, _);
    Tensor gA = local_tile(mA, cta_tiler, cta_coord, Step<_1, X, _1>{}); // (BM, BK, k)
    Tensor gB = local_tile(mB, cta_tiler, cta_coord, Step<X, _1, _1>{}); // (BN, BK, k)
    Tensor gC = local_tile(mC, cta_tiler, cta_coord, Step<_1, _1, X>{}); // (BM, BN)

    __shared__ TA smemA[cosize_v<ASmemLayout>];
    __shared__ TB smemB[cosize_v<BSmemLayout>];
    Tensor sA = make_tensor(make_smem_ptr(smemA), sA_layout);
    Tensor sB = make_tensor(make_smem_ptr(smemB), sB_layout);

    ThrCopy thr_copy_a = copy_a.get_slice(threadIdx.x);
    Tensor tAgA = thr_copy_a.partition_S(gA); // (CPY, CM, CK, k)
    Tensor tAsA = thr_copy_a.partition_D(sA); // (CPY, CM, CK)
    Tensor tArA = make_fragment_like(tAsA);

    ThrCopy thr_copy_b = copy_b.get_slice(threadIdx.x);
    Tensor tBgB = thr_copy_b.partition_S(gB); // (CPY, CN, CK, k)
    Tensor tBsB = thr_copy_b.partition_D(sB); // (CPY, CN, CK)
    Tensor tBrB = make_fragment_like(tBsB); // (CPY, CN, CK)

    CUTE_STATIC_ASSERT_V(size<1>(tAgA) == size<1>(tAsA)); // CM
    CUTE_STATIC_ASSERT_V(size<1>(tAgA) == size<1>(tArA)); // CM
    CUTE_STATIC_ASSERT_V(size<2>(tAgA) == size<2>(tAsA));                // CPY_K
    CUTE_STATIC_ASSERT_V(size<2>(tAgA) == size<2>(tArA));                // CPY_K
    CUTE_STATIC_ASSERT_V(size<1>(tBgB) == size<1>(tBsB));                // CPY_N
    CUTE_STATIC_ASSERT_V(size<1>(tBgB) == size<1>(tBrB));                // CPY_N
    CUTE_STATIC_ASSERT_V(size<2>(tBgB) == size<2>(tBsB));                // CPY_K
    CUTE_STATIC_ASSERT_V(size<2>(tBgB) == size<2>(tBrB));                // CPY_K

    // copy from k_tile_0 from gmem 2 rmem
    copy(copy_a, tAgA(_, _, _, 0), tArA);
    copy(copy_b, tBgB(_, _, _, 0), tBrB);

    ThrMMA thr_mma = mma.get_slice(threadIdx.x);
    Tensor tCsA = thr_mma.partition_A(sA); // (MMA, MMA_M, MMA_K)
    Tensor tCsB = thr_mma.partition_B(sB); // (MMA, MMA_N, MMA_K)
    Tensor tCgC = thr_mma.partition_C(gC); // (MMA, MMA_M, MMA_N)

    Tensor tCrA = thr_mma.make_fragment_A(tCsA);
    Tensor tCrB = thr_mma.make_fragment_B(tCsB);
    Tensor tCrC = thr_mma.make_fragment_C(tCgC);

    CUTE_STATIC_ASSERT_V(  shape(tCrC) ==   shape(tCgC));                // (MMA,MMA_M,MMA_K)
    CUTE_STATIC_ASSERT_V(  shape(tCrB) ==   shape(tCsB));                // (MMA,MMA_N,MMA_K)
    CUTE_STATIC_ASSERT_V(  shape(tCrC) ==   shape(tCgC));                // (MMA,MMA_M,MMA_N)
    CUTE_STATIC_ASSERT_V(size<1>(tCgC) == size<1>(tCsA));                // MMA_M
    CUTE_STATIC_ASSERT_V(size<2>(tCgC) == size<1>(tCsB));                // MMA_N
    CUTE_STATIC_ASSERT_V(size<2>(tCsA) == size<2>(tCsB));                // MMA_K

    clear(tCrC);

#if 0
    if(block0() && thread0()) {
        print("  mA : "); print(  mA); print("\n");
        print("  gA : "); print(  gA); print("\n");
        print("  sA : "); print(  sA); print("\n");
        print("tAgA : "); print(tAgA); print("\n");
        print("tAsA : "); print(tAsA); print("\n");
        print("tArA : "); print(tArA); print("\n");
        print("smemA size: ");print(cosize_v<ASmemLayout>);print("\n");
    }
#endif

#if 0
    if(block0() && thread0()) {
        print("  mB : "); print(  mB); print("\n");
        print("  gB : "); print(  gB); print("\n");
        print("  sB : "); print(  sB); print("\n");
        print("tBgB : "); print(tBgB); print("\n");
        print("tBsB : "); print(tBsB); print("\n");
        print("tBrB : "); print(tBrB); print("\n");
    }
#endif

#if 0
    if(block0() && thread0()) {
        print("  mC : "); print(  mC); print("\n");
        print("  gC : "); print(  gC); print("\n");
        print("tCsA : "); print(tCsA); print("\n");
        print("tCsB : "); print(tCsB); print("\n");
        print("tCgC : "); print(tCgC); print("\n");
        print("tCrC : "); print(tCrC); print("\n");
    }
#endif

    // copy tile0 from rmem 2 smem
    copy(tArA, tAsA);
    copy(tBrB, tBsB);
    __syncthreads();

    // load A B smem 2 rmem for k_block = 0
    copy(tCsA(_, _, 0), tCrA(_, _, 0));
    copy(tCsB(_, _, 0), tCrB(_, _, 0));

    auto K_TILE_MAX = size<3>(tAgA);
    auto K_BLOCK_MAX = size<2>(tCrA);

    CUTE_NO_UNROLL
    for (int k_tile = 0; k_tile < K_TILE_MAX; ++k_tile) {

        // pipeline for k-mode of the block registers
        CUTE_UNROLL
        for (int k_block = 0; k_block < K_BLOCK_MAX; ++k_block) {

            if (k_block == K_BLOCK_MAX - 1) {
                __syncthreads();
                copy(tArA, tAsA);
                copy(tBrB, tBsB);
                __syncthreads();
            }

            int k_block_next = (k_block + 1) % K_BLOCK_MAX;
            copy(tCsA(_, _, k_block_next), tCrA(_, _, k_block_next));
            copy(tCsB(_, _, k_block_next), tCrB(_, _, k_block_next));

            if (k_block == 0) {
                int k_tile_next = (k_tile + 1 < K_TILE_MAX) ? k_tile + 1 : k_tile;
                copy(copy_a, tAgA(_, _, _, k_tile_next), tArA);
                copy(copy_b, tBgB(_, _, _, k_tile_next), tBrB);
            }
            gemm(mma, tCrA(_, _, k_block), tCrB(_,_,k_block), tCrC);
        }
    }

    axpby(alpha, tCrC, beta, tCgC);

}

template <class TA, class TB, class TC,
          class Alpha, class Beta>
void
gemm(int m, int n, int k,
     TA const* A, int ldA,
     TB const* B, int ldB,
     TC      * C, int ldC,
     Alpha alpha, Beta beta)
{
    using namespace cute;
    auto M = int(m);
    auto N = int(n);
    auto K = int(k);
    auto prob_shape = make_shape(M, N, K);

    auto dA = make_stride(Int<1>{}, ldA);
    auto dB = make_stride(Int<1>{}, ldB);
    auto dC = make_stride(Int<1>{}, ldC);

    auto bM = Int<128>{};
    auto bN = Int<128>{};
    auto bK = Int< 8>{};
    auto cta_tiler = make_shape(bM, bN, bK);

    auto sA = make_layout(make_shape(bM, bK),
        make_stride(Int<1>{}, bM));
    auto sB = make_layout(make_shape(bN, bK),
        make_stride(Int<1>{}, bN));
    auto sC= make_layout(make_shape(bM, bN));

    TiledCopy copyA = make_tiled_copy(Copy_Atom<UniversalCopy<uint128_t>, TA>{},
        Layout<Shape<_32, _8>>{},
        Layout<Shape<_4, _1>>{});
    TiledCopy copyB = make_tiled_copy(Copy_Atom<UniversalCopy<uint128_t>, TB>{},
        Layout<Shape<_32, _8>>{},
        Layout<Shape<_4, _1>>{});

    TiledMMA mmaC = make_tiled_mma(UniversalFMA<TA, TB, TC>{},
        Layout<Shape<_16, _16, _1>>{});
    // TiledMMA mmaC = make_tiled_mma(MMA_Atom<SM80_16x8x16_F32F16F16F32_TN>{},
    //    Layout<Shape<_16, _16, _1>>{});

    dim3 dimBlock(size(mmaC));
    dim3 dimGrid(size(ceil_div(M, bM)), size(ceil_div(N, bN)));
    gemm_device<<<dimGrid, dimBlock>>>
    (prob_shape, cta_tiler,
     A, dA, sA, copyA,
     B, dB, sB, copyB,
     C, dC, sC, mmaC,
     alpha, beta);

}


int main()
{
    using namespace cute;
    using TA = float;
    using TB = float;
    using TC = float;
    const int m = 5120, n = 5120, k = 4096;

    thrust::host_vector<TA> h_A(m*k);
    thrust::host_vector<TB> h_B(k*n);
    thrust::host_vector<TC> h_C(m*n);

    for (int j = 0; j < m*k; ++j) h_A[j] = static_cast<TA>( 2*(rand() / double(RAND_MAX)) - 1 );
    for (int j = 0; j < n*k; ++j) h_B[j] = static_cast<TB>( 2*(rand() / double(RAND_MAX)) - 1 );
    for (int j = 0; j < m*n; ++j) h_C[j] = static_cast<TC>(-1);

    thrust::device_vector<TA> d_A = h_A;
    thrust::device_vector<TB> d_B = h_B;
    thrust::device_vector<TC> d_C = h_C;


    double gflops = (2.0*m*n*k) * 1e-9;

    const int timing_iterations = 10;
    GPU_Clock timer;

    const int ldA = m, ldB = n, ldC = m;

    // Run once warm up
    d_C = h_C;
    const float alpha = 1.0f, beta = 0.0f;
    // gemm(m, n, k,
    //          d_A.data().get(), ldA,
    //          d_B.data().get(), ldB,
    //          d_C.data().get(), ldC,
    //          alpha, beta);
    // CUTE_CHECK_LAST();
    thrust::host_vector<TC> cute_result = d_C;

    // Timing iterations
    timer.start();
    for (int i = 0; i < timing_iterations; ++i) {
        gemm(m, n, k,
                 d_A.data().get(), ldA,
                 d_B.data().get(), ldB,
                 d_C.data().get(), ldC,
                 alpha, beta);
    }
    double cute_time = timer.seconds() / timing_iterations;
    CUTE_CHECK_LAST();
    printf("CUTE_GEMM:     [%6.1f]GFlop/s  (%6.4f)ms\n", gflops / cute_time, cute_time*1000);

    return 0;
}

/*
CUTE_GEMM:     [1538.6]GFlop/s  (139.5782)ms

# fix k_block == 0
CUTE_GEMM:     [1311.4]GFlop/s  (163.7584)ms

# modify bk from 8 to 32
 */