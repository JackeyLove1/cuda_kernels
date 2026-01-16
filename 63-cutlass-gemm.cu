#include <cstdlib>
#include <cuda_runtime.h>
#include <cute/tensor.hpp>
#include <thrust/device_vector.h>
#include <thrust/device_ptr.h>
#include <thrust/host_vector.h>

#include "cutlass/util/GPU_Clock.hpp"
#include "cutlass/util/helper_cuda.hpp"

template <class ProblemShape, class CtaTiler,
          class TA, class AStride, class ASmemLayout, class AThreadLayout,
          class TB, class BStride, class BSmemLayout, class BThreadLayout,
          class TC, class CStride, class CSmemLayout, class CThreadLayout,
          class Alpha, class Beta>
__global__ static
__launch_bounds__(decltype(size(CThreadLayout{}))::value)
void
gemm_device(ProblemShape shape_MNK, CtaTiler cta_tiler,
            TA const* A, AStride dA, ASmemLayout sA_layout, AThreadLayout tA,
            TB const* B, BStride dB, BSmemLayout sB_layout, BThreadLayout tB,
            TC      * C, CStride dC, CSmemLayout          , CThreadLayout tC,
            Alpha alpha, Beta beta)
{
    using namespace cute;

    CUTE_STATIC_ASSERT_V(rank(shape_MNK) == Int<3>{}); // (M, N, K)
    CUTE_STATIC_ASSERT_V(rank(cta_tiler) == Int<3>{}); // (bM, bN, bK)

    CUTE_STATIC_ASSERT(is_static<AThreadLayout>::value);
    CUTE_STATIC_ASSERT(is_static<BThreadLayout>::value);
    CUTE_STATIC_ASSERT(is_static<CThreadLayout>::value);

    CUTE_STATIC_ASSERT_V(size(tA) == size(tB));
    CUTE_STATIC_ASSERT_V(size(tB) == size(tC));

    CUTE_STATIC_ASSERT_V(size<0>(cta_tiler) % size<0>(tA) == Int<0>{});   // BLK_M / THR_M
    CUTE_STATIC_ASSERT_V(size<2>(cta_tiler) % size<1>(tA) == Int<0>{});  // BLK_K / THR_K
    CUTE_STATIC_ASSERT_V(size<1>(cta_tiler) % size<0>(tB) == Int<0>{});  // BLK_N / THR_N
    CUTE_STATIC_ASSERT_V(size<2>(cta_tiler) % size<1>(tB) == Int<0>{});  // BLK_K / THR_K
    CUTE_STATIC_ASSERT_V(size<0>(cta_tiler) % size<0>(tC) == Int<0>{});  // BLK_M / THR_M
    CUTE_STATIC_ASSERT_V(size<1>(cta_tiler) % size<1>(tC) == Int<0>{});  // BLK_N / THR_N

    CUTE_STATIC_ASSERT(is_static<ASmemLayout>::value);
    CUTE_STATIC_ASSERT(is_static<BSmemLayout>::value);
    CUTE_STATIC_ASSERT(is_static<CSmemLayout>::value);

    CUTE_STATIC_ASSERT_V(size<0>(ASmemLayout{}) == size<0>(cta_tiler));  // BLK_M
    CUTE_STATIC_ASSERT_V(size<0>(CSmemLayout{}) == size<0>(cta_tiler));  // BLK_M
    CUTE_STATIC_ASSERT_V(size<0>(BSmemLayout{}) == size<1>(cta_tiler));  // BLK_N
    CUTE_STATIC_ASSERT_V(size<1>(CSmemLayout{}) == size<1>(cta_tiler));  // BLK_N
    CUTE_STATIC_ASSERT_V(size<1>(ASmemLayout{}) == size<2>(cta_tiler));  // BLK_K
    CUTE_STATIC_ASSERT_V(size<1>(BSmemLayout{}) == size<2>(cta_tiler));  // BLK_K

    CUTE_STATIC_ASSERT_V(congruent(select<0, 2>(shape_MNK), dA));
    CUTE_STATIC_ASSERT_V(congruent(select<1, 2>(shape_MNK), dB));
    CUTE_STATIC_ASSERT_V(congruent(select<1, 2>(shape_MNK), dC));

    Tensor mA = make_tensor(make_gmem_ptr(A), select<0,2>(shape_MNK), dA); // (M,K)
    Tensor mB = make_tensor(make_gmem_ptr(B), select<1,2>(shape_MNK), dB); // (N,K)
    Tensor mC = make_tensor(make_gmem_ptr(C), select<0,1>(shape_MNK), dC); // (M,N)

    auto cta_coord = make_coord(blockIdx.x, blockIdx.y, _);
    Tensor gA = local_tile(mA, cta_tiler, cta_coord, Step<_1, X, _1>{}); // (BM, BN, k)
    Tensor gB = local_tile(mB, cta_tiler, cta_coord, Step<X, _1, _1>{}); // (BN, BK, k)
    Tensor gC = local_tile(mC, cta_tiler, cta_coord, Step<_1, _1, X>{}); // (BM, BN)

    __shared__ TA smemA[cosize_v<ASmemLayout>];
    __shared__ TB smemB[cosize_v<BSmemLayout>];
    Tensor sA = make_tensor(make_smem_ptr(smemA), sA_layout);
    Tensor sB = make_tensor(make_smem_ptr(smemB), sB_layout);

    Tensor tAgA = local_partition(gA, tA, threadIdx.x); // (THR_M, THR_K, k)
    Tensor tAsA = local_partition(sA, tA, threadIdx.x); // (THR_M, THR_K)

    Tensor tBgB = local_partition(gB, tB, threadIdx.x); // (THR_N, THR_K, k)
    Tensor tBsB = local_partition(sB, tB, threadIdx.x); // (THR_N, THR_K)

    CUTE_STATIC_ASSERT_V(size<0>(tAgA) == size<0>(tAsA));
    CUTE_STATIC_ASSERT_V(size<1>(tAgA) == size<1>(tAsA));
    CUTE_STATIC_ASSERT_V(size<0>(tBgB) == size<0>(tBsB));
    CUTE_STATIC_ASSERT_V(size<1>(tBgB) == size<1>(tBsB));

    Tensor tCsA = local_partition(sA, tC, threadIdx.x, Step<_1, X>{});  // (TM, BK)
    Tensor tCsB = local_partition(sB, tC, threadIdx.x, Step<X, _1>{});  // (TN, BK)
    Tensor tCgC = local_partition(gC, tC, threadIdx.x, Step<_1, _1>{}); // (TM, TN)

    Tensor tCrC = make_tensor_like(tCgC);

    CUTE_STATIC_ASSERT_V(size<0>(tCrC) == size<0>(tCgC));                // THR_M
    CUTE_STATIC_ASSERT_V(size<0>(tCrC) == size<0>(tCsA));                // THR_M
    CUTE_STATIC_ASSERT_V(size<1>(tCrC) == size<1>(tCgC));                // THR_N
    CUTE_STATIC_ASSERT_V(size<1>(tCrC) == size<0>(tCsB));                // THR_N
    CUTE_STATIC_ASSERT_V(size<1>(tCsA) == size<1>(tCsB));                // BLK_K

    clear(tCrC);

    auto K_TILE_MAX = size<2>(tAgA);

    for (int k_tile = 0; k_tile < K_TILE_MAX; ++k_tile) {
        copy(tAgA(_, _, k_tile), tAsA); // A   (THR_M,THR_K) -> (THR_M,THR_K)
        copy(tBgB(_, _, k_tile), tCsB); // B   (THR_N,THR_K) -> (THR_N,THR_K)

        cp_async_fence();
        cp_async_wait<0>();
        __syncthreads(); // wait for all copy done

        gemm(tCsA, tCsB, tCrC);  // (THR_M,THR_N) += (THR_M,BLK_K) * (THR_N,BLK_K)
        __syncthreads();
    }

    axpby(alpha, tCrC, beta, tCgC);

}

template <class TA, class TB, class TC, class Alpha, class Beta>
void gemm(int m, int n, int k,
    TA const* A, int ldA,
    TB const *B, int ldB,
    TC      *C, int ldC,
    Alpha const alpha, Beta const beta)
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
    auto bK = Int<  8>{};
    auto cta_tiler = make_shape(bM, bN, bK);

    auto sA = make_layout(make_shape(bM, bK)); // m-major
    auto sB = make_layout(make_shape(bN, bK)); // n-major
    auto sC= make_layout(make_shape(bM, bN)); // m-major

    auto tA = make_layout(make_shape(Int<32>{}, Int<8>{}));
    auto tB = make_layout(make_shape(Int<32>{}, Int<8>{}));
    auto tC = make_layout(make_shape(Int<16>{}, Int<16>{}));

    dim3 dimBlock(size(tC));
    dim3 dimGrid(size(ceil_div(M, bM)), size(ceil_div(N, bN)));

    gemm_device<<<dimGrid, dimBlock>>>
      (prob_shape, cta_tiler,
       A, dA, sA, tA,
       B, dB, sB, tB,
       C, dC, sC, tC,
       alpha, beta);
}

// 计算单个元素的CPU参考值
template <class TA, class TB, class TC, class TI>
TC compute_ref_element(const thrust::host_vector<TA>& A, 
                       const thrust::host_vector<TB>& B,
                       const thrust::host_vector<TC>& C,
                       int m, int n, int k,
                       int row, int col,
                       int ldA, int ldB, int ldC,
                       TI alpha, TI beta)
{
    TC sum = 0;
    for (int i = 0; i < k; ++i) {
        // A是(M,K)列主序: A[row, i] = A[row + i * ldA]
        // B是(N,K)列主序: B[col, i] = B[col + i * ldB]
        // C = A * B^T，因为B存储为(N,K)而非(K,N)
        sum += static_cast<TC>(A[row + i * ldA]) * static_cast<TC>(B[col + i * ldB]);
    }
    return alpha * sum + beta * C[row + col * ldC];
}

// 随机抽样验证
template <class TA, class TB, class TC, class TI>
bool verify_random_samples(const thrust::host_vector<TA>& A,
                           const thrust::host_vector<TB>& B,
                           const thrust::host_vector<TC>& C,
                           const thrust::host_vector<TC>& result,
                           int m, int n, int k,
                           int ldA, int ldB, int ldC,
                           TI alpha, TI beta,
                           int num_samples = 100)
{
    printf("\n=== Random Sampling Verification ===\n");
    printf("Sampling %d random elements from %d x %d matrix...\n", num_samples, m, n);
    
    int errors = 0;
    float max_error = 0.0f;
    float avg_error = 0.0f;
    
    srand(12345);  // 固定种子以便复现
    
    for (int sample = 0; sample < num_samples; ++sample) {
        int row = rand() % m;
        int col = rand() % n;
        int idx = row + col * ldC;
        
        TC expected = compute_ref_element(A, B, C, m, n, k, row, col, ldA, ldB, ldC, alpha, beta);
        TC actual = result[idx];
        float error = std::abs(expected - actual);
        float rel_error = std::abs(expected) > 1e-5 ? error / std::abs(expected) : error;
        
        avg_error += rel_error;
        max_error = std::max(max_error, rel_error);
        
        if (rel_error > 1e-3) {  // 相对误差阈值
            if (errors < 10) {  // 只打印前10个错误
                printf("  [Error %d] Position (%d, %d): expected %.6f, got %.6f, rel_error = %.6e\n",
                       errors + 1, row, col, expected, actual, rel_error);
            }
            errors++;
        }
    }
    
    avg_error /= num_samples;
    
    printf("\nVerification Results:\n");
    printf("  Total samples: %d\n", num_samples);
    printf("  Errors (rel_error > 1e-3): %d\n", errors);
    printf("  Average relative error: %.6e\n", avg_error);
    printf("  Maximum relative error: %.6e\n", max_error);
    
    if (errors == 0) {
        printf("  ✓ PASSED - All sampled elements are correct!\n");
        return true;
    } else {
        printf("  ✗ FAILED - %d/%d samples have errors\n", errors, num_samples);
        return false;
    }
}

int main ()
{
    using namespace cute;
    // auto layout = make_layout(make_shape(4, 2), make_stride(1, 6));
    // print_layout(layout);print("\n");
    // std::cout << "layout size: " << (int)size(layout) << std::endl;
    // std::cout << "layout cosize: " << (int)cosize(layout) << std::endl;

    const int m = 5120;
    const int n = 5120;
    const int k = 4096;

    using TA = float;
    using TB = float;
    using TC = float;
    using TI = float;

    TI alpha = 1.0f;
    TI beta = 0.0f;

    cute::device_init(0);

    thrust::host_vector<TA> h_A(m * k);
    thrust::host_vector<TB> h_B(k * n);
    thrust::host_vector<TC> h_C(m * n);

    srand(42);  // 设置随机种子
    for (int j = 0; j < m*k; ++j) h_A[j] = static_cast<TA>( 2*(rand() / double(RAND_MAX)) - 1 );
    for (int j = 0; j < n*k; ++j) h_B[j] = static_cast<TB>( 2*(rand() / double(RAND_MAX)) - 1 );
    for (int j = 0; j < m*n; ++j) h_C[j] = static_cast<TC>(-1);

    thrust::device_vector<TA> d_A = h_A;
    thrust::device_vector<TB> d_B = h_B;
    thrust::device_vector<TC> d_C = h_C;

    int ldA = m, ldB = n, ldC = m;

    double gflops = (2.0 * m * n * k) * 1e-9;

    const int timing_iterations = 1;
    GPU_Clock timer;
    
    // Run once for correctness check
    printf("=== Running CUTLASS GEMM ===\n");
    printf("Problem size: M=%d, N=%d, K=%d\n", m, n, k);
    printf("Alpha=%.1f, Beta=%.1f\n\n", alpha, beta);
    
    d_C = h_C;
    gemm(m, n, k,
         d_A.data().get(), ldA,
         d_B.data().get(), ldB,
         d_C.data().get(), ldC,
         alpha, beta);
    CUTE_CHECK_LAST();
    thrust::host_vector<TC> cute_result = d_C;

    // 随机抽样验证
    bool verification_passed = verify_random_samples(
        h_A, h_B, h_C, cute_result,
        m, n, k, ldA, ldB, ldC,
        alpha, beta, 100);

    // Timing iterations
    printf("\n=== Performance Measurement ===\n");
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
    
    return verification_passed ? 0 : 1;
}