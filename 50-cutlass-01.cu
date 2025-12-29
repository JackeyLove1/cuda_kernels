#include <cutlass/gemm/device/gemm.h>
#include <cutlass/numeric_types.h>
#include <cutlass/util/host_tensor.h>

using half_t      = cutlass::half_t;
using ColumnMajor = cutlass::layout::ColumnMajor;

// Define the GEMM operation
using Gemm       = cutlass::gemm::device::Gemm<half_t,                         // ElementA
                                               ColumnMajor,                    // LayoutA
                                               half_t,                         // ElementB
                                               ColumnMajor,                    // LayoutB
                                               half_t,                         // ElementOutput
                                               ColumnMajor,                    // LayoutOutput
                                               float,                          // ElementAccumulator
                                               cutlass::arch::OpClassTensorOp, // tag indicating Tensor Cores
                                               cutlass::arch::Sm75 // tag indicating target GPU compute architecture
                                               >;
using HostTensor = cutlass::HostTensor<cutlass::half_t, cutlass::layout::ColumnMajor>;

int main()
{


    Gemm            gemm_op;
    cutlass::Status status;

    //
    // Define the problem size
    //
    constexpr int M = 512;
    constexpr int N = 256;
    constexpr int K = 128;

    constexpr float alpha = 1.25f;
    constexpr float beta  = -1.25f;

    //
    // Allocate device memory
    //

    HostTensor A({M, K});
    HostTensor B({K, N});
    HostTensor C({M, N});

    half_t const *ptrA = A.device_data();
    half_t const *ptrB = B.device_data();
    half_t const *ptrC = C.device_data();
    half_t       *ptrD = C.device_data();

    int lda = A.device_ref().stride(0);
    int ldb = B.device_ref().stride(0);
    int ldc = C.device_ref().stride(0);
    int ldd = C.device_ref().stride(0);
    //
    // Launch GEMM on the device
    //

    status = gemm_op({
        {M, N, K},
        {ptrA, lda},  // TensorRef to A device tensor
        {ptrB, ldb},  // TensorRef to B device tensor
        {ptrC, ldc},  // TensorRef to C device tensor
        {ptrD, ldd},  // TensorRef to D device tensor - may be the same as C
        {alpha, beta} // epilogue operation arguments
    });

    if (status != cutlass::Status::kSuccess) {
        return -1;
    }

    return 0;
}