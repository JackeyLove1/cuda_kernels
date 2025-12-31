#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm_universal.h"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/util/device_memory.h"

//
// CUTLASS 2.x API for Ampere (Sm80/Sm86)
//
// Note: CUTLASS 3.x CollectiveBuilder is primarily for Hopper (Sm90) architectures.
// For Ampere (Sm80/Sm86), using the standard CUTLASS 2.x GemmUniversal is the 
// recommended and most robust way to achieve high performance with Tensor Cores.
//

int main(int argc, char const **args) {

  //
  // Define the GEMM type
  //

  using ElementA    = cutlass::half_t;
  using LayoutA     = cutlass::layout::RowMajor;
  using ElementB    = cutlass::half_t;
  using LayoutB     = cutlass::layout::ColumnMajor;
  using ElementC    = cutlass::half_t;
  using LayoutC     = cutlass::layout::ColumnMajor;
  using ElementAccumulator = float;

  // Use Sm80 which is compatible with Sm86
  using ArchTag = cutlass::arch::Sm80;
  using OperatorClass = cutlass::arch::OpClassTensorOp;
  
  // Threadblock shape: 128x128x64
  using ThreadblockShape = cutlass::gemm::GemmShape<128, 128, 64>;
  using WarpShape        = cutlass::gemm::GemmShape<64, 64, 64>;
  using InstructionShape = cutlass::gemm::GemmShape<16, 8, 16>;

  // Epilogue operator
  using EpilogueOp = cutlass::epilogue::thread::LinearCombination<
      ElementC,
      128 / cutlass::sizeof_bits<ElementC>::value, // Alignment
      ElementAccumulator,
      ElementAccumulator
  >;

  // GemmUniversal configuration
  using Gemm = cutlass::gemm::device::GemmUniversal<
      ElementA, LayoutA,
      ElementB, LayoutB,
      ElementC, LayoutC,
      ElementAccumulator,
      OperatorClass,
      ArchTag,
      ThreadblockShape,
      WarpShape,
      InstructionShape,
      EpilogueOp
  >;

  Gemm gemm_op;
  cutlass::Status status;

  //
  // Define the problem size
  //

  int M = 512;
  int N = 256;
  int K = 128;

  float alpha = 1.25f;
  float beta = -1.25f;

  //
  // Allocate device memory
  //

  cutlass::DeviceAllocation<ElementA> block_A(M * K);
  cutlass::DeviceAllocation<ElementB> block_B(K * N);
  cutlass::DeviceAllocation<ElementC> block_C(M * N);
  cutlass::DeviceAllocation<ElementC> block_D(M * N); // ElementOutput is ElementC here

  // Leading dimensions
  // RowMajor A (MxK): ld = K
  int lda = K;
  // ColumnMajor B (KxN): ld = K (stride between columns)
  int ldb = K;
  // ColumnMajor C (MxN): ld = M (stride between columns)
  int ldc = M;
  int ldd = M;

  // Batch strides (not used for batch_count=1 but good to set correctly)
  long long batch_stride_A = static_cast<long long>(M) * K;
  long long batch_stride_B = static_cast<long long>(K) * N;
  long long batch_stride_C = static_cast<long long>(M) * N;
  long long batch_stride_D = static_cast<long long>(M) * N;

  //
  // Launch GEMM on the device
  //

  status = gemm_op({
    cutlass::gemm::GemmUniversalMode::kGemm,
    {M, N, K},
    1, // batch_count
    {alpha, beta},
    block_A.get(),
    block_B.get(),
    block_C.get(),
    block_D.get(),
    batch_stride_A,
    batch_stride_B,
    batch_stride_C,
    batch_stride_D,
    lda,
    ldb,
    ldc,
    ldd
  });

  if (status != cutlass::Status::kSuccess) {
    return -1;
  }

  return 0;
}
