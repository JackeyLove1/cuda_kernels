cmake_minimum_required(VERSION 3.18)

# ============================================================================
# How to configure and build this project:
#
# From WSL/Linux terminal:
#   cmake -DCMAKE_BUILD_TYPE=Debug \
#         -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc \
#         -DCMAKE_CUDA_ARCHITECTURES=86 \
#         -G "Unix Makefiles" \
#         -S /mnt/c/Software/Codes/cuda/cuda_kernels \
#         -B /mnt/c/Software/Codes/cuda/cuda_kernels/cmake-build-debug
#
#   cmake --build /mnt/c/Software/Codes/cuda/cuda_kernels/cmake-build-debug
#
# Or use CMake presets (see CMakePresets.json):
#   cmake --preset debug
#   cmake --build --preset debug
#
# Note: Use /usr/local/cuda instead of /usr/local/cuda-X.Y to avoid path issues
# ============================================================================

project(cuda_examples LANGUAGES CXX CUDA)

enable_language(CUDA)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

add_compile_options(-O3)
# add_compile_options(-g)
# add_compile_options($<$<COMPILE_LANGUAGE:CUDA>:-G>)
# Only apply -use_fast_math to CUDA code, not C++ code
add_compile_options($<$<COMPILE_LANGUAGE:CUDA>:-use_fast_math>)
# Provide a reasonable default if user didn't set it.
# You can override at configure time, e.g.:
#   cmake -S . -B build -DCMAKE_CUDA_ARCHITECTURES=86
 if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
   set(CMAKE_CUDA_ARCHITECTURES 80)
 endif()

# Force update if it is set to the old default 52 (Maxwell) which doesn't support scoped atomics
if(CMAKE_CUDA_ARCHITECTURES STREQUAL "52")
    set(CMAKE_CUDA_ARCHITECTURES 86 CACHE STRING "CUDA architectures" FORCE)
endif()

# Find Threads package (required by CUDAToolkit on some systems)
set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)

find_package(CUDAToolkit REQUIRED)

# Cutlass includes
include_directories(
  # ${CMAKE_CURRENT_SOURCE_DIR}
  /mnt/c/Software/Codes/cuda/cutlass/include
  /mnt/c/Software/Codes/cuda/cutlass/tools/util/include
  /mnt/c/Software/Codes/cuda/cutlass/examples/common
)



add_executable(cublas1
  cublas1.cu
)

add_executable(sorting sorting.cu)

target_link_libraries(cublas1 PRIVATE
  CUDA::cudart
  CUDA::cublas
)

target_link_libraries(sorting PRIVATE
  CUDA::cudart
)

# Nice-to-have warnings for the host compiler
if (MSVC)
  target_compile_options(cublas1 PRIVATE /W4)
else()
  target_compile_options(cublas1 PRIVATE -Wall -Wextra -Wpedantic)
endif()


add_executable(helloworld 01-helloworld.cu)
add_executable(vector_add 02-vector-add.cu)
add_executable(matrix_t 03-matrix-transpose.cu)
add_executable(wrap_lane 04-wrap-lane.cu)
add_executable(vector_add_cutlass 05-vector-add-cutlass.cu)
target_link_libraries(vector_add_cutlass PRIVATE CUDA::cudart)
add_executable(vector_add_2 06-vector-add.cu)
add_executable(matrix_t_2 07-matrix-transpose.cu)
add_executable(reduce_sum 08-reduce-sum.cu)
add_executable(reduce_sum_2 09-reduce-sum-1.cu)
add_executable(reduce_sum_3 10-reduce-sum-2.cu)
add_executable(reduce_sum_cub 11-cub-reduce-sum.cu)
add_executable(reduce_sum_4 12-reduce-sum-3.cu)
add_executable(dim 13-dim.cu)
add_executable(reduce_sum_5 14-reduce-5.cu)
add_executable(reduce_sum_6 14-reduce-tuned.cu)
add_executable(reduce_sum_7 15-reduce-sum-6.cu)
add_executable(reduce_sum_8 16-reduce-sum-7.cu)
add_executable(reduce_sum_9 17-reduce-sum-8.cu)
add_executable(reduce_sum_10 18-reduce-baseline.cu)
add_executable(fast_atomic_add 19-fast-atomic-add.cu)
add_executable(relu_1 20-relu.cu)
add_executable(relu_2 21-relu.cu)
add_executable(dot_1 22-dot-production.cu)
add_executable(sigmoid_1 23-sigmoid.cu)
add_executable(silu_1 24-silu.cu)
add_executable(sigmoid_2 25-sigmoid-2.cu)
add_executable(softmax_1 26-safe-softmax.cu)
add_executable(softmax_2 27-safe-softmax-optimized.cu)
add_executable(softmax_3 28-softmax-cub.cu)
add_executable(softmax_4 29-online-softmax.cu)
add_executable(gemm_1 30-gemm-cublas.cu)
target_link_libraries(gemm_1 PRIVATE CUDA::cudart CUDA::cublas)
add_executable(gemm_2 31-gemm-1.cu)
add_executable(gemm_3 32-gemm-2.cu)
add_executable(gemm_4 33-gemm-3.cu)
add_executable(gemm_5 34-gemm-4.cu)
add_executable(gemm_6 35-gemm-5.cu)
add_executable(gemm_7 36-gemm-6.cu)
add_executable(gemm_8 37-gemm-7.cu)
add_executable(gemm_9 38-gemm-8.cu)
add_executable(gemm_10 39-gemm-9.cu)
add_executable(gemm_11 40-gemm-10.cu)
add_executable(gemm_12 41-gemm-11.cu)
add_executable(gemm_13 42-gemm-12.cu)
add_executable(cublas_benchmark cublas_benchmark.cu)
target_link_libraries(cublas_benchmark PRIVATE CUDA::cudart CUDA::cublas)
add_executable(gemm_14 43-gemm-register-tiling.cu)
add_executable(gemm_15 44-gemm-13.cu)
add_executable(gemm_16 45-gemm-1.cu)
add_executable(gemm_17 46-gemm-15.cu)
add_executable(gemm_18 47-gemm-16.cu)
add_executable(gemm_19 48-gemm-17.cu)
add_executable(wgemm_1 49-wgemm-1.cu)
add_executable(cutlass_1 50-cutlass-01.cu)
add_executable(cutlass_2 51-cutlass-02.cu)
add_executable(cutlass_3 52-cutlass-03.cu)
target_link_libraries(cutlass_3 PRIVATE CUDA::cudart CUDA::cublas)
add_executable(cute_1 53-cutlass-04.cu)
add_executable(cute_2 54-cutlass-05.cu)
add_executable(cute_3 55-cutlass-06.cu)
add_executable(cute_4 56-cutlass-07.cu)
add_executable(cublas_1 57-cublas-fp16.cu)
target_link_libraries(cublas_1 PRIVATE CUDA::cudart CUDA::cublas)
add_executable(cute_5 58-hgemm.cu)
add_executable(cute_6 59-sm80-hgemm-1.cu)

find_package(OpenMP REQUIRED)
add_executable(openmp_1 60-openmp-01.cpp)
if(OpenMP_CXX_FOUND)
    target_link_libraries(openmp_1 PUBLIC OpenMP::OpenMP_CXX)
endif()