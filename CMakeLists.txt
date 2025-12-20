cmake_minimum_required(VERSION 3.18)

project(cuda_examples LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

add_compile_options(-O3)

# Provide a reasonable default if user didn't set it.
# You can override at configure time, e.g.:
#   cmake -S . -B build -DCMAKE_CUDA_ARCHITECTURES=86
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 86)
endif()

# Force update if it is set to the old default 52 (Maxwell) which doesn't support scoped atomics
if(CMAKE_CUDA_ARCHITECTURES STREQUAL "52")
    set(CMAKE_CUDA_ARCHITECTURES 86 CACHE STRING "CUDA architectures" FORCE)
endif()

find_package(CUDAToolkit REQUIRED)

# Cutlass includes
include_directories(
  /root/autodl-tmp/cutlass-main/include
  /root/autodl-tmp/cutlass-main/tools/util/include
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