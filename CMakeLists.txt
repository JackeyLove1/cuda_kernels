cmake_minimum_required(VERSION 3.18)

# ============================================================================
# How to configure and build this project:
#
# From WSL/Linux or Windows terminal:
#   cmake -DCMAKE_BUILD_TYPE=Debug \
#         -DCMAKE_CUDA_ARCHITECTURES=86 \
#         -S . \
#         -B build
#
#   cmake --build build
#
# Note: On Windows, make sure NVCC is in your PATH or set CMAKE_CUDA_COMPILER.
# ============================================================================

project(cuda_examples LANGUAGES CXX CUDA)

enable_language(CUDA)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

set(CMAKE_PREFIX_PATH "/opt/libtorch")
find_package(Torch REQUIRED)

# Only apply -use_fast_math to CUDA code, not C++ code
add_compile_options($<$<COMPILE_LANGUAGE:CUDA>:-use_fast_math>)

# Emit line-number info for profiling/debugging (CUDA only)
add_compile_options($<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>)

# ============================================================================
# 警告控制 (Warning Control)
# ============================================================================
# 方案1: 禁用所有警告（最激进）
# add_compile_options($<$<COMPILE_LANGUAGE:CXX>:-w>)
# add_compile_options($<$<COMPILE_LANGUAGE:CUDA>:-w>)

# 方案2: 禁用CUDA主机编译器的警告（推荐）
add_compile_options($<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-w>)

# NOTE:
# - Hardcoding SM120 breaks non-Blackwell builds and can also lead to runtime
#   CUTLASS "arch conditional MMA instruction" asserts if the generated code
#   doesn't match the required __CUDA_ARCH__.
if(NOT CMAKE_CUDA_ARCHITECTURES)
  # Prefer "native" when supported so the build matches the installed GPU.
  # (If your CMake is older and doesn't understand "native" for CUDA archs,
  # fall back to a conservative default.)
  if(CMAKE_VERSION VERSION_GREATER_EQUAL "3.24")
    set(CMAKE_CUDA_ARCHITECTURES native)
  else()
    set(CMAKE_CUDA_ARCHITECTURES 120)
  endif()
endif()

# Force update if it is set to the old default 52 (Maxwell) which doesn't support scoped atomics
#if(CMAKE_CUDA_ARCHITECTURES STREQUAL "52")
#    set(CMAKE_CUDA_ARCHITECTURES 86 CACHE STRING "CUDA architectures" FORCE)
#endif()

# Add current source directory to include path so all targets can access headers like cuda_array.h
include_directories(${CMAKE_CURRENT_SOURCE_DIR})

# Find Threads package (required by CUDAToolkit on some systems)
set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)

find_package(CUDAToolkit REQUIRED)

# Cutlass includes
if(NOT DEFINED CUTLASS_DIR)
  # Default to local 3rdparty/cutlass directory
  set(CUTLASS_DIR "${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/cutlass")
endif()

if(EXISTS "${CUTLASS_DIR}/include")
  include_directories(
    ${CUTLASS_DIR}/include
    ${CUTLASS_DIR}/tools/util/include
    ${CUTLASS_DIR}/examples/common
  )
  message(STATUS "CUTLASS found at: ${CUTLASS_DIR}")
else()
  message(WARNING "CUTLASS NOT found at: ${CUTLASS_DIR}. Some targets might fail to build.")
endif()




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
#if (MSVC)
#  target_compile_options(cublas1 PRIVATE /W4)
#else()
#  target_compile_options(cublas1 PRIVATE -Wall -Wextra -Wpedantic)
#endif()


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

# In your CMakeLists.txt
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --extended-lambda")
add_executable(cub_1 61-cub-device-reduce.cu)
target_compile_options(cub_1 PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>)
target_link_libraries(cub_1 PRIVATE CUDA::cudart)

add_executable(cub_2 62-cub-2.cu)
add_executable(cut_gemm_1 63-cutlass-gemm.cu)
add_executable(cut_gemm_2 64-cutlass-gemm.cu)
add_executable(cut_gemm_3 65-cutlass-gemm.cu)
add_executable(cut_gemm_4 66-cutlass-gemm.cu)
target_compile_options(cut_gemm_4 PRIVATE -lineinfo)

add_executable(hello_1 67-hello.cpp)
add_executable(cut_gemm_5 68-cutlass-gemm.cu)
add_executable(cut_gemm_6 69-cutlass-gemm.cu)
target_link_libraries(cut_gemm_6 PRIVATE
        CUDA::cudart
        CUDA::cublas)

find_package(pybind11 REQUIRED)
pybind11_add_module(hello 70-py-bind.cpp)
pybind11_add_module(pet 71-py-pet.cpp)

pybind11_add_module(cubind1 73-py-cuarray-bind.cpp 72-py-cuarray.cu)
set_target_properties(cubind1 PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_link_libraries(cubind1 PRIVATE CUDA::cudart)

add_executable(torch1 74-libtorch-1.cu)
#
# Libtorch: find_package(Torch) does NOT automatically add include paths to every target.
# Wire Torch into targets that actually use it so both builds and IDE tooling (compile_commands.json)
# can see <torch/...> headers.
target_link_libraries(torch1 PRIVATE ${TORCH_LIBRARIES})
target_include_directories(torch1 PRIVATE ${TORCH_INCLUDE_DIRS})

add_executable(cute_8 75-cute-gemm.cu)
target_link_libraries(cute_8 PRIVATE CUDA::cudart)

add_executable(cute_9 76-cute-copy.cu)
add_executable(cute_10 77-cute-gemm.cu)
add_executable(cute_11 78-cute-gemm.cu)
add_executable(cute_12 79-cute-gemm.cu)
add_executable(cute_13 80-cute-gemm.cu)
add_executable(cute_14 81-cute-gemm.cu)
add_executable(cute_15 82-cute-gemm.cu)
target_link_libraries(cute_15 PRIVATE CUDA::cudart CUDA::cublas)

add_executable(torch2 83-print-tensor.cu)
target_link_libraries(torch2 PRIVATE ${TORCH_LIBRARIES})

# Blackwell SM120 examples - only build if targeting that architecture
if(CMAKE_CUDA_ARCHITECTURES MATCHES "120|121")
  add_executable(blackwell1 84-blackwell-1.cu)
  add_executable(flashinfer4 88-flashinfer4.cu)
  message(STATUS "Building Blackwell SM120 examples (blackwell1, flashinfer4)")
else()
  message(STATUS "Skipping Blackwell SM120 examples - requires compute capability 120+, current: ${CMAKE_CUDA_ARCHITECTURES}")
endif()

# Regular flashinfer examples (work on any modern GPU)
add_executable(flashinfer1 85-flashinfer-1.cu)
add_executable(flashinfer2 86-flashinfer2.cu)

include_directories(${PROJECT_SOURCE_DIR}/include)
add_executable(flashinfer3 87-flashinfer3.cu)
#target_link_libraries(flashinfer3 PRIVATE ${TORCH_LIBRARIES})
#target_include_directories(flashinfer3 PRIVATE ${TORCH_INCLUDE_DIRS})

add_executable(flashinfer5 89-flashinfer5.cu)
add_executable(flashinfer6 90-flashinfer6.cu)
add_executable(flashinfer7 91-flashinfer7.cu)
add_executable(flashinfer8 92-flashinfer8-rmsnorm.cu)

add_executable(flashinfer9 94-torch-rmsnorm.cu)
target_link_libraries(flashinfer9 PRIVATE ${TORCH_LIBRARIES})
target_include_directories(flashinfer9 PRIVATE ${TORCH_INCLUDE_DIRS})

add_executable(flashinfer10 95-flashinfer-rope.cu)
#target_link_libraries(flashinfer10 PRIVATE ${TORCH_LIBRARIES})
#target_include_directories(flashinfer10 PRIVATE ${TORCH_INCLUDE_DIRS})

add_executable(flashinfer11 96-flashinfer-gemm1.cu)
add_executable(flashinfer12 97-flashinfer-dotp.cu)
add_executable(conv2d1 98-2D-Conv.cu)
add_executable(stencil 99-stencil.cu)