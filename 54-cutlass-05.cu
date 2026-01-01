#include <cstdlib>
#include <cstdio>
#include <cassert>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include <cute/tensor.hpp>

#include <cutlass/util/print_error.hpp>
#include <cutlass/util/GPU_Clock.hpp>
#include <cutlass/util/helper_cuda.hpp>

#define SPLIT_LINE print("\n%s\n", "=============================");

int main()
{
    using namespace cute;
    auto layout = Layout<Shape <_2,Shape <_1,_6>>,
                     Stride<_1,Stride<_6,_2>>>{};
    print(layout);
    SPLIT_LINE

    auto result = coalesce(layout);    // _12:_1
    print(result);
    SPLIT_LINE

    // 1. 定义 Layout A: Shape=(6,2), Stride=(8,2)
    auto layout_a = make_layout(make_shape(Int<6>{}, Int<2>{}),
        make_stride(Int<8>{}, Int<2>{}));
    // 2. 定义 Layout B: Shape=(4,3), Stride=(3,1)
    auto layout_b = make_layout(make_shape(Int<4>{}, Int<3>{}),
                                make_stride(Int<3>{}, Int<1>{}));
    auto layout_r = composition(layout_a,layout_b);
    print(layout_r);
    print_layout(layout_r);
    SPLIT_LINE

    // 行优先 (Row-Major): C/C++ 默认
    auto row_major = make_layout(
        make_shape(Int<4>{}, Int<8>{}),  // 4×8 矩阵
        make_stride(Int<8>{}, Int<1>{})   // stride: (8, 1)
    );

    // 列优先 (Column-Major): Fortran/BLAS 默认
    auto col_major = make_layout(
        make_shape(Int<4>{}, Int<8>{}),  // 4×8 矩阵
        make_stride(Int<1>{}, Int<4>{})   // stride: (1, 4)
    );

    std::cout << "Row-Major: " << row_major << std::endl;
    std::cout << "Col-Major: " << col_major << std::endl;
    std::cout << std::endl;

    // 验证访问模式
    std::cout << "Row-Major (4×8): 前16个元素的地址" << std::endl;
    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 8; ++j) {
            printf("%3d ", row_major(i, j));
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;

    Tensor A = make_tensor<float>(Shape <Shape < _4,_5>,Int<13>>{},
                              Stride<Stride<_12,_1>,    _64>{});
    print(A);
    for (int m0 = 0; m0 < size<0,0>(A); ++m0) {
        for (int m1 = 0; m1 < size<0,1>(A); ++m1) {
            for (int n = 0; n < size<1>(A); ++n) {
                A[make_coord(make_coord(m0,m1),n)] = n + 2 * m0;
                print("m0: %d, m1: %d, n: %d\n", m0, m1, n);
                print(make_coord(m0, m1));
                print(" | ");
                print(make_coord(make_coord(m0, m1), n));
                print(" | ");
                SPLIT_LINE
            }
        }
    }

    print_tensor(A);

    print(size(A));
    SPLIT_LINE
    print(size<0>(A));
    SPLIT_LINE
    print(size<1>(A));
    SPLIT_LINE
    print(size<0, 0>(A));
    SPLIT_LINE
    print(size<0, 1>(A));
    SPLIT_LINE

    Layout layout_a_1 = make_layout(Shape <Shape < _4,_5>,Int<13>>{},
                              Stride<Stride<_12,_1>,    _64>{});
    print_layout(layout_a_1);
    SPLIT_LINE
}