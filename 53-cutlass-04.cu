#include <cstdlib>
#include <cstdio>
#include <cassert>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include <cute/tensor.hpp>

#include <cutlass/util/print_error.hpp>
#include <cutlass/util/GPU_Clock.hpp>
#include <cutlass/util/helper_cuda.hpp>

#define SPLIT_LINE print("%s\n", "=======================");
int main()
{
    using namespace cute;
    auto layout = Layout<Shape<_2, Shape<_1, _6>>, Stride<_1, Stride<_6, _2>>>{};
    print_layout(layout);
    SPLIT_LINE

    auto layout2 = coalesce(layout, Step<_1, _2>{});
    print_layout(layout2);
    SPLIT_LINE

    // const auto layout3 = make_layout(make_shape(Int<16>{}, Int<8>{}), make_stride(Int<4>{}, Int<2>{}));
    // print_layout(layout3);
    // SPLIT_LINE

    const auto layout4 = make_layout(make_shape(Shape<_3>{}, Shape<_2, _3>{}));
    print_layout(layout4);
    auto shape = Shape<_3, Shape<_2, _3>>{};
    print(idx2crd(16, shape));
    SPLIT_LINE
}