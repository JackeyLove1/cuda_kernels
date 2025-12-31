#include <cstdlib>
#include <cstdio>
#include <cassert>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include <cute/tensor.hpp>

#include <cutlass/util/print_error.hpp>
#include <cutlass/util/GPU_Clock.hpp>
#include <cutlass/util/helper_cuda.hpp>

int main()
{
    using namespace cute;
    auto layout = Layout<Shape<_2, Shape<_1, _6>>, Stride<_1, Stride<_6, _2>>>{};
    print_layout(layout);

    print("%s\n", "=======================");

    auto layout2 = coalesce(layout, Step<_1, _2>{});
    print_layout(layout2);

    const auto layout3 = make_layout(make_shape(Int<2>{}, 4), make_stride(Int<12>{}, Int<1>{}));
    print_layout(layout3);
}