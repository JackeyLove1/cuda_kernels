#include <cstdlib>
#include <cstdio>
#include <cassert>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

// #include <cute/tensor.hpp>
//
// #include <cutlass/util/print_error.hpp>
// #include <cutlass/util/GPU_Clock.hpp>
// #include <cutlass/util/helper_cuda.hpp>
#include <thrust/device_ptr.h>
#include <thrust/host_vector.h>


#define SPLIT_LINE print("\n%s\n", "=============================");

int main()
{
    // using namespace cute;
    // auto layout = make_layout(make_shape(Int<128>{}, Int<256>{}));
    // auto shape = make_shape(Int<32>{}, Int<8>{});
    // auto tiled = tiled_divide(layout, shape);
    // print(tiled);
    // SPLIT_LINE
    // print(size(tiled));
    // SPLIT_LINE
    // print(size<0>(tiled));
    // SPLIT_LINE
    // print(size<1>(tiled));
    // SPLIT_LINE
    // print(size<2>(tiled));
    // SPLIT_LINE
    // print(tiled.shape());
    // SPLIT_LINE
    constexpr auto A = 1024;
    thrust::host_vector<int> h_a(A);
    for (int i = 0; i < A; i++) {
        h_a[i] = i;
    }
    thrust::device_vector<int> d_a = h_a;


}