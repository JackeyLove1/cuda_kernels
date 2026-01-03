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
    const float* ptr = nullptr;

    // ((_3,2),(2,_5,_2)):((4,1),(_2,13,100))
    Tensor A = make_tensor(ptr, make_shape (make_shape (Int<3>{},2), make_shape (       2,Int<5>{},Int<2>{})),
                                make_stride(make_stride(       4,1), make_stride(Int<2>{},      13,     100)));
    print(A);
    SPLIT_LINE
    print_layout(A.layout());
    SPLIT_LINE

    // ((2,_5,_2)):((_2,13,100))
    Tensor B = A(2,_);
    print(B);
    SPLIT_LINE

    // ((_3,_2)):((4,1))
    Tensor C = A(_,5);
    print(C);
    SPLIT_LINE

    // (_3,2):(4,1)
    Tensor D = A(make_coord(_,_),5);
    print(D);
    SPLIT_LINE

    // (_3,_5):(4,13)
    Tensor E = A(make_coord(_,1),make_coord(0,_,1));
    print(E);
    SPLIT_LINE

    // (2,2,_2):(1,_2,100)
    Tensor F = A(make_coord(2,_),make_coord(_,3,_));
    print(F);
    SPLIT_LINE

    Tensor gmem = make_tensor(ptr, make_shape(Int<8>{}, 16));  // (_8,16)
    Tensor rmem = make_tensor_like(gmem(_, 0));                // (_8)
    print(gmem);
    // print_layout(gmem.layout());
    SPLIT_LINE
    print(rmem);
    // print_layout(rmem.layout());
    SPLIT_LINE

}