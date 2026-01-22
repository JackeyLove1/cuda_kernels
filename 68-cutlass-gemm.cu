#include <cstdlib>
#include <cuda_runtime.h>
#include <cute/tensor.hpp>

int main() {
  using namespace cute;

  auto shape = Shape<_3, Shape<_2, _3>>{};
  print(shape);
  print("\n");
  auto stride = Stride<_3, Stride<_12, _1>>{};
  print(stride);
  print("\n");
  auto layout_1 = make_layout(shape, stride);
  print_layout(layout_1);
  print("\n");

  auto a = Layout<Shape<_2, Shape<_1, _6>>, Stride<_1, Stride<_6, _2>>>{};
  print_layout(a);
  print("\n");

  auto result = coalesce(a, Step<_1, _1>{});  // (_2,_6):(_1,_2)
  print_layout(result);
  print("\n");
  // Identical to
  auto same_r = make_layout(coalesce(layout<0>(a)), coalesce(layout<1>(a)));

  auto A = Layout<Shape<_6, _2>, Stride<_8, _2>>{};
  auto B = Layout<Shape<_4, _3>, Stride<_3, _1>>{};
  auto R = composition(A, B);
  print("\nA:\n");
  print_layout(A);
  print("\n");
  print("\nB:\n");
  print_layout(B);
  print("\n");
  print_layout(R);
  print("\n");

  auto layout_2 =
      make_layout(Shape<_4, Shape<_2, _2>>{}, Stride<_2, Stride<_1, _8>>{});
  print_layout(layout_2);
  print("\n");

  auto layout3 = Layout<Shape<Shape<_4, _2>, _4>, Stride<Stride<_8, _4>, _1>>{};
  print_layout(layout3);
  print("\n");

  using SM80_16x8_Row = Layout<Shape<Shape<_4, _8>, Shape<_2, _2>>,
                               Stride<Stride<_32, _1>, Stride<_16, _8>>>;
  print_layout(SM80_16x8_Row{});
  print("\n");
  print_latex(SM80_16x8_Row{}) print("\n");
}