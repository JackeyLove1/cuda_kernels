#include <cstdio>
#include <cstdlib>

#include <torch/extension.h>
#include <torch/all.h>

void printArray(torch::Tensor input) {
  int *ptr = static_cast<int*>(input.data_ptr());
  for(int i = 0; i < input.numel(); ++i) {
    printf("%d\n", ptr[i]);
  }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("print_array", &printArray, "");
}