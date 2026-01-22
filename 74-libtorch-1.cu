#include <ATen/core/TensorBody.h>
#include <ATen/ops/mm.h>
#include <torch/csrc/autograd/generated/variable_factories.h>
#include <torch/torch.h>
#include <iostream>
#include <ATen/ATen.h>

int main() {
  torch::Tensor tensor = torch::rand({2, 3});
  std::cout << tensor << std::endl;
  torch::Tensor a = torch::rand({4, 32});
  auto b = torch::rand({32, 8});
  auto c = torch::mm(a, b);
  std::cout << "out: " << c << std::endl;
}