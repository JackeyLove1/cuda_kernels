import torch
from torch.utils.cpp_extension import load

ext_module = load(name="print_tensor", sources=["83-print-tensor.cu"], verbose=True)
print("module dir: ", ext_module.__file__)
ext_module.print_array(torch.tensor([1,2,3,4,5]))
