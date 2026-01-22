#include <pybind11/pybind11.h>
#include <pybind11/stl_bind.h>
#include "cuda_array.h"

namespace py = pybind11;

PYBIND11_MODULE(cubind1, m) {
    py::class_<CudaArray>(m, "CudaArray")
        .def(py::init<size_t>())
        .def("copy_from_host", &CudaArray::copy_from_host)
        .def("copy_to_host", &CudaArray::copy_to_host)
        .def("sum", &CudaArray::sum) // 直接调用 Thrust 的 reduce
        .def_property_readonly("size", &CudaArray::size);

    m.def("add", &launch_add, "Add two CudaArrays using custom kernel");
}