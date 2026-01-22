#include <pybind11/pybind11.h>
#include <iostream>

namespace py = pybind11;

int add(int a, int b)
{
    return a + b;
}



PYBIND11_MODULE(hello, m)
{
    m.doc() = "hello world";
    m.def("add", &add, "A function that adds two integrers");
}