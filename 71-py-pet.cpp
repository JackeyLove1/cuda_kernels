#include <pybind11/pybind11.h>
#include <iostream>

namespace py = pybind11;

class Pet
{
    std::string _name;
    int _age;
public:
    Pet(std::string name, int age): _name(std::move(name)), _age(age){}

    const std::string & get_name() const
    {
        return _name;
    }

    const auto age() const
    {
        return _age;
    }

    void greet()
    {
        std::cout << "Say Hello from: " << _name << " age: " << _age << std::endl;
    }
};

PYBIND11_MODULE(pet, m) {
    py::class_<Pet>(m, "pet")
        .def(py::init<std::string, int>(), py::arg("name"), py::arg("age"))
        .def("greet", &Pet::greet)
        .def_property_readonly("age", &Pet::age);
}