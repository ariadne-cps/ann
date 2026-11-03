#include "pybind11.hpp"
#include "neural/neural.hpp"

void neural_submodule(pybind11::module& module) {
    module.def("neural_version", &Ariadne::neural_version);
}
