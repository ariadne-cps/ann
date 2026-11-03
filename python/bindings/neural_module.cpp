#include "pybind11.hpp"

void neural_submodule(pybind11::module& module);

void neural_module(pybind11::module& module) {
    neural_submodule(module);
}
