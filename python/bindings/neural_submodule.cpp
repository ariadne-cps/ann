/***************************************************************************
 *            python/bindings/neural_submodule.cpp
 *
 *  Copyright  2026  Luca Geretti
 *
 ****************************************************************************/

/*
 *  This file is part of Ariadne Neural.
 *
 *  Ariadne Neural is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 3 of the License, or
 *  (at your option) any later version.
 *
 *  Ariadne Neural is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with Ariadne Neural.  If not, see <https://www.gnu.org/licenses/>.
 */

#include "pybind11.hpp"

#include "neural/neural.hpp"
#include "neural/network.hpp"
#include "neural/backend.hpp"

#include <cstddef>
#include <vector>

using namespace Ariadne;

void neural_submodule(pybind11::module& module) {
    module.def("neural_version", &neural_version);

    pybind11::enum_<NeuralActivation>(module, "NeuralActivation")
        .value("Identity", NeuralActivation::Identity)
        .value("Relu", NeuralActivation::Relu)
        .value("Tanh", NeuralActivation::Tanh)
        .value("Sigmoid", NeuralActivation::Sigmoid)
        .value("Softplus", NeuralActivation::Softplus);

    pybind11::class_<NeuralDenseLayerSpec>(module, "NeuralDenseLayerSpec")
        .def(pybind11::init([](std::size_t outputs, NeuralActivation activation) {
            return NeuralDenseLayerSpec{outputs, activation};
        }), pybind11::arg("outputs"), pybind11::arg("activation"))
        .def_readonly("outputs", &NeuralDenseLayerSpec::outputs)
        .def_readonly("activation", &NeuralDenseLayerSpec::activation);

    pybind11::class_<NeuralDenseLayerLayout>(module, "NeuralDenseLayerLayout")
        .def_readonly("inputs", &NeuralDenseLayerLayout::inputs)
        .def_readonly("outputs", &NeuralDenseLayerLayout::outputs)
        .def_readonly("activation", &NeuralDenseLayerLayout::activation)
        .def_readonly("weights_offset", &NeuralDenseLayerLayout::weights_offset)
        .def_readonly("biases_offset", &NeuralDenseLayerLayout::biases_offset);

    pybind11::class_<NeuralNetwork>(module, "NeuralNetwork")
        .def(pybind11::init<std::size_t, std::vector<NeuralDenseLayerSpec>>(),
             pybind11::arg("input_dimension"), pybind11::arg("layers"))
        .def("input_dimension", &NeuralNetwork::input_dimension)
        .def("output_dimension", &NeuralNetwork::output_dimension)
        .def("parameter_count", &NeuralNetwork::parameter_count)
        .def("layers", &NeuralNetwork::layers);

    pybind11::class_<NeuralParameterSnapshot>(module, "NeuralParameterSnapshot")
        .def(pybind11::init<NeuralNetwork, std::vector<double>>(),
             pybind11::arg("network"), pybind11::arg("values"))
        .def("network", &NeuralParameterSnapshot::network,
             pybind11::return_value_policy::reference_internal)
        .def("values", &NeuralParameterSnapshot::export_values)
        .def("export_values", &NeuralParameterSnapshot::export_values);

    pybind11::enum_<NeuralBackendKind>(module, "NeuralBackendKind")
        .value("NativeCpu", NeuralBackendKind::NativeCpu)
        .value("AriadneValidated", NeuralBackendKind::AriadneValidated)
        .value("LibTorch", NeuralBackendKind::LibTorch)
        .value("Mlx", NeuralBackendKind::Mlx);

    pybind11::enum_<NeuralOperation>(module, "NeuralOperation")
        .value("ImportSnapshot", NeuralOperation::ImportSnapshot)
        .value("Inference", NeuralOperation::Inference)
        .value("BatchInference", NeuralOperation::BatchInference)
        .value("InputDerivatives", NeuralOperation::InputDerivatives)
        .value("ParameterGradients", NeuralOperation::ParameterGradients)
        .value("Training", NeuralOperation::Training)
        .value("ValidatedBounds", NeuralOperation::ValidatedBounds);

    module.def("neural_backend_available", &neural_backend_available,
               pybind11::arg("backend"));
    module.def("neural_backend_supports", &neural_backend_supports,
               pybind11::arg("backend"), pybind11::arg("operation"));

    pybind11::class_<NeuralExecutionSession>(module, "NeuralExecutionSession")
        .def(pybind11::init<NeuralParameterSnapshot, NeuralBackendKind>(),
             pybind11::arg("snapshot"),
             pybind11::arg("backend") = NeuralBackendKind::NativeCpu)
        .def("backend", &NeuralExecutionSession::backend)
        .def("snapshot", &NeuralExecutionSession::export_snapshot)
        .def("export_snapshot", &NeuralExecutionSession::export_snapshot)
        .def("supports", &NeuralExecutionSession::supports)
        .def("require", &NeuralExecutionSession::require);
}
