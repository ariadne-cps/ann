/***************************************************************************
 *            python/bindings/ann_submodule.cpp
 *
 *  Copyright  2026  Luca Geretti
 *
 ****************************************************************************/

/*
 *  This file is part of Ariadne ANN.
 *
 *  Ariadne ANN is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 3 of the License, or
 *  (at your option) any later version.
 *
 *  Ariadne ANN is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with Ariadne ANN.  If not, see <https://www.gnu.org/licenses/>.
 */

#include "pybind11.hpp"

#include "ann/ann.hpp"
#include "ann/network.hpp"
#include "ann/backend.hpp"
#include "ann/training.hpp"
#include "ann/serialization.hpp"

#include <cstddef>
#include <vector>

using namespace Ariadne;

void ann_submodule(pybind11::module& module) {
    module.def("ann_version", &ann_version);

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

    pybind11::enum_<NeuralGradientReduction>(module, "NeuralGradientReduction")
        .value("Sum", NeuralGradientReduction::Sum)
        .value("Mean", NeuralGradientReduction::Mean);

    pybind11::class_<NeuralGradientResult>(module, "NeuralGradientResult")
        .def_readonly("input_gradient", &NeuralGradientResult::input_gradient)
        .def_readonly("parameter_gradient", &NeuralGradientResult::parameter_gradient);

    pybind11::class_<NeuralBatchGradientResult>(module, "NeuralBatchGradientResult")
        .def_readonly("input_gradients", &NeuralBatchGradientResult::input_gradients)
        .def_readonly("parameter_gradient", &NeuralBatchGradientResult::parameter_gradient);

    pybind11::enum_<NeuralLossKind>(module, "NeuralLossKind")
        .value("MeanSquaredError", NeuralLossKind::MeanSquaredError)
        .value("BinaryCrossEntropyWithLogits", NeuralLossKind::BinaryCrossEntropyWithLogits);

    pybind11::enum_<NeuralOptimizerKind>(module, "NeuralOptimizerKind")
        .value("Sgd", NeuralOptimizerKind::Sgd)
        .value("AdamW", NeuralOptimizerKind::AdamW);

    pybind11::class_<NeuralOptimizerOptions>(module, "NeuralOptimizerOptions")
        .def(pybind11::init<>())
        .def_readwrite("kind", &NeuralOptimizerOptions::kind)
        .def_readwrite("learning_rate", &NeuralOptimizerOptions::learning_rate)
        .def_readwrite("momentum", &NeuralOptimizerOptions::momentum)
        .def_readwrite("weight_decay", &NeuralOptimizerOptions::weight_decay)
        .def_readwrite("beta1", &NeuralOptimizerOptions::beta1)
        .def_readwrite("beta2", &NeuralOptimizerOptions::beta2)
        .def_readwrite("epsilon", &NeuralOptimizerOptions::epsilon);

    pybind11::class_<NeuralTrainingCheckpoint>(module, "NeuralTrainingCheckpoint")
        .def_readonly("snapshot", &NeuralTrainingCheckpoint::snapshot)
        .def_readonly("options", &NeuralTrainingCheckpoint::options)
        .def_readonly("loss", &NeuralTrainingCheckpoint::loss)
        .def_readonly("trainable_layers", &NeuralTrainingCheckpoint::trainable_layers)
        .def_readonly("first_moment", &NeuralTrainingCheckpoint::first_moment)
        .def_readonly("second_moment", &NeuralTrainingCheckpoint::second_moment)
        .def_readonly("steps", &NeuralTrainingCheckpoint::steps);

    pybind11::class_<NeuralTrainingSession>(module, "NeuralTrainingSession")
        .def(pybind11::init<NeuralParameterSnapshot, NeuralOptimizerOptions, NeuralLossKind,
                            std::vector<bool>>(),
             pybind11::arg("initial"), pybind11::arg("options"),
             pybind11::arg("loss") = NeuralLossKind::MeanSquaredError,
             pybind11::arg("trainable_layers") = std::vector<bool>{})
        .def(pybind11::init<NeuralTrainingCheckpoint>(),
             pybind11::arg("checkpoint"))
        .def("steps", &NeuralTrainingSession::steps)
        .def("trainable_layers", &NeuralTrainingSession::trainable_layers)
        .def("export_snapshot", &NeuralTrainingSession::export_snapshot)
        .def("export_checkpoint", &NeuralTrainingSession::export_checkpoint)
        .def("loss_batch", &NeuralTrainingSession::loss_batch,
             pybind11::arg("inputs"), pybind11::arg("targets"))
        .def("train_batch", &NeuralTrainingSession::train_batch,
             pybind11::arg("inputs"), pybind11::arg("targets"));

    module.def("neural_initialize_parameters", &neural_initialize_parameters,
               pybind11::arg("network"), pybind11::arg("seed"));
    module.def("neural_replace_output_head", &neural_replace_output_head,
               pybind11::arg("source"), pybind11::arg("new_head"),
               pybind11::arg("seed"));
    module.def("neural_shuffled_indices", &neural_shuffled_indices,
               pybind11::arg("count"), pybind11::arg("seed"));

    module.def("neural_snapshot_sha256", &neural_snapshot_sha256,
               pybind11::arg("snapshot"));
    module.def("neural_save_snapshot", &neural_save_snapshot,
               pybind11::arg("snapshot"), pybind11::arg("manifest_path"),
               pybind11::arg("safetensors_path"));
    module.def("neural_load_snapshot", &neural_load_snapshot,
               pybind11::arg("manifest_path"), pybind11::arg("safetensors_path"));
    module.def("neural_save_training_checkpoint", &neural_save_training_checkpoint,
               pybind11::arg("session"), pybind11::arg("manifest_path"),
               pybind11::arg("safetensors_path"), pybind11::arg("optimizer_path"));
    module.def("neural_load_training_checkpoint", &neural_load_training_checkpoint,
               pybind11::arg("manifest_path"), pybind11::arg("safetensors_path"),
               pybind11::arg("optimizer_path"));
    module.def("neural_save_snapshot_atomic", &neural_save_snapshot_atomic,
               pybind11::arg("snapshot"), pybind11::arg("directory"));
    module.def("neural_load_snapshot_atomic", &neural_load_snapshot_atomic,
               pybind11::arg("directory"));
    module.def("neural_save_training_checkpoint_atomic",
               &neural_save_training_checkpoint_atomic,
               pybind11::arg("session"), pybind11::arg("directory"));
    module.def("neural_load_training_checkpoint_atomic",
               &neural_load_training_checkpoint_atomic,
               pybind11::arg("directory"));

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
        .def("infer", &NeuralExecutionSession::infer, pybind11::arg("input"))
        .def("infer_batch", &NeuralExecutionSession::infer_batch, pybind11::arg("inputs"))
        .def("differentiate", &NeuralExecutionSession::differentiate,
             pybind11::arg("input"), pybind11::arg("output_cotangent"))
        .def("input_jacobian", &NeuralExecutionSession::input_jacobian,
             pybind11::arg("input"))
        .def("differentiate_batch", &NeuralExecutionSession::differentiate_batch,
             pybind11::arg("inputs"), pybind11::arg("output_cotangents"),
             pybind11::arg("reduction") = NeuralGradientReduction::Sum)
        .def("supports", &NeuralExecutionSession::supports)
        .def("require", &NeuralExecutionSession::require);
}
