/***************************************************************************
 *            src/network.cpp
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

#include "neural/network.hpp"

#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>

namespace Ariadne {
namespace {

std::size_t checked_add(std::size_t first, std::size_t second) {
    if (first > std::numeric_limits<std::size_t>::max() - second) {
        throw std::overflow_error("Neural parameter offset overflow");
    }
    return first + second;
}

// The caller guarantees second != 0: every layer output count is validated
// positive before this is called, and the input dimension is positive too.
std::size_t checked_multiply(std::size_t first, std::size_t second) {
    if (first > std::numeric_limits<std::size_t>::max() / second) {
        throw std::overflow_error("Neural parameter count overflow");
    }
    return first * second;
}

} // namespace

std::string_view neural_activation_name(NeuralActivation activation) {
    switch (activation) {
        case NeuralActivation::Identity: return "identity";
        case NeuralActivation::Relu: return "relu";
        case NeuralActivation::Tanh: return "tanh";
        case NeuralActivation::Sigmoid: return "sigmoid";
        case NeuralActivation::Softplus: return "softplus";
        default: throw std::invalid_argument("Unknown neural activation");
    }
}

NeuralNetwork::NeuralNetwork(
    std::size_t input_dimension, std::vector<NeuralDenseLayerSpec> layers)
    : input_dimension_(input_dimension),
      output_dimension_(input_dimension),
      parameter_count_(0) {
    if (input_dimension == 0 || layers.empty()) {
        throw std::invalid_argument("A neural network requires inputs and layers");
    }
    layers_.reserve(layers.size());
    std::size_t inputs = input_dimension;
    for (auto const& layer : layers) {
        if (layer.outputs == 0) {
            throw std::invalid_argument("Neural layers require positive dimensions");
        }
        (void)neural_activation_name(layer.activation);
        const std::size_t weights = checked_multiply(inputs, layer.outputs);
        const std::size_t biases_offset = checked_add(parameter_count_, weights);
        const std::size_t next_offset = checked_add(biases_offset, layer.outputs);
        layers_.push_back(
            NeuralDenseLayerLayout{inputs, layer.outputs, layer.activation,
                                   parameter_count_, biases_offset});
        parameter_count_ = next_offset;
        inputs = layer.outputs;
    }
    output_dimension_ = inputs;
}

std::size_t NeuralNetwork::input_dimension() const noexcept {
    return input_dimension_;
}
std::size_t NeuralNetwork::output_dimension() const noexcept {
    return output_dimension_;
}
std::size_t NeuralNetwork::parameter_count() const noexcept {
    return parameter_count_;
}
std::vector<NeuralDenseLayerLayout> const& NeuralNetwork::layers() const noexcept {
    return layers_;
}

NeuralParameterSnapshot::NeuralParameterSnapshot(
    NeuralNetwork network, std::vector<double> values)
    : network_(std::move(network)), values_(std::move(values)) {
    if (values_.size() != network_.parameter_count()) {
        throw std::invalid_argument("Wrong number of neural parameters");
    }
    for (double value : values_) {
        if (!std::isfinite(value)) {
            throw std::invalid_argument("Neural parameters must be finite");
        }
    }
}

NeuralNetwork const& NeuralParameterSnapshot::network() const noexcept {
    return network_;
}
std::vector<double> const& NeuralParameterSnapshot::values() const noexcept {
    return values_;
}
std::vector<double> NeuralParameterSnapshot::export_values() const {
    return values_;
}

} // namespace Ariadne
