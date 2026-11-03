/***************************************************************************
 *            src/cpu/inference.cpp
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

#include "inference.hpp"
#include <cmath>
#include <stdexcept>
#include <utility>

namespace Ariadne::NeuralCpu {
namespace {
double activate(NeuralActivation activation, double x) {
    switch (activation) {
        case NeuralActivation::Identity: return x;
        case NeuralActivation::Relu: return x > 0.0 ? x : 0.0;
        case NeuralActivation::Tanh: return std::tanh(x);
        case NeuralActivation::Sigmoid:
            if (x >= 0.0) return 1.0 / (1.0 + std::exp(-x));
            else {
                const double e = std::exp(x);
                return e / (1.0 + e);
            }
        case NeuralActivation::Softplus:
            if (x > 0.0) return x + std::log1p(std::exp(-x));
            else return std::log1p(std::exp(x));
        default: throw std::invalid_argument("Unknown neural activation");
    }
}

void validate(NeuralNetwork const& network, std::vector<double> const& input) {
    if (input.size() != network.input_dimension())
        throw std::invalid_argument("Wrong number of neural input features");
    for (double x : input)
        if (!std::isfinite(x)) throw std::invalid_argument("Neural inputs must be finite");
}

std::vector<double> forward(NeuralParameterSnapshot const& snapshot,
                            std::vector<double> const& input) {
    std::vector<double> current = input;
    auto const& parameters = snapshot.values();
    for (auto const& layer : snapshot.network().layers()) {
        std::vector<double> next(layer.outputs);
        for (std::size_t output = 0; output < layer.outputs; ++output) {
            double sum = parameters[layer.biases_offset + output];
            const std::size_t offset = layer.weights_offset + output * layer.inputs;
            for (std::size_t feature = 0; feature < layer.inputs; ++feature) {
                sum = std::fma(parameters[offset + feature], current[feature], sum);
                if (!std::isfinite(sum))
                    throw std::overflow_error("Non-finite neural affine result");
            }
            next[output] = activate(layer.activation, sum);
            if (!std::isfinite(next[output]))
                throw std::overflow_error("Non-finite neural activation result");
        }
        current = std::move(next);
    }
    return current;
}
} // namespace

std::vector<double> infer(NeuralParameterSnapshot const& snapshot,
                          std::vector<double> const& input) {
    validate(snapshot.network(), input);
    return forward(snapshot, input);
}
std::vector<std::vector<double>> infer_batch(
    NeuralParameterSnapshot const& snapshot,
    std::vector<std::vector<double>> const& inputs) {
    for (auto const& row : inputs) validate(snapshot.network(), row);
    std::vector<std::vector<double>> result;
    result.reserve(inputs.size());
    for (auto const& row : inputs) result.push_back(forward(snapshot, row));
    return result;
}
} // namespace Ariadne::NeuralCpu
