/***************************************************************************
 *            src/inference.cpp
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

#include "neural/inference.hpp"

#include <cmath>
#include <stdexcept>
#include <utility>

namespace Ariadne::NeuralCpu {
namespace {

double sigmoid(double x) {
    if (x >= 0.0) return 1.0 / (1.0 + std::exp(-x));
    const double exp_x = std::exp(x);
    return exp_x / (1.0 + exp_x);
}

double activate(NeuralActivation activation, double x) {
    switch (activation) {
        case NeuralActivation::Identity: return x;
        case NeuralActivation::Relu: return x > 0.0 ? x : 0.0;
        case NeuralActivation::Tanh: return std::tanh(x);
        case NeuralActivation::Sigmoid: return sigmoid(x);
        case NeuralActivation::Softplus:
            if (x > 0.0) return x + std::log1p(std::exp(-x));
            return std::log1p(std::exp(x));
        default: throw std::invalid_argument("Unknown neural activation");
    }
}

double activation_derivative(NeuralActivation activation,
                             double preactivation, double activated) {
    switch (activation) {
        case NeuralActivation::Identity: return 1.0;
        case NeuralActivation::Relu: return preactivation > 0.0 ? 1.0 : 0.0;
        case NeuralActivation::Tanh: return 1.0 - activated * activated;
        case NeuralActivation::Sigmoid: return activated * (1.0 - activated);
        case NeuralActivation::Softplus: return sigmoid(preactivation);
        default: throw std::invalid_argument("Unknown neural activation");
    }
}

double finite_product(double lhs, double rhs) {
    const double value = lhs * rhs;
    if (!std::isfinite(value))
        throw std::overflow_error("Non-finite neural gradient product");
    return value;
}

double finite_fma(double lhs, double rhs, double initial) {
    const double value = std::fma(lhs, rhs, initial);
    if (!std::isfinite(value))
        throw std::overflow_error("Non-finite neural arithmetic result");
    return value;
}

void validate_input(NeuralNetwork const& network,
                    std::vector<double> const& input) {
    if (input.size() != network.input_dimension())
        throw std::invalid_argument("Wrong number of neural input features");
    for (double x : input)
        if (!std::isfinite(x))
            throw std::invalid_argument("Neural inputs must be finite");
}

void validate_cotangent(NeuralNetwork const& network,
                        std::vector<double> const& cotangent) {
    if (cotangent.size() != network.output_dimension())
        throw std::invalid_argument("Wrong number of output cotangent features");
    for (double x : cotangent)
        if (!std::isfinite(x))
            throw std::invalid_argument("Neural output cotangents must be finite");
}

void validate_reduction(NeuralGradientReduction reduction) {
    switch (reduction) {
        case NeuralGradientReduction::Sum:
        case NeuralGradientReduction::Mean: return;
        default: throw std::invalid_argument("Unknown neural gradient reduction");
    }
}

struct ForwardTrace {
    std::vector<std::vector<double>> activations;
    std::vector<std::vector<double>> preactivations;
};

/** @brief Shared forward kernel; optionally cache intermediates for reverse mode. */
std::vector<double> forward(NeuralParameterSnapshot const& snapshot,
                            std::vector<double> const& input,
                            ForwardTrace* trace = nullptr) {
    std::vector<double> current = input;
    if (trace) {
        trace->activations.reserve(snapshot.network().layers().size() + 1);
        trace->preactivations.reserve(snapshot.network().layers().size());
        trace->activations.push_back(input);
    }
    auto const& parameters = snapshot.values();
    for (auto const& layer : snapshot.network().layers()) {
        std::vector<double> next(layer.outputs);
        std::vector<double> preactivation;
        if (trace) preactivation.resize(layer.outputs);
        for (std::size_t output = 0; output < layer.outputs; ++output) {
            double sum = parameters[layer.biases_offset + output];
            const std::size_t offset = layer.weights_offset + output * layer.inputs;
            for (std::size_t feature = 0; feature < layer.inputs; ++feature)
                sum = finite_fma(parameters[offset + feature], current[feature], sum);
            if (trace) preactivation[output] = sum;
            next[output] = activate(layer.activation, sum);
            if (!std::isfinite(next[output]))
                throw std::overflow_error("Non-finite neural activation result");
        }
        if (trace) {
            trace->preactivations.push_back(std::move(preactivation));
            trace->activations.push_back(next);
        }
        current = std::move(next);
    }
    return current;
}

/** @brief Unchecked dimensions only: callers have validated the sample. */
NeuralGradientResult differentiate_validated(
    NeuralParameterSnapshot const& snapshot,
    std::vector<double> const& input,
    std::vector<double> const& cotangent) {
    ForwardTrace trace;
    (void)forward(snapshot, input, &trace);
    auto const& layers = snapshot.network().layers();
    auto const& parameters = snapshot.values();

    NeuralGradientResult result{
        std::vector<double>(snapshot.network().input_dimension(), 0.0),
        std::vector<double>(snapshot.network().parameter_count(), 0.0)
    };
    std::vector<double> upstream = cotangent;
    for (std::size_t index = layers.size(); index-- > 0;) {
        auto const& layer = layers[index];
        std::vector<double> previous(layer.inputs, 0.0);
        for (std::size_t o = 0; o < layer.outputs; ++o) {
            const double local = activation_derivative(
                layer.activation, trace.preactivations[index][o],
                trace.activations[index + 1][o]);
            const double delta = finite_product(upstream[o], local);
            result.parameter_gradient[layer.biases_offset + o] = delta;
            const std::size_t weight_offset = layer.weights_offset + o * layer.inputs;
            for (std::size_t i = 0; i < layer.inputs; ++i) {
                result.parameter_gradient[weight_offset + i] =
                    finite_product(delta, trace.activations[index][i]);
                previous[i] = finite_fma(
                    parameters[weight_offset + i], delta, previous[i]);
            }
        }
        upstream = std::move(previous);
    }
    result.input_gradient = std::move(upstream);
    return result;
}

} // namespace

std::vector<double> infer(NeuralParameterSnapshot const& snapshot,
                          std::vector<double> const& input) {
    validate_input(snapshot.network(), input);
    return forward(snapshot, input);
}

std::vector<std::vector<double>> infer_batch(
    NeuralParameterSnapshot const& snapshot,
    std::vector<std::vector<double>> const& inputs) {
    for (auto const& input : inputs) validate_input(snapshot.network(), input);
    std::vector<std::vector<double>> result;
    result.reserve(inputs.size());
    for (auto const& input : inputs)
        result.push_back(forward(snapshot, input));
    return result;
}

NeuralGradientResult differentiate(
    NeuralParameterSnapshot const& snapshot,
    std::vector<double> const& input,
    std::vector<double> const& output_cotangent) {
    validate_input(snapshot.network(), input);
    validate_cotangent(snapshot.network(), output_cotangent);
    return differentiate_validated(snapshot, input, output_cotangent);
}

std::vector<std::vector<double>> input_jacobian(
    NeuralParameterSnapshot const& snapshot,
    std::vector<double> const& input) {
    validate_input(snapshot.network(), input);
    std::vector<std::vector<double>> jacobian;
    jacobian.reserve(snapshot.network().output_dimension());
    std::vector<double> cotangent(snapshot.network().output_dimension(), 0.0);
    for (std::size_t o = 0; o < cotangent.size(); ++o) {
        cotangent[o] = 1.0;
        jacobian.push_back(differentiate_validated(
            snapshot, input, cotangent).input_gradient);
        cotangent[o] = 0.0;
    }
    return jacobian;
}

NeuralBatchGradientResult differentiate_batch(
    NeuralParameterSnapshot const& snapshot,
    std::vector<std::vector<double>> const& inputs,
    std::vector<std::vector<double>> const& output_cotangents,
    NeuralGradientReduction reduction) {
    validate_reduction(reduction);
    if (inputs.size() != output_cotangents.size())
        throw std::invalid_argument("Mismatched neural gradient batch sizes");
    if (inputs.empty() && reduction == NeuralGradientReduction::Mean)
        throw std::invalid_argument("Cannot mean-reduce an empty neural batch");
    for (std::size_t n = 0; n < inputs.size(); ++n) {
        validate_input(snapshot.network(), inputs[n]);
        validate_cotangent(snapshot.network(), output_cotangents[n]);
    }
    NeuralBatchGradientResult result{
        {}, std::vector<double>(snapshot.network().parameter_count(), 0.0)
    };
    result.input_gradients.reserve(inputs.size());
    for (std::size_t n = 0; n < inputs.size(); ++n) {
        auto gradient = differentiate_validated(
            snapshot, inputs[n], output_cotangents[n]);
        result.input_gradients.push_back(std::move(gradient.input_gradient));
        for (std::size_t p = 0; p < result.parameter_gradient.size(); ++p)
            result.parameter_gradient[p] = finite_fma(
                1.0, gradient.parameter_gradient[p], result.parameter_gradient[p]);
    }
    if (reduction == NeuralGradientReduction::Mean) {
        const double divisor = static_cast<double>(inputs.size());
        for (auto& row : result.input_gradients)
            for (double& value : row) value /= divisor;
        for (double& value : result.parameter_gradient) value /= divisor;
    }
    return result;
}

} // namespace Ariadne::NeuralCpu
