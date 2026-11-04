/***************************************************************************
 *            src/validated.cpp
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

#include "neural/validated.hpp"
#include "numeric/floats.hpp"

#include <cmath>
#include <stdexcept>
#include <utility>
#include <vector>

namespace Ariadne {
namespace {

using BoundsType = FloatDPBounds;

BoundsType exact(double value) {
    // FloatDP holds an exact copy of the binary64 value. We must not round
    // an ordinary decimal literal or convert a value via a string.
    return BoundsType(FloatDP(value));
}

BoundsType checked(BoundsType value) {
    if (!std::isfinite(value.lower_raw().get_d()) ||
        !std::isfinite(value.upper_raw().get_d()) ||
        value.lower_raw().get_d() > value.upper_raw().get_d())
        throw std::overflow_error("Nonfinite or invalid validated neural enclosure");
    return value;
}

BoundsType activate(BoundsType const& value, NeuralActivation activation) {
    switch (activation) {
        case NeuralActivation::Identity: return value;
        case NeuralActivation::Relu: return max(value, exact(0.0));
        case NeuralActivation::Tanh: return tanh(value);
        case NeuralActivation::Sigmoid: return sigmoid(value);
        case NeuralActivation::Softplus: return softplus(value);
        default: throw std::invalid_argument("Unsupported validated neural activation");
    }
}

} // namespace

std::vector<NeuralInterval> neural_validated_bounds(
    NeuralParameterSnapshot const& snapshot,
    std::vector<NeuralInterval> const& inputs) {
    auto const& network = snapshot.network();
    if (inputs.size() != network.input_dimension())
        throw std::invalid_argument("Wrong validated neural input dimension");

    std::vector<BoundsType> state;
    state.reserve(inputs.size());
    for (auto const& ivl : inputs) {
        if (!std::isfinite(ivl.lower) || !std::isfinite(ivl.upper) ||
            ivl.lower > ivl.upper)
            throw std::invalid_argument("Invalid validated neural input interval");
        state.emplace_back(FloatDP(ivl.lower), FloatDP(ivl.upper));
    }

    auto const& parameters = snapshot.values();
    for (auto const& layer : network.layers()) {
        std::vector<BoundsType> next;
        next.reserve(layer.outputs);
        for (std::size_t o = 0; o < layer.outputs; ++o) {
            auto value = exact(parameters[layer.biases_offset + o]);
            for (std::size_t i = 0; i < layer.inputs; ++i) {
                auto const weight =
                    exact(parameters[layer.weights_offset + o*layer.inputs + i]);
                value = checked(value + checked(weight * state[i]));
            }
            next.push_back(checked(activate(value, layer.activation)));
        }
        state.swap(next);
    }
    std::vector<NeuralInterval> result;
    result.reserve(state.size());
    for (auto const& bounds : state)
        result.push_back({bounds.lower_raw().get_d(),
                          bounds.upper_raw().get_d()});
    return result;
}

} // namespace Ariadne
