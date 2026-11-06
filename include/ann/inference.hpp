/***************************************************************************
 *            ann/inference.hpp
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

#ifndef ARIADNE_ANN_CPU_INFERENCE_HPP
#define ARIADNE_ANN_CPU_INFERENCE_HPP
#include "ann/network.hpp"
#include "ann/derivatives.hpp"
#include <vector>
namespace Ariadne::NeuralCpu {
std::vector<double> infer(NeuralParameterSnapshot const& snapshot,
                          std::vector<double> const& input);
std::vector<std::vector<double>> infer_batch(
    NeuralParameterSnapshot const& snapshot,
    std::vector<std::vector<double>> const& inputs);
NeuralGradientResult differentiate(
    NeuralParameterSnapshot const& snapshot,
    std::vector<double> const& input,
    std::vector<double> const& output_cotangent);
std::vector<std::vector<double>> input_jacobian(
    NeuralParameterSnapshot const& snapshot,
    std::vector<double> const& input);
NeuralBatchGradientResult differentiate_batch(
    NeuralParameterSnapshot const& snapshot,
    std::vector<std::vector<double>> const& inputs,
    std::vector<std::vector<double>> const& output_cotangents,
    NeuralGradientReduction reduction);
} // namespace Ariadne::NeuralCpu
#endif // ARIADNE_ANN_CPU_INFERENCE_HPP
