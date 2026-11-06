/***************************************************************************
 *            ann/derivatives.hpp
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

/** @file ann/derivatives.hpp
 *  @brief Reverse-mode gradients of a scalarized neural output and batch reductions.
 */
#ifndef ARIADNE_ANN_DERIVATIVES_HPP
#define ARIADNE_ANN_DERIVATIVES_HPP

#include <vector>

namespace Ariadne {

/** @brief Reduction over independently scalarized sample outputs. */
enum class NeuralGradientReduction {
    Sum,  ///< Sum sample objectives and all corresponding gradients.
    Mean  ///< Divide the sum by the nonzero number of samples.
};

/** @brief Gradient of output_cotangent dot network(input).
 *
 *  input_gradient has input_dimension entries. parameter_gradient has
 *  parameter_count entries, in the frozen snapshot's canonical layout.
 */
struct NeuralGradientResult {
    std::vector<double> input_gradient;
    std::vector<double> parameter_gradient;
};

/** @brief Batched VJP results for a sum or mean of per-sample objectives.
 *
 *  Each input_gradients row and the parameter_gradient are scaled by the
 *  requested reduction. No training state or optimizer is stored.
 */
struct NeuralBatchGradientResult {
    std::vector<std::vector<double>> input_gradients;
    std::vector<double> parameter_gradient;
};

} // namespace Ariadne
#endif // ARIADNE_ANN_DERIVATIVES_HPP
