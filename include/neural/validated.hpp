/***************************************************************************
 *            neural/validated.hpp
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

/** @file neural/validated.hpp
 *  @brief Public binary64 endpoint type for rigorously enclosed inference.
 */
#ifndef ARIADNE_NEURAL_VALIDATED_HPP
#define ARIADNE_NEURAL_VALIDATED_HPP

#include "neural/network.hpp"

#include <vector>

namespace Ariadne {

/** @brief Closed interval with exactly represented binary64 endpoints.
 *
 *  The input interval includes every real between its finite endpoints.
 *  Output intervals from the validated evaluator are rigorous enclosures,
 *  not approximate predictions. All returned endpoints are finite; numerical
 *  overflow is an error rather than a purported finite certificate.
 */
struct NeuralInterval {
    double lower;
    double upper;
};

/** @brief Validated interval-bound propagation using Ariadne Numeric.
 *
 *  Converts all model weights, biases and input endpoints exactly from
 *  binary64 into FloatDPBounds. Operations use outward rounding, including
 *  monotone activation ranges (Identity, Relu, Tanh, Sigmoid and Softplus).
 *  This is natural interval propagation, not a tight global optimizer.
 *
 *  @throws std::invalid_argument for wrong input shape, NaNs, infinities
 *  or reversed endpoints.
 *  @throws std::overflow_error if any intermediate bound becomes nonfinite.
 */
std::vector<NeuralInterval> neural_validated_bounds(
    NeuralParameterSnapshot const& snapshot,
    std::vector<NeuralInterval> const& inputs);

} // namespace Ariadne

#endif // ARIADNE_NEURAL_VALIDATED_HPP
