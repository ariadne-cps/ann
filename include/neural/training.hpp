/***************************************************************************
 *            neural/training.hpp
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

/** @file neural/training.hpp
 *  @brief Sequential native-CPU training of frozen-snapshot neural models.
 */
#ifndef ARIADNE_NEURAL_TRAINING_HPP
#define ARIADNE_NEURAL_TRAINING_HPP

#include "neural/network.hpp"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace Ariadne {

/** @brief Loss summed over output components and averaged over samples. */
enum class NeuralLossKind {
    MeanSquaredError,           ///< 0.5 * sum_outputs (prediction - target)^2.
    BinaryCrossEntropyWithLogits///< Independent binary labels, logits output.
};

/** @brief Concrete optimizer update rule, with no implicit schedule. */
enum class NeuralOptimizerKind {
    Sgd,  ///< Coupled L2 decay; optional classical momentum.
    AdamW ///< Decoupled weight decay with bias-corrected Adam moments.
};

/** @brief Optimizer parameters; unused algorithm-specific fields are checked. */
struct NeuralOptimizerOptions {
    NeuralOptimizerKind kind = NeuralOptimizerKind::Sgd;
    double learning_rate = 0.01; ///< Strictly positive, finite.
    double momentum = 0.0; ///< SGD: in [0,1). AdamW: must be zero.
    double weight_decay = 0.0; ///< Nonnegative. Coupled in SGD; decoupled in AdamW.
    double beta1 = 0.9; ///< AdamW first-moment factor in [0,1).
    double beta2 = 0.999; ///< AdamW second-moment factor in [0,1).
    double epsilon = 1e-8; ///< AdamW positive denominator offset.
};

/** @brief CPU mutable workspace, independent of all imported frozen snapshots.
 *
 *  Every training step processes a nonempty rectangular batch sequentially,
 *  computes a sample-mean loss, backpropagates all samples, then performs
 *  one optimizer update. If validation or arithmetic fails, parameters,
 *  moments and step count remain unchanged.
 *
 *  BinaryCrossEntropyWithLogits requires a final Identity activation.
 *  Labels for this loss must be finite within [0,1].
 */
class NeuralTrainingSession {
public:
    NeuralTrainingSession(NeuralParameterSnapshot initial,
                          NeuralOptimizerOptions options,
                          NeuralLossKind loss = NeuralLossKind::MeanSquaredError);

    /** @brief Number of completed optimizer steps, not training examples. */
    std::size_t steps() const noexcept;
    /** @brief Export a new, independent, immutable parameter snapshot. */
    NeuralParameterSnapshot export_snapshot() const;
    /** @brief Calculate sample-mean objective without updating anything. */
    double loss_batch(std::vector<std::vector<double>> const& inputs,
                      std::vector<std::vector<double>> const& targets) const;
    /** @brief Evaluate loss and apply exactly one step; return pre-update loss. */
    double train_batch(std::vector<std::vector<double>> const& inputs,
                       std::vector<std::vector<double>> const& targets);

private:
    NeuralParameterSnapshot snapshot_;
    NeuralOptimizerOptions options_;
    NeuralLossKind loss_;
    std::vector<double> first_moment_;
    std::vector<double> second_moment_;
    std::size_t steps_ = 0;
};

/** @brief Generate reproducible Xavier-uniform finite weights and zero biases.
 *
 *  Uses a fixed SplitMix64 integer generator; binary64 transcendental
 *  rounding may vary slightly across floating-point implementations.
 */
NeuralParameterSnapshot neural_initialize_parameters(NeuralNetwork network,
                                                       std::uint64_t seed);

/** @brief Return seeded Fisher-Yates permutation of 0..count-1. */
std::vector<std::size_t> neural_shuffled_indices(std::size_t count,
                                                 std::uint64_t seed);

} // namespace Ariadne
#endif // ARIADNE_NEURAL_TRAINING_HPP
