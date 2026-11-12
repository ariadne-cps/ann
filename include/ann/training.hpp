/***************************************************************************
 *            ann/training.hpp
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

/** @file ann/training.hpp
 *  @brief Sequential native-CPU training of frozen-snapshot neural models.
 */
#ifndef ARIADNE_ANN_TRAINING_HPP
#define ARIADNE_ANN_TRAINING_HPP

#include <cstddef>
#include <cstdint>
#include <vector>

#include "ann/network.hpp"

namespace Ariadne {

/** @brief Loss summed over output components and averaged over samples. */
enum class NeuralLossKind {
    MeanSquaredError,            ///< 0.5 * sum_outputs (prediction - target)^2.
    BinaryCrossEntropyWithLogits ///< Independent binary labels, logits output.
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
    double momentum = 0.0;       ///< SGD: in [0,1). AdamW: must be zero.
    double weight_decay = 0.0;   ///< Nonnegative. Coupled in SGD; decoupled in AdamW.
    double beta1 = 0.9;          ///< AdamW first-moment factor in [0,1).
    double beta2 = 0.999;        ///< AdamW second-moment factor in [0,1).
    double epsilon = 1e-8;       ///< AdamW positive denominator offset.
};

/** @brief Immutable-by-convention complete native-CPU training state.
 *
 *  Holds model snapshot, optimizer hyperparameters, loss, trainability policy,
 *  completed optimizer step count, and exact moment buffer values. Import
 *  through NeuralTrainingSession to enforce consistency. This is a value
 *  representation, not a certificate of training provenance.
 */
struct NeuralTrainingCheckpoint {
    NeuralParameterSnapshot snapshot;
    NeuralOptimizerOptions options;
    NeuralLossKind loss;
    std::vector<bool> trainable_layers;
    std::vector<double> first_moment;
    std::vector<double> second_moment;
    std::size_t steps;
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
    /** @brief Create a mutable session with an optional per-layer trainability mask.
     *
     *  An empty mask means every layer is trainable. Otherwise there must be
     *  exactly one boolean per layer; at least one must be trainable. Frozen
     *  layers retain their exact binary64 values even with weight decay.
     *  The mask cannot change within a session: to unfreeze layers, construct
     *  a new session from export_snapshot(), resetting optimizer moments.
     */
    NeuralTrainingSession(NeuralParameterSnapshot initial,
                          NeuralOptimizerOptions options,
                          NeuralLossKind loss = NeuralLossKind::MeanSquaredError,
                          std::vector<bool> trainable_layers = {});

    /** @brief Restore an entire checkpoint, including optimizer moments.
     *  Rejects malformed dimensions, nonfinite moments, invalid optimizer
     *  settings, and nonzero moments for frozen parameters. In particular,
     *  this is distinct from constructing a fresh session from a snapshot.
     */
    explicit NeuralTrainingSession(NeuralTrainingCheckpoint checkpoint);

    /** @brief Number of completed optimizer steps, not training examples. */
    std::size_t steps() const noexcept;
    /** @brief Per-layer update mask (true means trainable). */
    const std::vector<bool>& trainable_layers() const noexcept;
    /** @brief Export a new, independent, immutable parameter snapshot. */
    NeuralParameterSnapshot export_snapshot() const;
    /** @brief Export the complete optimizer state for exact resumption. */
    NeuralTrainingCheckpoint export_checkpoint() const;
    /** @brief Calculate sample-mean objective without updating anything. */
    double loss_batch(const std::vector<std::vector<double>>& inputs,
                      const std::vector<std::vector<double>>& targets) const;
    /** @brief Evaluate loss and apply exactly one step; return pre-update loss. */
    double train_batch(const std::vector<std::vector<double>>& inputs,
                       const std::vector<std::vector<double>>& targets);

  private:
    NeuralParameterSnapshot snapshot_;
    NeuralOptimizerOptions options_;
    NeuralLossKind loss_;
    std::vector<bool> trainable_layers_;
    std::vector<bool> trainable_parameters_;
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

/** @brief Return a new snapshot with the final dense layer replaced.
 *
 *  Every preceding layer, including its exact binary64 parameter bits,
 *  remains unchanged. The new output head is Xavier-uniform initialized
 *  using the supplied seed with zero biases. Changing the output layer
 *  creates a new model identity; no optimizer state is transferred.
 */
NeuralParameterSnapshot neural_replace_output_head(
    const NeuralParameterSnapshot& source,
    NeuralDenseLayerSpec new_head,
    std::uint64_t seed);

/** @brief Return seeded Fisher-Yates permutation of 0..count-1. */
std::vector<std::size_t> neural_shuffled_indices(std::size_t count,
                                                 std::uint64_t seed);

} // namespace Ariadne
#endif // ARIADNE_ANN_TRAINING_HPP
