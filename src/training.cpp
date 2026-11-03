/***************************************************************************
 *            src/training.cpp
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

#include "neural/training.hpp"
#include "neural/backend.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <utility>

namespace Ariadne {
namespace {

double finite(double x) {
    if (!std::isfinite(x))
        throw std::overflow_error("Non-finite neural training arithmetic");
    return x;
}

void validate_optimizer(NeuralOptimizerOptions const& opt) {
    if (!std::isfinite(opt.learning_rate) || opt.learning_rate <= 0.0
        || !std::isfinite(opt.weight_decay) || opt.weight_decay < 0.0
        || !std::isfinite(opt.momentum) || opt.momentum < 0.0 || opt.momentum >= 1.0
        || !std::isfinite(opt.beta1) || opt.beta1 < 0.0 || opt.beta1 >= 1.0
        || !std::isfinite(opt.beta2) || opt.beta2 < 0.0 || opt.beta2 >= 1.0
        || !std::isfinite(opt.epsilon) || opt.epsilon <= 0.0)
        throw std::invalid_argument("Invalid neural optimizer hyperparameter");
    switch (opt.kind) {
        case NeuralOptimizerKind::Sgd: return;
        case NeuralOptimizerKind::AdamW:
            if (opt.momentum != 0.0)
                throw std::invalid_argument("AdamW does not use SGD momentum");
            return;
        default: throw std::invalid_argument("Unknown neural optimizer");
    }
}

void validate_loss(NeuralNetwork const& model, NeuralLossKind kind) {
    switch (kind) {
        case NeuralLossKind::MeanSquaredError: return;
        case NeuralLossKind::BinaryCrossEntropyWithLogits:
            if (model.layers().back().activation != NeuralActivation::Identity)
                throw std::invalid_argument("BCEWithLogits requires Identity output");
            return;
        default: throw std::invalid_argument("Unknown neural loss");
    }
}

double sigmoid(double x) {
    if (x >= 0.0) return 1.0 / (1.0 + std::exp(-x));
    const double e = std::exp(x);
    return e / (1.0 + e);
}

struct LossEvaluation {
    double mean_loss;
    std::vector<std::vector<double>> output_cotangents;
};

LossEvaluation evaluate(NeuralExecutionSession const& execution,
                        NeuralLossKind kind,
                        std::vector<std::vector<double>> const& inputs,
                        std::vector<std::vector<double>> const& targets,
                        bool with_gradient) {
    if (inputs.empty() || inputs.size() != targets.size())
        throw std::invalid_argument("Neural training batch must be nonempty and matched");
    const std::size_t outputs = execution.snapshot().network().output_dimension();
    for (auto const& row : targets) {
        if (row.size() != outputs)
            throw std::invalid_argument("Wrong neural training target dimension");
        for (double value : row) {
            if (!std::isfinite(value) ||
                (kind == NeuralLossKind::BinaryCrossEntropyWithLogits &&
                 (value < 0.0 || value > 1.0)))
                throw std::invalid_argument("Invalid neural training target");
        }
    }

    auto predictions = execution.infer_batch(inputs);
    std::vector<std::vector<double>> cotangents;
    if (with_gradient) cotangents.reserve(inputs.size());
    double total = 0.0;
    for (std::size_t n = 0; n < inputs.size(); ++n) {
        std::vector<double> cotangent;
        if (with_gradient) cotangent.reserve(outputs);
        for (std::size_t o = 0; o < outputs; ++o) {
            const double y = predictions[n][o], target = targets[n][o];
            double sample_loss = 0.0;
            double derivative = 0.0;
            switch (kind) {
                case NeuralLossKind::MeanSquaredError: {
                    const double diff = finite(y - target);
                    sample_loss = finite(0.5 * finite(diff * diff));
                    derivative = diff;
                    break;
                }
                case NeuralLossKind::BinaryCrossEntropyWithLogits:
                    // Stable softplus(logit) - target*logit, with no cancellation
                    // between two large positive terms.
                    if (y >= 0.0) {
                        sample_loss = finite(
                            finite((1.0 - target) * y) + std::log1p(std::exp(-y)));
                    } else {
                        sample_loss = finite(
                            finite(-target * y) + std::log1p(std::exp(y)));
                    }
                    derivative = finite(sigmoid(y) - target);
                    break;
                default: throw std::invalid_argument("Unknown neural loss");
            }
            total = finite(total + sample_loss);
            if (with_gradient) cotangent.push_back(derivative);
        }
        if (with_gradient) cotangents.push_back(std::move(cotangent));
    }
    const double divisor = static_cast<double>(inputs.size());
    return {finite(total / divisor), std::move(cotangents)};
}

std::uint64_t splitmix64(std::uint64_t& state) noexcept {
    state += UINT64_C(0x9e3779b97f4a7c15);
    std::uint64_t z = state;
    z = (z ^ (z >> 30)) * UINT64_C(0xbf58476d1ce4e5b9);
    z = (z ^ (z >> 27)) * UINT64_C(0x94d049bb133111eb);
    return z ^ (z >> 31);
}

double uniform01(std::uint64_t& state) {
    return static_cast<double>(splitmix64(state) >> 11) *
        (1.0 / 9007199254740992.0);
}

} // namespace

NeuralTrainingSession::NeuralTrainingSession(
    NeuralParameterSnapshot initial,
    NeuralOptimizerOptions options,
    NeuralLossKind loss)
    : snapshot_(std::move(initial)),
      options_(options), loss_(loss),
      first_moment_(snapshot_.network().parameter_count(), 0.0),
      second_moment_(snapshot_.network().parameter_count(), 0.0) {
    validate_optimizer(options_);
    validate_loss(snapshot_.network(), loss_);
}

std::size_t NeuralTrainingSession::steps() const noexcept { return steps_; }

NeuralParameterSnapshot NeuralTrainingSession::export_snapshot() const {
    return snapshot_;
}

double NeuralTrainingSession::loss_batch(
    std::vector<std::vector<double>> const& inputs,
    std::vector<std::vector<double>> const& targets) const {
    NeuralExecutionSession execution(snapshot_);
    return evaluate(execution, loss_, inputs, targets, false).mean_loss;
}

double NeuralTrainingSession::train_batch(
    std::vector<std::vector<double>> const& inputs,
    std::vector<std::vector<double>> const& targets) {
    if (steps_ == std::numeric_limits<std::size_t>::max())
        throw std::overflow_error("Neural optimizer step count overflow");
    NeuralExecutionSession execution(snapshot_);
    auto evaluated = evaluate(execution, loss_, inputs, targets, true);
    auto gradients = execution.differentiate_batch(
        inputs, evaluated.output_cotangents, NeuralGradientReduction::Mean);
    auto updated = snapshot_.export_values();
    auto first = first_moment_;
    auto second = second_moment_;
    const std::size_t next_step = steps_ + 1;

    if (options_.kind == NeuralOptimizerKind::Sgd) {
        for (std::size_t i = 0; i < updated.size(); ++i) {
            const double regularized = finite(
                gradients.parameter_gradient[i] +
                finite(options_.weight_decay * updated[i]));
            first[i] = finite(
                options_.momentum * first[i] + regularized);
            updated[i] = finite(
                updated[i] - finite(options_.learning_rate * first[i]));
        }
    } else {
        const double correction1 = 1.0 - std::pow(
            options_.beta1, static_cast<double>(next_step));
        const double correction2 = 1.0 - std::pow(
            options_.beta2, static_cast<double>(next_step));
        const double decay_factor = finite(
            1.0 - finite(options_.learning_rate * options_.weight_decay));
        for (std::size_t i = 0; i < updated.size(); ++i) {
            const double g = gradients.parameter_gradient[i];
            first[i] = finite(options_.beta1 * first[i] +
                              (1.0 - options_.beta1) * g);
            second[i] = finite(options_.beta2 * second[i] +
                               (1.0 - options_.beta2) * finite(g * g));
            const double mhat = finite(first[i] / correction1);
            const double vhat = finite(second[i] / correction2);
            const double increment = finite(options_.learning_rate *
                                             (mhat / (std::sqrt(vhat) +
                                                      options_.epsilon)));
            updated[i] = finite(finite(updated[i] * decay_factor) - increment);
        }
    }

    // Validate the complete new snapshot before committing any mutable state.
    NeuralParameterSnapshot next(snapshot_.network(), std::move(updated));
    snapshot_ = std::move(next);
    first_moment_.swap(first);
    second_moment_.swap(second);
    steps_ = next_step;
    return evaluated.mean_loss;
}

NeuralParameterSnapshot neural_initialize_parameters(
    NeuralNetwork network, std::uint64_t seed) {
    std::vector<double> values(network.parameter_count(), 0.0);
    for (auto const& layer : network.layers()) {
        const double denom =
            static_cast<double>(layer.inputs) + static_cast<double>(layer.outputs);
        const double bound = std::sqrt(6.0 / denom);
        for (std::size_t o = 0; o < layer.outputs; ++o) {
            for (std::size_t i = 0; i < layer.inputs; ++i) {
                values[layer.weights_offset + o * layer.inputs + i] =
                    (2.0 * uniform01(seed) - 1.0) * bound;
            }
        }
    }
    return NeuralParameterSnapshot(std::move(network), std::move(values));
}

std::vector<std::size_t> neural_shuffled_indices(
    std::size_t count, std::uint64_t seed) {
    std::vector<std::size_t> indices(count);
    std::iota(indices.begin(), indices.end(), std::size_t{0});
    for (std::size_t i = count; i > 1; --i) {
        // Unbiased rejection sampling: draw from [0, 2^64) without modulo bias.
        const std::uint64_t bound = static_cast<std::uint64_t>(i);
        const std::uint64_t threshold =
            (std::uint64_t{0} - bound) % bound;
        std::uint64_t draw;
        do { draw = splitmix64(seed); } while (draw < threshold);
        const std::size_t j = static_cast<std::size_t>(draw % bound);
        std::swap(indices[i - 1], indices[j]);
    }
    return indices;
}

} // namespace Ariadne
