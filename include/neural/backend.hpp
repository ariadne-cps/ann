/***************************************************************************
 *            neural/backend.hpp
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

/** @file neural/backend.hpp
 *  @brief Backend capability contract and frozen model import session.
 *
 *  The backend contract is intentionally independent of any tensor runtime.
 *  A backend advertises only operations that have actually been implemented.
 */
#ifndef ARIADNE_NEURAL_BACKEND_HPP
#define ARIADNE_NEURAL_BACKEND_HPP

#include "neural/network.hpp"
#include "neural/derivatives.hpp"

namespace Ariadne {

/** @brief Execution implementations planned for Ariadne Neural.
 *
 *  Enumeration of a backend does not imply that it is built or available.
 *  LibTorch and MLX must remain optional integrations.
 */
enum class NeuralBackendKind {
    NativeCpu,       ///< Built-in, portable CPU execution engine.
    AriadneValidated,///< Ariadne rigorous numerical evaluator.
    LibTorch,        ///< Optional LibTorch C++ integration.
    Mlx              ///< Optional MLX C++ integration.
};

/** @brief Feature-granular operations that an implementation may support. */
enum class NeuralOperation {
    ImportSnapshot,   ///< Import and inspect a frozen canonical model.
    Inference,        ///< Evaluate one input sample approximately.
    BatchInference,   ///< Evaluate multiple samples approximately.
    InputDerivatives, ///< Compute input derivatives.
    ParameterGradients,///< Compute gradients with respect to parameters.
    Training,         ///< Perform one or more training steps.
    ValidatedBounds   ///< Enclose network outputs rigorously.
};

/** @brief Whether a backend implementation is actually compiled and usable.
 *
 *  NativeCpu supports snapshot import and approximate single/batch inference.
 */
bool neural_backend_available(NeuralBackendKind backend) noexcept;

/** @brief Whether a backend implements a particular operation.
 *
 *  An unavailable backend supports no operations. NativeCpu supports
 *  ImportSnapshot, Inference, BatchInference, InputDerivatives and
 *  ParameterGradients and Training in M4. This function reports backend-wide
 *  capabilities; Training is available only via the mutable
 *  NeuralTrainingSession, not via the frozen NeuralExecutionSession.
 */
bool neural_backend_supports(NeuralBackendKind backend, NeuralOperation operation) noexcept;

/** @brief A backend-selection and frozen-snapshot import boundary.
 *
 *  The session takes ownership of a snapshot copy or move and exposes no
 *  mutable access to its parameters. CPU inference and reverse-mode
 *  derivatives are available. The separate NeuralTrainingSession performs
 *  NativeCpu training; validated bounds remain unsupported.
 */
class NeuralExecutionSession {
public:
    /** @brief Import the model into a currently available backend.
     *  @param snapshot Canonical network and finite parameters.
     *  @param backend Selected implementation, defaulting to the native CPU.
     *  @throws std::invalid_argument if the backend is not available.
     */
    explicit NeuralExecutionSession(
        NeuralParameterSnapshot snapshot,
        NeuralBackendKind backend = NeuralBackendKind::NativeCpu);

    /** @brief Selected implementation identity. */
    NeuralBackendKind backend() const noexcept;
    /** @brief Read-only imported snapshot, owned by this session. */
    NeuralParameterSnapshot const& snapshot() const noexcept;
    /** @brief Export an independent frozen snapshot for another backend. */
    NeuralParameterSnapshot export_snapshot() const;
    /** @brief Query a concrete operation available through this frozen session.
     *
     *  Training is false here even if neural_backend_supports(NativeCpu,
     *  Training) is true: updates use a separate NeuralTrainingSession.
     */
    bool supports(NeuralOperation operation) const noexcept;

    /** @brief Evaluate a finite input vector of exactly input_dimension entries.
     *  @throws std::invalid_argument for invalid input shape or non-finite values.
     *  @throws std::overflow_error for a non-finite intermediate or output.
     */
    std::vector<double> infer(std::vector<double> const& input) const;

    /** @brief Evaluate rows of shape [batch, input_dimension].
     *  Empty batches are accepted, and all rows validated before evaluation.
     */
    std::vector<std::vector<double>> infer_batch(
        std::vector<std::vector<double>> const& inputs) const;

    /** @brief Reverse-mode vector-Jacobian product for c dot network(input).
     *
     *  Returns both input and canonical parameter gradients. ReLU's
     *  computational derivative at zero is defined as 0 (not a classical
     *  derivative). The snapshot is unchanged.
     *  @throws std::invalid_argument for wrong shape or non-finite input/cotangent.
     *  @throws std::overflow_error if any intermediate becomes non-finite.
     */
    NeuralGradientResult differentiate(
        std::vector<double> const& input,
        std::vector<double> const& output_cotangent) const;

    /** @brief Full input Jacobian [output_dimension, input_dimension].
     *
     *  Constructed from reverse-mode unit output cotangents, in row order.
     */
    std::vector<std::vector<double>> input_jacobian(
        std::vector<double> const& input) const;

    /** @brief Sum/mean per-sample VJPs, with corresponding input gradients.
     *
     *  Inputs and cotangents have the same batch size. Empty Sum batches
     *  yield empty input gradients and zero parameter gradients; empty Mean
     *  batches are invalid. No implicit loss function or optimizer update.
     */
    NeuralBatchGradientResult differentiate_batch(
        std::vector<std::vector<double>> const& inputs,
        std::vector<std::vector<double>> const& output_cotangents,
        NeuralGradientReduction reduction = NeuralGradientReduction::Sum) const;

    /** @brief Require an operation without silently falling back.
     *  @throws std::logic_error when the operation is unsupported.
     */
    void require(NeuralOperation operation) const;

private:
    NeuralParameterSnapshot snapshot_;
    NeuralBackendKind backend_;
};

} // namespace Ariadne

#endif // ARIADNE_NEURAL_BACKEND_HPP
