/***************************************************************************
 *            include/neural/backend.hpp
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
 *  In M1 only NativeCpu is available, solely for canonical snapshot import.
 */
bool neural_backend_available(NeuralBackendKind backend) noexcept;

/** @brief Whether a backend implements a particular operation.
 *
 *  An unavailable backend supports no operations. In M1 NativeCpu supports
 *  only ImportSnapshot; inference is intentionally deferred to M2.
 */
bool neural_backend_supports(NeuralBackendKind backend, NeuralOperation operation) noexcept;

/** @brief A backend-selection and frozen-snapshot import boundary.
 *
 *  The session takes ownership of a snapshot copy or move and exposes no
 *  mutable access to its parameters. This M1 contract is not an inference
 *  or training engine; those operations will be added by implementations
 *  that advertise the corresponding capabilities.
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
    /** @brief Query a concrete supported operation of this session. */
    bool supports(NeuralOperation operation) const noexcept;

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
