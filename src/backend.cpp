/***************************************************************************
 *            src/backend.cpp
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

#include "neural/backend.hpp"
#include "neural/inference.hpp"

#include <stdexcept>
#include <utility>

namespace Ariadne {

bool neural_backend_available(NeuralBackendKind backend) noexcept {
    switch (backend) {
        case NeuralBackendKind::NativeCpu:
            return true;
        case NeuralBackendKind::AriadneValidated:
        case NeuralBackendKind::LibTorch:
        case NeuralBackendKind::Mlx:
        default:
            return false;
    }
}

bool neural_backend_supports(
    NeuralBackendKind backend, NeuralOperation operation) noexcept {
    return neural_backend_available(backend)
        && (operation == NeuralOperation::ImportSnapshot
            || operation == NeuralOperation::Inference
            || operation == NeuralOperation::BatchInference);
}

NeuralExecutionSession::NeuralExecutionSession(
    NeuralParameterSnapshot snapshot, NeuralBackendKind backend)
    : snapshot_(std::move(snapshot)), backend_(backend) {
    if (!neural_backend_available(backend)) {
        throw std::invalid_argument("Requested neural backend is unavailable");
    }
}

NeuralBackendKind NeuralExecutionSession::backend() const noexcept {
    return backend_;
}
NeuralParameterSnapshot const& NeuralExecutionSession::snapshot() const noexcept {
    return snapshot_;
}
NeuralParameterSnapshot NeuralExecutionSession::export_snapshot() const {
    return snapshot_;
}
bool NeuralExecutionSession::supports(NeuralOperation operation) const noexcept {
    return neural_backend_supports(backend_, operation);
}
std::vector<double> NeuralExecutionSession::infer(std::vector<double> const& input) const {
    require(NeuralOperation::Inference);
    return NeuralCpu::infer(snapshot_, input);
}
std::vector<std::vector<double>> NeuralExecutionSession::infer_batch(
    std::vector<std::vector<double>> const& inputs) const {
    require(NeuralOperation::BatchInference);
    return NeuralCpu::infer_batch(snapshot_, inputs);
}
void NeuralExecutionSession::require(NeuralOperation operation) const {
    if (!supports(operation)) {
        throw std::logic_error("Neural backend does not support requested operation");
    }
}

} // namespace Ariadne
