/***************************************************************************
 *            neural/serialization.hpp
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

/** @file neural/serialization.hpp
 *  @brief Portable canonical snapshot storage using JSON and Safetensors.
 */
#ifndef ARIADNE_NEURAL_SERIALIZATION_HPP
#define ARIADNE_NEURAL_SERIALIZATION_HPP

#include "neural/network.hpp"
#include <string>

namespace Ariadne {

/** @brief Return the canonical SHA-256 identity of topology and parameter bits.
 *
 *  Hashes the domain-separated binary representation: input dimension and
 *  layer count (little-endian u64), each layer output dimension (u64) and
 *  activation identifier (one byte, Identity=0, Relu=1, Tanh=2,
 *  Sigmoid=3, Softplus=4), followed by every parameter's IEEE-754
 *  binary64 bit pattern as little-endian u64. The domain separator is
 *  "ariadne-neural-model" followed by a NUL byte.
 *  Result: 64 lowercase hexadecimal characters. This identifies model
 *  content, NOT its author or provenance (there is no signature).
 */
std::string neural_snapshot_sha256(NeuralParameterSnapshot const& snapshot);

/** @brief Write the current manifest JSON and its Safetensors data file.
 *
 *  Weights use per-layer tensors "layers.N.weight" [outputs,inputs] in
 *  row-major order and "layers.N.bias" [outputs]. All are F64 with exact
 *  IEEE-754 little-endian bit patterns. No implicit precision conversion.
 *
 *  Both output paths must be distinct. This is a two-file snapshot, not an
 *  atomic transaction: consumers must publish both files together.
 *  @throws std::runtime_error for I/O errors.
 *  @throws std::invalid_argument for a rejected path or oversized model.
 */
void neural_save_snapshot(NeuralParameterSnapshot const& snapshot,
                          std::string const& manifest_path,
                          std::string const& safetensors_path);

/** @brief Load the current JSON+Safetensors snapshot contract.
 *
 *  Requires a SHA-256 digest matching the reconstructed canonical model;
 *  earlier manifests without snapshot_sha256 are not accepted.
 *
 *  Rejects unknown manifest fields, missing/invalid digest, malformed or oversized JSON,
 *  duplicate/missing tensors, non-F64 values, inconsistent shapes/offsets,
 *  trailing bytes, nonfinite data and noncanonical metadata.
 *  @throws std::invalid_argument for malformed or incompatible content.
 *  @throws std::runtime_error for I/O errors.
 */
NeuralParameterSnapshot neural_load_snapshot(
    std::string const& manifest_path,
    std::string const& safetensors_path);

} // namespace Ariadne

#endif // ARIADNE_NEURAL_SERIALIZATION_HPP
