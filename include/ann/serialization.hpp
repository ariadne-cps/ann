/***************************************************************************
 *            ann/serialization.hpp
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

/** @file ann/serialization.hpp
 *  @brief Portable canonical snapshot storage using JSON and Safetensors.
 */
#ifndef ARIADNE_ANN_SERIALIZATION_HPP
#define ARIADNE_ANN_SERIALIZATION_HPP

#include <string>

#include "ann/network.hpp"
#include "ann/training.hpp"

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
std::string neural_snapshot_sha256(const NeuralParameterSnapshot& snapshot);

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
void neural_save_snapshot(const NeuralParameterSnapshot& snapshot,
                          const std::string& manifest_path,
                          const std::string& safetensors_path);

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
    const std::string& manifest_path,
    const std::string& safetensors_path);

/** @brief Save model files and a separate optimizer-state Safetensors file.
 *
 *  The third file stores optimizer moments as F64 and strict metadata
 *  specifying loss, hyperparameters, frozen layers, step count and hashes.
 *  All three paths must be distinct; publication is NOT atomic.
 */
void neural_save_training_checkpoint(
    const NeuralTrainingSession& session,
    const std::string& manifest_path,
    const std::string& safetensors_path,
    const std::string& optimizer_path);

/** @brief Recover exact native-CPU optimizer state and frozen model.
 *
 *  Requires the model snapshot hash and independent optimizer state hash
 *  to agree. Does not load untrusted executable or pickle data.
 */
NeuralTrainingSession neural_load_training_checkpoint(
    const std::string& manifest_path,
    const std::string& safetensors_path,
    const std::string& optimizer_path);

/** @brief Publish a new immutable model checkpoint directory via one rename.
 *
 *  Creates a private staging sibling directory, writes model.json and
 *  model.safetensors and verifies their content before publishing. The
 *  destination must not exist. Existing checkpoints are never intentionally
 *  overwritten. No schema or file format change; no crash-durability,
 *  multi-writer no-replace, or cross-filesystem guarantee is provided.
 */
void neural_save_snapshot_atomic(const NeuralParameterSnapshot& snapshot,
                                 const std::string& directory);

/** @brief Read a complete model checkpoint directory. */
NeuralParameterSnapshot neural_load_snapshot_atomic(const std::string& directory);

/** @brief Publish a complete new training checkpoint directory.
 *
 *  Writes model.json, model.safetensors and optimizer.safetensors to a
 *  private staging sibling, verifies the model and optimizer contents,
 *  then renames the directory into place. Destination must be absent.
 */
void neural_save_training_checkpoint_atomic(const NeuralTrainingSession& session,
                                            const std::string& directory);

/** @brief Restore a complete training checkpoint directory. */
NeuralTrainingSession neural_load_training_checkpoint_atomic(
    const std::string& directory);

} // namespace Ariadne

#endif // ARIADNE_ANN_SERIALIZATION_HPP
