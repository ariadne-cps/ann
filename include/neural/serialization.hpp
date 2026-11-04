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

/** @brief Write one version-1 manifest JSON and its Safetensors data file.
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

/** @brief Load the exact version-1 JSON+Safetensors snapshot contract.
 *
 *  Rejects unknown manifest fields/versions, malformed or oversized JSON,
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
