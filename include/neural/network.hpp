/***************************************************************************
 *            include/neural/network.hpp
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

/** @file neural/network.hpp
 *  @brief Canonical, backend-independent dense feed-forward network description.
 *
 *  No tensor engine or execution backend is included in this header.
 *  Definitions of non-template operations live in src/network.cpp.
 */
#ifndef ARIADNE_NEURAL_NETWORK_HPP
#define ARIADNE_NEURAL_NETWORK_HPP

#include <cstddef>
#include <string_view>
#include <vector>

namespace Ariadne {

/** @brief Supported elementwise activations of a dense neural layer.
 *
 *  This enumeration describes mathematical operations, not particular
 *  CPU, PyTorch, MLX or validated implementations. A selected execution
 *  backend may support only a subset of them.
 */
enum class NeuralActivation {
    Identity, ///< Identity: f(x) = x.
    Relu,     ///< Rectified linear unit: f(x) = max(0, x).
    Tanh,     ///< Hyperbolic tangent.
    Sigmoid,  ///< Logistic sigmoid.
    Softplus  ///< Softplus: log(1 + exp(x)).
};

/** @brief Return the stable descriptive name of an activation.
 *  @throws std::invalid_argument for an invalid enumerator.
 */
std::string_view neural_activation_name(NeuralActivation activation);

/** @brief User-facing specification of one dense output layer.
 *
 *  The input dimension is determined by the network input dimension or
 *  the preceding layer. Each layer has a bias vector of size @c outputs.
 */
struct NeuralDenseLayerSpec {
    std::size_t outputs;         ///< Number of output neurons, strictly positive.
    NeuralActivation activation; ///< Elementwise activation after the affine map.
};

/** @brief Derived, backend-neutral parameter indexing for one dense layer.
 *
 *  Weights have logical shape [outputs, inputs] and row-major linear order
 *  weight[o * inputs + i]. Biases immediately follow the layer's weights.
 *  A backend is free to copy and transpose these values into native storage.
 */
struct NeuralDenseLayerLayout {
    std::size_t inputs;         ///< Number of inputs to this layer.
    std::size_t outputs;        ///< Number of outputs of this layer.
    NeuralActivation activation;///< Activation applied after bias addition.
    std::size_t weights_offset; ///< First weight in the flattened parameter array.
    std::size_t biases_offset;  ///< First bias in the flattened parameter array.
};

/** @brief Immutable-by-interface topology and indexing of a dense network.
 *
 *  A network contains one or more dense layers and requires strictly
 *  positive input and output dimensions. The empty network is deliberately
 *  unsupported in M1; represent an identity function as a dense layer with
 *  identity activation if needed.
 *
 *  The canonical affine convention is:
 *  @code
 *    output[o] = bias[o] + sum_i weights[o * inputs + i] * input[i]
 *  @endcode
 *  followed by the layer's activation. Evaluation is not implemented in M1.
 */
class NeuralNetwork {
public:
    /** @brief Build a network and compute canonical parameter offsets.
     *  @param input_dimension Positive number of network inputs.
     *  @param layers Non-empty layer specifications, each with positive outputs.
     *  @throws std::invalid_argument on unsupported topology or activation.
     *  @throws std::overflow_error if a parameter count overflows size_t.
     */
    NeuralNetwork(std::size_t input_dimension, std::vector<NeuralDenseLayerSpec> layers);

    /** @brief Number of network inputs. */
    std::size_t input_dimension() const noexcept;
    /** @brief Number of network outputs. */
    std::size_t output_dimension() const noexcept;
    /** @brief Total number of weights plus biases, across all layers. */
    std::size_t parameter_count() const noexcept;
    /** @brief Read-only layer descriptions with computed parameter offsets. */
    std::vector<NeuralDenseLayerLayout> const& layers() const noexcept;

private:
    std::size_t input_dimension_;
    std::size_t output_dimension_;
    std::size_t parameter_count_;
    std::vector<NeuralDenseLayerLayout> layers_;
};

/** @brief A frozen collection of binary64 parameters and its network topology.
 *
 *  The values are *canonical model values*, not backend-owned tensors.
 *  Copying a snapshot copies both topology and parameters. No mutator is
 *  exposed: training will use a separate mutable backend working state.
 *  Binary64 values (including signed zero) are preserved on copying.
 *
 *  This initial representation admits only finite binary64 values.
 *  Other precision policies and versioned persistent serialization remain
 *  future work, as specified in doc/neural_design.md.
 */
class NeuralParameterSnapshot {
public:
    /** @brief Copy or move the topology and flattened parameter array.
     *  @param network Network with an established canonical parameter count.
     *  @param values Exactly network.parameter_count() finite binary64 values.
     *  @throws std::invalid_argument for a wrong count or non-finite value.
     */
    NeuralParameterSnapshot(NeuralNetwork network, std::vector<double> values);

    /** @brief Frozen canonical topology associated with the parameters. */
    NeuralNetwork const& network() const noexcept;
    /** @brief Read-only flattened canonical weights and biases. */
    std::vector<double> const& values() const noexcept;
    /** @brief Export an independent copy of the canonical parameter array. */
    std::vector<double> export_values() const;

private:
    NeuralNetwork network_;
    std::vector<double> values_;
};

} // namespace Ariadne

#endif // ARIADNE_NEURAL_NETWORK_HPP
