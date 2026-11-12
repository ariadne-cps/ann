/***************************************************************************
 *            tests/test_network.cpp
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
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with Ariadne ANN. If not, see <https://www.gnu.org/licenses/>.
 */

/** @file tests/test_network.cpp
 *  @brief Test canonical network layout and snapshots.
 */

#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

#include "ann/network.hpp"

#include "ariadne-testing.hpp"

using namespace Ariadne;

class TestNetwork : public Ariadne::Testing::ContextFixture {
  public:
    using ContextFixture::ContextFixture;

    void test_offsets_and_activations() {
        NeuralNetwork network(2, {
                                     {3,     NeuralActivation::Tanh},
                                     {1, NeuralActivation::Identity}
        });
        ARIADNE_TESTING_EXPECT_EQUAL(network.input_dimension(), std::size_t{2});
        ARIADNE_TESTING_EXPECT_EQUAL(network.output_dimension(), std::size_t{1});
        ARIADNE_TESTING_EXPECT_EQUAL(network.parameter_count(), std::size_t{13});
        ARIADNE_TESTING_EXPECT_EQUAL(network.layers().size(), std::size_t{2});
        const auto& first = network.layers().at(0);
        const auto& second = network.layers().at(1);
        ARIADNE_TESTING_EXPECT_EQUAL(first.inputs, std::size_t{2});
        ARIADNE_TESTING_EXPECT_EQUAL(first.outputs, std::size_t{3});
        ARIADNE_TESTING_EXPECT_EQUAL(first.weights_offset, std::size_t{0});
        ARIADNE_TESTING_EXPECT_EQUAL(first.biases_offset, std::size_t{6});
        ARIADNE_TESTING_EXPECT_EQUAL(second.inputs, std::size_t{3});
        ARIADNE_TESTING_EXPECT_EQUAL(second.outputs, std::size_t{1});
        ARIADNE_TESTING_EXPECT_EQUAL(second.weights_offset, std::size_t{9});
        ARIADNE_TESTING_EXPECT_EQUAL(second.biases_offset, std::size_t{12});
        ARIADNE_TESTING_EXPECT_EQUAL(first.activation, NeuralActivation::Tanh);
        ARIADNE_TESTING_EXPECT_EQUAL(neural_activation_name(NeuralActivation::Identity), "identity");
        ARIADNE_TESTING_EXPECT_EQUAL(neural_activation_name(NeuralActivation::Relu), "relu");
        ARIADNE_TESTING_EXPECT_EQUAL(neural_activation_name(NeuralActivation::Tanh), "tanh");
        ARIADNE_TESTING_EXPECT_EQUAL(neural_activation_name(NeuralActivation::Sigmoid), "sigmoid");
        ARIADNE_TESTING_EXPECT_EQUAL(neural_activation_name(NeuralActivation::Softplus), "softplus");
    }

    void test_invalid_topology() {
        ARIADNE_TESTING_EXPECT_THROWS(
            std::invalid_argument, (void)NeuralNetwork(0, {
                                                              {1, NeuralActivation::Identity}
        }));
        ARIADNE_TESTING_EXPECT_THROWS(
            std::invalid_argument, (void)NeuralNetwork(2, {}));
        ARIADNE_TESTING_EXPECT_THROWS(
            std::invalid_argument, (void)NeuralNetwork(2, {
                                                              {0, NeuralActivation::Identity}
        }));
        ARIADNE_TESTING_EXPECT_THROWS(
            std::invalid_argument, (void)NeuralNetwork(1, {
                                                              {2, static_cast<NeuralActivation>(77)}
        }));
        ARIADNE_TESTING_EXPECT_THROWS(
            std::invalid_argument, (void)neural_activation_name(static_cast<NeuralActivation>(100)));
        const std::size_t max = std::numeric_limits<std::size_t>::max();
        ARIADNE_TESTING_EXPECT_THROWS(
            std::overflow_error, (void)NeuralNetwork(2, {
                                                            {max, NeuralActivation::Identity}
        }));
        ARIADNE_TESTING_EXPECT_THROWS(
            std::overflow_error, (void)NeuralNetwork(1, {
                                                            {max, NeuralActivation::Identity}
        }));
    }

    void test_snapshot_copy_and_validation() {
        NeuralNetwork network(2, {
                                     {1, NeuralActivation::Identity}
        });
        std::vector<double> original{1.25, -0.0, 3.5};
        NeuralParameterSnapshot snapshot(network, original);
        original.at(0) = 99.0;
        ARIADNE_TESTING_EXPECT_EQUAL(snapshot.values().at(0), 1.25);
        ARIADNE_TESTING_PROVE(std::signbit(snapshot.values().at(1)));
        auto exported = snapshot.export_values();
        exported.at(2) = 7.0;
        ARIADNE_TESTING_EXPECT_EQUAL(snapshot.values().at(2), 3.5);
        NeuralParameterSnapshot copied = snapshot;
        ARIADNE_TESTING_EXPECT_EQUAL(copied.network().parameter_count(), std::size_t{3});
        ARIADNE_TESTING_EXPECT_EQUAL(copied.values().at(0), 1.25);
        ARIADNE_TESTING_EXPECT_THROWS(
            std::invalid_argument, (void)NeuralParameterSnapshot(network, {0.0, 1.0}));
        ARIADNE_TESTING_EXPECT_THROWS(
            std::invalid_argument, (void)NeuralParameterSnapshot(network, {0.0, std::numeric_limits<double>::infinity(), 0.0}));
        ARIADNE_TESTING_EXPECT_THROWS(
            std::invalid_argument, (void)NeuralParameterSnapshot(network, {0.0, std::numeric_limits<double>::quiet_NaN(), 0.0}));
    }
};

int main(int argc, char** argv) {
    ARIADNE_TESTING_SUITE("network");
    ARIADNE_TESTING_ADD_TEST(TestNetwork, test_offsets_and_activations);
    ARIADNE_TESTING_ADD_TEST(TestNetwork, test_invalid_topology);
    ARIADNE_TESTING_ADD_TEST(TestNetwork, test_snapshot_copy_and_validation);
    return ARIADNE_TESTING_RUN_TESTS_WITH_ARGS(argc, argv);
}
