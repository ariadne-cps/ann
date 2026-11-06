/***************************************************************************
 *            tests/test_backend.cpp
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

#include "ann/backend.hpp"
#include "ariadne-testing.hpp"

#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

using namespace Ariadne;

class TestBackend : public Ariadne::Testing::ContextFixture {
public:
    using ContextFixture::ContextFixture;

    void test_capabilities_are_truthful() {
        ARIADNE_TESTING_PROVE(neural_backend_available(NeuralBackendKind::NativeCpu));
        for (auto kind : {NeuralBackendKind::AriadneValidated,
                          NeuralBackendKind::LibTorch,
                          NeuralBackendKind::Mlx}) {
            ARIADNE_TESTING_PROVE(!neural_backend_available(kind));
            ARIADNE_TESTING_PROVE(!neural_backend_supports(kind, NeuralOperation::ImportSnapshot));
        }
        ARIADNE_TESTING_PROVE(neural_backend_supports(
            NeuralBackendKind::NativeCpu, NeuralOperation::ImportSnapshot));
        for (auto op : {NeuralOperation::Inference, NeuralOperation::BatchInference,
                        NeuralOperation::InputDerivatives,
                        NeuralOperation::ParameterGradients,
                        NeuralOperation::Training}) {
            ARIADNE_TESTING_PROVE(neural_backend_supports(NeuralBackendKind::NativeCpu, op));
        }
        for (auto op : {NeuralOperation::ValidatedBounds}) {
            ARIADNE_TESTING_PROVE(!neural_backend_supports(NeuralBackendKind::NativeCpu, op));
        }
        ARIADNE_TESTING_PROVE(!neural_backend_available(static_cast<NeuralBackendKind>(42)));
        ARIADNE_TESTING_PROVE(!neural_backend_supports(
            NeuralBackendKind::NativeCpu, static_cast<NeuralOperation>(42)));
    }

    void test_import_and_fail_fast() {
        NeuralNetwork network(1, {{1, NeuralActivation::Tanh}});
        NeuralParameterSnapshot parameters(network, {2.0, -3.0});
        NeuralExecutionSession session(parameters);
        ARIADNE_TESTING_EXPECT_EQUAL(session.backend(), NeuralBackendKind::NativeCpu);
        ARIADNE_TESTING_PROVE(session.supports(NeuralOperation::ImportSnapshot));
        ARIADNE_TESTING_EXPECT_NO_THROW(session.require(NeuralOperation::ImportSnapshot));
        ARIADNE_TESTING_EXPECT_NO_THROW(session.require(NeuralOperation::Inference));
        ARIADNE_TESTING_EXPECT_NO_THROW(session.require(NeuralOperation::BatchInference));
        ARIADNE_TESTING_PROVE(neural_backend_supports(
            NeuralBackendKind::NativeCpu, NeuralOperation::Training));
        ARIADNE_TESTING_PROVE(!session.supports(NeuralOperation::Training));
        ARIADNE_TESTING_EXPECT_THROWS(std::logic_error,
            session.require(NeuralOperation::Training));
        ARIADNE_TESTING_EXPECT_THROWS(std::logic_error,
            session.require(NeuralOperation::ValidatedBounds));
        auto exported = session.export_snapshot();
        ARIADNE_TESTING_EXPECT_EQUAL(exported.values().at(0), 2.0);
        ARIADNE_TESTING_EXPECT_EQUAL(exported.network().layers().at(0).outputs,
                                      std::size_t{1});
        ARIADNE_TESTING_EXPECT_THROWS(std::invalid_argument,
            (void)NeuralExecutionSession(parameters, NeuralBackendKind::LibTorch));
        ARIADNE_TESTING_EXPECT_THROWS(std::invalid_argument,
            (void)NeuralExecutionSession(parameters, NeuralBackendKind::Mlx));
        ARIADNE_TESTING_EXPECT_THROWS(std::invalid_argument,
            (void)NeuralExecutionSession(parameters, NeuralBackendKind::AriadneValidated));
    }

    void test_cpu_forward() {
        NeuralNetwork network(2, {{2, NeuralActivation::Relu},
                                  {1, NeuralActivation::Identity}});
        NeuralParameterSnapshot snapshot(network, {1.0, -1.0, -1.0, 1.0,
                                                   0.0, 0.0, 1.0, 1.0, 0.0});
        NeuralExecutionSession session(snapshot);
        ARIADNE_TESTING_EXPECT_EQUAL(session.infer({3.0, 1.0}).at(0), 2.0);
        ARIADNE_TESTING_EXPECT_EQUAL(session.infer({1.0, 4.0}).at(0), 3.0);
        auto batch = session.infer_batch({{3.0, 1.0}, {1.0, 4.0}});
        ARIADNE_TESTING_EXPECT_EQUAL(batch.size(), std::size_t{2});
        ARIADNE_TESTING_EXPECT_EQUAL(batch.at(0).at(0), 2.0);
        ARIADNE_TESTING_EXPECT_EQUAL(batch.at(1).at(0), 3.0);
        ARIADNE_TESTING_EXPECT_EQUAL(session.infer_batch({}).size(), std::size_t{0});
        ARIADNE_TESTING_EXPECT_EQUAL(session.export_snapshot().values().at(0), 1.0);
        ARIADNE_TESTING_EXPECT_THROWS(std::invalid_argument, session.infer({1.0}));
        ARIADNE_TESTING_EXPECT_THROWS(std::invalid_argument,
            session.infer({std::numeric_limits<double>::quiet_NaN(), 1.0}));
        ARIADNE_TESTING_EXPECT_THROWS(std::invalid_argument,
            session.infer_batch({{1.0, 1.0}, {2.0}}));
    }

    void test_stable_activations_and_overflow() {
        for (auto activation : {NeuralActivation::Tanh, NeuralActivation::Sigmoid,
                                NeuralActivation::Softplus}) {
            NeuralExecutionSession session(NeuralParameterSnapshot(
                NeuralNetwork(1, {{1, activation}}), {1.0, 0.0}));
            double negative = session.infer({-1000.0}).at(0);
            double positive = session.infer({1000.0}).at(0);
            ARIADNE_TESTING_PROVE(std::isfinite(negative));
            ARIADNE_TESTING_PROVE(std::isfinite(positive));
            if (activation == NeuralActivation::Sigmoid) {
                ARIADNE_TESTING_EXPECT_EQUAL(negative, 0.0);
                ARIADNE_TESTING_EXPECT_EQUAL(positive, 1.0);
            }
            if (activation == NeuralActivation::Softplus) {
                ARIADNE_TESTING_EXPECT_EQUAL(negative, 0.0);
                ARIADNE_TESTING_EXPECT_EQUAL(positive, 1000.0);
            }
        }
        NeuralExecutionSession overflow(NeuralParameterSnapshot(
            NeuralNetwork(1, {{1, NeuralActivation::Identity}}), {1e308, 0.0}));
        ARIADNE_TESTING_EXPECT_THROWS(std::overflow_error,
                                      overflow.infer({1e308}));
    }
};

int main(int argc, char** argv) {
    ARIADNE_TESTING_SUITE("backend");
    ARIADNE_TESTING_ADD_TEST(TestBackend, test_capabilities_are_truthful);
    ARIADNE_TESTING_ADD_TEST(TestBackend, test_import_and_fail_fast);
    ARIADNE_TESTING_ADD_TEST(TestBackend, test_cpu_forward);
    ARIADNE_TESTING_ADD_TEST(TestBackend, test_stable_activations_and_overflow);
    return ARIADNE_TESTING_RUN_TESTS_WITH_ARGS(argc, argv);
}
