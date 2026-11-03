/***************************************************************************
 *            tests/test_backend.cpp
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
#include "ariadne-testing.hpp"

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
        for (auto op : {NeuralOperation::Inference,
                        NeuralOperation::BatchInference,
                        NeuralOperation::InputDerivatives,
                        NeuralOperation::ParameterGradients,
                        NeuralOperation::Training,
                        NeuralOperation::ValidatedBounds}) {
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
        ARIADNE_TESTING_EXPECT_THROWS(std::logic_error,
            session.require(NeuralOperation::Inference));
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
};

int main(int argc, char** argv) {
    ARIADNE_TESTING_SUITE("backend");
    ARIADNE_TESTING_ADD_TEST(TestBackend, test_capabilities_are_truthful);
    ARIADNE_TESTING_ADD_TEST(TestBackend, test_import_and_fail_fast);
    return ARIADNE_TESTING_RUN_TESTS_WITH_ARGS(argc, argv);
}
