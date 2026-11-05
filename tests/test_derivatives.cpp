/***************************************************************************
 *            tests/test_derivatives.cpp
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

#include "ariadne-neural.hpp"
#include "ariadne-testing.hpp"

#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

using namespace Ariadne;

class TestDerivatives : public Ariadne::Testing::ContextFixture {
public:
    using ContextFixture::ContextFixture;

    static NeuralExecutionSession affine_session() {
        return NeuralExecutionSession(NeuralParameterSnapshot(
            NeuralNetwork(2, {{2, NeuralActivation::Identity}}),
            {2.0, 3.0, -1.0, 4.0, 5.0, 6.0}));
    }

    void test_affine_vjp_and_jacobian() {
        auto session = affine_session();
        auto result = session.differentiate({1.0, 2.0}, {7.0, -2.0});
        ARIADNE_TESTING_EXPECT_EQUAL(result.input_gradient.size(), std::size_t{2});
        ARIADNE_TESTING_EXPECT_EQUAL(result.input_gradient.at(0), 16.0);
        ARIADNE_TESTING_EXPECT_EQUAL(result.input_gradient.at(1), 13.0);
        std::vector<double> const expected{7.0, 14.0, -2.0, -4.0, 7.0, -2.0};
        ARIADNE_TESTING_EXPECT_EQUAL(result.parameter_gradient.size(), expected.size());
        for (std::size_t i = 0; i < expected.size(); ++i)
            ARIADNE_TESTING_EXPECT_EQUAL(result.parameter_gradient.at(i), expected.at(i));

        auto jacobian = session.input_jacobian({1.0, 2.0});
        ARIADNE_TESTING_EXPECT_EQUAL(jacobian.size(), std::size_t{2});
        ARIADNE_TESTING_EXPECT_EQUAL(jacobian.at(0).at(0), 2.0);
        ARIADNE_TESTING_EXPECT_EQUAL(jacobian.at(0).at(1), 3.0);
        ARIADNE_TESTING_EXPECT_EQUAL(jacobian.at(1).at(0), -1.0);
        ARIADNE_TESTING_EXPECT_EQUAL(jacobian.at(1).at(1), 4.0);
        ARIADNE_TESTING_EXPECT_EQUAL(session.export_snapshot().values().at(0), 2.0);
    }

    void test_batch_sum_mean_and_empty() {
        auto session = affine_session();
        auto sum = session.differentiate_batch(
            {{1.0, 2.0}, {3.0, 4.0}}, {{7.0, -2.0}, {1.0, 1.0}});
        std::vector<double> const expected{10.0, 18.0, 1.0, 0.0, 8.0, -1.0};
        ARIADNE_TESTING_EXPECT_EQUAL(sum.input_gradients.size(), std::size_t{2});
        ARIADNE_TESTING_EXPECT_EQUAL(sum.input_gradients.at(0).at(0), 16.0);
        ARIADNE_TESTING_EXPECT_EQUAL(sum.input_gradients.at(0).at(1), 13.0);
        ARIADNE_TESTING_EXPECT_EQUAL(sum.input_gradients.at(1).at(0), 1.0);
        ARIADNE_TESTING_EXPECT_EQUAL(sum.input_gradients.at(1).at(1), 7.0);
        for (std::size_t i = 0; i < expected.size(); ++i)
            ARIADNE_TESTING_EXPECT_EQUAL(sum.parameter_gradient.at(i), expected.at(i));

        auto mean = session.differentiate_batch(
            {{1.0, 2.0}, {3.0, 4.0}}, {{7.0, -2.0}, {1.0, 1.0}},
            NeuralGradientReduction::Mean);
        for (std::size_t i = 0; i < expected.size(); ++i)
            ARIADNE_TESTING_EXPECT_EQUAL(mean.parameter_gradient.at(i), expected.at(i) / 2.0);
        ARIADNE_TESTING_EXPECT_EQUAL(mean.input_gradients.at(0).at(0), 8.0);
        ARIADNE_TESTING_EXPECT_EQUAL(mean.input_gradients.at(1).at(1), 3.5);
        auto empty = session.differentiate_batch({}, {});
        ARIADNE_TESTING_EXPECT_EQUAL(empty.input_gradients.size(), std::size_t{0});
        for (double x : empty.parameter_gradient)
            ARIADNE_TESTING_EXPECT_EQUAL(x, 0.0);
        ARIADNE_TESTING_EXPECT_THROWS(std::invalid_argument,
            session.differentiate_batch({}, {}, NeuralGradientReduction::Mean));
    }

    void test_smooth_multilayer_finite_difference() {
        NeuralNetwork network(2, {{2, NeuralActivation::Tanh},
                                  {1, NeuralActivation::Sigmoid}});
        std::vector<double> parameters{0.7, -0.2, 0.3, 0.6, 0.1, -0.3,
                                       1.4, -0.8, 0.2};
        NeuralExecutionSession session(NeuralParameterSnapshot(network, parameters));
        std::vector<double> input{0.4, -0.6};
        auto gradient = session.differentiate(input, {1.0});
        const double step = 1e-6;
        for (std::size_t i = 0; i < input.size(); ++i) {
            auto plus = input;
            auto minus = input;
            plus[i] += step;
            minus[i] -= step;
            const double oracle =
                (session.infer(plus).at(0) - session.infer(minus).at(0)) / (2.0 * step);
            ARIADNE_TESTING_PROVE(std::abs(gradient.input_gradient.at(i) - oracle) < 1e-6);
        }
        for (std::size_t p = 0; p < parameters.size(); ++p) {
            auto plus = parameters;
            auto minus = parameters;
            plus[p] += step;
            minus[p] -= step;
            NeuralExecutionSession greater(NeuralParameterSnapshot(network, plus));
            NeuralExecutionSession lesser(NeuralParameterSnapshot(network, minus));
            const double oracle =
                (greater.infer(input).at(0) - lesser.infer(input).at(0)) / (2.0 * step);
            ARIADNE_TESTING_PROVE(std::abs(gradient.parameter_gradient.at(p) - oracle) < 1e-6);
        }
    }

    void test_relu_kink_and_stable_derivatives() {
        NeuralExecutionSession relu(NeuralParameterSnapshot(
            NeuralNetwork(1, {{1, NeuralActivation::Relu}}), {1.0, 0.0}));
        auto at_zero = relu.differentiate({0.0}, {3.0});
        ARIADNE_TESTING_EXPECT_EQUAL(at_zero.input_gradient.at(0), 0.0);
        ARIADNE_TESTING_EXPECT_EQUAL(at_zero.parameter_gradient.at(0), 0.0);
        ARIADNE_TESTING_EXPECT_EQUAL(at_zero.parameter_gradient.at(1), 0.0);
        auto above = relu.differentiate({2.0}, {3.0});
        ARIADNE_TESTING_EXPECT_EQUAL(above.input_gradient.at(0), 3.0);
        ARIADNE_TESTING_EXPECT_EQUAL(above.parameter_gradient.at(0), 6.0);
        ARIADNE_TESTING_EXPECT_EQUAL(above.parameter_gradient.at(1), 3.0);
        for (auto act : {NeuralActivation::Softplus, NeuralActivation::Sigmoid,
                         NeuralActivation::Tanh}) {
            NeuralExecutionSession session(NeuralParameterSnapshot(
                NeuralNetwork(1, {{1, act}}), {1.0, 0.0}));
            for (double x : {-1000.0, 1000.0}) {
                auto gradient = session.differentiate({x}, {1.0});
                ARIADNE_TESTING_PROVE(std::isfinite(gradient.input_gradient.at(0)));
                ARIADNE_TESTING_PROVE(std::isfinite(gradient.parameter_gradient.at(0)));
            }
        }
    }

    void test_invalid_shapes_and_nonfinite_gradients() {
        auto session = affine_session();
        ARIADNE_TESTING_EXPECT_THROWS(std::invalid_argument,
            session.differentiate({1.0}, {1.0, 1.0}));
        ARIADNE_TESTING_EXPECT_THROWS(std::invalid_argument,
            session.differentiate({1.0, 2.0}, {1.0}));
        ARIADNE_TESTING_EXPECT_THROWS(std::invalid_argument,
            session.differentiate({1.0, 2.0}, {1.0, std::numeric_limits<double>::infinity()}));
        ARIADNE_TESTING_EXPECT_THROWS(std::invalid_argument,
            session.differentiate_batch({{1.0, 2.0}}, {}));
        ARIADNE_TESTING_EXPECT_THROWS(std::invalid_argument,
            session.differentiate_batch({{1.0, 2.0}}, {{1.0}}));
        ARIADNE_TESTING_EXPECT_THROWS(std::invalid_argument,
            session.differentiate_batch({}, {}, static_cast<NeuralGradientReduction>(77)));
        NeuralExecutionSession large(NeuralParameterSnapshot(
            NeuralNetwork(1, {{1, NeuralActivation::Identity}}), {1e308, 0.0}));
        ARIADNE_TESTING_EXPECT_THROWS(std::overflow_error,
            large.differentiate({1.0}, {1e308}));
        // Every forward value is finite, but delta * activation overflows:
        // the weight gradient is 1e308 * 1e308.
        NeuralExecutionSession huge(NeuralParameterSnapshot(
            NeuralNetwork(1, {{1, NeuralActivation::Identity}}), {1.0, 0.0}));
        ARIADNE_TESTING_EXPECT_THROWS(std::overflow_error,
            huge.differentiate({1e308}, {1e308}));
    }
};

int main(int argc, char** argv) {
    ARIADNE_TESTING_SUITE("derivatives");
    ARIADNE_TESTING_ADD_TEST(TestDerivatives, test_affine_vjp_and_jacobian);
    ARIADNE_TESTING_ADD_TEST(TestDerivatives, test_batch_sum_mean_and_empty);
    ARIADNE_TESTING_ADD_TEST(TestDerivatives, test_smooth_multilayer_finite_difference);
    ARIADNE_TESTING_ADD_TEST(TestDerivatives, test_relu_kink_and_stable_derivatives);
    ARIADNE_TESTING_ADD_TEST(TestDerivatives, test_invalid_shapes_and_nonfinite_gradients);
    return ARIADNE_TESTING_RUN_TESTS_WITH_ARGS(argc, argv);
}
