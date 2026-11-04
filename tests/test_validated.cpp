/***************************************************************************
 *            tests/test_validated.cpp
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

class TestValidated : public Testing::ContextFixture {
public:
    using ContextFixture::ContextFixture;

    void test_affine_corner_oracle_and_cancellation() {
        NeuralParameterSnapshot snapshot(
            NeuralNetwork(2,{{1,NeuralActivation::Identity}}),
            {2.0,-3.0,0.5});
        NeuralExecutionSession session(snapshot,NeuralBackendKind::AriadneValidated);
        ARIADNE_TESTING_PROVE(session.supports(NeuralOperation::ValidatedBounds));
        ARIADNE_TESTING_PROVE(!session.supports(NeuralOperation::Inference));
        ARIADNE_TESTING_PROVE(!session.supports(NeuralOperation::Training));
        auto result=session.validated_bounds({{-1.0,2.0},{4.0,5.0}});
        ARIADNE_TESTING_EXPECT_EQUAL(result.size(),std::size_t{1});
        // Exact affine extrema: 0.5 + 2*(-1) - 3*5 = -16.5;
        //                       0.5 + 2*2 - 3*4 = -7.5.
        ARIADNE_TESTING_PROVE(result[0].lower <= -16.5);
        ARIADNE_TESTING_PROVE(result[0].upper >= -7.5);
        ARIADNE_TESTING_PROVE(result[0].lower > -17.0);
        ARIADNE_TESTING_PROVE(result[0].upper < -7.0);
        ARIADNE_TESTING_EXPECT_THROWS(std::logic_error,
            (void)session.infer({1.0,2.0}));
        NeuralExecutionSession approximate(snapshot);
        ARIADNE_TESTING_EXPECT_THROWS(std::logic_error,
            (void)approximate.validated_bounds({{-1.0,2.0},{4.0,5.0}}));

        // 0.1 and 0.2 are exact binary64 *parameters*, not the
        // corresponding ideal decimal numbers. Cancellation exposes
        // outward rounding at the third operation.
        NeuralParameterSnapshot cancellation(
            NeuralNetwork(1,{{1,NeuralActivation::Identity}}),
            {0.1,0.2});
        auto enclosed=neural_validated_bounds(cancellation,{{3.0,3.0}});
        const double exact_sample=NeuralExecutionSession(cancellation).infer({3.0})[0];
        ARIADNE_TESTING_PROVE(enclosed[0].lower <= exact_sample);
        ARIADNE_TESTING_PROVE(enclosed[0].upper >= exact_sample);
    }

    void test_activation_ranges_and_sampling() {
        for (auto activation : {NeuralActivation::Identity,NeuralActivation::Relu,
                                NeuralActivation::Tanh,NeuralActivation::Sigmoid,
                                NeuralActivation::Softplus}) {
            NeuralParameterSnapshot snapshot(
                NeuralNetwork(1,{{1,activation}}),{1.25,-0.5});
            auto bounds=neural_validated_bounds(snapshot,{{-3.0,4.0}});
            ARIADNE_TESTING_EXPECT_EQUAL(bounds.size(),std::size_t{1});
            ARIADNE_TESTING_PROVE(bounds[0].lower <= bounds[0].upper);
            NeuralExecutionSession cpu(snapshot);
            for (int i=0;i<=100;++i) {
                const double x=-3.0+7.0*static_cast<double>(i)/100.0;
                const double y=cpu.infer({x}).at(0);
                ARIADNE_TESTING_PROVE(bounds[0].lower <= y);
                ARIADNE_TESTING_PROVE(bounds[0].upper >= y);
            }
            if (activation==NeuralActivation::Relu) {
                ARIADNE_TESTING_EXPECT_EQUAL(bounds[0].lower,0.0);
                ARIADNE_TESTING_PROVE(bounds[0].upper >= 4.5);
            }
            if (activation==NeuralActivation::Sigmoid) {
                ARIADNE_TESTING_PROVE(bounds[0].lower >= 0.0);
                ARIADNE_TESTING_PROVE(bounds[0].upper <= 1.0);
            }
            if (activation==NeuralActivation::Tanh) {
                ARIADNE_TESTING_PROVE(bounds[0].lower >= -1.0);
                ARIADNE_TESTING_PROVE(bounds[0].upper <= 1.0);
            }
            if (activation==NeuralActivation::Softplus) {
                ARIADNE_TESTING_PROVE(bounds[0].lower >= 0.0);
            }
        }
    }

    void test_composition_and_strict_rejections() {
        NeuralParameterSnapshot model(
            NeuralNetwork(2,{{2,NeuralActivation::Relu},
                             {1,NeuralActivation::Identity}}),
            {1.0,-1.0,-1.0,1.0,0.0,0.0,1.0,1.0,0.0});
        auto bounds=neural_validated_bounds(model,{{1.0,3.0},{0.0,2.0}});
        ARIADNE_TESTING_EXPECT_EQUAL(bounds.size(),std::size_t{1});
        for (int i=0;i<=20;++i)
            for (int j=0;j<=20;++j) {
                const double x=1.0+0.1*i;
                const double y=0.1*j;
                const double z=NeuralExecutionSession(model).infer({x,y})[0];
                ARIADNE_TESTING_PROVE(bounds[0].lower <= z);
                ARIADNE_TESTING_PROVE(bounds[0].upper >= z);
            }
        ARIADNE_TESTING_EXPECT_THROWS(std::invalid_argument,
            (void)neural_validated_bounds(model,{{0.0,1.0}}));
        ARIADNE_TESTING_EXPECT_THROWS(std::invalid_argument,
            (void)neural_validated_bounds(model,{{2.0,1.0},{0.0,1.0}}));
        ARIADNE_TESTING_EXPECT_THROWS(std::invalid_argument,
            (void)neural_validated_bounds(model,{{0.0,
                  std::numeric_limits<double>::infinity()},{0.0,1.0}}));
        ARIADNE_TESTING_EXPECT_THROWS(std::invalid_argument,
            (void)neural_validated_bounds(model,{{0.0,
                  std::numeric_limits<double>::quiet_NaN()},{0.0,1.0}}));
        NeuralParameterSnapshot overflow(
            NeuralNetwork(1,{{1,NeuralActivation::Identity}}),{1e308,0.0});
        ARIADNE_TESTING_EXPECT_THROWS(std::overflow_error,
            (void)neural_validated_bounds(overflow,{{1e308,1e308}}));
    }
};

int main(int argc,char** argv) {
    ARIADNE_TESTING_SUITE("validated");
    ARIADNE_TESTING_ADD_TEST(TestValidated,test_affine_corner_oracle_and_cancellation);
    ARIADNE_TESTING_ADD_TEST(TestValidated,test_activation_ranges_and_sampling);
    ARIADNE_TESTING_ADD_TEST(TestValidated,test_composition_and_strict_rejections);
    return ARIADNE_TESTING_RUN_TESTS_WITH_ARGS(argc,argv);
}
