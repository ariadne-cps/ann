/***************************************************************************
 *            tests/test_training.cpp
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

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

using namespace Ariadne;

class TestTraining : public Testing::ContextFixture {
public:
    using ContextFixture::ContextFixture;

    static NeuralParameterSnapshot affine(double weight = 1.0, double bias = 0.0) {
        return NeuralParameterSnapshot(
            NeuralNetwork(1, {{1, NeuralActivation::Identity}}), {weight, bias});
    }

    static NeuralOptimizerOptions sgd(double learning_rate = 0.1) {
        NeuralOptimizerOptions o;
        o.kind = NeuralOptimizerKind::Sgd;
        o.learning_rate = learning_rate;
        return o;
    }

    void test_sgd_exact_update_and_freezing() {
        auto frozen = affine();
        NeuralTrainingSession trainer(frozen, sgd());
        ARIADNE_TESTING_EXPECT_EQUAL(trainer.steps(), std::size_t{0});
        ARIADNE_TESTING_EXPECT_EQUAL(trainer.loss_batch({{2.0}}, {{0.0}}), 2.0);
        const double initial_loss = trainer.train_batch({{2.0}}, {{0.0}});
        ARIADNE_TESTING_EXPECT_EQUAL(initial_loss, 2.0);
        auto updated = trainer.export_snapshot().values();
        ARIADNE_TESTING_PROVE(std::abs(updated.at(0) - 0.6) < 1e-12);
        ARIADNE_TESTING_PROVE(std::abs(updated.at(1) + 0.2) < 1e-12);
        ARIADNE_TESTING_EXPECT_EQUAL(frozen.values().at(0), 1.0);
        ARIADNE_TESTING_EXPECT_EQUAL(frozen.values().at(1), 0.0);
        ARIADNE_TESTING_EXPECT_EQUAL(trainer.steps(), std::size_t{1});
        ARIADNE_TESTING_PROVE(trainer.loss_batch({{2.0}}, {{0.0}}) < initial_loss);
    }

    void test_sgd_momentum_and_weight_decay() {
        auto opt = sgd();
        opt.momentum = 0.5;
        opt.weight_decay = 0.1;
        NeuralTrainingSession trainer(affine(), opt);
        (void)trainer.train_batch({{2.0}}, {{0.0}});
        // g = (4,2) + decay*(1,0) = (4.1,2).
        auto updated = trainer.export_snapshot().values();
        ARIADNE_TESTING_PROVE(std::abs(updated.at(0) - 0.59) < 1e-12);
        ARIADNE_TESTING_PROVE(std::abs(updated.at(1) + 0.2) < 1e-12);
        // Second step checks the momentum buffer is retained.
        (void)trainer.train_batch({{2.0}}, {{0.0}});
        auto after = trainer.export_snapshot().values();
        ARIADNE_TESTING_PROVE(std::isfinite(after.at(0)));
        ARIADNE_TESTING_PROVE(after.at(0) != updated.at(0));
        ARIADNE_TESTING_EXPECT_EQUAL(trainer.steps(), std::size_t{2});
    }

    void test_adamw_independent_one_step_oracle() {
        NeuralOptimizerOptions opt;
        opt.kind = NeuralOptimizerKind::AdamW;
        opt.learning_rate = 0.1;
        opt.weight_decay = 0.1;
        opt.beta1 = 0.0;
        opt.beta2 = 0.0;
        opt.epsilon = 1.0;
        NeuralTrainingSession trainer(affine(), opt);
        (void)trainer.train_batch({{2.0}}, {{0.0}});
        auto p = trainer.export_snapshot().values();
        // g=(4,2), mhat=g, vhat=g^2, decoupled decay factor=.99.
        ARIADNE_TESTING_PROVE(std::abs(p.at(0) - 0.91) < 1e-12);
        ARIADNE_TESTING_PROVE(std::abs(p.at(1) + 0.1 * 2.0 / 3.0) < 1e-12);
        ARIADNE_TESTING_EXPECT_EQUAL(trainer.steps(), std::size_t{1});
    }

    void test_minibatch_scaling_and_convergence() {
        NeuralTrainingSession trainer(affine(0.0, 0.0), sgd());
        // For x=1, two identical observations yield the same mean gradient.
        (void)trainer.train_batch({{1.0}, {1.0}}, {{2.0}, {2.0}});
        auto p = trainer.export_snapshot().values();
        ARIADNE_TESTING_PROVE(std::abs(p.at(0) - 0.2) < 1e-12);
        ARIADNE_TESTING_PROVE(std::abs(p.at(1) - 0.2) < 1e-12);
        for (int i = 0; i < 70; ++i)
            (void)trainer.train_batch({{1.0}}, {{2.0}});
        ARIADNE_TESTING_PROVE(trainer.loss_batch({{1.0}}, {{2.0}}) < 1e-10);
    }

    void test_bce_logits_and_classification_convergence() {
        NeuralTrainingSession trainer(
            affine(0.0, 0.0), sgd(0.2),
            NeuralLossKind::BinaryCrossEntropyWithLogits);
        double initial = trainer.loss_batch({{1.0}, {-1.0}}, {{1.0}, {0.0}});
        ARIADNE_TESTING_PROVE(std::abs(initial - std::log(2.0)) < 1e-12);
        (void)trainer.train_batch({{1.0}, {-1.0}}, {{1.0}, {0.0}});
        ARIADNE_TESTING_PROVE(std::abs(
            trainer.export_snapshot().values().at(0) - 0.1) < 1e-12);
        for (int i = 0; i < 80; ++i)
            (void)trainer.train_batch({{1.0}, {-1.0}}, {{1.0}, {0.0}});
        ARIADNE_TESTING_PROVE(
            trainer.loss_batch({{1.0}, {-1.0}}, {{1.0}, {0.0}}) < 0.1);
        NeuralTrainingSession extremes(
            affine(1.0, 0.0), sgd(0.1),
            NeuralLossKind::BinaryCrossEntropyWithLogits);
        ARIADNE_TESTING_PROVE(
            std::isfinite(extremes.loss_batch({{1000.0}, {-1000.0}},
                                              {{1.0}, {0.0}})));
    }

    void test_deterministic_initialization_and_shuffling() {
        NeuralNetwork model(2, {{3, NeuralActivation::Tanh},
                                {1, NeuralActivation::Identity}});
        auto a = neural_initialize_parameters(model, 31415);
        auto b = neural_initialize_parameters(model, 31415);
        auto c = neural_initialize_parameters(model, 31416);
        ARIADNE_TESTING_PROVE(a.values() == b.values());
        ARIADNE_TESTING_PROVE(a.values() != c.values());
        for (auto const& layer : model.layers()) {
            for (std::size_t j = 0; j < layer.outputs; ++j)
                ARIADNE_TESTING_EXPECT_EQUAL(
                    a.values().at(layer.biases_offset + j), 0.0);
        }
        auto perm = neural_shuffled_indices(32, 123);
        auto repeated = neural_shuffled_indices(32, 123);
        ARIADNE_TESTING_PROVE(perm == repeated);
        std::sort(perm.begin(), perm.end());
        for (std::size_t i = 0; i < perm.size(); ++i)
            ARIADNE_TESTING_EXPECT_EQUAL(perm[i], i);
        ARIADNE_TESTING_EXPECT_EQUAL(
            neural_shuffled_indices(0, 123).size(), std::size_t{0});
    }

    void test_validation_and_failure_atomicity() {
        auto opt = sgd();
        opt.learning_rate = 0.0;
        ARIADNE_TESTING_EXPECT_THROWS(
            std::invalid_argument, (void)NeuralTrainingSession(affine(), opt));
        opt = sgd();
        opt.kind = static_cast<NeuralOptimizerKind>(77);
        ARIADNE_TESTING_EXPECT_THROWS(
            std::invalid_argument, (void)NeuralTrainingSession(affine(), opt));
        opt = sgd();
        ARIADNE_TESTING_EXPECT_THROWS(
            std::invalid_argument, (void)NeuralTrainingSession(
                affine(), opt, static_cast<NeuralLossKind>(77)));
        NeuralTrainingSession trainer(affine(), sgd());
        ARIADNE_TESTING_EXPECT_THROWS(
            std::invalid_argument, trainer.train_batch({}, {}));
        ARIADNE_TESTING_EXPECT_THROWS(
            std::invalid_argument, trainer.train_batch({{2.0}}, {{1.0, 2.0}}));
        ARIADNE_TESTING_EXPECT_THROWS(
            std::invalid_argument, trainer.train_batch({{2.0}}, {{std::numeric_limits<double>::infinity()}}));
        ARIADNE_TESTING_EXPECT_EQUAL(trainer.steps(), std::size_t{0});
        ARIADNE_TESTING_EXPECT_EQUAL(trainer.export_snapshot().values().at(0), 1.0);
        ARIADNE_TESTING_EXPECT_EQUAL(trainer.export_snapshot().values().at(1), 0.0);
        NeuralTrainingSession overflow(affine(1e308, 0.0), sgd());
        ARIADNE_TESTING_EXPECT_THROWS(
            std::overflow_error, overflow.train_batch({{1.0}}, {{0.0}}));
        ARIADNE_TESTING_EXPECT_EQUAL(overflow.steps(), std::size_t{0});
        ARIADNE_TESTING_EXPECT_EQUAL(overflow.export_snapshot().values().at(0), 1e308);
    }
};

int main(int argc, char** argv) {
    ARIADNE_TESTING_SUITE("training");
    ARIADNE_TESTING_ADD_TEST(TestTraining, test_sgd_exact_update_and_freezing);
    ARIADNE_TESTING_ADD_TEST(TestTraining, test_sgd_momentum_and_weight_decay);
    ARIADNE_TESTING_ADD_TEST(TestTraining, test_adamw_independent_one_step_oracle);
    ARIADNE_TESTING_ADD_TEST(TestTraining, test_minibatch_scaling_and_convergence);
    ARIADNE_TESTING_ADD_TEST(TestTraining, test_bce_logits_and_classification_convergence);
    ARIADNE_TESTING_ADD_TEST(TestTraining, test_deterministic_initialization_and_shuffling);
    ARIADNE_TESTING_ADD_TEST(TestTraining, test_validation_and_failure_atomicity);
    return ARIADNE_TESTING_RUN_TESTS_WITH_ARGS(argc, argv);
}
