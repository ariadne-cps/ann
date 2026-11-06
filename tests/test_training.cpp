/***************************************************************************
 *            tests/test_training.cpp
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

#include "ariadne-ann.hpp"
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
        // For count=3 the rejection threshold is 2^64 mod 3 == 1. This seed
        // makes the first splitmix64 draw exactly 0, below the threshold, so
        // the unbiased rejection-sampling retry runs before a valid draw.
        auto retried = neural_shuffled_indices(3, 7046029254386353131ULL);
        std::sort(retried.begin(), retried.end());
        ARIADNE_TESTING_PROVE(
            (retried == std::vector<std::size_t>{0u, 1u, 2u}));
    }

    void test_optimizer_loss_and_batch_validation() {
        const double nan = std::numeric_limits<double>::quiet_NaN();
        const double inf = std::numeric_limits<double>::infinity();
        auto expect_invalid = [this](NeuralOptimizerOptions const& o) {
            ARIADNE_TESTING_EXPECT_THROWS(std::invalid_argument,
                (void)NeuralTrainingSession(affine(), o));
        };
        // Each clause of validate_optimizer, exercised one at a time.
        for (double bad : {nan, inf}) {
            auto o = sgd(); o.learning_rate = bad; expect_invalid(o);
            o = sgd(); o.weight_decay = bad; expect_invalid(o);
            o = sgd(); o.momentum = bad; expect_invalid(o);
            o = sgd(); o.beta1 = bad; expect_invalid(o);
            o = sgd(); o.beta2 = bad; expect_invalid(o);
            o = sgd(); o.epsilon = bad; expect_invalid(o);
        }
        auto o = sgd(); o.learning_rate = -1.0; expect_invalid(o);
        o = sgd(); o.weight_decay = -0.5; expect_invalid(o);
        o = sgd(); o.momentum = -0.5; expect_invalid(o);
        o = sgd(); o.momentum = 1.0; expect_invalid(o);
        o = sgd(); o.beta1 = -0.5; expect_invalid(o);
        o = sgd(); o.beta1 = 1.0; expect_invalid(o);
        o = sgd(); o.beta2 = -0.5; expect_invalid(o);
        o = sgd(); o.beta2 = 1.0; expect_invalid(o);
        o = sgd(); o.epsilon = 0.0; expect_invalid(o);
        o = sgd(); o.epsilon = -1.0; expect_invalid(o);
        // AdamW does not use SGD momentum.
        o = sgd(); o.kind = NeuralOptimizerKind::AdamW; o.momentum = 0.5;
        expect_invalid(o);
        // BCEWithLogits requires an Identity output layer.
        NeuralParameterSnapshot tanh_head(
            NeuralNetwork(1, {{1, NeuralActivation::Tanh}}), {1.0, 0.0});
        ARIADNE_TESTING_EXPECT_THROWS(std::invalid_argument,
            (void)NeuralTrainingSession(tanh_head, sgd(),
                NeuralLossKind::BinaryCrossEntropyWithLogits));
        // Non-empty batch whose sizes disagree.
        NeuralTrainingSession trainer(affine(), sgd());
        ARIADNE_TESTING_EXPECT_THROWS(std::invalid_argument,
            trainer.train_batch({{2.0}}, {}));
        // BCE targets must lie in [0, 1].
        NeuralTrainingSession bce(affine(), sgd(),
            NeuralLossKind::BinaryCrossEntropyWithLogits);
        ARIADNE_TESTING_EXPECT_THROWS(std::invalid_argument,
            bce.train_batch({{1.0}}, {{1.5}}));
        ARIADNE_TESTING_EXPECT_THROWS(std::invalid_argument,
            bce.train_batch({{1.0}}, {{-0.5}}));
    }

    void test_checkpoint_restore_validation() {
        NeuralNetwork net(1, {{1, NeuralActivation::Identity},
                              {1, NeuralActivation::Identity}});
        const auto snapshot = NeuralParameterSnapshot(net, {2.0, 1.0, 3.0, -1.0});
        NeuralOptimizerOptions opt = sgd(0.05);
        opt.momentum = 0.5;
        opt.weight_decay = 0.1;
        NeuralTrainingSession session(snapshot, opt,
            NeuralLossKind::MeanSquaredError, {false, true});
        (void)session.train_batch({{2.0}}, {{0.0}});
        const std::size_t count = net.parameter_count();

        // A valid checkpoint restores steps, moments and trainability.
        auto restored = NeuralTrainingSession(session.export_checkpoint());
        ARIADNE_TESTING_EXPECT_EQUAL(restored.steps(), session.steps());
        ARIADNE_TESTING_PROVE(restored.export_checkpoint().first_moment ==
                              session.export_checkpoint().first_moment);
        ARIADNE_TESTING_PROVE(restored.export_checkpoint().second_moment ==
                              session.export_checkpoint().second_moment);
        ARIADNE_TESTING_PROVE(restored.trainable_layers() ==
                              std::vector<bool>({false, true}));

        const auto base = session.export_checkpoint();
        auto expect_invalid = [this](NeuralTrainingCheckpoint const& c) {
            ARIADNE_TESTING_EXPECT_THROWS(std::invalid_argument,
                (void)NeuralTrainingSession(c));
        };
        // Wrong optimizer-state dimensions.
        {
            auto c = base;
            c.first_moment.resize(count - 1u);
            expect_invalid(c);
        }
        {
            auto c = base;
            c.second_moment.resize(count + 1u);
            expect_invalid(c);
        }
        // Non-finite moments.
        {
            auto c = base;
            c.first_moment.at(0) = std::numeric_limits<double>::quiet_NaN();
            expect_invalid(c);
        }
        {
            auto c = base;
            c.second_moment.at(1) = std::numeric_limits<double>::infinity();
            expect_invalid(c);
        }
        // AdamW rejects a negative second moment.
        {
            NeuralOptimizerOptions a = sgd(0.05);
            a.kind = NeuralOptimizerKind::AdamW;
            NeuralTrainingSession adam(snapshot, a);
            (void)adam.train_batch({{2.0}}, {{0.0}});
            auto c = adam.export_checkpoint();
            c.second_moment.at(2) = -1e-9;
            expect_invalid(c);
        }
        // A frozen layer must keep exactly zero moments.
        {
            auto c = base;  // trainable_layers == {false, true}
            c.first_moment.at(0) = 0.5;  // parameter 0 is in the frozen layer
            expect_invalid(c);
        }
        {
            auto c = base;
            c.second_moment.at(0) = 0.5;
            expect_invalid(c);
        }
        // SGD rejects any nonzero second moment.
        {
            auto c = base;
            c.second_moment.at(2) = 0.5;
            expect_invalid(c);
        }
        // steps == 0 requires exactly zero moments (checked with AdamW, whose
        // second moments are legitimately positive).
        {
            NeuralOptimizerOptions a = sgd(0.05);
            a.kind = NeuralOptimizerKind::AdamW;
            NeuralTrainingSession adam(snapshot, a);
            (void)adam.train_batch({{2.0}}, {{0.0}});
            auto c = adam.export_checkpoint();
            c.steps = 0;
            expect_invalid(c);  // nonzero first moment
            std::fill(c.first_moment.begin(), c.first_moment.end(), 0.0);
            expect_invalid(c);  // nonzero second moment
            std::fill(c.second_moment.begin(), c.second_moment.end(), 0.0);
            NeuralTrainingSession fresh(std::move(c));
            ARIADNE_TESTING_EXPECT_EQUAL(fresh.steps(), std::size_t{0});
        }
        // A saturated step counter cannot advance.
        {
            auto c = base;
            std::fill(c.first_moment.begin(), c.first_moment.end(), 0.0);
            std::fill(c.second_moment.begin(), c.second_moment.end(), 0.0);
            c.steps = std::numeric_limits<std::size_t>::max();
            NeuralTrainingSession saturated(std::move(c));
            ARIADNE_TESTING_EXPECT_THROWS(std::overflow_error,
                saturated.train_batch({{2.0}}, {{0.0}}));
        }
    }

    void test_m5_layer_freeze_sgd_and_adamw() {
        NeuralNetwork model(1, {{1, NeuralActivation::Identity},
                                {1, NeuralActivation::Identity}});
        NeuralParameterSnapshot source(model, {2.0, 1.0, 3.0, -1.0});
        for (NeuralOptimizerKind kind :
             {NeuralOptimizerKind::Sgd, NeuralOptimizerKind::AdamW}) {
            NeuralOptimizerOptions opt = sgd(0.05);
            opt.kind = kind;
            opt.weight_decay = 0.25;
            opt.momentum = kind == NeuralOptimizerKind::Sgd ? 0.5 : 0.0;
            NeuralTrainingSession trainer(source, opt,
                NeuralLossKind::MeanSquaredError, {false, true});
            ARIADNE_TESTING_PROVE(trainer.trainable_layers() ==
                                  std::vector<bool>({false, true}));
            auto const initial = trainer.loss_batch({{2.0}}, {{0.0}});
            for (int i = 0; i < 6; ++i)
                (void)trainer.train_batch({{2.0}}, {{0.0}});
            auto const result = trainer.export_snapshot();
            ARIADNE_TESTING_EXPECT_EQUAL(result.values().at(0), 2.0);
            ARIADNE_TESTING_EXPECT_EQUAL(result.values().at(1), 1.0);
            ARIADNE_TESTING_PROVE(result.values().at(2) != 3.0);
            ARIADNE_TESTING_PROVE(trainer.loss_batch({{2.0}}, {{0.0}}) < initial);
            ARIADNE_TESTING_EXPECT_EQUAL(trainer.steps(), std::size_t{6});
            ARIADNE_TESTING_PROVE(source.values() ==
                                  std::vector<double>({2.0, 1.0, 3.0, -1.0}));
        }
    }

    void test_m5_head_replacement_and_fine_tuning() {
        NeuralNetwork model(2, {{2, NeuralActivation::Tanh},
                                {1, NeuralActivation::Identity}});
        auto source = neural_initialize_parameters(model, 11);
        auto replaced = neural_replace_output_head(
            source, {2, NeuralActivation::Identity}, 123);
        auto repeated = neural_replace_output_head(
            source, {2, NeuralActivation::Identity}, 123);
        ARIADNE_TESTING_EXPECT_EQUAL(replaced.network().input_dimension(),
                                     std::size_t{2});
        ARIADNE_TESTING_EXPECT_EQUAL(replaced.network().output_dimension(),
                                     std::size_t{2});
        ARIADNE_TESTING_EXPECT_EQUAL(replaced.network().parameter_count(),
                                     std::size_t{12});
        ARIADNE_TESTING_PROVE(replaced.values() == repeated.values());
        for (std::size_t i = 0; i < 6; ++i)
            ARIADNE_TESTING_EXPECT_EQUAL(
                replaced.values().at(i), source.values().at(i));
        for (std::size_t i = 10; i < 12; ++i)
            ARIADNE_TESTING_EXPECT_EQUAL(replaced.values().at(i), 0.0);
        NeuralTrainingSession trainer(replaced, sgd(0.02),
                                      NeuralLossKind::MeanSquaredError,
                                      {false, true});
        auto const before = replaced.values();
        (void)trainer.train_batch({{1.0, -1.0}}, {{1.0, -1.0}});
        auto const after = trainer.export_snapshot().export_values();
        for (std::size_t i = 0; i < 6; ++i)
            ARIADNE_TESTING_EXPECT_EQUAL(after.at(i), before.at(i));
        ARIADNE_TESTING_PROVE(after != before);
        ARIADNE_TESTING_EXPECT_EQUAL(trainer.steps(), std::size_t{1});
        // Recreating a session from a frozen snapshot explicitly resets moments.
        NeuralTrainingSession unfreeze(trainer.export_snapshot(), sgd(),
                                       NeuralLossKind::MeanSquaredError,
                                       {true, true});
        ARIADNE_TESTING_EXPECT_EQUAL(unfreeze.steps(), std::size_t{0});
    }

    void test_m5_invalid_freeze_and_head() {
        NeuralNetwork model(1, {{1, NeuralActivation::Identity},
                                {1, NeuralActivation::Identity}});
        auto const source = NeuralParameterSnapshot(model, {2.0, 1.0, 3.0, 0.0});
        ARIADNE_TESTING_EXPECT_THROWS(
            std::invalid_argument,
            (void)NeuralTrainingSession(source, sgd(),
                NeuralLossKind::MeanSquaredError, {true}));
        ARIADNE_TESTING_EXPECT_THROWS(
            std::invalid_argument,
            (void)NeuralTrainingSession(source, sgd(),
                NeuralLossKind::MeanSquaredError, {false, false}));
        ARIADNE_TESTING_EXPECT_THROWS(
            std::invalid_argument,
            (void)neural_replace_output_head(
                source, {0, NeuralActivation::Identity}, 123));
        ARIADNE_TESTING_EXPECT_THROWS(
            std::invalid_argument,
            (void)neural_replace_output_head(
                source, {1, static_cast<NeuralActivation>(99)}, 123));
        ARIADNE_TESTING_PROVE(
            neural_replace_output_head(
                source, {1, NeuralActivation::Identity}, 5).values().size() ==
            source.values().size());
        ARIADNE_TESTING_PROVE(source.values() ==
                              std::vector<double>({2.0, 1.0, 3.0, 0.0}));
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
    ARIADNE_TESTING_ADD_TEST(TestTraining, test_optimizer_loss_and_batch_validation);
    ARIADNE_TESTING_ADD_TEST(TestTraining, test_checkpoint_restore_validation);
    ARIADNE_TESTING_ADD_TEST(TestTraining, test_m5_layer_freeze_sgd_and_adamw);
    ARIADNE_TESTING_ADD_TEST(TestTraining, test_m5_head_replacement_and_fine_tuning);
    ARIADNE_TESTING_ADD_TEST(TestTraining, test_m5_invalid_freeze_and_head);
    ARIADNE_TESTING_ADD_TEST(TestTraining, test_validation_and_failure_atomicity);
    return ARIADNE_TESTING_RUN_TESTS_WITH_ARGS(argc, argv);
}
