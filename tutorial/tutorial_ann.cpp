/***************************************************************************
 *            tutorial/tutorial_ann.cpp
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

/** @file tutorial/tutorial_ann.cpp
 *  @brief Demonstrate ANN inference and training.
 */

#include "ariadne-ann.hpp"

#include <iostream>
#include <cstdio>
#include <filesystem>
#include <string>
#include <vector>

int main() {
    using namespace Ariadne;
    NeuralNetwork network(2, {{2, NeuralActivation::Relu},
                              {1, NeuralActivation::Identity}});
    NeuralParameterSnapshot parameters(
        network, {1.0, -1.0, -1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0});
    NeuralExecutionSession session(parameters);

    std::cout << "Ariadne ANN " << ann_version() << '\n'
              << "Network dimensions: " << network.input_dimension()
              << " -> " << network.output_dimension() << '\n'
              << "Parameters: " << network.parameter_count() << '\n'
              << "Native CPU snapshot import: "
              << session.supports(NeuralOperation::ImportSnapshot) << '\n'
              << "Inference available: "
              << session.supports(NeuralOperation::Inference) << '\n'
              << "Single inference: " << session.infer({3.0, 1.0}).at(0) << '\n'
              << "Batch inference: " << session.infer_batch({{3.0, 1.0}, {1.0, 4.0}}).at(1).at(0) << '\n'
              << "Input derivative of sum of outputs: "
              << session.differentiate({3.0, 1.0}, {1.0}).input_gradient.at(0) << '\n';

    NeuralOptimizerOptions optimizer;
    optimizer.learning_rate = 0.1;
    NeuralNetwork regression(1, {{1, NeuralActivation::Identity}});
    NeuralParameterSnapshot regression_start(regression, {0.0, 0.0});
    NeuralTrainingSession trainer(regression_start, optimizer);
    for (int step = 0; step < 70; ++step) {
        (void)trainer.train_batch({{1.0}}, {{2.0}});
    }
    std::cout << "Training steps: " << trainer.steps()
              << " | final MSE: " << trainer.loss_batch({{1.0}}, {{2.0}}) << '\n';
    // Store the frozen, trained model in two open files and reconstruct it.
    auto const prefix = std::filesystem::temp_directory_path() /
        "ariadne_neural_tutorial";
    const std::string manifest = prefix.string() + ".json";
    const std::string tensors = prefix.string() + ".safetensors";
    auto const trained = trainer.export_snapshot();
    neural_save_snapshot(trained, manifest, tensors);
    auto const restored = neural_load_snapshot(manifest, tensors);
    std::cout << "Snapshot SHA-256: " << neural_snapshot_sha256(restored) << '\n';
    std::cout << "Restored parameters: " << restored.network().parameter_count()
              << " | first weight: " << restored.values().at(0) << '\n';
    // Replace the output head, freeze the trained feature layer and fine-tune.
    NeuralNetwork base(1, {{2, NeuralActivation::Tanh},
                           {1, NeuralActivation::Identity}});
    auto pretrained = neural_initialize_parameters(base, 19);
    auto transferred = neural_replace_output_head(
        pretrained, {2, NeuralActivation::Identity}, 29);
    NeuralTrainingSession fine_tuning(transferred, optimizer,
        NeuralLossKind::MeanSquaredError, {false, true});
    auto const preserved_prefix = transferred.values().at(0);
    for (int step = 0; step < 20; ++step)
        (void)fine_tuning.train_batch({{1.0}}, {{1.0, -1.0}});
    std::cout << "Fine-tuning steps: " << fine_tuning.steps()
              << " | preserved feature weight: "
              << (fine_tuning.export_snapshot().values().at(0) == preserved_prefix)
              << '\n';
    // Persist the complete optimizer state, then restore a resumable session.
    const std::string optimizer_state=prefix.string()+".optimizer.safetensors";
    neural_save_training_checkpoint(trainer,manifest,tensors,optimizer_state);
    auto resumed=neural_load_training_checkpoint(manifest,tensors,optimizer_state);
    std::cout << "Recovered optimizer steps: " << resumed.steps() << '\n';
    // Publish a separate complete checkpoint directory in one rename.
    const auto atomic_dir = prefix.string() + ".checkpoint";
    // Use a fresh destination; a published checkpoint is never overwritten.
    neural_save_training_checkpoint_atomic(trainer, atomic_dir);
    std::cout << "Atomic directory optimizer steps: "
              << neural_load_training_checkpoint_atomic(atomic_dir).steps() << '\n';
    (void)std::filesystem::remove_all(atomic_dir);
    (void)std::remove(optimizer_state.c_str());
    (void)std::remove(manifest.c_str());
    (void)std::remove(tensors.c_str());
    return 0;
}
