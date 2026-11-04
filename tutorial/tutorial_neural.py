# ***************************************************************************
# *            tutorial/tutorial_neural.py
# *
# *  Copyright  2026  Luca Geretti
# *
# ***************************************************************************
#
# This file is part of Ariadne Neural.
#
# Ariadne Neural is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# Ariadne Neural is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with Ariadne Neural.  If not, see <https://www.gnu.org/licenses/>.

"""Native CPU inference, training and canonical model persistence."""

from pathlib import Path
from tempfile import TemporaryDirectory

from pyariadne import (
    NeuralActivation,
    NeuralDenseLayerSpec,
    NeuralExecutionSession,
    NeuralOptimizerOptions,
    NeuralTrainingSession,
    NeuralNetwork,
    NeuralOperation,
    NeuralParameterSnapshot,
    neural_version,
    neural_save_snapshot,
    neural_load_snapshot,
    neural_initialize_parameters,
    neural_replace_output_head,
    NeuralLossKind,
)

if __name__ == "__main__":
    model = NeuralNetwork(2, [
        NeuralDenseLayerSpec(2, NeuralActivation.Relu),
        NeuralDenseLayerSpec(1, NeuralActivation.Identity),
    ])
    snapshot = NeuralParameterSnapshot(model, [1, -1, -1, 1, 0, 0, 1, 1, 0])
    session = NeuralExecutionSession(snapshot)
    print("Ariadne Neural", neural_version())
    print("Network dimensions:", model.input_dimension(), "->", model.output_dimension())
    print("Parameters:", model.parameter_count())
    print("Native CPU snapshot import:", session.supports(NeuralOperation.ImportSnapshot))
    print("Inference available:", session.supports(NeuralOperation.Inference))
    print('Single inference:', session.infer([3.0, 1.0]))
    print('Batch inference:', session.infer_batch([[3.0, 1.0], [1.0, 4.0]]))
    print("Output gradient:", session.differentiate([3.0, 1.0], [1.0]).input_gradient)

    regression = NeuralNetwork(1, [NeuralDenseLayerSpec(1, NeuralActivation.Identity)])
    optimizer = NeuralOptimizerOptions()
    optimizer.learning_rate = 0.1
    trainer = NeuralTrainingSession(NeuralParameterSnapshot(regression, [0.0, 0.0]), optimizer)
    for _ in range(70):
        trainer.train_batch([[1.0]], [[2.0]])
    print("Training steps:", trainer.steps(), "final MSE:", trainer.loss_batch([[1.0]], [[2.0]]))

    with TemporaryDirectory(prefix="ariadne-neural-") as directory:
        manifest = str(Path(directory) / "model.json")
        tensors = str(Path(directory) / "model.safetensors")
        neural_save_snapshot(trainer.export_snapshot(), manifest, tensors)
        restored = neural_load_snapshot(manifest, tensors)
        print("Reloaded parameter count:", restored.network().parameter_count())

    # Fine-tune a new output head while retaining the trained feature layer.
    features = NeuralNetwork(1, [
        NeuralDenseLayerSpec(2, NeuralActivation.Tanh),
        NeuralDenseLayerSpec(1, NeuralActivation.Identity),
    ])
    pretrained = neural_initialize_parameters(features, 19)
    transferred = neural_replace_output_head(
        pretrained, NeuralDenseLayerSpec(2, NeuralActivation.Identity), 29)
    tuning = NeuralTrainingSession(
        transferred, optimizer, NeuralLossKind.MeanSquaredError, [False, True])
    for _ in range(20):
        tuning.train_batch([[1.0]], [[1.0, -1.0]])
    print("Fine-tuning steps:", tuning.steps(),
          "preserved feature weight:",
          tuning.export_snapshot().values()[0] == transferred.values()[0])
