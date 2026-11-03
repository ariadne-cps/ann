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

"""M2 canonical model and reference CPU inference."""

from pyariadne import (
    NeuralActivation,
    NeuralDenseLayerSpec,
    NeuralExecutionSession,
    NeuralNetwork,
    NeuralOperation,
    NeuralParameterSnapshot,
    neural_version,
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
