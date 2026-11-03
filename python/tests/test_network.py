# ***************************************************************************
# *            python/tests/test_network.py
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

"""M1 canonical model, snapshot and backend API tests."""

import math
import pytest

from pyariadne import (
    NeuralActivation,
    NeuralBackendKind,
    NeuralDenseLayerSpec,
    NeuralExecutionSession,
    NeuralNetwork,
    NeuralOperation,
    NeuralParameterSnapshot,
    neural_backend_available,
    neural_backend_supports,
)


def test_model_offsets_and_snapshot():
    network = NeuralNetwork(2, [
        NeuralDenseLayerSpec(3, NeuralActivation.Tanh),
        NeuralDenseLayerSpec(1, NeuralActivation.Identity),
    ])
    assert network.input_dimension() == 2
    assert network.output_dimension() == 1
    assert network.parameter_count() == 13
    assert [(layer.inputs, layer.outputs, layer.weights_offset, layer.biases_offset)
            for layer in network.layers()] == [(2, 3, 0, 6), (3, 1, 9)]
    params = [float(n) for n in range(13)]
    frozen = NeuralParameterSnapshot(network, params)
    params[0] = -10.0
    assert frozen.values()[0] == 0.0
    copied = frozen.export_values()
    copied[0] = -7.0
    assert frozen.values()[0] == 0.0
    session = NeuralExecutionSession(frozen)
    assert session.backend() == NeuralBackendKind.NativeCpu
    assert session.supports(NeuralOperation.ImportSnapshot)
    assert not session.supports(NeuralOperation.Inference)
    assert session.export_snapshot().values() == frozen.values()
    with pytest.raises(RuntimeError):
        session.require(NeuralOperation.Inference)


def test_invalid_shapes_and_parameters():
    with pytest.raises(ValueError):
        NeuralNetwork(0, [NeuralDenseLayerSpec(1, NeuralActivation.Identity)])
    with pytest.raises(ValueError):
        NeuralNetwork(2, [])
    with pytest.raises(ValueError):
        NeuralNetwork(2, [NeuralDenseLayerSpec(0, NeuralActivation.Identity)])
    model = NeuralNetwork(1, [NeuralDenseLayerSpec(1, NeuralActivation.Relu)])
    with pytest.raises(ValueError):
        NeuralParameterSnapshot(model, [1.0])
    with pytest.raises(ValueError):
        NeuralParameterSnapshot(model, [math.inf, 0.0])
    with pytest.raises(ValueError):
        NeuralParameterSnapshot(model, [math.nan, 0.0])


def test_backend_capability_rejection():
    assert neural_backend_available(NeuralBackendKind.NativeCpu)
    assert neural_backend_supports(NeuralBackendKind.NativeCpu,
                                   NeuralOperation.ImportSnapshot)
    for kind in (NeuralBackendKind.AriadneValidated,
                 NeuralBackendKind.LibTorch, NeuralBackendKind.Mlx):
        assert not neural_backend_available(kind)
        assert not neural_backend_supports(kind, NeuralOperation.ImportSnapshot)
        with pytest.raises(ValueError):
            NeuralExecutionSession(
                NeuralParameterSnapshot(
                    NeuralNetwork(1, [NeuralDenseLayerSpec(1, NeuralActivation.Identity)]),
                    [1.0, 0.0]),
                kind)
