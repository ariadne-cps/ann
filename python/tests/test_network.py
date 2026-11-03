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
    NeuralGradientReduction,
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
    assert session.supports(NeuralOperation.Inference)
    assert session.supports(NeuralOperation.BatchInference)
    assert session.export_snapshot().values() == frozen.values()
    session.require(NeuralOperation.Inference)
    with pytest.raises(RuntimeError):
        session.require(NeuralOperation.Training)


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


def test_cpu_inference_and_batch():
    network = NeuralNetwork(2, [NeuralDenseLayerSpec(2, NeuralActivation.Relu),
                                NeuralDenseLayerSpec(1, NeuralActivation.Identity)])
    session = NeuralExecutionSession(NeuralParameterSnapshot(
        network, [1.0, -1.0, -1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0]))
    assert session.infer([3, 1]) == [2.0]
    assert session.infer_batch([[3, 1], [1, 4]]) == [[2.0], [3.0]]
    assert session.infer_batch([]) == []
    with pytest.raises(ValueError):
        session.infer([1.0])
    with pytest.raises(ValueError):
        session.infer_batch([[1.0, 2.0], [math.inf, 2.0]])
    assert session.export_snapshot().values()[0] == 1.0


def test_stable_cpu_activations():
    for activation in (NeuralActivation.Sigmoid, NeuralActivation.Softplus,
                       NeuralActivation.Tanh):
        session = NeuralExecutionSession(NeuralParameterSnapshot(
            NeuralNetwork(1, [NeuralDenseLayerSpec(1, activation)]), [1.0, 0.0]))
        negative, positive = session.infer([-1000])[0], session.infer([1000])[0]
        assert math.isfinite(negative) and math.isfinite(positive)
        if activation == NeuralActivation.Sigmoid:
            assert (negative, positive) == (0.0, 1.0)
        if activation == NeuralActivation.Softplus:
            assert (negative, positive) == (0.0, 1000.0)
    overflow = NeuralExecutionSession(NeuralParameterSnapshot(
        NeuralNetwork(1, [NeuralDenseLayerSpec(1, NeuralActivation.Identity)]),
        [1e308, 0.0]))
    with pytest.raises(OverflowError):
        overflow.infer([1e308])


def test_m3_affine_derivatives_and_batch():
    network = NeuralNetwork(2, [NeuralDenseLayerSpec(2, NeuralActivation.Identity)])
    session = NeuralExecutionSession(NeuralParameterSnapshot(
        network, [2.0, 3.0, -1.0, 4.0, 5.0, 6.0]))
    assert session.supports(NeuralOperation.InputDerivatives)
    assert session.supports(NeuralOperation.ParameterGradients)
    assert not session.supports(NeuralOperation.Training)
    gradient = session.differentiate([1.0, 2.0], [7.0, -2.0])
    assert gradient.input_gradient == [16.0, 13.0]
    assert gradient.parameter_gradient == [7.0, 14.0, -2.0, -4.0, 7.0, -2.0]
    assert session.input_jacobian([1.0, 2.0]) == [[2.0, 3.0], [-1.0, 4.0]]
    summed = session.differentiate_batch(
        [[1.0, 2.0], [3.0, 4.0]], [[7.0, -2.0], [1.0, 1.0]])
    assert summed.input_gradients == [[16.0, 13.0], [1.0, 7.0]]
    assert summed.parameter_gradient == [10.0, 18.0, 1.0, 0.0, 8.0, -1.0]
    averaged = session.differentiate_batch(
        [[1.0, 2.0], [3.0, 4.0]], [[7.0, -2.0], [1.0, 1.0]],
        NeuralGradientReduction.Mean)
    assert averaged.parameter_gradient == [5.0, 9.0, 0.5, 0.0, 4.0, -0.5]
    assert averaged.input_gradients == [[8.0, 6.5], [0.5, 3.5]]
    assert session.differentiate_batch([], []).parameter_gradient == [0.0] * 6
    with pytest.raises(ValueError):
        session.differentiate_batch([], [], NeuralGradientReduction.Mean)
    with pytest.raises(ValueError):
        session.differentiate([1.0, 2.0], [1.0])


def test_m3_relu_at_kink_and_snapshot():
    session = NeuralExecutionSession(NeuralParameterSnapshot(
        NeuralNetwork(1, [NeuralDenseLayerSpec(1, NeuralActivation.Relu)]),
        [1.0, 0.0]))
    gradient = session.differentiate([0.0], [3.0])
    assert gradient.input_gradient == [0.0]
    assert gradient.parameter_gradient == [0.0, 0.0]
    assert session.export_snapshot().values() == [1.0, 0.0]
