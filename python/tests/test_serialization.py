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

"""Versioned JSON manifest and Safetensors interoperability tests (no optional package)."""

import json
import math
import struct

import pytest

from pyariadne import (
    NeuralActivation,
    NeuralDenseLayerSpec,
    NeuralExecutionSession,
    NeuralNetwork,
    NeuralParameterSnapshot,
    neural_load_snapshot,
    neural_save_snapshot,
)


def make_snapshot():
    return NeuralParameterSnapshot(
        NeuralNetwork(2, [
            NeuralDenseLayerSpec(2, NeuralActivation.Tanh),
            NeuralDenseLayerSpec(1, NeuralActivation.Identity),
        ]),
        [1.5, -0.0, 0.25, -2.0, 0.0, -1.25, 2.5, 3.5, 4.5],
    )


def raw_bits(values):
    return [struct.pack("<d", value) for value in values]


def test_roundtrip_and_open_safetensors_layout(tmp_path):
    manifest_path = tmp_path / "model.json"
    weights_path = tmp_path / "model.safetensors"
    snapshot = make_snapshot()
    neural_save_snapshot(snapshot, str(manifest_path), str(weights_path))

    manifest = json.loads(manifest_path.read_text())
    assert manifest == {
        "format": "ariadne-neural",
        "version": 1,
        "numeric": "IEEE754-binary64",
        "input_dimension": 2,
        "layers": [
            {"outputs": 2, "activation": "tanh"},
            {"outputs": 1, "activation": "identity"},
        ],
        "tensors": "safetensors",
    }

    binary = weights_path.read_bytes()
    header_size = struct.unpack("<Q", binary[:8])[0]
    header = json.loads(binary[8:8 + header_size])
    assert header == {
        "layers.0.weight": {
            "dtype": "F64", "shape": [2, 2], "data_offsets": [0, 32],
        },
        "layers.0.bias": {
            "dtype": "F64", "shape": [2], "data_offsets": [32, 48],
        },
        "layers.1.weight": {
            "dtype": "F64", "shape": [1, 2], "data_offsets": [48, 64],
        },
        "layers.1.bias": {
            "dtype": "F64", "shape": [1], "data_offsets": [64, 72],
        },
    }
    assert binary[8 + header_size:] == b"".join(raw_bits(snapshot.values()))

    loaded = neural_load_snapshot(str(manifest_path), str(weights_path))
    assert raw_bits(loaded.values()) == raw_bits(snapshot.values())
    assert math.copysign(1.0, loaded.values()[1]) == -1.0
    assert NeuralExecutionSession(loaded).infer([0.5, 0.25]) == (
        NeuralExecutionSession(snapshot).infer([0.5, 0.25])
    )

    # A second export must have exactly the same bytes, not just equal numbers.
    neural_save_snapshot(loaded, str(manifest_path), str(weights_path))
    assert binary == weights_path.read_bytes()


@pytest.mark.parametrize("corruption", [
    "unsupported-version",
    "extra-field",
    "duplicate-key",
    "unknown-activation",
    "bad-shape",
    "bad-type",
    "bad-offset",
    "truncated",
    "trailing-byte",
    "nonfinite-payload",
])
def test_rejects_bad_snapshot(tmp_path, corruption):
    manifest_path = tmp_path / "model.json"
    weights_path = tmp_path / "model.safetensors"
    neural_save_snapshot(make_snapshot(), str(manifest_path), str(weights_path))
    manifest = json.loads(manifest_path.read_text())
    binary = weights_path.read_bytes()
    count = struct.unpack("<Q", binary[:8])[0]
    header = json.loads(binary[8:8 + count])
    payload = binary[8 + count:]

    if corruption == "unsupported-version":
        manifest["version"] = 99
    elif corruption == "extra-field":
        manifest["extension"] = 1
    elif corruption == "duplicate-key":
        manifest_path.write_text(manifest_path.read_text().replace(
            '"version":1', '"version":1,"version":1'
        ))
    elif corruption == "unknown-activation":
        manifest["layers"][0]["activation"] = "unsupported"
    elif corruption == "bad-shape":
        header["layers.0.weight"]["shape"] = [1, 4]
    elif corruption == "bad-type":
        header["layers.0.weight"]["dtype"] = "F32"
    elif corruption == "bad-offset":
        header["layers.0.bias"]["data_offsets"] = [24, 40]
    elif corruption == "truncated":
        payload = payload[:-1]
    elif corruption == "trailing-byte":
        payload += b"\0"
    elif corruption == "nonfinite-payload":
        payload = struct.pack("<d", math.inf) + payload[8:]

    if corruption not in {"duplicate-key"}:
        manifest_path.write_text(json.dumps(manifest))
    new_header = json.dumps(header, separators=(",", ":")).encode("ascii")
    weights_path.write_bytes(struct.pack("<Q", len(new_header)) + new_header + payload)
    with pytest.raises(ValueError):
        neural_load_snapshot(str(manifest_path), str(weights_path))


def test_rejects_same_path_and_missing_file(tmp_path):
    snapshot = make_snapshot()
    file_path = str(tmp_path / "same.json")
    with pytest.raises(ValueError):
        neural_save_snapshot(snapshot, file_path, file_path)
    with pytest.raises(RuntimeError):
        neural_load_snapshot(str(tmp_path / "missing.json"),
                             str(tmp_path / "missing.safetensors"))
