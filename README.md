# Ariadne Neural

[![macOS Status](https://github.com/ariadne-cps/neural/actions/workflows/macos.yml/badge.svg)](https://github.com/ariadne-cps/neural/actions/workflows/macos.yml)
[![Ubuntu Status](https://github.com/ariadne-cps/neural/actions/workflows/ubuntu.yml/badge.svg)](https://github.com/ariadne-cps/neural/actions/workflows/ubuntu.yml)
[![Windows Status](https://github.com/ariadne-cps/neural/actions/workflows/win.yml/badge.svg)](https://github.com/ariadne-cps/neural/actions/workflows/win.yml)
[![Sanitizers Status](https://github.com/ariadne-cps/neural/actions/workflows/sanitizers.yml/badge.svg)](https://github.com/ariadne-cps/neural/actions/workflows/sanitizers.yml)
[![Coverage Status](https://github.com/ariadne-cps/neural/actions/workflows/coverage.yml/badge.svg)](https://github.com/ariadne-cps/neural/actions/workflows/coverage.yml)
[![codecov](https://codecov.io/gh/ariadne-cps/neural/branch/main/graph/badge.svg)](https://codecov.io/gh/ariadne-cps/neural)

Ariadne Neural is a standalone C++20 library for dense feed-forward neural networks.
Its current implementation provides canonical network topology, frozen binary64
parameter snapshots and native-CPU inference for individual samples and batches,
reverse-mode input/parameter derivatives, input Jacobians, and per-batch gradient
sum or mean reductions, plus native-CPU sequential minibatch training with
MSE and binary cross entropy with logits, SGD (optionally with momentum)
and AdamW. Deterministic seeded initialization and sample shuffling are provided.
Supported layer activations are identity, ReLU, tanh, sigmoid and softplus.
M6 introduces rigorous, directed-rounding interval-bound propagation via
Ariadne Numeric, independently from approximate CPU inference.
Fine-tuning supports per-layer trainability masks (with exact freezing of
unchanged parameters for SGD and AdamW) and deterministic replacement of the
final dense output head. A new training session resets optimizer state; masks
are not saved in model-only snapshots. An optional three-file training
checkpoint preserves the optimizer state and trainability mask.
C++ and Python interfaces expose model metadata, parameter snapshots, backend
capability queries, CPU inference, derivatives and training.

## Dependencies

Direct Git submodules, pinned to specific commits:

- [Numeric](https://github.com/ariadne-cps/numeric) (`ss/num`, rigorous interval bounds).
- [Threading](https://github.com/ariadne-cps/threading) (`ss/thr`, for scheduling and Logging).
- [Configuration](https://github.com/ariadne-cps/configuration) (`ss/cfg`, for CMake).
- [Testing](https://github.com/ariadne-cps/testing) (`ss/tst`, for tests).
- [PythonCommon](https://github.com/ariadne-cps/python-common) (`ss/pyc`, for optional Python bindings and pybind11).

The C++ neural model, inference, Jacobian, training and serialization code
does not depend on Function or Symbolic. Function-based and symbolic adapters
belong in the planned `neuro-symbolic` repository. Threading remains a direct
dependency for the planned parallel CPU implementation and the Ariadne
command-line/logger benchmark.

## Layout

- `include/neural/`: public headers, including `inference.hpp`.
- `include/ariadne-neural.hpp`: umbrella header.
- `src/`: implementation sources, including `inference.cpp`.
- `tests/`: C++ tests.
- `python/bindings/` and `python/tests/`: Python bindings and tests.
- `tutorial/`: C++ and Python examples.
- `doc/neural_design.md`: design journal and roadmap.

## Build and test

```sh
git clone https://github.com/ariadne-cps/neural.git
cd neural
cmake -S . -B build -DCMAKE_BUILD_TYPE=Debug
cmake --build build --parallel --target tests
ctest --test-dir build --output-on-failure
```

CMake initializes the required Git submodules.

To build and install the release library and Python bindings:

```sh
cmake -S . -B build-release -DCMAKE_BUILD_TYPE=Release
cmake --build build-release --parallel --target ariadne-neural bindings
cmake --install build-release --prefix /path/to/install
```

Python bindings require Python 3.11+ development headers and are composed into
the `pyariadne` module. Set `-DBUILD_PYTHON_BINDINGS=OFF` to disable them.

## Validated interval-bound propagation (M6)

A separate `AriadneValidated` backend uses Ariadne Numeric's outwardly
rounded `FloatDPBounds` to enclose every output for a finite input box.
All model parameters and input endpoints are treated as exact binary64 values;
activation intervals use directed-rounding implementations for ReLU, tanh,
sigmoid and softplus. No approximate CPU evaluation is silently substituted.

```python
from pyariadne import (
    NeuralBackendKind, NeuralExecutionSession, NeuralInterval,
)
validated = NeuralExecutionSession(snapshot, NeuralBackendKind.AriadneValidated)
bounds = validated.validated_bounds([
    NeuralInterval(-1.0, 1.0), NeuralInterval(0.0, 2.0)
])
```

Only `ImportSnapshot` and `ValidatedBounds` capabilities are exposed for
`AriadneValidated`. Interval-bound propagation may overestimate the output
range because repeated variables lose dependencies. Nonfinite intermediate
bounds are rejected; no validated Jacobian, affine/Taylor enclosure or
tightness guarantee is claimed. Function/Symbolic integration will belong to
the future `neuro-symbolic` repository.

## Canonical model persistence (M5)

Ariadne Neural saves a frozen network in two open files: a versioned JSON
manifest describing the dense topology and activations, and a standard
Safetensors file containing named, row-major F64 weights and biases.
The binary64 parameter bits (including signed zero) are preserved exactly.
C++ and Python expose `neural_save_snapshot` and
`neural_load_snapshot` for this canonical format.

```python
neural_save_snapshot(snapshot, "model.json", "model.safetensors")
restored = neural_load_snapshot("model.json", "model.safetensors")
```

The current-only schema and strict size/type validation rules are recorded
in [the design journal](doc/neural_design.md). File pairs must be published together. Each manifest contains a canonical
SHA-256 snapshot digest (exposed as `neural_snapshot_sha256`) binding topology,
activations and exact binary64 parameter bits. The digest is a content
identifier, not an authenticated signature. There is no legacy manifest
version support, atomic two-file commit, ONNX import/export yet; optimizer state is stored separately
in a three-file training checkpoint.

## Resumable training checkpoints (M5)

A model snapshot contains **only topology and frozen weights**. To resume a
native CPU training session, save its complete optimizer state too:

```python
neural_save_training_checkpoint(
    trainer, "model.json", "model.safetensors", "optimizer.safetensors")
resumed = neural_load_training_checkpoint(
    "model.json", "model.safetensors", "optimizer.safetensors")
```

The optimizer Safetensors file holds the F64 moment buffers and exact
optimizer hyperparameters (binary64-encoded), loss, frozen-layer mask, step
count, model digest and independent state SHA-256 digest. A resumed SGD
momentum or AdamW session starts with the same update state rather than
resetting the moments. The files must all belong to the same checkpoint;
tampering and mix-and-match snapshots are rejected by hash checks, but hashes
do not authenticate who created them. Publishing three files atomically is
not supported. The checkpoint format is for the native CPU training
implementation and has no external-backend interoperability guarantee.

## Atomic checkpoint directory publication (M5)

For a new, immutable checkpoint, use the directory-based API instead of
writing multiple files directly:

```python
neural_save_training_checkpoint_atomic(trainer, "checkpoints/step-100")
resumed = neural_load_training_checkpoint_atomic("checkpoints/step-100")
```

The model-only equivalents are `neural_save_snapshot_atomic` and
`neural_load_snapshot_atomic`. The parent directory must exist and the
destination must **not** exist. Each saver writes a private staging
directory beside the destination, validates all files and then renames
the directory once. The files remain the same JSON/Safetensors format.
Previously published checkpoints are not intentionally overwritten;
use a new destination for subsequent steps.

Directory rename gives atomic visibility only when the underlying
filesystem supports it. This does not provide crash-durable persistence
or portable atomic no-replace semantics for concurrent writers; clients
must serialize writers to the same destination. Low-level multi-file
save APIs remain available but are not atomic.

## Performance benchmark

Build a Release executable to compare the same three training algorithms on
matched, seeded regression networks:

```sh
cmake -S . -B build-release -DCMAKE_BUILD_TYPE=Release
cmake --build build-release --parallel --target benchmark_optimizers
./build-release/benchmarks/benchmark_optimizers \
    --iterations 50 --samples 7 --batch-size 16 --width 32 \
    --seed 42 --seeds 3 --target-loss 0.001 --max-steps 500 \
    --sgd-lr 0.01 --sgd-momentum 0.9 --adamw-lr 0.01 --concurrency 0
```

The optimizer variants are **SGD**, **SGD with classical momentum**, and
**AdamW**. Each variant receives the same initialized 8 -> hidden -> 1 network
and the same fixed synthetic regression batch at each seed. The loss is the
half-squared-error sum over output features averaged across samples; it is
not the usual unscaled mean squared error.

Two experiments are measured independently. The *fixed-step* experiment
reports median nanoseconds per complete training update and the final loss
after `--iterations` updates. The *time-to-target* experiment reports
milliseconds and the number of updates needed to first reach
`--target-loss`, up to `--max-steps`. It includes an extra loss evaluation
after each update, so the timing figures from the two experiments must not
be compared as equivalent operation costs. A target not reached within
the budget is reported as **censored**, including its final loss; it is
not silently omitted or assigned an invented completion time.

Both experiments use one unmeasured warm-up, odd-sample median timing,
rotating optimizer execution order, and reproducibility checks. `--seeds`
runs paired experiments on consecutive initial seeds. The summary reports
hit counts for all seeds, but only summarizes time-to-target when **every**
paired seed reached the threshold. Each optimizer has configurable learning
rate, momentum (SGD only), and weight decay. Defaults do not imply equal
hyperparameter tuning.

Arguments, `--help`, logging and `--concurrency` are handled by Ariadne's
`ThreadingCommandLineInterface`. Native M4 training remains sequential;
only `--concurrency 0` is accepted. A lightweight Debug CI smoke test
checks the benchmark without imposing timing or convergence thresholds.
For meaningful comparisons record compiler, build flags and hardware.

## License

GNU General Public License v3.0; see [LICENSE](LICENSE).
