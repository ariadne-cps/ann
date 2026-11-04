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
Fine-tuning supports per-layer trainability masks (with exact freezing of
unchanged parameters for SGD and AdamW) and deterministic replacement of the
final dense output head. A new training session resets optimizer state; masks
and optimizer state are not saved in JSON/Safetensors checkpoints.
C++ and Python interfaces expose model metadata, parameter snapshots, backend
capability queries, CPU inference, derivatives and training.

## Dependencies

Direct Git submodules, pinned to specific commits:

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

The version-1 schema and strict size/type validation rules are recorded
in [the design journal](doc/neural_design.md). File pairs must be published
together; no cryptographic model identity, atomic two-file commit,
optimizer-state checkpoint or ONNX import/export is implemented yet.

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
