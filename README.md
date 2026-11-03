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
C++ and Python interfaces expose model metadata, parameter snapshots, backend
capability queries, CPU inference, derivatives and training.

## Dependencies

Direct Git submodules, pinned to specific commits:

- [Function](https://github.com/ariadne-cps/function) (`ss/fun`).
- [Threading](https://github.com/ariadne-cps/threading) (`ss/thr`).
- [Configuration](https://github.com/ariadne-cps/configuration) (`ss/cfg`).
- [Testing](https://github.com/ariadne-cps/testing) (`ss/tst`, for tests).

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

## Performance benchmark

Build the native CPU optimizer benchmark separately from correctness tests:

```sh
cmake -S . -B build-release -DCMAKE_BUILD_TYPE=Release
cmake --build build-release --parallel --target benchmark_optimizers
./build-release/benchmarks/benchmark_optimizers \
    --iterations 50 --samples 7 --batch-size 16 --width 32 \
    --seed 42 --sgd-lr 0.01 --adamw-lr 0.01 --concurrency 0
```

The benchmark uses `ThreadingCommandLineInterface` (including logging,
`--help`, and `--concurrency`) and reports results through Ariadne logging.
Both optimizers train the **same seeded 8–32–1 network and synthetic MSE
dataset**. Each measured sample starts with identical frozen parameters;
a warm-up run is excluded, execution order alternates, and the median
nanoseconds per training step is reported along with the final loss after the
same number of steps. Use `--sgd-momentum` and `--weight-decay` to alter
optimizer options. `--width` changes the hidden size and `--batch-size`
changes mini-batch size; both apply to both optimizers.

This measures **end-to-end sequential training**, including forward/backward
passes and optimizer state/snapshot updates, rather than just optimizer update
arithmetic. The training implementation is not parallelized: only
`--concurrency 0` is accepted. Final losses reflect the configured learning
rates and are not a matched-quality comparison. The tiny CI smoke test
checks execution, not benchmark performance or a speedup threshold.

## License

GNU General Public License v3.0; see [LICENSE](LICENSE).
