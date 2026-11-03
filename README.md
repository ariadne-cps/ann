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
sum or mean reductions.
Supported layer activations are identity, ReLU, tanh, sigmoid and softplus.
C++ and Python interfaces expose model metadata, parameter snapshots, backend
capability queries and CPU inference.

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

## License

GNU General Public License v3.0; see [LICENSE](LICENSE).
